"""Post-training evals for one run dir: zero-shot HellaSwag / ARC-Easy / PIQA / LAMBADA, plus the
attention-noise probe (attention mass on the answer span vs the rest of the context in a key-retrieval
prompt; for DiffAttn the effective map is A1 - lambda*A2). Writes <run>/evals.json."""
import argparse
import json
import os

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, apply_rope

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 on CPU and on sm80+; fp16 on older GPUs (Kaggle T4)
AMP = torch.bfloat16 if DEV == "cpu" or torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def load(run, device):
    st = torch.load(os.path.join(run, "final.pt"), map_location=device)
    m = GPT(GPTConfig(**st["cfg"])).to(device)
    m.load_state_dict(st["model"])
    return m.eval()


@torch.no_grad()
def score(m, items, device, bs=64):
    """Per item: sum and mean log-prob of each continuation; LAMBADA-style greedy exact match too."""
    seqs = []
    for i, it in enumerate(items):
        for j, c in enumerate(it["conts"]):
            s = (it["ctx"] + c)[-m.cfg.seq_len:]
            seqs.append((i, j, s, len(c)))
    res = {}
    for b in range(0, len(seqs), bs):
        chunk = seqs[b:b + bs]
        L = max(len(s) for _, _, s, _ in chunk)
        x = torch.full((len(chunk), L), 50256, dtype=torch.long)
        for r, (_, _, s, _) in enumerate(chunk):
            x[r, :len(s)] = torch.tensor(s)
        x = x.to(device)
        with torch.autocast(DEV, dtype=AMP):
            logits = m(x).float()
        lp = F.log_softmax(logits[:, :-1], -1)
        for r, (i, j, s, nc) in enumerate(chunk):
            pos = torch.arange(len(s) - nc - 1, len(s) - 1, device=device)
            tgt = x[r, pos + 1]
            tok_lp = lp[r, pos, tgt]
            greedy = bool((lp[r, pos].argmax(-1) == tgt).all())
            res[(i, j)] = (tok_lp.sum().item(), tok_lp.mean().item(), greedy)
    out = {}
    n = len(items)
    acc = sum(int(np.argmax([res[(i, j)][0] for j in range(len(it["conts"]))]) == it["label"])
              for i, it in enumerate(items))
    accn = sum(int(np.argmax([res[(i, j)][1] for j in range(len(it["conts"]))]) == it["label"])
               for i, it in enumerate(items))
    out.update(n=n, acc=acc / n, acc_norm=accn / n)
    if all(len(it["conts"]) == 1 for it in items):  # LAMBADA
        out = dict(n=n, acc=sum(res[(i, 0)][2] for i in range(n)) / n,
                   ppl=float(np.exp(-np.mean([res[(i, 0)][0] for i in range(n)]))))
    return out


@torch.no_grad()
def last_row_attention(m, idx):
    """Per layer, head-averaged attention of the LAST query position over all keys: (n_layer, T)."""
    x = m.wte(idx)
    rows = []
    for blk in m.blocks:
        h = blk.n1(x)
        a = blk.attn
        B, T, C = h.shape
        if m.cfg.arch == "diff":
            q = a.q(h).view(B, T, 2 * a.h, a.hd).transpose(1, 2)
            k = a.k(h).view(B, T, 2 * a.h, a.hd).transpose(1, 2)
            q, k = apply_rope(q, m.cos, m.sin), apply_rope(k, m.cos, m.sin)
            s = (q[:, :, -1:] @ k.transpose(-1, -2)).float() / a.hd ** 0.5  # (B, 2H, 1, T)
            A = torch.softmax(s, -1).view(B, a.h, 2, T)
            eff = A[:, :, 0] - a.lam() * A[:, :, 1]
        else:
            q, k, _ = a.qkv(h).view(B, T, 3, a.h, a.hd).permute(2, 0, 3, 1, 4)
            if a.qk_norm:
                q, k = a.q_norm(q), a.k_norm(k)
            q, k = apply_rope(q, m.cos, m.sin), apply_rope(k, m.cos, m.sin)
            s = (q[:, :, -1:] @ k.transpose(-1, -2)).float() / a.hd ** 0.5
            eff = torch.softmax(s, -1)[:, :, 0]
        rows.append(eff.mean(1)[0])
        x = blk(x, m.cos, m.sin)
    return torch.stack(rows)


NAMES = ["Alice", "Bruno", "Chen", "Dara", "Elif", "Farid", "Gita", "Hugo", "Ines", "Jonas"]


@torch.no_grad()
def needle_probe(m, val, device, n=200, ctx_len=900):
    """Key-retrieval prompt: filler from held-out val text, one needle sentence at a random depth, then
    the query. Measures (head-avg, per layer) effective attention mass of the final position on the
    needle's number tokens ('answer') vs all other context tokens excluding the first token and the
    query ('noise'), and greedy retrieval accuracy of the first number token."""
    enc = tiktoken.get_encoding("gpt2")
    rng = np.random.default_rng(1234)  # identical probe set for every run
    ans_mass, noise_mass, hits = [], [], 0
    for t in range(n):
        name = NAMES[t % len(NAMES)]
        num = str(rng.integers(1000, 9999))
        pre = enc.encode_ordinary(f" The secret number of {name} is")
        numtok = enc.encode_ordinary(" " + num)
        needle = pre + numtok + enc.encode_ordinary(".")
        query = enc.encode_ordinary(f" What is the secret number of {name}? The secret number of {name} is")
        start = int(rng.integers(0, len(val) - ctx_len - 1))
        filler = [int(z) for z in val[start:start + ctx_len - len(needle) - len(query)]]
        d = int(rng.integers(0, len(filler)))
        toks = filler[:d] + needle + filler[d:] + query
        apos = list(range(d + len(pre), d + len(pre) + len(numtok)))
        idx = torch.tensor([toks], device=device)
        att = last_row_attention(m, idx)  # (L, T)
        T = len(toks)
        mask_noise = torch.ones(T, dtype=torch.bool, device=device)
        mask_noise[0] = False
        mask_noise[apos] = False
        mask_noise[T - len(query):] = False
        ans_mass.append(att[:, apos].sum(1).cpu().numpy())
        noise_mass.append(att[:, mask_noise].sum(1).cpu().numpy())
        with torch.autocast(DEV, dtype=AMP):
            pred = m(idx)[0, -1].argmax().item()
        hits += int(pred == numtok[0])
    am, nm = np.mean(ans_mass, 0), np.mean(noise_mass, 0)
    return dict(n=n, ctx_len=ctx_len, retrieval_acc=hits / n, answer_mass_per_layer=am.tolist(),
                noise_mass_per_layer=nm.tolist(), answer_mass_mean=float(am.mean()),
                noise_mass_mean=float(nm.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data_dir", default="/data")
    a = ap.parse_args()
    device = DEV
    m = load(a.run, device)
    out = {}
    for task in ("hellaswag", "arc_easy", "piqa", "lambada"):
        items = json.load(open(os.path.join(a.data_dir, "evals", f"{task}.json")))
        out[task] = score(m, items, device)
        print(task, out[task], flush=True)
    val = np.memmap(os.path.join(a.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    out["needle"] = needle_probe(m, val, device, ctx_len=min(900, int(0.88 * m.cfg.seq_len)))
    print("needle", {k: v for k, v in out["needle"].items() if "per_layer" not in k}, flush=True)
    json.dump(out, open(os.path.join(a.run, "evals.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
