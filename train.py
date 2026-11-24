"""DDP trainer. Launch: torchrun --nproc_per_node=N train.py --arch diff --lr 1.5e-3 --seed 0 --out /vol/runs/x
Data: uint16 token shards (*.bin) in --data_dir; val.bin held out. Per seed, all 1024-token windows are
globally shuffled; the global batch at step s is identical for every arm and world size.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

PEAK = {"H100": 989.5e12, "T4": 65e12, "A100": 312e12, "P100": 18.7e12, "cpu": 1e12}  # dense half-precision


def peak_flops(name):
    return next((v for k, v in PEAK.items() if k in name), float("nan"))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=["std", "qknorm", "diff"])
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data_dir", default="/data")
    p.add_argument("--max_steps", type=int, default=0, help="0 = use the whole training pool once")
    p.add_argument("--schedule_steps", type=int, default=0, help="LR schedule length (default max_steps)")
    p.add_argument("--batch_tokens", type=int, default=524288)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--micro_bs", type=int, default=32)
    p.add_argument("--warmup_frac", type=float, default=0.02)
    p.add_argument("--final_lr_frac", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_small_tokens", type=int, default=2_000_000)
    p.add_argument("--val_full_every", type=int, default=1000)
    p.add_argument("--val_tokens", type=int, default=20_000_000)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--deadline", type=float, default=0, help="unix time; checkpoint + PAUSED and exit when passed")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--mlp_hidden", type=int, default=1408)
    return p.parse_args()


class Data:
    def __init__(self, data_dir, seq, seed):
        self.seq = seq
        files = sorted(glob.glob(os.path.join(data_dir, "train_*.bin")))
        assert files, f"no train shards in {data_dir}"
        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        rng = np.random.default_rng(seed)
        idx = []
        for s, sh in enumerate(self.shards):
            nwin = (len(sh) - 1) // seq
            idx.append(np.stack([np.full(nwin, s, np.int64), np.arange(nwin, dtype=np.int64) * seq], 1))
        idx = np.concatenate(idx)
        self.idx = idx[rng.permutation(len(idx))]  # global window shuffle, fixed per seed: (shard, offset)
        self.files = files

    def batch(self, start, n):
        rows = self.idx[start:start + n]
        assert len(rows) == n, "ran out of training windows"
        buf = np.stack([self.shards[s][o:o + self.seq + 1] for s, o in rows]).astype(np.int64)
        t = torch.from_numpy(buf)
        return t[:, :-1], t[:, 1:]


def lr_at(step, total, peak, warmup_frac, final_frac):
    w = max(1, int(warmup_frac * total))
    if step < w:
        return peak * (step + 1) / w
    p = min(1.0, (step - w) / max(1, total - w))
    return peak * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model, val, ntok, seq, micro_bs, rank, world, device, dev_type, amp_dtype):
    """Mean loss over the first ntok tokens of val.bin, trimmed to whole (world*micro_bs) batches so the
    compiled graph never sees a ragged batch. Returns (loss, n_windows_used)."""
    per = world * micro_bs
    nwin = (min(ntok, len(val) - 1) // seq) // per * per
    mine = list(range(rank, nwin, world))
    tot = torch.zeros(2, device=device, dtype=torch.float64)
    for i in range(0, len(mine), micro_bs):
        ws = mine[i:i + micro_bs]
        buf = torch.from_numpy(np.stack([val[w * seq:w * seq + seq + 1] for w in ws]).astype(np.int64)).to(device)
        with torch.autocast(dev_type, dtype=amp_dtype):
            loss = model(buf[:, :-1], buf[:, 1:])
        tot[0] += loss.double() * len(ws)
        tot[1] += len(ws)
    if world > 1:
        dist.all_reduce(tot)
    return (tot[0] / tot[1]).item(), int(tot[1].item())


def main():
    a = get_args()
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    local = int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.cuda.is_available()
    device = f"cuda:{local}" if cuda else "cpu"
    if cuda:
        torch.cuda.set_device(device)
    dev_type = "cuda" if cuda else "cpu"
    amp_dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    gpu_name = torch.cuda.get_device_name() if cuda else "cpu"
    peak = peak_flops(gpu_name)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    per_step = a.batch_tokens // a.seq
    assert per_step % (a.micro_bs * world) == 0
    accum = per_step // (a.micro_bs * world)

    data = Data(a.data_dir, a.seq, a.seed)
    pool_steps = len(data.idx) // per_step
    max_steps = a.max_steps or pool_steps
    assert max_steps <= pool_steps, (max_steps, pool_steps)
    sched = a.schedule_steps or max_steps
    val = np.memmap(os.path.join(a.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    torch.manual_seed(a.seed)  # init depends on seed only, identical across ranks
    cfg = GPTConfig(arch=a.arch, seq_len=a.seq, d_model=a.d_model, n_layer=a.n_layer, n_head=a.n_head,
                    mlp_hidden=a.mlp_hidden)
    model = GPT(cfg).to(device)
    n_params, fpt, hwfpt = model.n_params(), model.flops_per_token(), model.hw_flops_per_token()

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=cuda)
    scaler = torch.amp.GradScaler(dev_type, enabled=(a.dtype == "fp16"))

    os.makedirs(a.out, exist_ok=True)
    ck = os.path.join(a.out, "ckpt.pt")
    step = 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if "scaler" in st:
            scaler.load_state_dict(st["scaler"])
        step = st["step"]
        if rank == 0:
            print(f"resumed from step {step}", flush=True)
    raw = model
    if a.compile:
        model = torch.compile(model)
    cmodel = model  # compiled, un-wrapped: used for eval
    if ddp:
        model = DDP(model, device_ids=[local])

    logf = open(os.path.join(a.out, "log.jsonl"), "a") if rank == 0 else None
    if rank == 0:
        meta = dict(vars(a), resumed_from=step, n_params=n_params, flops_per_token=fpt, hw_flops_per_token=hwfpt, world=world,
                    accum=accum, max_steps=max_steps, schedule_steps=sched, pool_steps=pool_steps,
                    train_tokens=max_steps * a.batch_tokens, gpu=gpu_name, peak_flops=peak,
                    torch=torch.__version__, shards=[os.path.basename(f) for f in data.files])
        json.dump(meta, open(os.path.join(a.out, f"meta_from{step}.json"), "w"), indent=1)
        print(json.dumps(meta)[:2000], flush=True)

    def log(d):
        if rank == 0:
            logf.write(json.dumps(d) + "\n")
            logf.flush()
            print(json.dumps(d), flush=True)

    t0, tok_since = time.time(), 0
    loss_acc = torch.zeros((), device=device)
    n_ok = torch.zeros((), device=device)  # steps in this log window whose loss was finite
    n_win = 0  # steps in this log window
    diverged = False
    hit_deadline = False
    while step < max_steps:
        lr = lr_at(step, sched, a.lr, a.warmup_frac, a.final_lr_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        base = step * per_step
        step_loss = torch.zeros((), device=device)
        for m in range(accum):
            x, y = data.batch(base + (m * world + rank) * a.micro_bs, a.micro_bs)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if ddp:
                model.require_backward_grad_sync = (m == accum - 1)
            with torch.autocast(dev_type, dtype=amp_dtype):
                loss = model(x, y) / accum
            scaler.scale(loss).backward()
            step_loss += loss.detach()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), a.clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        n_win += 1
        tok_since += a.batch_tokens
        fin = torch.isfinite(step_loss)  # fp16: an overflowed step is skipped by GradScaler, not fatal
        loss_acc += torch.where(fin, step_loss, torch.zeros_like(step_loss))
        n_ok += fin.float()

        last = step == max_steps
        if step % a.log_every == 0 or last:
            if ddp:
                dist.all_reduce(loss_acc)
                dist.all_reduce(n_ok)
            ok = n_ok.item()
            train_loss = loss_acc.item() / ok if ok > 0 else float("nan")
            skipped = n_win * world - ok
            loss_acc.zero_()
            n_ok.zero_()
            n_win = 0
            if cuda:
                torch.cuda.synchronize()
            dt = time.time() - t0
            tps = tok_since / dt
            full = last or step % a.val_full_every == 0
            vl, vwin = evaluate(cmodel, val, a.val_tokens if full else a.val_small_tokens, a.seq, a.micro_bs, rank,
                              world, device, dev_type, amp_dtype)
            rec = dict(step=step, tokens=step * a.batch_tokens, lr=lr, train_loss=train_loss, grad_norm=gn.item(),
                       tok_per_s=tps, mfu=tps * fpt / (world * peak),
                       hw_mfu=tps * hwfpt / (world * peak), loss_scale=scaler.get_scale(),
                       nonfinite_steps=skipped, lambdas=raw.lambdas())
            rec["val_loss_full" if full else "val_loss_2m"] = vl
            rec["val_windows"] = vwin
            # diverged: no finite step in the window, non-finite val loss, or fp16 loss scale collapsed
            diverged = (not math.isfinite(train_loss)) or (not math.isfinite(vl)) or \
                (a.dtype == "fp16" and scaler.get_scale() < 1.0)
            if diverged:
                rec["diverged"] = True
            log(rec)
            if diverged:
                break
            t0, tok_since = time.time(), 0
        if a.deadline and not last:
            flag = torch.tensor(float(time.time() > a.deadline), device=device)
            if ddp:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)  # all ranks stop on the same step
            if flag.item() > 0:
                hit_deadline = True
                break
        if rank == 0 and (step % a.ckpt_every == 0) and not last:
            torch.save({"model": raw.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    if rank == 0 and not diverged and (step < sched or hit_deadline):
        # partial run (e.g. throughput pilot): leave a resumable checkpoint, do not mark DONE
        torch.save({"model": raw.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        open(os.path.join(a.out, "PAUSED"), "w").write(str(step))
    elif rank == 0:
        torch.save({"model": raw.state_dict(), "step": step, "cfg": vars(cfg)}, os.path.join(a.out, "final.pt"))
        for f in (ck, os.path.join(a.out, "PAUSED")):
            if os.path.exists(f):
                os.remove(f)
        open(os.path.join(a.out, "DIVERGED" if diverged else "DONE"), "w").write(str(step))
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
