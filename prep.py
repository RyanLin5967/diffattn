"""Tokenize one FineWeb-Edu sample-10BT parquet file (GPT-2 BPE, EOT before each doc) into
train_00.bin + val.bin (first VAL_TOKENS worth of documents, disjoint from train), and tokenize the
zero-shot eval sets into evals/*.json. Runs in a CPU container."""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
import requests
import tiktoken
from huggingface_hub import hf_hub_download

EOT = 50256
VAL_TOKENS = 10_000_000


def tok_parquet(out):
    enc = tiktoken.get_encoding("gpt2")
    path = hf_hub_download("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet", repo_type="dataset",
                           local_dir="/tmp/hf")
    pf = pq.ParquetFile(path)
    val, train = [], []
    nval = ntr = ndoc_val = ndoc_tr = 0
    for rb in pf.iter_batches(batch_size=8192, columns=["text"]):
        texts = rb.column("text").to_pylist()
        for ids in enc.encode_ordinary_batch(texts, num_threads=8):
            arr = np.array([EOT] + ids, dtype=np.uint16)
            if nval < VAL_TOKENS:
                val.append(arr)
                nval += len(arr)
                ndoc_val += 1
            else:
                train.append(arr)
                ntr += len(arr)
                ndoc_tr += 1
    np.concatenate(val).tofile(os.path.join(out, "val.bin"))
    np.concatenate(train).tofile(os.path.join(out, "train_00.bin"))
    meta = dict(source="HuggingFaceFW/fineweb-edu sample/10BT/000_00000.parquet", tokenizer="gpt2",
                val_tokens=nval, val_docs=ndoc_val, train_tokens=ntr, train_docs=ndoc_tr)
    json.dump(meta, open(os.path.join(out, "data_meta.json"), "w"), indent=1)
    print(meta, flush=True)


def jl(url):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return [json.loads(x) for x in r.text.splitlines() if x.strip()]


def pq_rows(repo, fname):
    p = hf_hub_download(repo, fname, repo_type="dataset", local_dir="/tmp/hf_evals")
    return pq.read_table(p).to_pylist()


def evals(out):
    enc = tiktoken.get_encoding("gpt2")
    E = lambda s: enc.encode_ordinary(s)  # noqa: E731
    os.makedirs(os.path.join(out, "evals"), exist_ok=True)
    tasks = {}

    hs = pq_rows("Rowan/hellaswag", "data/validation-00000-of-00001.parquet")
    tasks["hellaswag"] = [dict(ctx=E(x["ctx"]), conts=[E(" " + e) for e in x["endings"]], label=int(x["label"]))
                          for x in hs]

    items = []
    for x in pq_rows("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet"):
        labels = list(x["choices"]["label"])
        if x["answerKey"] not in labels:
            continue
        items.append(dict(ctx=E(f"Question: {x['question']}\nAnswer:"),
                          conts=[E(" " + t) for t in x["choices"]["text"]], label=labels.index(x["answerKey"])))
    tasks["arc_easy"] = items

    pv = jl("https://yonatanbisk.com/piqa/data/valid.jsonl")
    pl = requests.get("https://yonatanbisk.com/piqa/data/valid-labels.lst", timeout=120).text.split()
    assert len(pv) == len(pl)
    tasks["piqa"] = [dict(ctx=E(f"Question: {x['goal']}\nAnswer:"), conts=[E(" " + x["sol1"]), E(" " + x["sol2"])],
                          label=int(lab)) for x, lab in zip(pv, pl)]

    items = []
    for x in pq_rows("EleutherAI/lambada_openai", "default/test/default.parquet"):
        ctx, word = x["text"].rsplit(" ", 1)
        items.append(dict(ctx=E(ctx), conts=[E(" " + word)], label=0))
    tasks["lambada"] = items

    for k, v in tasks.items():
        assert len(v) > 500, (k, len(v))
        json.dump(v, open(os.path.join(out, "evals", f"{k}.json"), "w"))
        print(k, len(v), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--evals_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.evals_only:
        tok_parquet(a.out)
    evals(a.out)
