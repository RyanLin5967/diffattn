"""Pre-registered analysis (docs/prereg.md). Reads runs/<name>/{log.jsonl,meta_from0.json,evals.json,DONE}
and writes analysis/results.json + results.md + figures. Refuses (exit 1) on missing or unfinished runs.

  python3 analysis/analyze.py sweep  <runs_dir>   -> best LR per arm from the sweep
  python3 analysis/analyze.py main   <runs_dir>   -> full table, tests, figures
"""
import glob
import itertools
import json
import math
import os
import sys

import numpy as np

ARMS = ["std", "qknorm", "diff"]
HERE = os.path.dirname(os.path.abspath(__file__))


def load_run(d):
    recs = {}
    for l in open(os.path.join(d, "log.jsonl")):  # a resumed run may re-log steps: keep the last record
        r = json.loads(l)
        recs[r["step"]] = r
    log = [recs[k] for k in sorted(recs)]
    meta = json.load(open(os.path.join(d, "meta_from0.json")))
    full = [r for r in log if "val_loss_full" in r]
    ev = os.path.join(d, "evals.json")
    return dict(name=os.path.basename(d), meta=meta, log=log, done=os.path.exists(os.path.join(d, "DONE")),
                diverged=os.path.exists(os.path.join(d, "DIVERGED")) or any(r.get("diverged") for r in log),
                final_val=full[-1]["val_loss_full"] if full else float("nan"),
                final_step=log[-1]["step"] if log else 0,
                evals=json.load(open(ev)) if os.path.exists(ev) else None)


def runs(root, prefix):
    out = [load_run(d) for d in sorted(glob.glob(os.path.join(root, prefix + "*"))) if
           os.path.exists(os.path.join(d, "log.jsonl"))]
    if not out:
        sys.exit(f"REFUSE: no runs matching {prefix}* in {root}")
    return out


def welch(a, b):
    from scipy import stats
    t = stats.ttest_ind(a, b, equal_var=False)
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = np.mean(a) - np.mean(b)
    h = stats.t.ppf(0.975, df) * math.sqrt(va + vb)
    return dict(diff=diff, ci95=[diff - h, diff + h], t=float(t.statistic), df=df, p=float(t.pvalue))


def paired_perm(a, b):
    """Exact two-sided sign-flip test on per-seed differences (a_i - b_i)."""
    d = np.asarray(a) - np.asarray(b)
    obs = abs(d.mean())
    stats_ = [abs((d * np.array(s)).mean()) for s in itertools.product([1, -1], repeat=len(d))]
    p = float(np.mean([s >= obs - 1e-15 for s in stats_]))
    return dict(mean_diff=float(d.mean()), p=p, n_patterns=len(stats_), min_attainable_p=2 / len(stats_))


def mde(sd_a, sd_b, n, alpha=0.05, power=0.8):
    from scipy import stats
    df = 2 * n - 2
    return float((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * math.sqrt((sd_a ** 2 + sd_b ** 2) / n))


def sweep(root):
    rs = runs(root, "sweep_")
    best = {}
    rows = []
    for r in rs:
        arm, lr = r["meta"]["arch"], r["meta"]["lr"]
        ok = r["done"] and not r["diverged"] and math.isfinite(r["final_val"])
        rows.append(dict(arm=arm, lr=lr, final_val=r["final_val"], done=r["done"], diverged=r["diverged"]))
        if ok and (arm not in best or r["final_val"] < best[arm]["final_val"]):
            best[arm] = dict(lr=lr, final_val=r["final_val"])
    for arm in ARMS:
        lrs = sorted(x["lr"] for x in rows if x["arm"] == arm)
        if arm in best and lrs:
            best[arm]["at_edge"] = best[arm]["lr"] in (lrs[0], lrs[-1])
    out = dict(rows=sorted(rows, key=lambda x: (x["arm"], x["lr"])), best=best)
    json.dump(out, open(os.path.join(HERE, "sweep.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    missing = [a for a in ARMS if a not in best]
    if missing:
        sys.exit(f"REFUSE: no finished sweep run for {missing}")


def main(root):
    rs = runs(root, "main_")
    by = {a: sorted([r for r in rs if r["meta"]["arch"] == a], key=lambda r: r["meta"]["seed"]) for a in ARMS}
    unfinished = [r["name"] for r in rs if not r["done"] and not r["diverged"]]
    if unfinished:
        sys.exit(f"REFUSE: unfinished main runs: {unfinished}")
    failed = {a: [r["meta"]["seed"] for r in by[a] if r["diverged"]] for a in ARMS}
    # pre-registered (prereg amendment 2): diverged runs are reported per arm and excluded from the loss
    # tests; Welch uses each arm's finished runs, the paired test uses seeds finished in both arms.
    ok = {a: {r["meta"]["seed"]: r for r in by[a] if r["done"] and not r["diverged"]} for a in ARMS}
    v = {a: np.array([ok[a][s]["final_val"] for s in sorted(ok[a])]) for a in ARMS}
    by = {a: [ok[a][s] for s in sorted(ok[a])] for a in ARMS}
    seeds = {a: sorted(ok[a]) for a in ARMS}
    if any(len(v[a]) < 2 for a in ARMS):
        sys.exit(f"REFUSE: fewer than 2 finished runs in some arm: {seeds}")
    m0 = by["std"][0]["meta"]
    res = dict(seeds_finished=seeds, failed_runs=failed,
               setup=dict(n_params={a: by[a][0]["meta"]["n_params"] for a in ARMS},
                          flops_per_token={a: by[a][0]["meta"]["flops_per_token"] for a in ARMS},
                          hw_flops_per_token={a: by[a][0]["meta"]["hw_flops_per_token"] for a in ARMS},
                          lr={a: by[a][0]["meta"]["lr"] for a in ARMS},
                          train_tokens=m0["train_tokens"], batch_tokens=m0["batch_tokens"], seq=m0["seq"],
                          gpu=m0["gpu"]),
               per_arm={a: dict(vals=v[a].tolist(), mean=float(v[a].mean()), sd=float(v[a].std(ddof=1)))
                        for a in ARMS})
    pairs = [("std", "diff"), ("std", "qknorm"), ("qknorm", "diff")]
    def paired(a, b):
        common = sorted(set(seeds[a]) & set(seeds[b]))
        return paired_perm([ok[a][s]["final_val"] for s in common], [ok[b][s]["final_val"] for s in common])
    res["tests"] = {f"{a}-{b}": dict(welch=welch(v[a], v[b]), paired_perm=paired(a, b),
                                     mde_80pct_power=mde(v[a].std(ddof=1), v[b].std(ddof=1),
                                                         min(len(v[a]), len(v[b]))))
                    for a, b in pairs}
    gain = v["std"].mean() - v["diff"].mean()
    rec = (v["std"].mean() - v["qknorm"].mean()) / gain if gain != 0 else float("nan")
    # paired seed bootstrap (seeds share data order across arms): resample seed indices jointly, keep every
    # resample; percentile CIs of the gain G and of the qknorm gain Q, and of R = Q/G conditional on G > 0
    common = sorted(set(seeds["std"]) & set(seeds["qknorm"]) & set(seeds["diff"]))
    M = {a: np.array([ok[a][s]["final_val"] for s in common]) for a in ARMS}
    rng = np.random.default_rng(0)
    G, Q = [], []
    for _ in range(20000):
        ix = rng.integers(0, len(common), len(common))
        G.append(M["std"][ix].mean() - M["diff"][ix].mean())
        Q.append(M["std"][ix].mean() - M["qknorm"][ix].mean())
    G, Q = np.array(G), np.array(Q)
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    boots = Q[G > 0] / G[G > 0]
    res["diffattn_gain_over_std"] = float(gain)
    res["qknorm_recovery_frac"] = float(rec)
    res["gain_boot95"] = pct(G)
    res["qknorm_gain_boot95"] = pct(Q)
    res["qknorm_recovery_boot95_given_gain_pos"] = pct(boots) if len(boots) > 1000 else None
    res["boot_frac_gain_positive"] = float((G > 0).mean())
    # downstream + needle
    tasks = ["hellaswag", "arc_easy", "piqa", "lambada"]
    ds = {}
    for a in ARMS:
        evs = [r["evals"] for r in by[a] if r["evals"]]
        if len(evs) != len(by[a]):
            sys.exit(f"REFUSE: missing evals.json for arm {a}")
        ds[a] = {t: dict(mean=float(np.mean([e[t]["acc_norm" if t in ("hellaswag", "arc_easy") else "acc"]
                                              for e in evs])),
                         sd=float(np.std([e[t]["acc_norm" if t in ("hellaswag", "arc_easy") else "acc"]
                                          for e in evs], ddof=1))) for t in tasks}
        ds[a]["needle_answer_share"] = float(np.mean([e["needle"]["answer_share_mean"] for e in evs]))
        ds[a]["needle_answer_share_per_layer"] = np.mean([e["needle"]["answer_share_per_layer"] for e in evs],
                                                         0).tolist()
        ds[a]["needle_answer_mass"] = float(np.mean([e["needle"]["answer_mass_mean"] for e in evs]))
        ds[a]["needle_noise_mass"] = float(np.mean([e["needle"]["noise_mass_mean"] for e in evs]))
        ds[a]["needle_retrieval_acc"] = float(np.mean([e["needle"]["retrieval_acc"] for e in evs]))
        ds[a]["needle_answer_mass_per_layer"] = np.mean([e["needle"]["answer_mass_per_layer"] for e in evs],
                                                        0).tolist()
        ds[a]["needle_noise_mass_per_layer"] = np.mean([e["needle"]["noise_mass_per_layer"] for e in evs],
                                                       0).tolist()
    res["downstream"] = ds
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    figures(by)
    print(json.dumps({k: res[k] for k in ("per_arm", "tests", "diffattn_gain_over_std", "qknorm_recovery_frac",
                                          "gain_boot95", "qknorm_recovery_boot95_given_gain_pos")}, indent=1))


def figures(by):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    col = dict(std="#888888", qknorm="#1f77b4", diff="#d62728")
    for a in ARMS:
        # two different val sets are logged (2M-token prefix at most steps, full 10M set at val_full_every
        # and the last step); never join them in one line
        for key, style in (("val_loss_2m", "line"), ("val_loss_full", "marker")):
            curves = [[(x["tokens"], x[key]) for x in r["log"] if key in x] for r in by[a]]
            n = min(len(c) for c in curves)
            t = np.array([p[0] for p in curves[0][:n]]) / 1e6
            y = np.array([[p[1] for p in c[:n]] for c in curves])
            if style == "line":
                ax[0].plot(t, y.mean(0), color=col[a], label=f"{a} (2M-token val prefix)")
                ax[0].fill_between(t, y.min(0), y.max(0), color=col[a], alpha=0.25)
            else:
                ax[0].errorbar(t, y.mean(0), yerr=y.std(0, ddof=1), fmt="o", ms=4, color=col[a], capsize=3,
                               label=f"{a} (full 10M val, mean±sd)")
    ax[0].set_xlabel("tokens (M)")
    ax[0].set_ylabel("val loss")
    ax[0].set_ylim(None, np.nanmean([r["final_val"] for r in by["std"]]) + 0.6)
    ax[0].legend(fontsize=6)
    lam = np.array([[x["lambdas"] for x in r["log"]] for r in by["diff"]])  # seeds, logs, layers
    t = np.array([x["tokens"] for x in by["diff"][0]["log"]]) / 1e6
    for li in range(lam.shape[2]):
        ax[1].plot(t, lam[:, :, li].mean(0), label=f"L{li}")
    ax[1].set_xlabel("tokens (M)")
    ax[1].set_ylabel("lambda (mean over seeds)")
    ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "curves.png"), dpi=150)


if __name__ == "__main__":
    {"sweep": sweep, "main": main}[sys.argv[1]](sys.argv[2])
