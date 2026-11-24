"""Generate Kaggle plans for the sweep and main runs from the pre-registered budget.
  python3 make_plans.py sweep <main_tokens>
  python3 make_plans.py main  <main_tokens> <lr_std> <lr_qknorm> <lr_diff> <seeds e.g. 0,1,2> <slug>
Lanes are balanced greedily by estimated cost (diff ~ the measured slowdown factor)."""
import json
import sys

BATCH = 65536
COMMON = ["--dtype", "fp16", "--batch_tokens", str(BATCH), "--micro_bs", "MB", "--log_every", "100",
          "--val_small_tokens", "2000000", "--val_full_every", "1000", "--val_tokens", "10000000",
          "--ckpt_every", "1000000"]
COST = {"std": 1.0, "qknorm": 1.0, "diff": 1.15}  # replaced by pilot ratios in amendment if different


def spec(name, arch, lr, seed, steps, mb, evals):
    args = [mb if a == "MB" else a for a in COMMON] + ["--arch", arch, "--lr", str(lr), "--seed", str(seed),
                                                       "--max_steps", str(steps)]
    return {"name": name, "args": args, "evals": evals, "cost": COST[arch] * steps}


def lanes(specs, n=2):
    out, load = [[] for _ in range(n)], [0.0] * n
    for s in sorted(specs, key=lambda s: -s["cost"]):
        i = load.index(min(load))
        out[i].append(s)
        load[i] += s["cost"]
    return out


if __name__ == "__main__":
    kind, tokens = sys.argv[1], int(float(sys.argv[2]))
    mb = "16"
    steps = tokens // BATCH
    if kind == "sweep":
        sw = max(1, steps // 4)
        specs = [spec(f"sweep_{a}_lr{lr:g}", a, lr, 100, sw, mb, False) for a in ("std", "qknorm", "diff")
                 for lr in (1e-3, 2e-3, 4e-3)]
        plan = {"slug": "diffattn-sweep", "pre": [], "lanes": lanes(specs)}
        json.dump(plan, open("plans/sweep.json", "w"), indent=1)
    else:
        lr = dict(std=float(sys.argv[3]), qknorm=float(sys.argv[4]), diff=float(sys.argv[5]))
        seeds = [int(x) for x in sys.argv[6].split(",")]
        specs = [spec(f"main_{a}_s{s}", a, lr[a], s, steps, mb, True) for s in seeds for a in lr]
        plan = {"slug": sys.argv[7], "pre": [], "lanes": lanes(specs)}
        json.dump(plan, open(f"plans/{sys.argv[7]}.json", "w"), indent=1)
    for i, l in enumerate(plan["lanes"]):
        print(f"lane{i}: {len(l)} runs, cost {sum(s['cost'] for s in l):.0f} step-units", [s["name"] for s in l])
