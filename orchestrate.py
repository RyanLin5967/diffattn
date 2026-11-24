"""In-kernel orchestration on a 2-GPU Kaggle session. Two stages:

  python orchestrate.py A <out_dir> <main_hours>
      unit tests + GPU parity -> throughput pilot per arm -> pre-registered budget formula ->
      LR sweep (+ edge extension) -> LR selection. Writes budget.json, sweep.json.
  python orchestrate.py B <out_dir> <budget.json> <sweep.json> <arch:seed,arch:seed,...>
      main runs (best LR per arm, given seeds, full budget) with evals.

Budget formula (pre-registered in docs/prereg.md, amendment 1): choose main steps S so that the
slower of the two GPU lanes finishes all 15 main runs in <= main_hours, using the measured pilot
throughput per arm, +4% for periodic small-val evals, +measured compile/startup and eval time per run.
Sweep runs use S // 4 steps.
"""
import json
import math
import os
import subprocess
import sys
import time

DATA = os.environ.get("ORCH_DATA", "/tmp/data")
BATCH = 65536
COMMON = ["--dtype", "fp16", "--batch_tokens", str(BATCH), "--log_every", "100", "--val_small_tokens", "500000",
          "--val_full_every", "1000", "--val_tokens", "10000000", "--ckpt_every", "500"]
SESSION_START = time.time()
DEADLINE = SESSION_START + float(os.environ.get("SESSION_HOURS", "11.3")) * 3600  # Kaggle cap is 12h
ARMS = ["std", "qknorm", "diff"]
SWEEP_LRS = [1e-3, 2e-3, 4e-3]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


EXTRA = os.environ.get("ORCH_EXTRA", "").split()  # CPU dry-run only: tiny model overrides


def train_cmd(out, args):
    return [sys.executable, "train.py", "--data_dir", DATA, "--out", out] + [str(x) for x in args] + EXTRA


def run_lanes(lanes, root):
    """lanes: list (one per GPU) of lists of jobs {name, args, evals}. Each lane runs sequentially."""
    def lane_proc(i, jobs):
        lines = ["worst=0"]
        for j in jobs:
            o = os.path.join(root, j["name"])
            lines.append(f"mkdir -p {o} && touch {o}/lane_start")
            # skip finished runs (resume across sessions); stop starting runs after the deadline
            lines.append(f"if [ ! -f {o}/DONE ] && [ ! -f {o}/DIVERGED ] && [ $(date +%s) -lt {int(DEADLINE)} ]; then "
                         + " ".join(train_cmd(o, j["args"] + ["--deadline", DEADLINE]))
                         + f" >> {o}/stdout.txt 2>&1; r=$?; echo train_rc=$r >> {o}/rc.txt; [ $r -ne 0 ] && worst=$r; fi")
            if j.get("evals"):
                lines.append(f"if [ -f {o}/final.pt ] && [ ! -f {o}/evals.json ]; then {sys.executable} evals.py "
                             f"--run {o} --data_dir {DATA} >> {o}/evals_stdout.txt 2>&1; r=$?; "
                             f"echo eval_rc=$r >> {o}/rc.txt; [ $r -ne 0 ] && worst=$r; fi")
        lines.append("exit $worst")
        path = f"/tmp/lane_{os.path.basename(root)}_{i}.sh"
        open(path, "w").write("\n".join(lines) + "\n")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        return subprocess.Popen(["bash", path], env=env)
    procs = [lane_proc(i, jobs) for i, jobs in enumerate(lanes) if jobs]
    while any(p.poll() is None for p in procs):
        time.sleep(120)
        for j in sum(lanes, []):
            f = os.path.join(root, j["name"], "log.jsonl")
            if os.path.exists(f):
                ls = open(f).read().strip().splitlines()
                if ls:
                    r = json.loads(ls[-1])
                    log(j["name"], "step", r["step"], "train", round(r["train_loss"], 4), "tok/s", int(r["tok_per_s"]))
    rcs = [p.returncode for p in procs]
    log("lanes done, rc", rcs)
    return rcs


def last_log(d):
    f = os.path.join(d, "log.jsonl")
    if not os.path.exists(f):
        return None
    ls = open(f).read().strip().splitlines()
    return [json.loads(x) for x in ls] if ls else None


def balance(jobs, cost, n=2):
    lanes, load = [[] for _ in range(n)], [0.0] * n
    for j in sorted(jobs, key=lambda j: -cost(j)):
        i = load.index(min(load))
        lanes[i].append(j)
        load[i] += cost(j)
    return lanes, load


def final_val(d):
    lg = last_log(d) or []
    full = [r for r in lg if "val_loss_full" in r]
    ok = os.path.exists(os.path.join(d, "DONE")) and full and math.isfinite(full[-1]["val_loss_full"])
    return full[-1]["val_loss_full"] if ok else float("inf")


def stage_a(out, main_hours):
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    rc = 0 if os.environ.get("ORCH_DRYRUN") else \
        subprocess.run("python -m pytest -q tests && python tests/gpu_parity.py", shell=True).returncode
    assert rc == 0, "unit tests / GPU parity failed"
    # ---- pilot: 60 steps per arm (2 GPUs), micro_bs 16, fallback 8 on failure
    pilot = os.path.join(out, "pilot")
    mb = {}
    for try_mb in (16, 8):
        todo = [a for a in ARMS if a not in mb]
        if not todo:
            break
        jobs = [{"name": f"{a}_mb{try_mb}", "args": COMMON + ["--arch", a, "--lr", "2e-3", "--seed", "100",
                                                            "--micro_bs", try_mb, "--max_steps", 60,
                                                            "--schedule_steps", 4000, "--log_every", 20]}
                for a in todo]
        run_lanes([jobs[0::2], jobs[1::2]], pilot)
        for a in todo:
            lg = last_log(os.path.join(pilot, f"{a}_mb{try_mb}"))
            if lg and lg[-1]["step"] == 60:
                mb[a] = try_mb
    assert set(mb) == set(ARMS), f"pilot failed for {set(ARMS) - set(mb)}"
    tps = {a: last_log(os.path.join(pilot, f"{a}_mb{mb[a]}"))[-1]["tok_per_s"] for a in ARMS}
    mfu = {a: last_log(os.path.join(pilot, f"{a}_mb{mb[a]}"))[-1]["mfu"] for a in ARMS}
    # fixed overhead per run (process start, data index, compile, final full val): wall time of the pilot
    # run minus 60 steps at steady-state throughput
    over = {}
    for a in ARMS:
        d = os.path.join(pilot, f"{a}_mb{mb[a]}")
        wall = os.path.getmtime(os.path.join(d, "rc.txt")) - os.path.getmtime(os.path.join(d, "lane_start"))
        over[a] = max(0.0, wall - 60 * BATCH / tps[a])
    # eval smoke: 30-step complete diff run with evals, timed
    es = os.path.join(out, "evalsmoke")
    te = time.time()
    run_lanes([[{"name": "diff_evals", "args": COMMON + ["--arch", "diff", "--lr", "2e-3", "--seed", "100",
                                                         "--micro_bs", mb["diff"], "--max_steps", 30],
                 "evals": True}], []], es)
    eval_s = time.time() - te
    assert os.path.exists(os.path.join(es, "diff_evals", "evals.json")), "eval smoke failed"
    # ---- pre-registered budget formula
    per_run_fixed = max(over.values()) + eval_s

    def lane_hours(S):
        jobs = [(a, s) for a in ARMS for s in range(5)]
        cost = lambda j: S * BATCH / tps[j[0]] * 1.04 + per_run_fixed  # noqa: E731
        _, load = balance(jobs, cost)
        return max(load) / 3600

    S = 100
    while lane_hours(S + 100) <= main_hours:
        S += 100
    budget = dict(main_steps=S, main_tokens=S * BATCH, sweep_steps=S // 4, micro_bs=mb, pilot_tok_per_s=tps,
                  pilot_mfu=mfu, per_run_fixed_s=per_run_fixed, eval_s=eval_s, main_hours=main_hours,
                  predicted_main_lane_hours=lane_hours(S), stage_a_elapsed_s=time.time() - t0)
    json.dump(budget, open(os.path.join(out, "budget.json"), "w"), indent=1)
    log("TOKEN_PLAN", budget)
    # ---- sweep
    sw = os.path.join(out, "sweep")
    ss = budget["sweep_steps"]

    def sjob(a, lr):
        return {"name": f"sweep_{a}_lr{lr:g}", "arch": a,
                "args": COMMON + ["--arch", a, "--lr", lr, "--seed", 100, "--micro_bs", mb[a], "--max_steps", ss]}
    jobs = [sjob(a, lr) for a in ARMS for lr in SWEEP_LRS]
    lanes, _ = balance(jobs, lambda j: 1 / tps[j["arch"]])
    run_lanes(lanes, sw)
    res = {a: {lr: final_val(os.path.join(sw, f"sweep_{a}_lr{lr:g}")) for lr in SWEEP_LRS} for a in ARMS}
    # edge rule: extend by one factor-2 step in the direction of the edge (pre-registered)
    ext = []
    for a in ARMS:
        best = min(res[a], key=res[a].get)
        lrs = sorted(res[a])
        if math.isfinite(res[a][best]) and best == lrs[0]:
            ext.append(sjob(a, lrs[0] / 2))
        elif math.isfinite(res[a][best]) and best == lrs[-1]:
            ext.append(sjob(a, lrs[-1] * 2))
    if ext:
        lanes, _ = balance(ext, lambda j: 1 / tps[j["arch"]])
        run_lanes(lanes, sw)
        for j in ext:
            lr = float(j["args"][j["args"].index("--lr") + 1])
            res[j["arch"]][lr] = final_val(os.path.join(sw, j["name"]))
    best = {a: min(res[a], key=res[a].get) for a in ARMS}
    out_sw = dict(results={a: {str(k): v for k, v in res[a].items()} for a in ARMS}, best_lr=best,
                  extended=[j["name"] for j in ext])
    json.dump(out_sw, open(os.path.join(out, "sweep.json"), "w"), indent=1)
    log("SWEEP", out_sw)
    for a in ARMS:
        assert math.isfinite(res[a][best[a]]), f"no finished sweep run for {a}"


def stage_b(out, budget_path, sweep_path, runs):
    """runs: list of (arch, seed) to run in this session (main runs may be split across sessions)."""
    budget, sw = json.load(open(budget_path)), json.load(open(sweep_path))
    S, mb, tps = budget["main_steps"], budget["micro_bs"], budget["pilot_tok_per_s"]
    jobs = [{"name": f"main_{a}_s{s}", "arch": a, "evals": True,
             "args": COMMON + ["--arch", a, "--lr", sw["best_lr"][a], "--seed", s, "--micro_bs", mb[a],
                               "--max_steps", S]} for a, s in runs]
    lanes, load = balance(jobs, lambda j: 1 / tps[j["arch"]])
    log("lanes", [[j["name"] for j in l] for l in lanes])
    run_lanes(lanes, os.path.join(out, "main"))


if __name__ == "__main__":
    if sys.argv[1] == "A":
        stage_a(sys.argv[2], float(sys.argv[3]))
    else:
        stage_b(sys.argv[2], sys.argv[3], sys.argv[4],
                [(x.split(":")[0], int(x.split(":")[1])) for x in sys.argv[5].split(",")])
