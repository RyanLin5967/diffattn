"""Build a self-contained Kaggle script kernel: the repo code is embedded (base64 tar) and a job spec says
what to run. Kaggle is the compute (free); nothing here touches Modal.

  python3 kaggle_build.py prep  -> kernels/prep   (CPU + internet: tokenize data into /kaggle/working/data)
  python3 kaggle_build.py plan plans/pilot.json -> kernels/<slug>   (GPU T4x2)

plan json: {"slug": "...", "lanes": [[spec, ...], [spec, ...]], "pre": ["cmd", ...]}
spec: {"name": "...", "args": [...], "evals": bool}. Lane i runs sequentially on GPU i.
"""
import base64
import io
import json
import os
import sys
import tarfile

USER = "ryanlin5967"
ROOT = os.path.dirname(os.path.abspath(__file__))
FILES = ["model.py", "train.py", "evals.py", "prep.py", "orchestrate.py", "tests/test_diffattn.py", "tests/gpu_parity.py"]
EXTRA = ["results/budget.json", "results/sweep.json"]  # included when present (stage B)
DATA_KERNEL = f"{USER}/diffattn-data-prep"

RUNNER = r'''
import base64, io, json, os, subprocess, sys, tarfile, time, glob, shutil
CODE = "/tmp/code"
os.makedirs(CODE, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(B64))).extractall(CODE)
OUT = "/kaggle/working"
def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, cwd=CODE, **kw).returncode
sh("pip install -q tiktoken pytest 2>&1 | tail -2")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv || true")
sh("python -c 'import torch;print(torch.__version__, torch.cuda.device_count())'")
if JOB["kind"] == "prep":
    rc = sh(f"python prep.py --out {OUT}/data")
    sys.exit(rc)
src = [p for p in glob.glob("/kaggle/input/**/train_00.bin", recursive=True)]
assert src, "data kernel output not mounted"
os.makedirs("/tmp/data", exist_ok=True)
d = os.path.dirname(src[0])
for f in os.listdir(d):
    p = os.path.join(d, f)
    (shutil.copytree if os.path.isdir(p) else shutil.copy)(p, os.path.join("/tmp/data", f))
print("data:", os.listdir("/tmp/data"), flush=True)
if JOB["kind"] == "cmd":
    rc = sh(JOB["cmd"] + f" 2>&1 | tee {OUT}/orchestrate.log; exit ${{PIPESTATUS[0]}}", executable="/bin/bash")
    for p in glob.glob(f"{OUT}/**/final.pt", recursive=True):
        os.remove(p)  # keep kernel output small; evals already ran in-session
    sys.exit(rc)
status = {"pre": []}
for c in JOB.get("pre", []):
    status["pre"].append({"cmd": c, "rc": sh(c)})
json.dump(status, open(f"{OUT}/status.json", "w"), indent=1)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor"
lane_scripts = []
for i, lane in enumerate(JOB["lanes"]):
    lines = ["set -x"]
    for sp in lane:
        o = f"{OUT}/runs/{sp['name']}"
        lines.append(f"mkdir -p {o}")
        lines.append(f"python train.py --data_dir /tmp/data --out {o} {' '.join(map(str, sp['args']))} > {o}/stdout.txt 2>&1; echo train_rc=$? >> {o}/rc.txt")
        if sp.get("evals"):
            lines.append(f"[ -f {o}/final.pt ] && python evals.py --run {o} --data_dir /tmp/data > {o}/evals_stdout.txt 2>&1; echo eval_rc=$? >> {o}/rc.txt")
        if sp.get("drop_final"):
            lines.append(f"rm -f {o}/final.pt")
    path = f"/tmp/lane{i}.sh"
    open(path, "w").write("\n".join(lines) + "\n")
    lane_scripts.append(path)
procs = [subprocess.Popen(f"CUDA_VISIBLE_DEVICES={i} bash {p} > {OUT}/lane{i}.log 2>&1", shell=True, cwd=CODE)
         for i, p in enumerate(lane_scripts)]
while any(p.poll() is None for p in procs):
    time.sleep(60)
    for f in sorted(glob.glob(f"{OUT}/runs/*/log.jsonl")):
        try:
            last = open(f).read().strip().splitlines()[-1]
            print(os.path.basename(os.path.dirname(f)), last[:200], flush=True)
        except Exception:
            pass
status["lanes_rc"] = [p.returncode for p in procs]
json.dump(status, open(f"{OUT}/status.json", "w"), indent=1)
print("ALL DONE", status, flush=True)
'''


def code_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for f in FILES + [e for e in EXTRA if os.path.exists(os.path.join(ROOT, e))]:
            t.add(os.path.join(ROOT, f), arcname=f)
    return base64.b64encode(buf.getvalue()).decode()


def build(slug, title, job, gpu, sources):
    d = os.path.join(ROOT, "kernels", slug)
    os.makedirs(d, exist_ok=True)
    script = f"B64 = {code_b64()!r}\nJOB = {json.dumps(job)!r}\nimport json as _j\nJOB = _j.loads(JOB)\n" + RUNNER
    open(os.path.join(d, "run.py"), "w").write(script)
    meta = {"id": f"{USER}/{slug}", "title": title, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": "true", "enable_gpu": "true" if gpu else "false",
            "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4" if gpu else "",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": sources, "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    print(d)


if __name__ == "__main__":
    if sys.argv[1] == "stage":
        slug, cmd = sys.argv[2], sys.argv[3]
        build(slug, slug, {"kind": "cmd", "cmd": cmd}, True, [DATA_KERNEL])
    elif sys.argv[1] == "prep":
        build("diffattn-data-prep", "diffattn data prep", {"kind": "prep"}, False, [])
    else:
        plan = json.load(open(sys.argv[2]))
        plan["kind"] = "train"
        build(plan["slug"], plan["slug"], plan, True, [DATA_KERNEL])
