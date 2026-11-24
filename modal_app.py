"""Modal entrypoints. Budget ~$5 total, so: one parquet file of data, single-GPU containers that each
run several training runs back to back (one compile cache per container), evals in the same container.

  modal run modal_app.py::prep          # CPU: tokenize data + eval sets into volume diffattn-data
  modal run modal_app.py::tests         # CPU: unit tests
  modal run modal_app.py::gpu_check     # 1xH100: flash-path parity test + pause/resume smoke + throughput pilot
  modal run modal_app.py::launch --plan plans/sweep.json   # one container per group in the plan
"""
import json
import os
import subprocess
import sys
import threading
import time

import modal

app = modal.App("diffattn")
data_vol = modal.Volume.from_name("diffattn-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("diffattn-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "numpy", "tiktoken", "pyarrow", "huggingface_hub", "pytest", "scipy",
                 "datasets", "requests")
    .add_local_dir(os.path.dirname(os.path.abspath(__file__)), "/root/code",
                   ignore=["runs_local", ".git", "paper", "__pycache__", "*.pt"])
)
CODE = "/root/code"


def _sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, cwd=CODE, **kw)


@app.function(image=image, cpu=8, memory=16384, volumes={"/vdata": data_vol}, timeout=3600)
def prep_data():
    _sh([sys.executable, "prep.py", "--out", "/vdata"])
    data_vol.commit()


@app.function(image=image, cpu=4, timeout=900)
def run_tests():
    _sh([sys.executable, "-m", "pytest", "-q", "tests"])


def _stage_data():
    os.makedirs("/data", exist_ok=True)
    t = time.time()
    for f in os.listdir("/vdata"):
        if f.endswith(".bin"):
            _sh(["cp", f"/vdata/{f}", f"/data/{f}"])
    if os.path.isdir("/vdata/evals"):
        _sh(["cp", "-r", "/vdata/evals", "/data/evals"])
    print(f"staged data in {time.time() - t:.1f}s", flush=True)


def _committer(stop):
    while not stop.wait(300):
        try:
            runs_vol.commit()
        except Exception as e:  # noqa: BLE001
            print("commit failed", e, flush=True)


def _run_group(specs):
    """specs: list of dicts {out, args:[...], evals:bool}. Runs sequentially; skips runs already DONE."""
    _stage_data()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor"
    stop = threading.Event()
    th = threading.Thread(target=_committer, args=(stop,), daemon=True)
    th.start()
    results = []
    try:
        for sp in specs:
            out = sp["out"]
            done = os.path.exists(os.path.join(out, "DONE")) or os.path.exists(os.path.join(out, "DIVERGED"))
            if not done or sp.get("force"):
                t = time.time()
                r = subprocess.run([sys.executable, "train.py", "--data_dir", "/data", "--out", out] + sp["args"],
                                   cwd=CODE)
                results.append({"out": out, "train_rc": r.returncode, "train_s": time.time() - t})
                runs_vol.commit()
                if r.returncode != 0:
                    continue
            if sp.get("evals") and os.path.exists(os.path.join(out, "final.pt")) and \
                    not os.path.exists(os.path.join(out, "evals.json")):
                t = time.time()
                r = subprocess.run([sys.executable, "evals.py", "--run", out, "--data_dir", "/data"], cwd=CODE)
                results.append({"out": out, "eval_rc": r.returncode, "eval_s": time.time() - t})
                runs_vol.commit()
    finally:
        stop.set()
        runs_vol.commit()
    return results


@app.function(image=image, gpu="H100", cpu=8, memory=32768, volumes={"/vdata": data_vol, "/runs": runs_vol},
              timeout=4 * 3600, retries=modal.Retries(max_retries=1))
def train_group(specs):
    return _run_group(specs)


@app.function(image=image, gpu="H100", cpu=8, memory=32768, volumes={"/vdata": data_vol, "/runs": runs_vol},
              timeout=1800)
def gpu_check(pilot_specs):
    _sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    _sh([sys.executable, "-m", "pytest", "-q", "tests"])
    _sh([sys.executable, "tests/gpu_parity.py"])
    return _run_group(pilot_specs)


@app.local_entrypoint()
def prep():
    prep_data.remote()


@app.local_entrypoint()
def tests():
    run_tests.remote()


@app.local_entrypoint()
def check(plan: str):
    print(json.dumps(gpu_check.remote(json.load(open(plan))), indent=1))


@app.local_entrypoint()
def launch(plan: str):
    groups = json.load(open(plan))
    calls = [train_group.spawn(g) for g in groups]
    for c in calls:
        print(json.dumps(c.get(), indent=1), flush=True)
