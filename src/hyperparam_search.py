# src/hyperparam_search.py  (robust version: uses same Python executable)
import itertools, subprocess, json, os, time, sys
from datetime import datetime, timezone

grid = {
    "lr": [1e-3, 5e-4, 1e-4],
    "batch_size": [32, 64],
    "model": ["transformer", "grud"]
}

OUT_DIR = "hyper_results"
os.makedirs(OUT_DIR, exist_ok=True)

def run_one(params):
    cmd = [
        sys.executable,  # <<< ensures same Python interpreter (venv)
        "src/train_local.py",
        "--index", "data/processed/patients/index_with_labels.pt",
        "--epochs", "6",
        "--lr", str(params['lr']),
        "--batch_size", str(params['batch_size']),
        "--model", params['model'],
        "--run_name", f"hp_lr{params['lr']}_bs{params['batch_size']}_{params['model']}"
    ]
    start = datetime.now(timezone.utc).isoformat()
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        status = "SUCCESS"
        err = ""
    except subprocess.CalledProcessError as e:
        status = "FAILED"
        err = f"Returncode={e.returncode}"
    except Exception as e:
        status = "FAILED"
        err = str(e)
    end = datetime.now(timezone.utc).isoformat()
    record = {"params": params, "status": status, "err": err, "start": start, "end": end}
    fname = os.path.join(OUT_DIR, f"run_{int(time.time())}.json")
    with open(fname, "w") as fh:
        json.dump(record, fh, indent=2)
    print("Recorded:", fname)

def main():
    keys = list(grid.keys())
    for vals in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        run_one(params)

if __name__ == "__main__":
    main()
