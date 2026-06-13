# src/hyperparam_search.py
"""
Hyperparameter grid search.
Accepts an optional YAML config file for the search grid; falls back to built-in defaults.

Usage:
  python src/hyperparam_search.py                        # use built-in grid
  python src/hyperparam_search.py --config grid.yaml     # use custom grid

grid.yaml example:
  lr: [0.001, 0.0005, 0.0001]
  batch_size: [32, 64]
  model: [transformer, grud]
  epochs: [6]
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DEFAULT_GRID = {
    "lr": [1e-3, 5e-4, 1e-4],
    "batch_size": [32, 64],
    "model": ["transformer", "grud"],
    "epochs": [6],
}

OUT_DIR = str(PROJECT_ROOT / "hyper_results")
os.makedirs(OUT_DIR, exist_ok=True)


def load_grid(config_path=None):
    if config_path is None:
        return DEFAULT_GRID
    try:
        import yaml
    except ImportError:
        print("WARNING: pyyaml not installed. Using default grid.")
        return DEFAULT_GRID
    with open(config_path) as f:
        grid = yaml.safe_load(f)
    # Ensure all values are lists
    for k, v in grid.items():
        if not isinstance(v, list):
            grid[k] = [v]
    return grid


def run_one(params):
    epochs = params.get("epochs", 6)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "train_local.py"),
        "--index", str(PROJECT_ROOT / "data" / "processed" / "patients" / "index_with_labels.pt"),
        "--epochs", str(epochs),
        "--lr", str(params["lr"]),
        "--batch_size", str(params["batch_size"]),
        "--model", params["model"],
        "--run_name", f"hp_lr{params['lr']}_bs{params['batch_size']}_{params['model']}",
    ]
    start = datetime.now(timezone.utc).isoformat()
    print(f"[HYPER] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        status, err = "SUCCESS", ""
    except subprocess.CalledProcessError as e:
        status, err = "FAILED", f"rc={e.returncode}"
    except Exception as e:
        status, err = "FAILED", str(e)
    end = datetime.now(timezone.utc).isoformat()
    record = {"params": params, "status": status, "err": err, "start": start, "end": end}
    fname = os.path.join(OUT_DIR, f"run_{int(time.time())}.json")
    with open(fname, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"[HYPER] Recorded: {fname}  status={status}")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to YAML grid config file")
    args = ap.parse_args()

    grid = load_grid(args.config)
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"[HYPER] Running {len(combos)} combinations ...")

    results = []
    for vals in combos:
        params = dict(zip(keys, vals))
        rec = run_one(params)
        results.append(rec)

    success = sum(1 for r in results if r["status"] == "SUCCESS")
    print(f"\n[HYPER] Done: {success}/{len(results)} runs succeeded.")
    print(f"[HYPER] Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
