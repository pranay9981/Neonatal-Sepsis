"""
Local federated learning simulation.
Starts the FL server and N clients as subprocesses automatically — no manual terminal juggling.

Usage:
  python scripts/run_fl_sim.py \
    --client_indexes data/processed/clients/client1/index.pt data/processed/clients/client2/index.pt \
    --model transformer --rounds 5 --local_epochs 1
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def run_fl_sim(
    client_indexes,
    model="transformer",
    rounds=5,
    local_epochs=1,
    batch_size=32,
    host="127.0.0.1",
    port=8080,
    save_dir=None,
    checkpoints_dir=None,
    round_timeout=300,
    n_features=40,
    seq_len=48,
    client_startup_delay=6.0,
    strategy="fedavg",
):
    save_dir = save_dir or str(PROJECT_ROOT / "server_out")
    checkpoints_dir = checkpoints_dir or str(PROJECT_ROOT / "checkpoints")
    n_clients = len(client_indexes)

    env = {
        **os.environ,
        "PYTHONPATH": str(SRC_DIR) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    server_cmd = [
        sys.executable,
        str(SRC_DIR / "fl_server.py"),
        "--model", model,
        "--rounds", str(rounds),
        "--host", host,
        "--port", str(port),
        "--min_clients", str(n_clients),
        "--n_features", str(n_features),
        "--seq_len", str(seq_len),
        "--save_dir", save_dir,
        "--checkpoints_dir", checkpoints_dir,
        "--round_timeout", str(round_timeout),
        "--strategy", strategy,
    ]

    client_cmds = [
        [
            sys.executable,
            str(SRC_DIR / "fl_client.py"),
            "--index", str(idx),
            "--server_address", f"{host}:{port}",
            "--model", model,
            "--local_epochs", str(local_epochs),
            "--batch_size", str(batch_size),
            "--n_features", str(n_features),
            "--seq_len", str(seq_len),
        ]
        for idx in client_indexes
    ]

    print(f"[FL-SIM] Starting server on {host}:{port} for {rounds} rounds ...")
    server_proc = subprocess.Popen(server_cmd, env=env, cwd=str(PROJECT_ROOT))

    print(f"[FL-SIM] Waiting {client_startup_delay}s for server to initialise ...")
    time.sleep(client_startup_delay)

    if server_proc.poll() is not None:
        print(f"[FL-SIM] ERROR: Server exited early with code {server_proc.returncode}")
        sys.exit(1)

    client_procs = []
    for i, cmd in enumerate(client_cmds):
        print(f"[FL-SIM] Starting client {i + 1}/{n_clients} ...")
        proc = subprocess.Popen(cmd, env=env, cwd=str(PROJECT_ROOT))
        client_procs.append(proc)
        time.sleep(1.0)

    all_procs = [server_proc] + client_procs

    try:
        rc_server = server_proc.wait()
        if rc_server != 0:
            print(f"[FL-SIM] WARNING: Server exited with code {rc_server}")
        else:
            print("[FL-SIM] Server completed successfully.")

        for i, proc in enumerate(client_procs):
            try:
                rc = proc.wait(timeout=120)
                print(f"[FL-SIM] Client {i + 1} finished (rc={rc})")
            except subprocess.TimeoutExpired:
                print(f"[FL-SIM] Client {i + 1} timed out — terminating.")
                proc.terminate()

        best_pt = Path(save_dir) / "global_best.pt"
        if best_pt.exists():
            print(f"[FL-SIM] Best model saved at: {best_pt}")
        else:
            print(f"[FL-SIM] WARNING: global_best.pt not found at {best_pt}")

        return str(best_pt)

    except KeyboardInterrupt:
        print("\n[FL-SIM] Interrupted — terminating all processes ...")
        for proc in all_procs:
            try:
                proc.terminate()
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Local FL simulation — starts server + N clients automatically")
    ap.add_argument("--client_indexes", nargs="+", required=True, help="Paths to client index.pt files")
    ap.add_argument("--model", choices=["transformer", "grud"], default="transformer")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--round_timeout", type=int, default=300)
    ap.add_argument("--n_features", type=int, default=40)
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--checkpoints_dir", default=None)
    ap.add_argument("--strategy", choices=["fedavg", "fedbn"], default="fedavg")
    args = ap.parse_args()

    run_fl_sim(
        client_indexes=args.client_indexes,
        model=args.model,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        host=args.host,
        port=args.port,
        round_timeout=args.round_timeout,
        n_features=args.n_features,
        seq_len=args.seq_len,
        save_dir=args.save_dir,
        checkpoints_dir=args.checkpoints_dir,
        strategy=args.strategy,
    )
