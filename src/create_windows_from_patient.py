# src/create_windows_from_patient.py
import torch, os, argparse
from pathlib import Path

def create_windows(input_pt, out_dir, seq_len=48, step=1):
    os.makedirs(out_dir, exist_ok=True)
    d = torch.load(input_pt)
    X = d['X'].numpy()
    y = int(d.get('y', 0))
    T = X.shape[0]
    count = 0
    for start in range(0, max(1, T - seq_len + 1), step):
        window = X[start:start+seq_len]
        pid = f"{Path(input_pt).stem}_w{start}"
        torch.save({'X': torch.tensor(window.astype('float32')), 'y': y, 'meta': {'patient_id': pid}}, os.path.join(out_dir, f"{pid}.pt"))
        count += 1
    print("Created", count, "windows from", input_pt)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="data/processed/patients_windows")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()
    create_windows(args.input, args.out_dir, seq_len=args.seq_len, step=args.step)
