# src/generate_synthetic_patients.py
import os, torch, numpy as np, argparse
from tqdm import trange
from pathlib import Path

def generate_patient(seq_len=48, n_features=36, sepsis_prob=0.08):
    base = np.random.normal(loc=0.0, scale=1.0, size=(seq_len, n_features)).astype(np.float32)
    for f in range(n_features):
        base[:, f] = np.convolve(base[:, f], np.ones(3)/3, mode='same')
    offsets = np.random.uniform(-1.0, 1.0, size=(n_features,)).astype(np.float32)
    X = base + offsets
    y = 1 if np.random.rand() < sepsis_prob else 0
    if y == 1:
        t = np.arange(seq_len)
        spike = np.exp(-((t - seq_len*0.7)**2)/(2*(seq_len*0.05)**2))
        for f in range(min(5, n_features)):
            X[:, f] += 3.0 * spike
    return X, int(y)

def main(out_dir="data/processed/patients", n_patients=1000, seq_len=48, n_features=36, sepsis_prob=0.08, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)
    for i in trange(n_patients, desc="Generating synthetic patients"):
        X, y = generate_patient(seq_len=seq_len, n_features=n_features, sepsis_prob=sepsis_prob)
        pid = f"syn_{i:06d}"
        torch.save({'X': torch.tensor(X), 'y': int(y), 'meta': {'patient_id': pid, 'features': [f"f{j}" for j in range(n_features)]}}, os.path.join(out_dir, f"{pid}.pt"))
    files = sorted([str(p) for p in Path(out_dir).glob("*.pt") if not p.name.startswith("index")])
    ys = []
    for p in files:
        d = torch.load(p)
        ys.append(float(d['y']))
    torch.save({'x_paths': files, 'y': ys}, os.path.join(out_dir, "index_with_labels.pt"))
    print("Generated", n_patients, "synthetic patients to", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed/patients")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--n_features", type=int, default=36)
    ap.add_argument("--sepsis_prob", type=float, default=0.08)
    args = ap.parse_args()
    main(out_dir=args.out_dir, n_patients=args.n, seq_len=args.seq_len, n_features=args.n_features, sepsis_prob=args.sepsis_prob)
