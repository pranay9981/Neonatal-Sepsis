# src/split_clients.py
import os, glob, shutil
import torch
import argparse
from math import floor
from utils import ensure_dir
import random

def split_into_clients(processed_folder, out_root, n_clients=3, seed=42):
    ensure_dir(out_root)
    idx = os.path.join(processed_folder, "index_with_labels.pt")
    d = torch.load(idx)
    x_paths = d['x_paths']
    random.seed(seed)
    random.shuffle(x_paths)
    counts = [len(x_paths)//n_clients] * n_clients
    for i in range(len(x_paths) % n_clients):
        counts[i] += 1
    cur = 0
    clients = []
    for i,c in enumerate(counts):
        client_folder = os.path.join(out_root, f"client{i+1}")
        ensure_dir(client_folder)
        selected = x_paths[cur:cur+c]
        for p in selected:
            shutil.copy(p, client_folder)
        new_index = [os.path.join(client_folder, os.path.basename(p)) for p in selected]
        torch.save({'x_paths': new_index, 'y': [torch.load(p).get('y',0) for p in selected]}, os.path.join(client_folder, "index.pt"))
        clients.append(client_folder)
        cur += c
    print(f"Split {len(x_paths)} patients into {n_clients} clients.")
    return clients

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_folder", required=True)
    ap.add_argument("--out_root", default="data/processed/clients")
    ap.add_argument("--n_clients", type=int, default=3)
    args = ap.parse_args()
    split_into_clients(args.processed_folder, args.out_root, args.n_clients)
