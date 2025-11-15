# src/lmdb_packer.py
import lmdb, pickle, glob, os, argparse, math
from tqdm import tqdm

def pack_folder(in_folder, out_folder, shard_size=5000):
    os.makedirs(out_folder, exist_ok=True)
    files = sorted([f for f in glob.glob(os.path.join(in_folder, "*.pt")) if not os.path.basename(f).startswith("index")])
    n = len(files)
    n_shards = math.ceil(n / shard_size)
    idx_specs = []
    for s in range(n_shards):
        shard_files = files[s*shard_size:(s+1)*shard_size]
        shard_path = os.path.join(out_folder, f"shard_{s:03d}.lmdb")
        env = lmdb.open(shard_path, map_size=1024**4)
        with env.begin(write=True) as txn:
            for p in tqdm(shard_files, desc=f"Packing shard {s}"):
                key = os.path.splitext(os.path.basename(p))[0]
                obj = None
                with open(p, 'rb') as fh:
                    obj = fh.read()
                txn.put(key.encode('utf-8'), obj)
                idx_specs.append(f"lmdb://{shard_path}#{key}")
        env.close()
    # write index file referencing lmdb specs
    return idx_specs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_folder", default="data/processed/patients")
    ap.add_argument("--out_folder", default="data/processed/lmdb_shards")
    ap.add_argument("--shard_size", type=int, default=5000)
    args = ap.parse_args()
    specs = pack_folder(args.in_folder, args.out_folder, shard_size=args.shard_size)
    # Save the index for quick use
    import torch
    torch.save({'x_paths': specs, 'y': []}, os.path.join(args.out_folder, "index_lmdb.pt"))
    print("Wrote LMDB shards and index at", args.out_folder)
