# src/build_index_with_labels.py
import torch, glob, os, argparse

def build_index(processed_folder, out_path=None):
    if out_path is None:
        out_path = os.path.join(processed_folder, "index_with_labels.pt")
    files = sorted(glob.glob(os.path.join(processed_folder, "*.pt")))
    files = [f for f in files if not f.endswith("index_with_labels.pt") and not os.path.basename(f).startswith("index")]
    x_paths = []
    ys = []
    for p in files:
        try:
            d = torch.load(p)
            x_paths.append(p)
            y = d.get('y', 0)
            try:
                ys.append(float(y))
            except:
                try:
                    ys.append(float(y.item()))
                except:
                    ys.append(float(int(y)))
        except Exception as e:
            print("Skipping", p, "due to", e)
    torch.save({'x_paths': x_paths, 'y': ys}, out_path)
    print("Wrote index with", len(x_paths), "patients to", out_path)
    return out_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_folder", default="data/processed/patients")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    build_index(args.processed_folder, args.out)
