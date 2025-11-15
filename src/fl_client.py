# src/fl_client.py
"""
Flower client implementation that:
 - builds local model (transformer or grud)
 - trains locally for some epochs on its index
 - returns parameters and metrics (auroc, auprc when possible)
Usage:
  python src/fl_client.py --index data/processed/clients/client1/index.pt --server_address 127.0.0.1:8080 --model transformer --device cpu --local_epochs 1 --batch_size 32
"""
import argparse
import time
import os
import json
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import flwr as fl

# Import dataset and models from your repo
try:
    from dataset import PatientDataset
except Exception:
    PatientDataset = None

try:
    from model import TimeSeriesTransformer
except Exception:
    TimeSeriesTransformer = None

try:
    from model_grud import GRUD
except Exception:
    GRUD = None

from sklearn.metrics import roc_auc_score, average_precision_score

# -----------------------
# Utils: convert state_dict <-> list of numpy arrays (ordered)
# -----------------------
def state_dict_to_ndarrays(sd: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    arrs = []
    for k, v in sd.items():
        arrs.append(v.cpu().numpy())
    return arrs

def ndarrays_to_state_dict_by_order(model: torch.nn.Module, arrays: List[np.ndarray]) -> Dict[str, torch.Tensor]:
    sd = model.state_dict()
    keys = list(sd.keys())
    map_len = min(len(keys), len(arrays))
    new_sd = {}
    for k, arr in zip(keys[:map_len], arrays[:map_len]):
        t = torch.tensor(arr)
        if t.shape != sd[k].shape:
            try:
                t = t.view(sd[k].shape)
            except Exception:
                if t.ndim == 2 and tuple(t.T.shape) == tuple(sd[k].shape):
                    t = t.T
                else:
                    raise RuntimeError(f"Cannot map array for key {k}: got {tuple(t.shape)}, expected {tuple(sd[k].shape)}")
        new_sd[k] = t
    sd.update(new_sd)
    return sd

def build_model(model_name: str, n_features: int, seq_len: int, device: str):
    if model_name == "transformer":
        assert TimeSeriesTransformer is not None, "TimeSeriesTransformer not importable"
        return TimeSeriesTransformer(n_features=n_features, seq_len=seq_len).to(device)
    elif model_name == "grud":
        assert GRUD is not None, "GRUD not importable"
        return GRUD(n_features=n_features).to(device)
    else:
        raise ValueError("Unknown model: " + str(model_name))

# -----------------------
# Flower NumPyClient
# -----------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, index_path: str, model_name: str, device: str = "cpu", batch_size: int = 32, lr: float = 1e-3, local_epochs: int = 1, n_features: Optional[int] = None, seq_len: Optional[int] = None):
        assert PatientDataset is not None, "PatientDataset not importable"
        self.index_path = index_path
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.local_epochs = local_epochs

        # load dataset
        ds = PatientDataset(self.index_path, mode="transformer" if model_name=="transformer" else "grud")
        # infer n_features and seq_len if not provided
        if n_features is None:
            # try to infer from dataset first item
            try:
                x0, *_ = ds[0]
                # x0 may be series length x features or batch-like: handle common shapes
                arr = np.asarray(x0)
                if arr.ndim == 2:
                    seq_len, n_features = arr.shape
                elif arr.ndim == 1:
                    n_features = arr.shape[0]
                    seq_len = 1
                else:
                    n_features = arr.shape[-1]
                    seq_len = arr.shape[0] if arr.ndim >= 2 else None
            except Exception:
                n_features = n_features or 48
                seq_len = seq_len or 48
        if seq_len is None:
            seq_len = seq_len or 48

        self.n_features = int(n_features)
        self.seq_len = int(seq_len)

        # build model
        self.model = build_model(self.model_name, self.n_features, self.seq_len, self.device)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

        # dataloaders
        self.train_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.eval_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        print(f"[CLIENT] Inferred n_features={self.n_features}, seq_len={self.seq_len} from dataset.")

    # get_parameters: return list of numpy arrays
    def get_parameters(self) -> List[np.ndarray]:
        sd = self.model.state_dict()
        arrs = state_dict_to_ndarrays(sd)
        return arrs

    # set model params from list of numpy arrays
    def set_parameters(self, arrays: List[np.ndarray]):
        sd = ndarrays_to_state_dict_by_order(self.model, arrays)
        self.model.load_state_dict(sd)

    # fit: receive parameters, set them, train locally, return parameters and number of examples and metrics
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        # set incoming global parameters
        if parameters is not None:
            try:
                self.set_parameters(parameters)
            except Exception as e:
                print(f"[CLIENT][WARN] Could not set incoming parameters: {e}")

        self.model.train()
        device = torch.device(self.device)
        self.model.to(device)
        for epoch in range(self.local_epochs):
            running_loss = 0.0
            n_samples = 0
            for batch in self.train_loader:
                # handle transformer vs grud shapes
                if self.model_name == "transformer":
                    Xb, yb = batch
                    Xb = Xb.to(device).float()
                    yb = yb.to(device).float().view(-1)
                    logits = self.model(Xb)
                else:
                    Xb, Mb, Db, yb = batch
                    Xb = Xb.to(device).float()
                    Mb = Mb.to(device).float()
                    Db = Db.to(device).float()
                    yb = yb.to(device).float().view(-1)
                    logits = self.model(Xb, Mb, Db)

                loss = self.loss_fn(logits.view(-1), yb)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                running_loss += float(loss.item()) * len(yb)
                n_samples += len(yb)

            if n_samples > 0:
                ep_loss = running_loss / n_samples
                print(f"[CLIENT][TRAIN] epoch {epoch+1}/{self.local_epochs} loss={ep_loss:.4f}")

        # return updated parameters
        out_params = state_dict_to_ndarrays(self.model.state_dict())
        # compute metrics on local eval set
        metrics = self.evaluate_local()
        
        num_examples = len(self.train_loader.dataset)
        
        return out_params, num_examples, metrics

    # evaluate: remote evaluation requested by server -> return num_examples, metrics
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, float]]:
        if parameters is not None:
            try:
                self.set_parameters(parameters)
            except Exception as e:
                print(f"[CLIENT][WARN] Could not set parameters in evaluate: {e}")
        
        metrics = self.evaluate_local()
        n = len(self.eval_loader.dataset)
        
        # --- START OF FIX ---
        # Pop the loss from the metrics dict to return it as the first element
        # This matches the required (loss, num_examples, metrics) signature
        loss = metrics.pop("loss", float("nan"))
        
        return loss, n, metrics
        # --- END OF FIX ---

    def evaluate_local(self) -> Dict[str, float]:
        # Runs a forward pass on eval loader and returns metrics (loss, auroc, auprc)
        self.model.eval()
        device = torch.device(self.device)
        ys = []
        preds = []
        total_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for batch in self.eval_loader:
                if self.model_name == "transformer":
                    Xb, yb = batch
                    Xb = Xb.to(device).float()
                    logits = self.model(Xb).view(-1)
                    yb = yb.to(device).float().view(-1)
                else:
                    Xb, Mb, Db, yb = batch
                    Xb = Xb.to(device).float()
                    Mb = Mb.to(device).float()
                    Db = Db.to(device).float()
                    logits = self.model(Xb, Mb, Db).view(-1)
                    yb = yb.to(device).float().view(-1)

                # Calculate loss directly from tensors on device
                loss = self.loss_fn(logits, yb)
                
                # Get numpy versions for metric calculation
                yb_arr = yb.cpu().numpy().reshape(-1)
                logits_np = logits.cpu().numpy().reshape(-1)

                ys.append(yb_arr)
                preds.append(logits_np)
                
                total_loss += float(loss.item()) * len(yb_arr)
                n_samples += len(yb_arr)
                
        if n_samples == 0:
            print("[CLIENT][WARN] No samples found in evaluate_local. Returning empty metrics.")
            return {"loss": float("nan"), "auroc": float("nan"), "auprc": float("nan")}
            
        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        # convert logits -> probs
        probs = 1.0 / (1.0 + np.exp(-preds))
        metrics = {}
        # loss average
        metrics["loss"] = total_loss / n_samples
        
        # compute AUROC / AUPRC if both classes present
        if len(set(ys)) > 1:
            try:
                metrics["auroc"] = float(roc_auc_score(ys, probs))
            except Exception:
                metrics["auroc"] = float("nan")
            try:
                metrics["auprc"] = float(average_precision_score(ys, probs))
            except Exception:
                metrics["auprc"] = float("nan")
        else:
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")
            
        return metrics

# -----------------------
# CLI entry: wrapper that calls fl.client.start_client with client.to_client()
# -----------------------
def start_client(index_path: str, server_address: str, model_name: str, device: str = "cpu", batch_size: int = 32, lr: float = 1e-3, local_epochs: int = 1, n_features: Optional[int] = None, seq_len: Optional[int] = None, max_retries: int = 20, retry_delay: float = 2.0):
    client = FlowerClient(index_path=index_path, model_name=model_name, device=device, batch_size=batch_size, lr=lr, local_epochs=local_epochs, n_features=n_features, seq_len=seq_len)
    client_obj = client.to_client()  # convert NumPyClient to modern Client wrapper
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[CLIENT] Connecting to {server_address} (attempt {attempt}) ...")
            fl.client.start_client(server_address=server_address, client=client_obj)
            break
        except Exception as e:
            print(f"[CLIENT][WARN] start_client attempt {attempt} failed: {e}")
            if attempt >= max_retries:
                raise
            time.sleep(retry_delay)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--server_address", required=True)
    ap.add_argument("--model", choices=["transformer", "grud"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--n_features", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    args = ap.parse_args()
    
    start_client(index_path=args.index, server_address=args.server_address, model_name=args.model, device=args.device, batch_size=args.batch_size, lr=args.lr, local_epochs=args.local_epochs, n_features=args.n_features, seq_len=args.seq_len)