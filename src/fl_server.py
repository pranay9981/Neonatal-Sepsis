# src/fl_server.py
"""
Flower server with self-initialized parameters and per-round PT saving + best selection.

Usage (example):
  python src/fl_server.py \
    --rounds 10 \
    --host 127.0.0.1 \
    --port 8080 \
    --min_clients 3 \
    --model transformer \
    --n_features 40 \
    --seq_len 48 \
    --save_dir ./server_out \
    --checkpoints_dir ./checkpoints \
    --best_name global_best.pt \
    --round_timeout 60

Notes:
 - The server will construct a local untrained model (based on --model, --n_features, --seq_len)
   and use that model's parameters as INITIAL PARAMETERS (so it doesn't have to request them
   from a client at startup).
 - Each round the aggregated parameters are converted into a model.state_dict and saved as:
     {checkpoints_dir}/global_round_{r}.pt
 - The server tries to choose the best model (numeric metric) across rounds; supported metric keys:
     auc, val_auc, val_loss, loss (in that preference order).
 - If parameter->state_dict mapping fails because shapes mismatch, it will log a warning and skip saving that round's PT.
"""

import argparse
import os
import math
import json
import tempfile
from typing import List, Any, Dict

import numpy as np
import torch

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# Try importing models from your repo
try:
    from model import TimeSeriesTransformer
except Exception:
    TimeSeriesTransformer = None

try:
    from model_grud import GRUD
except Exception:
    GRUD = None


def build_model(model_name: str, n_features: int = None, seq_len: int = None, device: str = "cpu"):
    """Construct a model instance for given model_name (transformer or grud)."""
    if model_name == "transformer":
        assert TimeSeriesTransformer is not None, "TimeSeriesTransformer not importable"
        kwargs = {}
        if n_features is not None:
            kwargs["n_features"] = n_features
        if seq_len is not None:
            kwargs["seq_len"] = seq_len
        model = TimeSeriesTransformer(**kwargs)
    elif model_name == "grud":
        assert GRUD is not None, "GRUD not importable"
        kwargs = {}
        if n_features is not None:
            kwargs["n_features"] = n_features
        model = GRUD(**kwargs)
    else:
        raise ValueError("Unknown model: " + str(model_name))
    model.to(device)
    model.eval()
    return model


def state_dict_to_ndarrays_by_order(sd: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    """
    Convert model.state_dict() -> list of numpy arrays in key order.
    This list ordering will be used to create Flower Parameters.
    """
    arrays = []
    for k in sd.keys():
        t = sd[k].cpu().numpy()
        arrays.append(t)
    return arrays


def arrays_to_state_dict_by_order(model: torch.nn.Module, arrays: List[np.ndarray]) -> Dict[str, torch.Tensor]:
    """
    Map a list of numpy arrays -> model.state_dict() keys in order (best-effort).
    Returns a state_dict compatible dict (torch.Tensor values) which can be loaded into the model.
    Raises if mapping fails.
    """
    sd = model.state_dict()
    keys = list(sd.keys())
    if len(arrays) != len(keys):
        # We'll still attempt partial mapping by min len
        print(f"[WARN] arrays count {len(arrays)} != state_dict keys {len(keys)}. Attempting partial mapping.")
    map_len = min(len(keys), len(arrays))
    new_sd = {}
    for k, arr in zip(keys[:map_len], arrays[:map_len]):
        t = torch.tensor(arr)
        if t.shape != sd[k].shape:
            # try to reshape if counts match
            if t.numel() == sd[k].numel():
                try:
                    t = t.view(sd[k].shape)
                except Exception:
                    raise RuntimeError(f"Cannot reshape array for key {k} from {tuple(t.shape)} -> {tuple(sd[k].shape)}")
            else:
                raise RuntimeError(f"Shape mismatch for key {k}: array {tuple(t.shape)} vs expected {tuple(sd[k].shape)}")
        new_sd[k] = t
    # Update original sd with new mapped tensors
    sd.update(new_sd)
    return sd


class SaveEveryRoundFedAvg(fl.server.strategy.FedAvg):
    """
    FedAvg extension that saves aggregated model parameters each round to PT files,
    and keeps track of a "best" model according to an aggregated numeric evaluation metric.
    """

    def __init__(
        self,
        model_name: str,
        n_features: int,
        seq_len: int,
        save_dir: str,
        checkpoints_dir: str,
        best_name: str = "global_best.pt",
        metric_priority: List[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.n_features = n_features
        self.seq_len = seq_len
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.best_name = best_name
        
        # --- START OF FIX ---
        # Changed the metric priority list to match the keys provided by the client ("auroc", "auprc")
        self.metric_priority = metric_priority or ["auroc", "auprc", "loss"]
        # --- END OF FIX ---
        
        self.best_metric_value = None
        self.best_round = None
        self.best_path = None

    def aggregate_fit(self, server_round: int, results, failures):
        """Call parent to aggregate, then save aggregated parameters as PT checkpoint."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is None:
            # nothing to save
            return None, None

        # Convert Parameters -> ndarray list
        try:
            nds = parameters_to_ndarrays(aggregated_parameters)
        except Exception as e:
            print(f"[SERVER][WARN] Could not convert aggregated parameters to ndarrays: {e}")
            return aggregated_parameters, aggregated_metrics

        # Build a model instance locally to map arrays -> state_dict
        try:
            model = build_model(self.model_name, n_features=self.n_features, seq_len=self.seq_len, device="cpu")
        except Exception as e:
            print(f"[SERVER][WARN] Failed to build model for saving: {e}")
            return aggregated_parameters, aggregated_metrics

        # Map arrays -> state_dict
        try:
            sd = arrays_to_state_dict_by_order(model, nds)
        except Exception as e:
            print(f"[SERVER][WARN] Failed to map arrays -> state_dict: {e}")
            # still return aggregated parameters so training can continue
            return aggregated_parameters, aggregated_metrics

        # Save PT checkpoint for this round
        ckpt_path = os.path.join(self.checkpoints_dir, f"global_round_{server_round}.pt")
        try:
            torch.save(sd, ckpt_path)
            print(f"[SERVER] Saved PT: {ckpt_path}")
        except Exception as e:
            print(f"[SERVER][WARN] Failed to save checkpoint for round {server_round}: {e}")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        """
        Call parent to aggregate evaluation metrics, then check for 'best' model selection.
        Parent returns (aggregated, metric) per FedAvg implementation. We will examine the aggregated metric.
        """
        aggregated, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # aggregated_metrics is typically a dict or float depending on flwr version
        # We'll try to extract a numeric value keyed by our priority.
        val = None
        chosen_key = None # Keep track of which key we found
        
        if isinstance(aggregated_metrics, dict):
            for key in self.metric_priority:
                if key in aggregated_metrics and aggregated_metrics[key] is not None:
                    try:
                        val = float(aggregated_metrics[key])
                        chosen_key = key
                        break
                    except (ValueError, TypeError):
                        pass # Ignore non-numeric metrics
                        
        if val is None and aggregated is not None:
             # Fallback to the primary aggregated metric (like loss) if no dict keys matched
            try:
                val = float(aggregated)
                chosen_key = "loss" # Assume the primary metric is loss
            except (ValueError, TypeError):
                val = None

        if val is not None and not np.isnan(val):
            # interpret metric sign: for 'loss' and 'val_loss' smaller is better; for AUC-like larger is better
            better = False
            if self.best_metric_value is None:
                better = True
            else:
                if chosen_key in ["loss", "val_loss"]:
                    better = (val < self.best_metric_value)
                else:
                    # Assume higher is better for all other keys (like auroc, auprc)
                    better = (val > self.best_metric_value)

            if better:
                # update best info and copy latest PT to best_name
                # source checkpoint for this round:
                src = os.path.join(self.checkpoints_dir, f"global_round_{server_round}.pt")
                dst = os.path.join(self.save_dir, self.best_name)
                try:
                    if os.path.exists(src):
                        # copy file
                        import shutil
                        shutil.copyfile(src, dst)
                        self.best_metric_value = val
                        self.best_round = server_round
                        self.best_path = dst
                        print(f"[SERVER][ROUND {server_round}] New best model (metric={chosen_key}={val:.4f}). Saved -> {dst}")
                    else:
                        print(f"[SERVER][WARN] Best candidate checkpoint missing for round {server_round}: {src}")
                except Exception as e:
                    print(f"[SERVER][WARN] Failed to copy best checkpoint: {e}")
        else:
            print(f"[SERVER][ROUND {server_round}] No usable numeric metric found in {aggregated_metrics}; skipping best selection.")

        return aggregated, aggregated_metrics


def build_initial_parameters_from_model(model_name: str, n_features: int, seq_len: int):
    """Build fl.common.Parameters (ndarrays) from a local fresh model."""
    model = build_model(model_name, n_features=n_features, seq_len=seq_len, device="cpu")
    sd = model.state_dict()
    nds = state_dict_to_ndarrays_by_order(sd)
    params = ndarrays_to_parameters(nds)
    return params


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--min_clients", type=int, default=1, help="min available clients to begin")
    ap.add_argument("--model", choices=["transformer", "grud"], default="transformer")
    ap.add_argument("--n_features", type=int, default=40)
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--save_dir", type=str, default="./server_out")
    ap.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    ap.add_argument("--best_name", type=str, default="global_best.pt")
    ap.add_argument("--fraction_fit", type=float, default=1.0)
    ap.add_argument("--fraction_evaluate", type=float, default=1.0)
    ap.add_argument("--min_fit_clients", type=int, default=None)
    ap.add_argument("--min_evaluate_clients", type=int, default=None)
    ap.add_argument("--round_timeout", type=int, default=None, help="Optional round timeout (seconds)")
    args = ap.parse_args()
    return args


def main():
    args = parse_args()
    
    # If min_fit or min_evaluate are not set, default them to min_clients
    if args.min_fit_clients is None:
        args.min_fit_clients = args.min_clients
    if args.min_evaluate_clients is None:
        args.min_evaluate_clients = args.min_clients

    server_address = f"{args.host}:{args.port}"
    print(f"[SERVER] Starting Flower server on {server_address} for {args.rounds} rounds...")
    print(f"[SERVER] Will require min_available_clients = {args.min_clients} to begin")

    # Build initial parameters from local model so server doesn't need a client to provide them
    try:
        initial_parameters = build_initial_parameters_from_model(args.model, args.n_features, args.seq_len)
        print("[SERVER] Built initial parameters from local untrained model.")
    except Exception as e:
        initial_parameters = None
        print(f"[SERVER][WARN] Failed to build initial parameters from model: {e}. Server will request from a client instead.")

    # Build a strategy instance
    # The 'initial_parameters' argument is passed to the strategy, NOT to start_server
    strategy = SaveEveryRoundFedAvg(
        model_name=args.model,
        n_features=args.n_features,
        seq_len=args.seq_len,
        save_dir=args.save_dir,
        checkpoints_dir=args.checkpoints_dir,
        best_name=args.best_name,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_clients,
        initial_parameters=initial_parameters,
    )

    # Build server config compatible with flwr
    server_config = fl.server.ServerConfig(num_rounds=args.rounds, round_timeout=args.round_timeout)

    # Ensure directories exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    # Start server (using compat start_server wrapper)
    try:
        fl.server.start_server(
            server_address=server_address,
            config=server_config,
            strategy=strategy,
        )
    except Exception as e:
        print(f"[SERVER][ERROR] Failed to start server: {e}")
        raise


if __name__ == "__main__":
    main()