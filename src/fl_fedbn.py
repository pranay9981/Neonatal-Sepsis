# src/fl_fedbn.py
"""
FedBN strategy for Flower: keeps BatchNorm parameters local (not aggregated).

FedBN (Li et al., 2021) prevents global aggregation of BatchNorm statistics,
allowing each hospital's model to preserve its local feature distribution.
This is more effective than FedProx on heterogeneous (non-IID) data when
BatchNorm layers are present.

Usage in fl_server.py:
  from fl_fedbn import FedBNStrategy
  strategy = FedBNStrategy(model_name=..., ...)
"""
import os
import shutil
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from logging_config import get_logger

logger = get_logger(__name__)


class FedBNStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg variant that excludes BatchNorm parameters from aggregation.
    Each client keeps its own BN statistics; only non-BN params are averaged.
    """

    BN_KEYWORDS = ("running_mean", "running_var", "num_batches_tracked",
                   "batchnorm", "batch_norm", "layer_norm", "layernorm")

    def __init__(self, model_name: str, n_features: int, seq_len: int,
                 save_dir: str, checkpoints_dir: str,
                 best_name: str = "global_best.pt", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.n_features = n_features
        self.seq_len = seq_len
        self.save_dir = save_dir
        self.checkpoints_dir = checkpoints_dir
        self.best_name = best_name
        self.best_metric = None
        self.best_round = None
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

    def _is_bn_key(self, key: str) -> bool:
        return any(kw in key.lower() for kw in self.BN_KEYWORDS)

    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            return None, {}

        # Separate BN keys from the rest using key ordering from first result
        weights_results = [(fl.common.parameters_to_ndarrays(fit_res.parameters),
                            fit_res.num_examples)
                           for _, fit_res in results]

        # Build reference model to get key names
        try:
            from fl_server import build_model, arrays_to_state_dict_by_order
            ref_model = build_model(self.model_name, self.n_features, self.seq_len)
            keys = list(ref_model.state_dict().keys())
        except Exception as e:
            logger.error(
                "FedBN: could not introspect model keys (%s) — "
                "falling back to FedAvg. BatchNorm/LayerNorm params WILL be globally averaged.", e
            )
            return super().aggregate_fit(server_round, results, failures)

        # Compute weighted average for non-BN parameters only
        total_examples = sum(n for _, n in weights_results)
        if total_examples == 0:
            logger.warning("FedBN: total_examples == 0, skipping round %d", server_round)
            return None, {}
        for client_idx, (w, _) in enumerate(weights_results):
            if len(w) != len(keys):
                logger.warning(
                    "FedBN: client %d weight count mismatch (%d vs %d), falling back to FedAvg",
                    client_idx, len(w), len(keys),
                )
                return super().aggregate_fit(server_round, results, failures)
        agg_arrays = []
        first_client_arrays = weights_results[0][0]
        for i, key in enumerate(keys):
            if self._is_bn_key(key):
                if i < len(first_client_arrays):
                    agg_arrays.append(first_client_arrays[i])
                else:
                    agg_arrays.append(ref_model.state_dict()[key].cpu().numpy())
            else:
                weighted = np.zeros_like(weights_results[0][0][i])
                for w, n in weights_results:
                    weighted += w[i] * (n / total_examples)
                agg_arrays.append(weighted)

        aggregated_params = ndarrays_to_parameters(agg_arrays)

        # Save round checkpoint
        try:
            sd = arrays_to_state_dict_by_order(ref_model, agg_arrays)
            ckpt = os.path.join(self.checkpoints_dir, f"global_round_{server_round}.pt")
            torch.save(sd, ckpt)
            logger.info("FedBN: saved round %d checkpoint", server_round)
        except Exception as e:
            logger.warning("FedBN: could not save checkpoint: %s", e)

        return aggregated_params, {}

    def aggregate_evaluate(self, server_round: int, results, failures):
        agg, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Prefer AUROC/AUPRC (higher = better); fall back to negated loss so the
        # same "higher is better" comparison works in both cases.
        val = None
        for key in ("auroc", "auprc"):
            if isinstance(metrics, dict) and key in metrics:
                try:
                    val = float(metrics[key])
                    break
                except (ValueError, TypeError):
                    pass

        if val is None and agg is not None:
            try:
                val = -float(agg)  # negate: lower loss → higher score
            except (ValueError, TypeError):
                pass

        if val is not None and not np.isnan(val):
            if self.best_metric is None or val > self.best_metric:
                src = os.path.join(self.checkpoints_dir, f"global_round_{server_round}.pt")
                dst = os.path.join(self.save_dir, self.best_name)
                if os.path.exists(src):
                    shutil.copyfile(src, dst)
                    self.best_metric = val
                    self.best_round = server_round
                    logger.info("FedBN: new best (round %d, metric=%.4f)", server_round, val)
        return agg, metrics
