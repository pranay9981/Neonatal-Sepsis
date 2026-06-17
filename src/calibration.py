# src/calibration.py
"""
Post-hoc probability calibration.

Temperature Scaling: fit a single scalar T on the validation set such that
  calibrated_prob = sigmoid(logit / T)
minimises NLL. A T > 1 softens overconfident predictions; T < 1 sharpens them.

Usage:
  from calibration import TemperatureScaler
  ts = TemperatureScaler()
  ts.fit(val_logits, val_labels)   # numpy arrays
  cal_probs = ts.calibrate(logits)
  ts.save("threshold.json")        # appends temperature key
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler:
    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 100) -> float:
        """
        Optimise temperature to minimise negative log-likelihood on val set.
        Returns the fitted temperature.
        """
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)

        T = nn.Parameter(torch.ones(1))
        opt = optim.LBFGS([T], lr=lr, max_iter=max_iter)
        loss_fn = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = loss_fn(logits_t / T.clamp(min=1e-6), labels_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(T.clamp(min=1e-6).item())
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.array(logits, dtype=np.float64) / self.temperature))

    def save(self, path: str):
        try:
            with open(path) as f:
                d = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            d = {}
        d["temperature"] = self.temperature
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(d, f, indent=2)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        ts = cls()
        with open(path) as f:
            d = json.load(f)
        temp = float(d.get("temperature", 1.0))
        if temp <= 0:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Loaded temperature %.4f is non-positive; resetting to 1.0", temp
            )
            temp = 1.0
        ts.temperature = temp
        return ts
