# src/model_grud.py
"""
GRU-D: Recurrent Neural Networks for Multivariate Time Series with Missing Values.
(Che et al., 2018 — https://www.nature.com/articles/s41598-018-24271-9)

Faithful implementation of three key mechanisms:
1. Per-feature input decay (γ_x): missing values decay toward the empirical mean,
   not toward zero. The decay rate is feature-specific based on time-since-last-obs.
2. Tracking last observed value (x_last): imputation uses the most recently
   observed value as the starting point for decay, rather than always decaying from mean.
3. Hidden-state decay (γ_h): the previous hidden state is decayed before the GRU
   gates, capturing that stale context is less reliable.

set_empirical_mean() should be called with the per-feature training mean (from
scaler.json) before training begins. Defaults to zeros if not called.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUD(nn.Module):
    def __init__(self, n_features, hidden_size=128, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        # Per-feature input decay: delta (F,) → gamma_x (F,)
        self.W_gamma_x = nn.Linear(n_features, n_features, bias=True)

        # Hidden-state decay: scalar mean-delta (1,) → gamma_h (H,)
        self.W_gamma_h = nn.Linear(1, hidden_size, bias=True)

        # Project imputed input to hidden space
        self.input_proj = nn.Linear(n_features, hidden_size)

        # Standard GRU gates operating in hidden space
        self.z_x = nn.Linear(hidden_size, hidden_size)
        self.z_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.r_x = nn.Linear(hidden_size, hidden_size)
        self.r_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_x = nn.Linear(hidden_size, hidden_size)
        self.h_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Empirical per-feature mean — registered as a non-trainable buffer.
        # Call set_empirical_mean() after loading the scaler.
        self.register_buffer('x_mean', torch.zeros(n_features))

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def set_empirical_mean(self, x_mean: torch.Tensor):
        """Load the training-set per-feature mean into the buffer."""
        self.x_mean.copy_(x_mean.to(self.x_mean.device))

    def forward(self, x, mask, deltas):
        """
        x:      (B, T, F) imputed features (0 where missing)
        mask:   (B, T, F) 1=observed, 0=missing
        deltas: (B, T, F) time since last observation per feature (in timesteps)
        Returns logits: (B,)
        """
        B, T, _ = x.shape
        device = x.device
        x_mean = self.x_mean.to(device)  # (n_features,)

        h = torch.zeros(B, self.hidden_size, device=device)

        # x_last tracks the most recently observed value per sample per feature.
        # Initialised to the empirical mean, matching the GRU-D paper.
        x_last = x_mean.unsqueeze(0).expand(B, -1).clone()  # (B, F)

        for t in range(T):
            xt = x[:, t, :]       # (B, F)
            mt = mask[:, t, :]    # (B, F)
            dt = deltas[:, t, :]  # (B, F)

            # Per-feature input decay coefficient γ_x ∈ (0, 1]
            gamma_x = torch.exp(-F.relu(self.W_gamma_x(dt)))  # (B, F)

            # GRU-D imputation: x̂ = m⊙x + (1−m)⊙(γ_x⊙x_last + (1−γ_x)⊙x̄)
            x_hat = mt * xt + (1.0 - mt) * (gamma_x * x_last + (1.0 - gamma_x) * x_mean)

            # Advance x_last: use freshly observed value; otherwise keep previous.
            x_last = mt * xt + (1.0 - mt) * x_last

            # Scalar delta for hidden decay (mean across features)
            delta_scalar = dt.mean(dim=-1, keepdim=True)  # (B, 1)
            gamma_h = torch.exp(-F.relu(self.W_gamma_h(delta_scalar)))  # (B, H)
            h_decayed = gamma_h * h  # hidden-state decay

            # GRU update using decayed hidden state
            x_h = self.input_proj(x_hat)  # (B, H)
            z = torch.sigmoid(self.z_x(x_h) + self.z_h(h_decayed))
            r = torch.sigmoid(self.r_x(x_h) + self.r_h(h_decayed))
            h_tilde = torch.tanh(self.h_x(x_h) + self.h_h(r * h_decayed))
            h = (1.0 - z) * h_decayed + z * h_tilde
            h = self.dropout(h)

        return self.classifier(h).squeeze(-1)  # (B,)
