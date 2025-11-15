# src/model_grud.py
"""
GRU-D style model (simplified, robust).
- Projects input features to hidden_size.
- Projects masks to hidden_size so shapes match.
- Projects deltas to a scalar per timestep then derives per-hidden gamma via a linear layer.
This keeps shapes consistent and avoids runtime size errors on mixed feature/hidden dims.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUDCell(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # GRU-like gates working in hidden space
        self.z_x = nn.Linear(hidden_size, hidden_size)
        self.z_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.r_x = nn.Linear(hidden_size, hidden_size)
        self.r_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_x = nn.Linear(hidden_size, hidden_size)
        self.h_h = nn.Linear(hidden_size, hidden_size, bias=False)
        # decay from scalar delta to hidden gating (gamma)
        self.decay = nn.Linear(1, hidden_size)

    def forward(self, x_h, mask_h, delta_scalar, h):
        """
        x_h: (B, H)  - projected input at time t
        mask_h: (B, H) - projected mask at time t
        delta_scalar: (B, 1) - scalar delta (time since last observation) for timestep t
        h: (B, H) - previous hidden
        """
        # compute gamma from delta_scalar -> (B, H)
        gamma = torch.exp(-F.relu(self.decay(delta_scalar)))  # (B, H)
        # apply simple decay to the input when missing: x_hat = mask*x + (1-mask)*(gamma * x_h)
        x_hat = mask_h * x_h + (1.0 - mask_h) * (gamma * x_h)

        z = torch.sigmoid(self.z_x(x_hat) + self.z_h(h))
        r = torch.sigmoid(self.r_x(x_hat) + self.r_h(h))
        h_tilde = torch.tanh(self.h_x(x_hat) + self.h_h(r * h))
        h_new = (1 - z) * h + z * h_tilde
        return h_new

class GRUD(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=1, dropout=0.1):
        """
        n_features: number of raw features (F)
        hidden_size: GRU hidden dimension (H)
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # project raw input features -> hidden space
        self.input_proj = nn.Linear(n_features, hidden_size)

        # project mask (F) -> hidden space so mask shape matches projected input
        self.mask_proj = nn.Linear(n_features, hidden_size)

        # project deltas (F) -> scalar per timestep (we'll average across features)
        # then decay layer in GRUDCell will map scalar -> hidden gamma
        self.delta_proj = nn.Linear(n_features, 1)

        # stack of GRUDCells (we keep a simple single-layer cell by default)
        self.cells = nn.ModuleList([GRUDCell(hidden_size) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

        # classifier on final hidden
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask, deltas):
        """
        x: (B, T, F) - raw features (may have NaNs replaced by zeros by dataset)
        mask: (B, T, F) - 1.0 if observed, 0.0 if missing
        deltas: (B, T, F) or (B, T, 1) - time-since-last-observation (dataset provides)
        Returns logits: (B,)
        """
        B, T, F = x.shape
        device = x.device

        # Project inputs to hidden space
        x_h = self.input_proj(x)            # (B, T, H)
        mask_h = self.mask_proj(mask.float())  # (B, T, H)

        # Create timestep scalar delta: if deltas has shape (B,T,F) average across features -> (B,T,1)
        if deltas is None:
            delta_scalar = torch.zeros((B, T, 1), device=device)
        else:
            if deltas.dim() == 3 and deltas.shape[2] == F:
                # average per-timestep across features (robust simple choice)
                delta_scalar = deltas.mean(dim=2, keepdim=True)  # (B, T, 1)
            elif deltas.dim() == 3 and deltas.shape[2] == 1:
                delta_scalar = deltas  # already (B,T,1)
            elif deltas.dim() == 2:
                delta_scalar = deltas.unsqueeze(-1)  # (B,T,1)
            else:
                # fallback: zeros
                delta_scalar = torch.zeros((B, T, 1), device=device)

        # initial hidden state
        h = torch.zeros(B, self.hidden_size, device=device)

        # iterate over timesteps
        for t in range(T):
            xt = x_h[:, t, :]            # (B, H)
            mt = mask_h[:, t, :]         # (B, H)
            dt = delta_scalar[:, t, :]   # (B, 1)

            # pass through stacked cells (simple stack: feed output to next cell)
            h_current = h
            for i, cell in enumerate(self.cells):
                h_current = cell(xt, mt, dt, h_current)
            h = self.dropout(h_current)

        logits = self.classifier(h).squeeze(-1)  # (B,)
        return logits
