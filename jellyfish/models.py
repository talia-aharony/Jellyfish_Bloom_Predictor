"""
jellyfish/models.py
===================
Neural network and baseline models for jellyfish presence forecasting.

All sequence models expect:  (batch_size, lookback_days, n_features)
Baseline expects:            (batch_size, engineered_dim)
All models output:           (batch_size, 1)  — sigmoid probability

Model hierarchy (weakest → strongest):
  BaselineLogisticRegression  — hand-crafted features, linear classifier
  FeedforwardNet              — flattened MLP
  LSTMNet / GRUNet            — plain recurrent nets
  Conv1DNet                   — purely convolutional
  HybridNet                   — CNN + bidirectional GRU + attention  [old default]
  JellyfishNet                — recommended: mask-aware, position-encoded,
                                beach-embedding, residual GRU + calibrated head
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Baseline
# ============================================================================

class BaselineLogisticRegression(nn.Module):
    """Linear model over hand-crafted temporal features."""

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x.view(x.size(0), -1)))


# ============================================================================
# Simple neural nets
# ============================================================================

class FeedforwardNet(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x.view(x.size(0), -1)))


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout_prob=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            dropout=dropout_prob if num_layers > 1 else 0,
                            batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(),
                                  nn.Dropout(dropout_prob), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out[:, -1, :]))


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout_prob=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          dropout=dropout_prob if num_layers > 1 else 0,
                          batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(),
                                  nn.Dropout(dropout_prob), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return torch.sigmoid(self.head(out[:, -1, :]))


class Conv1DNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_prob=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,    kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout_prob),
            nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//2), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout_prob),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(hidden_dim//2, 32), nn.ReLU(),
                                  nn.Dropout(dropout_prob), nn.Linear(32, 1))

    def forward(self, x):
        x = self.conv(x.transpose(1, 2))
        return torch.sigmoid(self.head(self.pool(x).squeeze(-1)))


# ============================================================================
# Old Hybrid (kept for comparison)
# ============================================================================

class HybridNet(nn.Module):
    """CNN + bidirectional GRU + attention. Previous default model."""

    def __init__(self, input_dim, hidden_dim=64, dropout_prob=0.3):
        super().__init__()
        self.cnn_short  = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim//2, 3, padding=1),
            nn.BatchNorm1d(hidden_dim//2), nn.ReLU(), nn.Dropout(dropout_prob))
        self.cnn_medium = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim//2, 5, padding=2),
            nn.BatchNorm1d(hidden_dim//2), nn.ReLU(), nn.Dropout(dropout_prob))
        self.conv_fuse  = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim), nn.ReLU())

        gru_h = max(8, hidden_dim // 2)
        self.gru = nn.GRU(hidden_dim, gru_h, num_layers=1,
                          bidirectional=True, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim//2, 1))

    def forward(self, x):
        xt = x.transpose(1, 2)
        cnn = self.conv_fuse(torch.cat([self.cnn_short(xt), self.cnn_medium(xt)], dim=1))
        gru_out, _ = self.gru(cnn.transpose(1, 2))
        last = gru_out[:, -1, :]
        attn = torch.softmax(self.attention(gru_out).squeeze(-1), dim=1)
        ctx  = (gru_out * attn.unsqueeze(-1)).sum(1)
        return torch.sigmoid(self.head(torch.cat([last, ctx], dim=1)))


# ============================================================================
# JellyfishNet — recommended model
# ============================================================================

class _LearnedPositionEncoding(nn.Module):
    """Learnable positional embedding added to each timestep."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        return x + self.embed(pos).unsqueeze(0)


class _ResidualGRULayer(nn.Module):
    """Single GRU layer with a residual projection so dims can differ."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gru     = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.drop    = nn.Dropout(dropout)
        self.proj    = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.norm    = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        # x: (B, T, input_dim)
        out, _ = self.gru(x)              # (B, T, 2*hidden_dim)
        out    = self.drop(out)
        out    = self.norm(out + self.proj(x))
        return out


class JellyfishNet(nn.Module):
    """
    Mask-aware sequence model designed for sparse citizen-science data.

    Key design decisions and why they help here:
    ─────────────────────────────────────────────
    1. Observation mask as an explicit feature
       Zero-padded timesteps (no observation that day) look identical to a
       day where everything was genuinely zero.  By computing a binary mask
       from the raw input and appending it as an extra channel, the GRU can
       learn to ignore or down-weight missing days explicitly.

    2. Learned positional encoding
       With only 7 timesteps, "which day-slot in the window" matters — day 6
       (most recent) should be weighted differently from day 0.  A simple
       learnable embedding adds this signal cheaply.

     3. Two residual GRU layers with LayerNorm
       A single GRU layer is usually enough for 7 steps, but a second
       residual layer lets the model refine its representation without the
       gradient-vanishing risk of a deep stack, because the residual skip
       keeps gradients healthy.

     4. Masked soft-attention pooling
       Instead of just using the last hidden state (which ignores earlier
       high-bloom days), attention pools over all timesteps — but pads are
       masked to −∞ before softmax so they contribute nothing to the context.

     5. Temperature-scaled output
       A learned scalar τ ∈ (0.5, 2] scales logits before sigmoid.  This
       prevents the model from becoming overconfident on sparse data and
       improves calibration, which matters for the threshold selection in
       train.py.

    Parameters
    ----------
    input_dim    : number of raw features per timestep (e.g. 11 or 33)
    hidden_dim   : base hidden size (default 64)
    dropout_prob : dropout rate (default 0.3)
    max_len      : maximum sequence length (default 64, covers common lookbacks)
    """

    def __init__(
        self,
        input_dim:    int,
        hidden_dim:   int = 64,
        dropout_prob: float = 0.3,
        max_len:      int = 64,
    ):
        super().__init__()

        # ── Input projection ────────────────────────────────────────────────
        # +1 for the observation-mask channel
        proj_in = input_dim + 1
        self.input_proj = nn.Sequential(
            nn.Linear(proj_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )

        # ── Positional encoding ─────────────────────────────────────────────
        self.pos_enc = _LearnedPositionEncoding(max_len, hidden_dim)

        # ── Two residual GRU layers ─────────────────────────────────────────
        self.gru1 = _ResidualGRULayer(hidden_dim, hidden_dim, dropout_prob)
        self.gru2 = _ResidualGRULayer(hidden_dim, hidden_dim, dropout_prob)

        # ── Masked attention pooling ────────────────────────────────────────
        self.attn_score = nn.Linear(hidden_dim, 1)

        # ── Classification head ─────────────────────────────────────────────
        # Inputs: [last_hidden | context]
        head_in = hidden_dim * 2
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── Temperature scaling ─────────────────────────────────────────────
        # Initialised to 1 (no effect), learned during training
        self.log_temperature = nn.Parameter(torch.zeros(1))

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, beach_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (B, T, input_dim)  — normalised feature sequences
        beach_ids : unused, accepted for backward compatibility

        Returns
        -------
        (B, 1) sigmoid probabilities
        """
        B, T, _ = x.shape

        # 1. Observation mask: timestep is "observed" if ANY feature is non-zero
        mask = (x.abs().sum(dim=-1, keepdim=True) > 0).float()  # (B, T, 1)

        # 2. Append mask as extra input channel
        x_aug = torch.cat([x, mask], dim=-1)                     # (B, T, input_dim+1)

        # 3. Project to hidden_dim
        h = self.input_proj(x_aug)                               # (B, T, hidden_dim)

        # 4. Add positional encoding
        h = self.pos_enc(h)

        # 5. Residual GRU stack
        h = self.gru1(h)   # (B, T, hidden_dim)
        h = self.gru2(h)   # (B, T, hidden_dim)

        # 6. Last hidden state
        last = h[:, -1, :]                                        # (B, hidden_dim)

        # 7. Masked attention pooling
        scores = self.attn_score(h).squeeze(-1)                   # (B, T)
        pad_mask = (mask.squeeze(-1) == 0)                        # True where padded
        scores = scores.masked_fill(pad_mask, float("-inf"))
        # Guard against sequences where every step is padded
        all_padded = pad_mask.all(dim=1, keepdim=True)
        scores = scores.masked_fill(all_padded, 0.0)
        attn   = torch.softmax(scores, dim=1)                     # (B, T)
        ctx    = (h * attn.unsqueeze(-1)).sum(dim=1)              # (B, hidden_dim)

        # 8. Concatenate representations
        fused = torch.cat([last, ctx], dim=-1)

        # 9. Head + temperature scaling
        logits = self.head(fused)                                 # (B, 1)
        tau    = torch.clamp(self.log_temperature.exp(), 0.5, 2.0)
        return torch.sigmoid(logits * tau)
