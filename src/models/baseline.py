"""
1D CNN baseline operating on binned density profiles.

Input: [B, 3, 200] tensor — 3 channels (stellar density, mean pm1, mean pm2)
       each binned into 200 phi1 bins.
Output: [B, embedding_dim] embedding fed to the classifier head or SBI.

Must achieve >70% accuracy on the 4-class DM model classification task
before the GNN encoder is considered.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_1d_profile(
    phi1: np.ndarray,
    pm1: np.ndarray,
    pm2: np.ndarray,
    n_bins: int = 200,
    phi1_min: float = -100.0,
    phi1_max: float = 20.0,
) -> np.ndarray:
    """Bin stream stars into a 3-channel 1D profile.

    Channel 0: Gaussian-smoothed stellar density (stars/bin).
    Channel 1: Mean pm1 per bin (0 where empty).
    Channel 2: Mean pm2 per bin (0 where empty).

    Returns [3, n_bins] float32 array.
    """
    bins = np.linspace(phi1_min, phi1_max, n_bins + 1)
    density, _ = np.histogram(phi1, bins=bins)
    density = density.astype(np.float32)

    pm1_sum = np.zeros(n_bins, dtype=np.float32)
    pm2_sum = np.zeros(n_bins, dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.float32)
    idx = np.searchsorted(bins[:-1], phi1, side="right") - 1
    idx = np.clip(idx, 0, n_bins - 1)
    np.add.at(pm1_sum, idx, pm1)
    np.add.at(pm2_sum, idx, pm2)
    np.add.at(counts, idx, 1)
    safe = counts > 0
    pm1_mean = np.where(safe, pm1_sum / np.where(safe, counts, 1), 0.0)
    pm2_mean = np.where(safe, pm2_sum / np.where(safe, counts, 1), 0.0)

    # Smooth density with a narrow Gaussian kernel (sigma ~ 2 bins)
    from scipy.ndimage import gaussian_filter1d  # noqa: PLC0415
    density_smooth = gaussian_filter1d(density, sigma=2.0)

    # Normalise each channel to zero mean, unit variance
    def _norm(x):
        mu, s = x.mean(), x.std()
        return (x - mu) / (s + 1e-6)

    return np.stack([_norm(density_smooth), _norm(pm1_mean), _norm(pm2_mean)], axis=0)


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(self.bn2(self.conv2(x)))
        return F.relu(x + res)


class DensityProfileCNN(nn.Module):
    """1D CNN encoder for stellar density profiles.

    Architecture:
        Input conv: [B, 3, 200] -> [B, 32, 200]
        4 multi-scale conv branches (kernels 3, 5, 9, 15) + concatenate
        ResBlocks -> global avg pool -> MLP -> embedding

    Args:
        n_bins: Number of phi1 bins (default 200).
        embedding_dim: Output embedding dimension.
        n_classes: If >0, add a classification head for DM model prediction.
    """

    def __init__(
        self,
        n_bins: int = 200,
        in_channels: int = 3,
        embedding_dim: int = 128,
        n_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.embedding_dim = embedding_dim

        # Initial feature extraction
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # Multi-scale branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            for k in [3, 5, 9, 15]
        ])
        # 4 branches * 64 = 256 channels after concat

        self.res1 = ResBlock1D(256, 5, dropout)
        self.res2 = ResBlock1D(256, 5, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pooling

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, n_classes) if n_classes > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning embeddings [B, embedding_dim]."""
        x = self.input_proj(x)
        branches = [b(x) for b in self.branches]
        # Ensure all branches have same length (pad if kernel gives different size)
        x = torch.cat(branches, dim=1)  # [B, 256, ~200]
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x).squeeze(-1)  # [B, 256]
        return self.mlp(x)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for 4-class DM model classification."""
        emb = self.forward(x)
        if self.classifier is None:
            raise RuntimeError("No classification head; set n_classes > 0.")
        return self.classifier(emb)
