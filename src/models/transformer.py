"""
Transformer encoder for stellar streams (optional ensemble member).

Treats the stream as a sequence of stars ordered by phi1.
Positional encoding uses phi1 as a continuous physical coordinate (not learned).

Integration status:
    **Experimental / not wired into the training pipeline.**
    The GNN (GINEConv) is the primary encoder used for all training, inference,
    and results. This transformer exists as an optional ensemble member for
    future work (e.g. weighted average of GNN + transformer embeddings before
    SBI). To integrate it:
    1. Set ``model.transformer.enabled = true`` in ``config/training.yaml``.
    2. Add a ``--model transformer`` branch to ``scripts/train.py``.
    3. Use ``collate_for_transformer()`` as the DataLoader collate function.
    4. At inference time, average GNN and transformer embeddings before SBI.

Notes:
    - O(N^2) attention → subsamples to max_stars (default 3000) for large streams.
    - Use as ensemble member alongside GNN, not as primary model.
    - Requires padded batching (see ``collate_for_transformer``), unlike the
      GNN which uses PyG's native batching.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousPositionalEncoding(nn.Module):
    """Encode stream longitude phi1 as a continuous positional signal.

    Uses sinusoidal encoding: [sin(phi1/L), cos(phi1/L), phi1/L]
    where L is the stream length scale (default 120 deg = full phi1 range).
    """

    def __init__(self, d_model: int, L: float = 120.0) -> None:
        super().__init__()
        self.L = L
        self.proj = nn.Linear(3, d_model)

    def forward(self, phi1: torch.Tensor) -> torch.Tensor:
        """Args:
            phi1: [B, N] stream longitudes in degrees.
        Returns:
            [B, N, d_model] positional encoding.
        """
        phi1_norm = phi1 / self.L
        enc = torch.stack([
            torch.sin(phi1_norm),
            torch.cos(phi1_norm),
            phi1_norm,
        ], dim=-1)  # [B, N, 3]
        return self.proj(enc)


class StreamTransformer(nn.Module):
    """Transformer encoder mapping a stream star sequence to a fixed embedding.

    Args:
        n_node_features: Per-star input feature dimension (default 11).
        d_model: Transformer model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        embedding_dim: Output embedding size.
        dropout: Dropout probability.
        max_stars: Subsample streams longer than this (O(N^2) guard).
        phi1_scale: Denominator for positional encoding (degrees).
    """

    def __init__(
        self,
        n_node_features: int = 11,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        max_stars: int = 3000,
        phi1_scale: float = 120.0,
    ) -> None:
        super().__init__()
        self.max_stars = max_stars
        self.d_model = d_model

        # Input projection: node features -> d_model
        self.input_proj = nn.Linear(n_node_features, d_model)

        # Continuous positional encoding using phi1
        self.pos_enc = ContinuousPositionalEncoding(d_model, L=phi1_scale)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output: mean over tokens -> MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, embedding_dim),
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a padded batch of stream sequences.

        Args:
            x: [B, N, n_node_features] — padded star features (batch_first).
            src_key_padding_mask: [B, N] bool — True for padding positions.

        Returns:
            [B, embedding_dim] embeddings.
        """
        # Subsample if too long (O(N^2) guard)
        if x.shape[1] > self.max_stars:
            idx = torch.randperm(x.shape[1])[:self.max_stars]
            x = x[:, idx, :]
            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask[:, idx]

        # Sort by phi1 (index 0 in feature vector)
        phi1 = x[:, :, 0]  # [B, N]
        sort_idx = torch.argsort(phi1, dim=1)
        x = torch.gather(x, 1, sort_idx.unsqueeze(-1).expand_as(x))
        phi1_sorted = torch.gather(phi1, 1, sort_idx)

        # Embed node features + positional encoding
        h = self.input_proj(x)  # [B, N, d_model]
        h = h + self.pos_enc(phi1_sorted)

        # Transformer encoding
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)

        # Mean pool over non-padding positions
        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).float().unsqueeze(-1)  # [B, N, 1]
            h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            h = h.mean(dim=1)

        return self.output_mlp(h)


def collate_for_transformer(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate PyG Data objects into padded tensors for the transformer.

    Args:
        batch: List of Data objects with x [N_i, 11].

    Returns:
        x_padded: [B, N_max, 11]
        padding_mask: [B, N_max] bool, True for padding positions.
    """
    max_n = max(d.x.shape[0] for d in batch)
    n_feat = batch[0].x.shape[1]
    B = len(batch)

    x_padded = torch.zeros(B, max_n, n_feat)
    mask = torch.ones(B, max_n, dtype=torch.bool)

    for i, d in enumerate(batch):
        n = d.x.shape[0]
        x_padded[i, :n] = d.x
        mask[i, :n] = False  # False = valid position

    return x_padded, mask
