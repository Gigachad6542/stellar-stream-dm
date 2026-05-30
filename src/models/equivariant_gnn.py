"""
E(n)-Equivariant Graph Neural Network for stellar stream analysis.

This module provides an equivariant GNN architecture that respects rotational
and translational symmetries of the 3D spatial coordinates. Standard GINEConv
treats coordinates as arbitrary features; equivariant layers guarantee that
rotating or translating the input produces correspondingly transformed outputs.

For stellar streams, the relevant symmetry is SE(3) — the group of 3D rotations
and translations — since the physics of subhalo impacts does not depend on the
orientation of the stream in galactocentric coordinates.

Architecture:
    - EGNN layers (Satorras et al. 2021) operating on 3D positions + scalar features
    - Position updates respect E(n) equivariance
    - Scalar messages remain invariant
    - Edge features incorporate relative distance (invariant scalar)

Key advantage over base GINEConv:
    The equivariant architecture guarantees that a stream rotated in 3D will
    produce the same embedding (up to the corresponding rotation of vector
    features). This eliminates the need for rotational data augmentation and
    provides a stronger inductive bias for the physical problem.

References:
    - Satorras, Hoogeboom, Welling 2021: "E(n) Equivariant Graph Neural Networks"
    - Brandstetter, Hesselink et al. 2022: "Geometric and Physical Quantities
      improve E(3) Equivariant Message Passing"
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data


class EGNNLayer(nn.Module):
    """Single E(n)-equivariant graph neural network layer.

    Implements the EGNN update from Satorras et al. 2021:
      - message: m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2, a_ij)
      - position update: x_i' = x_i + C * sum_j (x_i - x_j) * phi_x(m_ij)
      - node update: h_i' = phi_h(h_i, sum_j m_ij)

    where phi_e, phi_x, phi_h are MLPs and a_ij are edge attributes.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int = 0,
        coords_weight: float = 1.0,
        attention: bool = True,
        normalize: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coords_weight = coords_weight
        self.attention = attention
        self.normalize = normalize

        # Edge model: computes messages
        edge_input_dim = 2 * hidden_dim + 1 + edge_attr_dim  # h_i, h_j, dist^2, edge_attr
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Coordinate update: scalar weights for position shifts
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        # Initialize coord MLP with small weights for stability
        nn.init.xavier_uniform_(self.coord_mlp[0].weight, gain=0.001)
        nn.init.xavier_uniform_(self.coord_mlp[2].weight, gain=0.001)

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention (optional)
        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        h: torch.Tensor,          # [N, hidden_dim] node features
        x: torch.Tensor,          # [N, 3] coordinates
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor | None = None,  # [E, edge_attr_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            h: Node scalar features [N, hidden_dim]
            x: Node coordinates [N, 3]
            edge_index: Edge connectivity [2, E]
            edge_attr: Optional edge attributes [E, edge_attr_dim]

        Returns:
            Updated (h, x) tensors.
        """
        row, col = edge_index  # row=source, col=target

        # Compute relative positions and squared distances
        coord_diff = x[row] - x[col]  # [E, 3]
        radial = (coord_diff ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

        # Edge model input
        edge_input = torch.cat([h[row], h[col], radial], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)

        # Compute messages
        messages = self.edge_mlp(edge_input)  # [E, hidden_dim]

        # Attention
        if self.attention:
            att_weights = self.att_mlp(messages)  # [E, 1]
            messages = messages * att_weights

        # Coordinate update (equivariant)
        coord_weights = self.coord_mlp(messages)  # [E, 1]
        if self.normalize:
            # Normalize by distance to prevent explosions
            norm = torch.sqrt(radial + 1e-8)
            coord_diff = coord_diff / norm

        # Aggregate coordinate updates
        coord_agg = torch.zeros_like(x)  # [N, 3]
        coord_agg.index_add_(0, col, coord_diff * coord_weights)
        x = x + self.coords_weight * coord_agg

        # Node update (invariant)
        msg_agg = torch.zeros_like(h)  # [N, hidden_dim]
        msg_agg.index_add_(0, col, messages)

        h = h + self.node_mlp(torch.cat([h, msg_agg], dim=-1))

        return h, x


class EquivariantStreamGNN(nn.Module):
    """E(n)-Equivariant GNN encoder for stellar streams.

    Processes 3D coordinates (x, y, z in galactocentric frame) equivariantly
    while maintaining invariant scalar features (velocities, uncertainties).

    The embedding is invariant (not equivariant) — suitable for downstream
    inference tasks that should not depend on stream orientation.

    Architecture:
        - Input: 3D coords [N,3] + scalar features [N, scalar_dim]
        - n_layers EGNN layers updating both coords and features
        - Global pooling (mean + max) → invariant embedding
        - Output MLP → final embedding_dim

    Args:
        scalar_input_dim: Dimension of non-coordinate node features.
        coord_dim: Coordinate dimension (3 for 3D).
        hidden_dim: Hidden layer dimension.
        embedding_dim: Output embedding dimension.
        n_layers: Number of EGNN layers.
        edge_attr_dim: Edge attribute dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        scalar_input_dim: int = 15,  # 18 total - 3 coordinates
        coord_dim: int = 3,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        n_layers: int = 6,
        edge_attr_dim: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.scalar_input_dim = scalar_input_dim
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        # Project scalar features to hidden dim
        self.scalar_embed = nn.Sequential(
            nn.Linear(scalar_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # EGNN layers
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                edge_attr_dim=edge_attr_dim,
                coords_weight=1.0 / (i + 1),  # Decrease coord updates in later layers
                attention=True,
                normalize=True,
            )
            for i in range(n_layers)
        ])

        # Layer norms (applied to scalar features)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output MLP: pool_dim → embedding_dim
        pool_dim = 2 * hidden_dim  # mean + max
        self.output_mlp = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass.

        Expects data to have:
            data.x: [N, scalar_input_dim + coord_dim] (coords are first 3 columns)
            data.edge_index: [2, E]
            data.edge_attr: [E, edge_attr_dim] (optional)
            data.batch: [N] batch assignment

        Returns:
            Invariant embedding [B, embedding_dim]
        """
        # Split coordinates from scalar features
        coords = data.x[:, :self.coord_dim]  # [N, 3]
        scalars = data.x[:, self.coord_dim:]  # [N, scalar_input_dim]

        # Project scalars to hidden dim
        h = self.scalar_embed(scalars)  # [N, hidden_dim]

        # Edge attributes
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None

        # EGNN message passing
        for i, (layer, norm) in enumerate(zip(self.egnn_layers, self.layer_norms)):
            h_in = h
            h, coords = layer(h, coords, data.edge_index, edge_attr)
            h = norm(h)
            h = self.dropout(h)
            if i >= 1:  # Residual from layer 2 onward
                h = h + h_in

        # Global pooling (invariant)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        h_mean = global_mean_pool(h, batch)  # [B, hidden_dim]
        h_max = global_max_pool(h, batch)    # [B, hidden_dim]
        pooled = torch.cat([h_mean, h_max], dim=-1)  # [B, 2*hidden_dim]

        # Output embedding
        embedding = self.output_mlp(pooled)  # [B, embedding_dim]
        return embedding

    def get_coord_trajectory(self, data: Data) -> list[torch.Tensor]:
        """Return coordinate trajectory through layers (for visualization).

        Useful for understanding how the equivariant updates transform the
        input geometry.
        """
        coords = data.x[:, :self.coord_dim]
        scalars = data.x[:, self.coord_dim:]
        h = self.scalar_embed(scalars)
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None

        trajectory = [coords.detach().clone()]
        for layer, norm in zip(self.egnn_layers, self.layer_norms):
            h, coords = layer(h, coords, data.edge_index, edge_attr)
            h = norm(h)
            trajectory.append(coords.detach().clone())

        return trajectory
