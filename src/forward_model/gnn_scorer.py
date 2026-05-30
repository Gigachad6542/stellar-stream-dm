"""
GNN-based profile feature scorer for the forward model.

Bridges the forward model pipeline to the trained GNN encoder by:
    1. Converting StreamParticles -> PyG Data graph
    2. Passing through a pre-trained GNN encoder to get embeddings
    3. Computing Euclidean distance between simulated and observed embeddings

This gives a learned similarity metric that captures morphological patterns
the hand-crafted scorers (density, gap, kinematic) might miss — learned
directly from the training simulations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from ..data.dataset import (
    STREAM_NAMES,
    _STREAM_NAME_TO_IDX,
    FeatureNormalizer,
    build_knn_graph,
)
from ..simulation.stream_gen import StreamParticles

log = logging.getLogger(__name__)


def stream_particles_to_data(
    particles: StreamParticles,
    stream_name: str = "GD1",
    membership_prob: Optional[np.ndarray] = None,
) -> Data:
    """Convert StreamParticles to a PyG Data object for GNN inference.

    Builds the 18-dimensional node feature vector expected by the encoder:
        [phi1, phi2, dist, pm1, pm2, vrad,
         e_dist, e_pm1, e_pm2, e_vrad,
         membership_prob,
         stream_GD1, stream_Pal5, ..., stream_Sylgr]

    For simulated streams, error columns are set to typical Gaia values
    and membership_prob defaults to 1.0.

    Args:
        particles: StreamParticles from simulation or observation.
        stream_name: Stream name for one-hot encoding.
        membership_prob: Optional per-star membership probabilities.

    Returns:
        PyG Data object ready for graph construction + GNN forward pass.
    """
    n = len(particles.phi1)

    # Build one-hot stream encoding
    stream_idx = _STREAM_NAME_TO_IDX.get(stream_name, 0)
    onehot = np.zeros((n, len(STREAM_NAMES)), dtype=np.float32)
    onehot[:, stream_idx] = 1.0

    # Membership probability
    if membership_prob is None:
        mem = np.ones(n, dtype=np.float32)
    else:
        mem = np.asarray(membership_prob, dtype=np.float32)

    # Typical Gaia DR3 uncertainties for simulated data
    e_dist = np.full(n, 0.3, dtype=np.float32)     # ~0.3 kpc distance error
    e_pm1 = np.full(n, 0.1, dtype=np.float32)       # ~0.1 mas/yr PM error
    e_pm2 = np.full(n, 0.1, dtype=np.float32)
    e_vrad = np.full(n, 2.0, dtype=np.float32)      # ~2 km/s RV error

    # Stack: [phi1, phi2, dist, pm1, pm2, vrad, e_dist, e_pm1, e_pm2, e_vrad, mem, onehot(7)]
    features = np.column_stack([
        particles.phi1.astype(np.float32),
        particles.phi2.astype(np.float32),
        particles.dist.astype(np.float32),
        particles.pm1.astype(np.float32),
        particles.pm2.astype(np.float32),
        particles.vrad.astype(np.float32),
        e_dist, e_pm1, e_pm2, e_vrad,
        mem,
        onehot,
    ])  # [n, 18]

    x = torch.tensor(features, dtype=torch.float32)
    batch = torch.zeros(n, dtype=torch.long)

    return Data(x=x, batch=batch)


def obs_particles_to_data(
    obs: dict,
    stream_name: str = "GD1",
) -> Data:
    """Convert observed particle dict to a PyG Data object.

    Args:
        obs: Dict with keys phi1, phi2, pm1, pm2, dist, vrad,
             and optionally membership_prob.

    Returns:
        PyG Data object ready for graph construction + GNN forward pass.
    """
    n = len(obs["phi1"])

    stream_idx = _STREAM_NAME_TO_IDX.get(stream_name, 0)
    onehot = np.zeros((n, len(STREAM_NAMES)), dtype=np.float32)
    onehot[:, stream_idx] = 1.0

    mem = obs.get("membership_prob", np.ones(n))

    # Use typical errors (real errors from HDF5 would be better but aren't
    # always available in the forward model's filtered particle dict)
    e_dist = np.full(n, 0.3, dtype=np.float32)
    e_pm1 = np.full(n, 0.1, dtype=np.float32)
    e_pm2 = np.full(n, 0.1, dtype=np.float32)
    e_vrad = np.full(n, 2.0, dtype=np.float32)

    features = np.column_stack([
        np.asarray(obs["phi1"], dtype=np.float32),
        np.asarray(obs["phi2"], dtype=np.float32),
        np.asarray(obs["dist"], dtype=np.float32),
        np.asarray(obs["pm1"], dtype=np.float32),
        np.asarray(obs["pm2"], dtype=np.float32),
        np.asarray(obs["vrad"], dtype=np.float32),
        e_dist, e_pm1, e_pm2, e_vrad,
        np.asarray(mem, dtype=np.float32),
        onehot,
    ])

    x = torch.tensor(features, dtype=torch.float32)
    batch = torch.zeros(n, dtype=torch.long)

    return Data(x=x, batch=batch)


class GNNProfileScorer:
    """Scores forward model candidates using GNN embedding distance.

    Loads a pre-trained GNN encoder checkpoint, computes embeddings for the
    observed stream (once) and each simulated candidate, then returns the
    Euclidean distance between them.

    Usage:
        scorer = GNNProfileScorer("checkpoints/gnn_best.pt")
        scorer.set_observed_embedding(obs_particles, stream_name="GD1")
        dist = scorer.score(sim_particles)
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/gnn_best.pt",
        device: str = "cpu",
        k_neighbors: int = 16,
    ):
        self.device = torch.device(device)
        self.k = k_neighbors
        self.encoder = None
        self.normalizer = None
        self.obs_embedding = None
        self._checkpoint_path = checkpoint_path

        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        """Load the GNN encoder from a training checkpoint."""
        from ..models.gnn import StreamGNNEncoder

        ckpt_path = Path(path)
        if not ckpt_path.exists():
            log.warning("GNN checkpoint not found at %s; scorer will be disabled", path)
            return

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        gnn_cfg = cfg.get("model", {}).get("gnn", {})

        # Build encoder with same architecture as training
        self.encoder = StreamGNNEncoder(
            n_node_features=cfg.get("graph", {}).get("n_node_features", 18),
            n_edge_features=cfg.get("graph", {}).get("n_edge_features", 5),
            hidden_dim=gnn_cfg.get("hidden_dim", 256),
            embedding_dim=gnn_cfg.get("embedding_dim", 128),
            n_layers=gnn_cfg.get("n_layers", 6),
            dropout=gnn_cfg.get("dropout", 0.1),
        )

        # Load weights (filter out non-encoder keys if checkpoint has multi-task heads)
        state_dict = ckpt["model_state_dict"]
        encoder_keys = {k for k in state_dict if k.startswith("encoder.")}
        if encoder_keys:
            # Multi-task checkpoint: extract encoder weights
            encoder_state = {k.replace("encoder.", "", 1): v
                             for k, v in state_dict.items() if k.startswith("encoder.")}
            self.encoder.load_state_dict(encoder_state, strict=False)
        else:
            # Standalone encoder checkpoint
            self.encoder.load_state_dict(state_dict, strict=False)

        self.encoder.to(self.device)
        self.encoder.eval()

        # Load normalizer if available
        if "normalizer_mean" in ckpt and "normalizer_std" in ckpt:
            self.normalizer = FeatureNormalizer(
                ckpt["normalizer_mean"].numpy(),
                ckpt["normalizer_std"].numpy(),
            )

        log.info("GNN scorer loaded from %s (embedding_dim=%d)",
                 path, gnn_cfg.get("embedding_dim", 128))

    @property
    def is_available(self) -> bool:
        """True if the encoder loaded successfully."""
        return self.encoder is not None

    @torch.no_grad()
    def _embed(self, data: Data) -> Optional[np.ndarray]:
        """Compute embedding for a single Data object.

        If the graph is too large (>max_stars), subsample to avoid memory issues.
        Returns None if embedding contains NaN (graceful degradation).
        """
        max_stars = 3000  # match training max

        # Subsample if too many stars
        n = data.x.shape[0]
        if n > max_stars:
            idx = torch.randperm(n)[:max_stars]
            data = Data(
                x=data.x[idx],
                batch=torch.zeros(max_stars, dtype=torch.long),
            )

        # Replace any NaN/Inf in features with 0
        data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)

        # Build k-NN graph
        edge_index, edge_attr = build_knn_graph(data.x, self.k, self.normalizer)
        data.edge_index = edge_index.to(self.device)
        data.edge_attr = edge_attr.to(self.device)
        data.x = data.x.to(self.device)
        data.batch = data.batch.to(self.device)

        embedding = self.encoder(data)  # [1, embedding_dim]
        emb_np = embedding.cpu().numpy().flatten()

        # Guard against NaN embeddings
        if not np.isfinite(emb_np).all():
            log.warning("  GNN embedding contains NaN/Inf; returning None")
            return None

        return emb_np

    def set_observed_embedding(
        self,
        obs_particles: dict,
        stream_name: str = "GD1",
    ) -> None:
        """Compute and cache the observed stream's GNN embedding.

        Call this once during prepare(), then reuse for all candidates.
        """
        if not self.is_available:
            return

        data = obs_particles_to_data(obs_particles, stream_name)
        self.obs_embedding = self._embed(data)
        if self.obs_embedding is not None:
            log.info("  GNN observed embedding computed (dim=%d, norm=%.2f)",
                     len(self.obs_embedding), float(np.linalg.norm(self.obs_embedding)))
        else:
            log.warning("  GNN observed embedding failed (NaN); scorer disabled")

    def score(
        self,
        sim_particles: StreamParticles,
        stream_name: str = "GD1",
    ) -> float:
        """Compute embedding distance between simulated and observed streams.

        Returns:
            Euclidean distance in embedding space (lower = more similar).
            Returns 0.0 if scorer is not available or embeddings are invalid.
        """
        if not self.is_available or self.obs_embedding is None:
            return 0.0

        data = stream_particles_to_data(sim_particles, stream_name)
        sim_embedding = self._embed(data)

        if sim_embedding is None:
            return 0.0

        dist = float(np.linalg.norm(sim_embedding - self.obs_embedding))
        return dist
