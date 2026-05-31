"""
PyTorch Dataset and DataLoader for simulated and real stellar stream data.

Both classes return torch_geometric.data.Data objects with identical structure
so the trained model applies to real data without modification.

Graph construction: k-NN in (phi1_norm, phi2_norm, pm1_norm, pm2_norm) 4D space.
Node features (18): phi1, phi2, dist, pm1, pm2, vrad, e_dist, e_pm1, e_pm2, e_vrad,
                     membership_prob, + 7 stream one-hot (GD1, Pal5, Orphan, ATLAS, Jhelum, Fjorm, Sylgr)
Edge features (5): delta_phi1, delta_phi2, delta_pm1, delta_pm2, euclidean_dist_4d
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from torch_geometric.data import Data
def _fallback_knn_graph(x, k, batch=None, loop=False, flow='source_to_target',
                        cosine=False, num_workers=1):
    """Pure-torch k-NN graph when torch-cluster is unavailable or broken."""
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    all_src, all_dst = [], []
    for g in batch.unique():
        mask = batch == g
        idx = mask.nonzero(as_tuple=True)[0]
        x_g = x[idx]
        dist = torch.cdist(x_g, x_g)
        if not loop:
            dist.fill_diagonal_(float('inf'))
        _, topk_idx = dist.topk(k, dim=1, largest=False)
        n_g = x_g.size(0)
        row = torch.arange(n_g, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)
        col = topk_idx.reshape(-1)
        # Map back to global indices
        all_src.append(idx[row])
        all_dst.append(idx[col])
    src = torch.cat(all_src)
    dst = torch.cat(all_dst)
    if flow == 'source_to_target':
        return torch.stack([src, dst], dim=0)
    else:
        return torch.stack([dst, src], dim=0)


# Try importing knn_graph from torch_geometric (requires torch-cluster).
# If torch-cluster is missing or broken (ABI mismatch), fall back gracefully.
try:
    from torch_geometric.nn import knn_graph as _pyg_knn_graph
    # Verify it actually works (pyg uses a lazy proxy that only errors on call)
    import torch_geometric.typing
    if torch_geometric.typing.WITH_TORCH_CLUSTER:
        knn_graph = _pyg_knn_graph
    else:
        knn_graph = _fallback_knn_graph
except (ImportError, OSError):
    knn_graph = _fallback_knn_graph
# h5py MUST be imported after torch_geometric on Windows — importing it before
# triggers a DLL conflict (STATUS_ACCESS_VIOLATION 0xC0000005) in the loky workers.
import h5py

log = logging.getLogger(__name__)

NODE_FEATURE_NAMES = ["phi1", "phi2", "dist", "pm1", "pm2", "vrad",
                      "e_dist", "e_pm1", "e_pm2", "e_vrad", "membership_prob",
                      "stream_GD1", "stream_Pal5", "stream_Orphan", "stream_ATLAS",
                      "stream_Jhelum", "stream_Fjorm", "stream_Sylgr"]
N_NODE_FEATURES = 18
N_EDGE_FEATURES = 5
CONSTRUCTION_FEATURE_IDX = [0, 1, 3, 4]  # phi1, phi2, pm1, pm2

# Ordered stream names for one-hot encoding — must match config/streams.yaml order.
# Index 0→GD1, 1→Pal5, ..., 6→Sylgr.
STREAM_NAMES = ["GD1", "Pal5", "Orphan", "ATLAS", "Jhelum", "Fjorm", "Sylgr"]
N_STREAMS = len(STREAM_NAMES)
_STREAM_NAME_TO_IDX = {name: i for i, name in enumerate(STREAM_NAMES)}


def stream_one_hot(stream_name: str, n_stars: int) -> np.ndarray:
    """Return [n_stars, 7] one-hot array for the given stream name."""
    idx = _STREAM_NAME_TO_IDX.get(stream_name, -1)
    onehot = np.zeros((n_stars, N_STREAMS), dtype=np.float32)
    if idx >= 0:
        onehot[:, idx] = 1.0
    else:
        log.warning("Unknown stream name '%s' — one-hot will be all zeros", stream_name)
    return onehot


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class FeatureNormalizer:
    """Standardize node features using per-feature mean/std.

    Attributes:
        mean: Per-feature mean values [n_features].
        std: Per-feature std values [n_features], clamped to min 0.01.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        if len(mean) != len(std):
            raise ValueError(f"mean ({len(mean)}) and std ({len(std)}) must have same length")
        self.mean = torch.tensor(mean, dtype=torch.float32)
        # Clamp std to a minimum of 0.01 to prevent near-constant features
        # (e.g. membership_prob=1.0 in all sims) from producing extreme values
        # when applied to real data with slightly different distributions.
        self.std = torch.tensor(std, dtype=torch.float32).clamp(min=0.01)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def __repr__(self) -> str:
        return f"FeatureNormalizer(n_features={len(self.mean)})"

    @classmethod
    def from_hdf5(cls, h5_path: str | Path, stream_name: str) -> "FeatureNormalizer":
        with h5py.File(str(h5_path), "r") as f:
            mean = f[f"streams/{stream_name}/meta/normalization/feature_mean"][:]
            std = f[f"streams/{stream_name}/meta/normalization/feature_std"][:]
        return cls(mean, std)

    @classmethod
    def from_dataset_stats(cls, means: np.ndarray, stds: np.ndarray) -> "FeatureNormalizer":
        return cls(means, stds)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_knn_graph_batched(
    batch_data,
    k: int,
    normalizer: Optional[FeatureNormalizer] = None,
    orbital_features: bool = False,
    _norm_cache: dict = {},  # noqa: B006 — intentional mutable default as device cache
) -> None:
    """Build k-NN graphs for an entire PyG batch IN-PLACE on the current device.

    Called in the training loop after batch.to(device) so the knn_graph runs
    on GPU — orders of magnitude faster than 32 serial CPU calls in __getitem__.

    Edge construction is wrapped in torch.no_grad() to prevent autograd from
    tracking the edge-index/edge-attr tensors — this avoids stale computation
    graph accumulation across epochs that would blow up GPU VRAM.

    Args:
        batch_data: PyG Batch object (with x, batch attributes set).
        k: Number of nearest neighbours.
        normalizer: If provided, normalize before computing k-NN distances.
        orbital_features: If True, append delta_R_cyl, delta_z_cyl as additional
            edge features. Requires R_cyl and z_cyl to be in node features at
            positions 11 and 12 (after membership_prob, before stream one-hot).
    """
    x = batch_data.x  # [total_nodes, 18], already on device
    b = batch_data.batch  # [total_nodes], already on device
    device = x.device

    with torch.no_grad():
        # Construct features used for k-NN (phi1, phi2, pm1, pm2)
        x_construct = x[:, CONSTRUCTION_FEATURE_IDX]  # [N_total, 4]
        if normalizer is not None:
            # Cache normalizer stats on the target device to avoid repeated
            # CPU→GPU copies inside the hot training loop.
            cache_key = (id(normalizer), str(device))
            if cache_key not in _norm_cache:
                _norm_cache[cache_key] = (
                    normalizer.mean[CONSTRUCTION_FEATURE_IDX].to(device),
                    normalizer.std[CONSTRUCTION_FEATURE_IDX].to(device),
                )
            mean, std = _norm_cache[cache_key]
            x_norm = (x_construct - mean) / std
        else:
            x_norm = x_construct

        # One vectorized knn_graph call for all graphs in the batch (respects batch)
        edge_index = knn_graph(x_norm, k=k, batch=b, loop=False)

        # Edge features from raw (unnormalised) construction features
        src, dst = edge_index
        delta = x_construct[dst] - x_construct[src]   # [E, 4]
        dist_4d = delta.norm(dim=1, keepdim=True)      # [E, 1]
        edge_parts = [delta, dist_4d]                  # [E, 5] base

        # Orbital edge features (Feature 4): delta_R_cyl, delta_z_cyl
        if orbital_features:
            # R_cyl at index 11, z_cyl at index 12 (after 11 physical features)
            r_cyl = x[:, 11:12]   # [N_total, 1]
            z_cyl = x[:, 12:13]   # [N_total, 1]
            delta_r = r_cyl[dst] - r_cyl[src]   # [E, 1]
            delta_z = z_cyl[dst] - z_cyl[src]   # [E, 1]
            edge_parts.extend([delta_r, delta_z])

        edge_attr = torch.cat(edge_parts, dim=1)  # [E, 5 or 7]

    batch_data.edge_index = edge_index
    batch_data.edge_attr = edge_attr


def build_knn_graph(
    node_features: torch.Tensor,
    k: int,
    normalizer: Optional[FeatureNormalizer] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build k-NN graph edges from the 4 construction features.

    Args:
        node_features: [N, 11] node feature tensor.
        k: Number of nearest neighbours.
        normalizer: If provided, normalise before computing k-NN distances.

    Returns:
        edge_index: [2, E] LongTensor.
        edge_attr: [E, 5] FloatTensor of (delta_phi1, delta_phi2, delta_pm1, delta_pm2, dist_4d).
    """
    x_construct = node_features[:, CONSTRUCTION_FEATURE_IDX]  # [N, 4]
    if normalizer is not None:
        x_norm = normalizer(node_features)[:, CONSTRUCTION_FEATURE_IDX]
    else:
        x_norm = x_construct

    edge_index = knn_graph(x_norm, k=k, loop=False)  # [2, N*k]

    # Compute edge features from raw (unnormalised) construction features
    src, dst = edge_index
    delta = node_features[dst][:, CONSTRUCTION_FEATURE_IDX] - node_features[src][:, CONSTRUCTION_FEATURE_IDX]
    dist_4d = delta.norm(dim=1, keepdim=True)
    edge_attr = torch.cat([delta, dist_4d], dim=1)  # [E, 5]

    return edge_index, edge_attr


def profile_feature_dim(
    n_bins: int = 48,
    feature_set: str = "compact",
    include_stream_onehot: bool = True,
) -> int:
    """Return the graph-level profile feature width for a feature set."""
    feature_set = feature_set.lower()
    if feature_set == "compact":
        return n_bins + 14
    if feature_set == "summary":
        return 3 + 11 * 7 + n_bins + 6 + 4 * 4 + (7 if include_stream_onehot else 0)
    raise ValueError(f"Unknown profile feature_set: {feature_set!r}")


def build_profile_features_batched(
    batch_data,
    n_bins: int = 48,
    eps: float = 1e-6,
    feature_set: str = "compact",
    include_stream_onehot: bool = True,
) -> None:
    """Build compact 1D stream-profile features for every graph in a batch.

    The profile branch is a signal-ladder aid: it gives the model direct access
    to along-stream density roughness, which is exactly where gap/no-gap
    information lives. Features are written in-place as ``batch_data.profile_x``.
    """
    feature_set = feature_set.lower()
    x = batch_data.x
    batch = batch_data.batch
    device = x.device
    n_graphs = int(getattr(batch_data, "num_graphs", int(batch.max().item()) + 1))
    out_dim = profile_feature_dim(n_bins, feature_set, include_stream_onehot)

    cached = getattr(batch_data, "profile_x", None)
    if cached is not None:
        cached = cached.to(device=device, dtype=x.dtype)
        if cached.ndim == 1 and cached.numel() == n_graphs * out_dim:
            cached = cached.view(n_graphs, out_dim)
        elif cached.ndim == 1 and n_graphs == 1 and cached.numel() == out_dim:
            cached = cached.view(1, out_dim)
        if cached.shape == (n_graphs, out_dim):
            batch_data.profile_x = cached
            return
        raise ValueError(
            f"Cached profile_x has shape {tuple(cached.shape)}, expected "
            f"({n_graphs}, {out_dim}) for feature_set={feature_set!r}."
        )

    rows: list[torch.Tensor] = []

    for graph_idx in range(n_graphs):
        xi = x[batch == graph_idx]
        if xi.numel() == 0:
            rows.append(torch.zeros(out_dim, dtype=x.dtype, device=device))
            continue

        phi1 = xi[:, 0]
        mem = xi[:, 10].clamp(0.0, 1.0) if xi.shape[1] > 10 else torch.ones_like(phi1)
        lo = torch.quantile(phi1, 0.01)
        hi = torch.quantile(phi1, 0.99)
        width = torch.clamp(hi - lo, min=eps)
        bin_idx = torch.floor((phi1 - lo) / width * n_bins).long().clamp(0, n_bins - 1)

        hist = torch.zeros(n_bins, dtype=x.dtype, device=device)
        hist.scatter_add_(0, bin_idx, mem)
        h_norm = hist / (hist.mean() + eps)

        diff = torch.diff(h_norm)
        roughness = torch.stack([
            h_norm.std(unbiased=False),
            h_norm.min(),
            torch.quantile(h_norm, 0.05),
            (h_norm < 0.5).float().mean(),
            (h_norm < 0.25).float().mean(),
            diff.abs().max() if len(diff) else torch.tensor(0.0, dtype=x.dtype, device=device),
        ])

        if feature_set == "summary":
            parts: list[torch.Tensor] = [
                torch.tensor(float(len(phi1)), dtype=x.dtype, device=device),
                mem.mean(),
                mem.std(unbiased=False) if len(mem) > 1 else torch.tensor(0.0, dtype=x.dtype, device=device),
            ]
            q_levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=x.dtype, device=device)
            for col in range(min(11, xi.shape[1])):
                values = xi[:, col]
                finite = values[torch.isfinite(values)]
                if finite.numel() == 0:
                    parts.extend([torch.zeros((), dtype=x.dtype, device=device)] * 7)
                    continue
                quantiles = torch.quantile(finite, q_levels)
                parts.extend([
                    finite.mean(),
                    finite.std(unbiased=False) if finite.numel() > 1 else torch.tensor(0.0, dtype=x.dtype, device=device),
                    *[q for q in quantiles],
                ])
            parts.extend([v for v in h_norm])
            parts.extend([v for v in roughness])

            for col in [1, 3, 4, 2]:  # phi2, pm1, pm2, dist
                values = xi[:, col]
                ok = torch.isfinite(values) & torch.isfinite(phi1)
                if ok.sum() < 20:
                    parts.extend([torch.zeros((), dtype=x.dtype, device=device)] * 4)
                    continue
                xpos = phi1[ok]
                ypos = values[ok]
                blo = torch.quantile(xpos, 0.01)
                bhi = torch.quantile(xpos, 0.99)
                bwidth = torch.clamp(bhi - blo, min=eps)
                medians: list[torch.Tensor] = []
                missing = 0
                for bin_i in range(16):
                    b0 = blo + bwidth * (bin_i / 16.0)
                    b1 = blo + bwidth * ((bin_i + 1) / 16.0)
                    in_bin = (xpos >= b0) & (xpos < b1 if bin_i < 15 else xpos <= b1)
                    if in_bin.any():
                        medians.append(torch.median(ypos[in_bin]))
                    else:
                        medians.append(torch.zeros((), dtype=x.dtype, device=device))
                        missing += 1
                med = torch.stack(medians)
                med_diff = torch.diff(med)
                parts.extend([
                    med.std(unbiased=False),
                    med_diff.std(unbiased=False) if med_diff.numel() > 1 else torch.tensor(0.0, dtype=x.dtype, device=device),
                    med.max() - med.min(),
                    torch.tensor(float(missing), dtype=x.dtype, device=device),
                ])

            if include_stream_onehot:
                if xi.shape[1] >= 18:
                    parts.extend([v for v in xi[0, 11:18]])
                else:
                    parts.extend([torch.zeros((), dtype=x.dtype, device=device)] * 7)
            rows.append(torch.stack(parts))
            continue

        if feature_set != "compact":
            raise ValueError(f"Unknown profile feature_set: {feature_set!r}")

        scalar_parts = []
        weight_sum = mem.sum().clamp_min(eps)
        for col in [1, 3, 4, 2]:  # phi2, pm1, pm2, dist
            values = xi[:, col]
            mean = (values * mem).sum() / weight_sum
            var = (((values - mean) ** 2) * mem).sum() / weight_sum
            scalar_parts.extend([mean, torch.sqrt(var.clamp_min(0.0))])

        scalar = torch.stack(scalar_parts)
        rows.append(torch.cat([h_norm, roughness, scalar], dim=0))

    batch_data.profile_x = torch.stack(rows, dim=0)


# ---------------------------------------------------------------------------
# Simulation dataset
# ---------------------------------------------------------------------------

class StreamSimDataset(Dataset):
    """Dataset backed by HDF5 simulation files.

    Expects files in output_dir matching pattern chunk_*.h5,
    each containing /simulations/{run_id}/ groups.

    When preload_ram=True (default), all raw feature arrays and labels are
    loaded into CPU memory at __init__ time. This eliminates per-sample HDF5
    I/O during training, which is the dominant bottleneck on Windows where
    num_workers=0 forces serial data loading. Typical cost: ~3 GB RAM for 46K
    sims; __getitem__ then just indexes a list and builds the k-NN graph.
    """

    def __init__(
        self,
        sim_dir: str | Path,
        k_neighbors: int = 16,
        normalizer: Optional[FeatureNormalizer] = None,
        max_stars: int = 3000,
        augment: bool = False,
        cache_dir: Optional[str | Path] = None,
        preload_ram: bool = True,
        use_orbital_features: bool = False,
        profile_features_path: Optional[str | Path] = None,
        downsample_seed: Optional[int] = None,
        label_schema: str = "compact",
        timeline_n_bins: int = 5,
        error_dr: Optional[dict] = None,
    ) -> None:
        self.sim_dir = Path(sim_dir)
        self.k = k_neighbors
        self.normalizer = normalizer
        self.max_stars = max_stars
        self.augment = augment
        # Error domain randomization: draw realistic, VARYING per-star
        # measurement errors at load time so the e_* features are not
        # near-constant. Near-constant features get their std clamped by the
        # normalizer, which makes any real-data offset explode to ~100 sigma
        # (the cause of p_impact=1.0 saturation on real streams). Applied
        # deterministically per simulation (seeded) so the normalizer and
        # training are reproducible. See changelog 2026-05-30 overconfidence fix.
        self.error_dr = error_dr if (error_dr and error_dr.get("enabled")) else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.preload_ram = preload_ram
        self._use_orbital_features = use_orbital_features
        self._profile_features: Optional[np.ndarray] = None
        self.profile_features_path = Path(profile_features_path) if profile_features_path else None
        self.downsample_seed = downsample_seed
        self.label_schema = label_schema.lower()
        if self.label_schema not in {"compact", "timeline"}:
            raise ValueError(f"Unknown label_schema: {label_schema!r}")
        self.timeline_n_bins = int(timeline_n_bins)

        self._index = self._build_index()
        log.info("StreamSimDataset: %d simulations (orbital_features=%s)",
                 len(self._index), use_orbital_features)
        if self.profile_features_path is not None:
            self._load_profile_features(self.profile_features_path)

        # RAM preload: load all raw arrays at init, keep HDF5 closed during training
        self._ram_x: Optional[list[np.ndarray]] = None
        self._ram_y: Optional[list[torch.Tensor]] = None
        if preload_ram:
            self._preload_all_into_ram()

    def _build_index(self) -> list[tuple[Path, str, str]]:
        """Return list of (hdf5_path, run_id, stream_name) triples."""
        index = []
        for h5_file in sorted(self.sim_dir.glob("**/*.h5")):
            with h5py.File(str(h5_file), "r") as f:
                if "simulations" not in f:
                    continue
                for run_id in f["simulations"].keys():
                    grp = f[f"simulations/{run_id}"]
                    sname = grp.attrs.get("stream_name", b"unknown")
                    if isinstance(sname, bytes):
                        sname = sname.decode()
                    index.append((h5_file, run_id, sname))
        return index

    def _load_profile_features(self, path: Path) -> None:
        """Load graph-level cached profile features aligned to dataset index."""
        if not path.exists():
            raise FileNotFoundError(f"Profile feature cache not found: {path}")
        if path.suffix.lower() == ".npy":
            features = np.load(str(path), mmap_mode="r")
        else:
            payload = np.load(str(path), allow_pickle=False)
            if "features" not in payload:
                raise KeyError(f"Profile feature cache {path} does not contain a 'features' array")
            features = payload["features"]
        if features.ndim != 2:
            raise ValueError(f"Profile feature cache must be 2-D; got shape {features.shape}")
        if len(features) != len(self._index):
            raise ValueError(
                f"Profile feature cache length {len(features)} does not match dataset length "
                f"{len(self._index)}"
            )
        self._profile_features = features
        log.info("Loaded profile feature cache %s with shape %s", path, features.shape)

    def _preload_all_into_ram(self) -> None:
        """Load all simulation arrays into CPU RAM.

        Groups by HDF5 file to minimise file open/close overhead.
        Stores _ram_x as a list of [N_i, 18] float32 arrays (truncated to
        max_stars, with stream one-hot appended) and _ram_y as a list of
        [3] float32 tensors.
        """
        log.info("Preloading %d simulations into RAM (this takes ~30-60s)...", len(self._index))
        from collections import defaultdict  # noqa: PLC0415
        file_groups: dict[Path, list[tuple[int, str, str]]] = defaultdict(list)
        for idx, (h5_path, run_id, sname) in enumerate(self._index):
            file_groups[h5_path].append((idx, run_id, sname))

        xs: list[Optional[np.ndarray]] = [None] * len(self._index)
        ys: list[Optional[torch.Tensor]] = [None] * len(self._index)

        for h5_path, items in file_groups.items():
            with h5py.File(str(h5_path), "r") as f:
                for idx, run_id, sname in items:
                    grp = f[f"simulations/{run_id}"]
                    x = self._load_node_features(grp["stream_data"])
                    y = self._load_labels(grp["labels"])
                    # Downsample now so RAM usage reflects max_stars, not raw sim size
                    x = self._downsample(x, idx)
                    # Append stream one-hot encoding (7 dims)
                    x = np.column_stack([x, stream_one_hot(sname, len(x))])
                    xs[idx] = x
                    ys[idx] = y

        self._ram_x = xs  # type: ignore[assignment]
        self._ram_y = ys  # type: ignore[assignment]
        total_mb = sum(x.nbytes for x in self._ram_x) / 1024 ** 2
        log.info("RAM preload complete: %.0f MB used", total_mb)

    def get_dm_model_labels(self) -> np.ndarray:
        """Return raw DM model indices without loading node features.

        This is used for stratified train/validation/test splitting. Reading
        only labels avoids accidentally doing a full HDF5 feature pass over
        100K simulations before training starts.
        """
        labels = np.empty(len(self._index), dtype=np.int64)
        from collections import defaultdict  # noqa: PLC0415

        file_groups: dict[Path, list[tuple[int, str]]] = defaultdict(list)
        for idx, (h5_path, run_id, _sname) in enumerate(self._index):
            file_groups[h5_path].append((idx, run_id))

        for h5_path, items in file_groups.items():
            with h5py.File(str(h5_path), "r") as f:
                for idx, run_id in items:
                    labels[idx] = int(f[f"simulations/{run_id}/labels/dm_model_idx"][()])

        return labels

    def get_label_matrix(self) -> np.ndarray:
        """Return the compact numeric label matrix without loading node features.

        Columns match ``_load_labels``. The default compact schema is:
        [dm_model_idx, log_m_sub_mean, n_subhalos, log_t_since_last_impact_gyr,
        log10_M_hm, impact_strength].

        With label_schema="timeline", additional columns are appended:
        [n_detectable_impacts, effective_n_impacts,
        log_t_since_strongest_impact_gyr, strongest_impact_t_since_gyr,
        timeline_effective_impacts...].
        """
        labels_list: list[Optional[np.ndarray]] = [None] * len(self._index)
        from collections import defaultdict  # noqa: PLC0415

        file_groups: dict[Path, list[tuple[int, str]]] = defaultdict(list)
        for idx, (h5_path, run_id, _sname) in enumerate(self._index):
            file_groups[h5_path].append((idx, run_id))

        for h5_path, items in file_groups.items():
            with h5py.File(str(h5_path), "r") as f:
                for idx, run_id in items:
                    labels_list[idx] = self._load_labels(
                        f[f"simulations/{run_id}/labels"]
                    ).numpy().astype(np.float32)

        if any(label is None for label in labels_list):
            raise RuntimeError("Failed to load one or more simulation label rows")
        return np.vstack(labels_list).astype(np.float32)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Data:
        if self._ram_x is not None:
            # Fast path: data already in RAM (with stream one-hot), no HDF5 I/O
            x = self._ram_x[idx].copy()  # copy so augmentation doesn't corrupt the cache
            y = self._ram_y[idx]  # type: ignore[index]
        else:
            # Slow path: read from HDF5 on disk
            h5_path, run_id, sname = self._index[idx]
            cache_key = hashlib.md5(f"{h5_path}:{run_id}".encode()).hexdigest()
            if self.cache_dir is not None:
                cached = self._try_load_cache(cache_key)
                if cached is not None:
                    return cached
            with h5py.File(str(h5_path), "r") as f:
                grp = f[f"simulations/{run_id}"]
                x = self._load_node_features(grp["stream_data"])
                y = self._load_labels(grp["labels"])
            x = self._downsample(x, idx)
            # Append stream one-hot encoding (7 dims)
            x = np.column_stack([x, stream_one_hot(sname, len(x))])

        if self.error_dr is not None:
            x = self._apply_error_dr(x, idx)
        if self.augment:
            x = self._augment(x)

        x_t = torch.tensor(x, dtype=torch.float32)

        # Graph construction is deferred to the training loop where it runs
        # on GPU for the entire batch at once (via build_knn_graph_batched in
        # train.py). Building per-sample on CPU here is the dominant bottleneck
        # (~7ms × 32 samples/batch = ~224ms/batch; GPU does it in <5ms total).
        data = Data(x=x_t, y=y, run_id=self._index[idx][1])
        if self._profile_features is not None:
            profile = np.asarray(self._profile_features[idx], dtype=np.float32).copy()
            data.profile_x = torch.from_numpy(profile).view(1, -1)

        if self._ram_x is None and self.cache_dir is not None:
            # Cache the raw features only (graph rebuilt on GPU at train time)
            self._save_cache(cache_key, data)

        return data

    def _load_node_features(self, sd: h5py.Group) -> np.ndarray:
        def _g(name, fill=0.0):
            if name in sd:
                arr = sd[name][:].astype(np.float32)
                # Replace NaN (e.g. missing Gaia RVS measurements) with fill value
                arr = np.where(np.isfinite(arr), arr, fill)
                return arr
            return np.full(sd["phi1"].shape, fill, dtype=np.float32)

        cols = [
            _g("phi1"), _g("phi2"), _g("dist"),
            _g("pm1"), _g("pm2"), _g("vrad", fill=0.0),  # 0 = unknown RV
            _g("e_dist", fill=1.0), _g("e_pm1", fill=0.1),
            _g("e_pm2", fill=0.1), _g("e_vrad", fill=1.0),
            _g("membership_prob", fill=1.0),
        ]

        # Orbital features (Feature 4): R_cyl, z_cyl from galactocentric coords.
        # Stored by generate_training_data.py if available; default = Solar (8.0, 0.0).
        if getattr(self, "_use_orbital_features", False):
            cols.append(_g("R_cyl", fill=8.0))  # kpc, default = Solar R
            cols.append(_g("z_cyl", fill=0.0))  # kpc, default = disk plane

        return np.column_stack(cols)

    def _load_labels(self, labels: h5py.Group) -> torch.Tensor:
        """Load compact label tensor from HDF5 labels group.

        Returns compact 6-element tensor by default:
            [dm_model_idx, log_m_sub_mean, n_subhalos, log_t_since_last_impact,
             log10_M_hm, impact_strength]

        With label_schema="timeline", appends:
            [n_detectable_impacts, effective_n_impacts,
             log_t_since_strongest_impact_gyr, strongest_impact_t_since_gyr,
             timeline_effective_impacts...]

        Backward-compatible: old HDF5 files without impact_strength get 0.0.
        """
        dm_idx = int(labels["dm_model_idx"][()])
        log_m = float(labels["log_m_sub_mean"][()])
        n_sub = float(labels["n_subhalos"][()])
        # Guard against NaN labels (e.g., sims with no subhalos) — NaN in loss
        # silently corrupts gradients and destroys model weights.
        if not np.isfinite(log_m):
            log_m = 0.0
        if not np.isfinite(n_sub):
            n_sub = 0.0
        # Perturbation age (Feature 5): backward-compatible with old HDF5 files
        if "log_t_since_last_impact_gyr" in labels.attrs:
            log_t = float(labels.attrs["log_t_since_last_impact_gyr"])
            if not np.isfinite(log_t):
                log_t = 1.0  # log10(10 Gyr) — uninformative default
        elif "log_t_since_last_impact_gyr" in labels:
            log_t = float(labels["log_t_since_last_impact_gyr"][()])
            if not np.isfinite(log_t):
                log_t = 1.0
        else:
            log_t = 1.0  # old HDF5 without this field
        if "log10_M_hm" in labels.attrs:
            log_mhm = float(labels.attrs["log10_M_hm"])
        elif "log10_M_hm" in labels:
            log_mhm = float(labels["log10_M_hm"][()])
        else:
            # Backward-compatible default for old HDF5 files. CDM/SIDM are
            # unsuppressed references, represented by the lower prior edge.
            # Suppressed-model datasets should be regenerated for real v2
            # training; this value keeps mixed old/new files shape-stable.
            log_mhm = 4.0 if dm_idx in (0, 3) else log_m
        if not np.isfinite(log_mhm):
            log_mhm = 4.0 if dm_idx in (0, 3) else log_m
        # Impact strength score (computed by compute_impact_strength.py)
        # Encodes the morphological detectability of the strongest encounter.
        impact_strength = float(labels.attrs.get("impact_strength", 0.0))
        if not np.isfinite(impact_strength):
            impact_strength = 0.0
        base = [dm_idx, log_m, n_sub, log_t, log_mhm, impact_strength]
        if self.label_schema == "compact":
            return torch.tensor(base, dtype=torch.float32)

        n_detectable = float(labels.attrs.get("n_detectable_impacts", n_sub))
        effective_n = float(labels.attrs.get("effective_n_impacts", n_sub))
        log_t_strongest = float(labels.attrs.get("log_t_since_strongest_impact_gyr", log_t))
        strongest_t = float(labels.attrs.get("strongest_impact_t_since_gyr", 10.0 ** log_t))

        if not np.isfinite(n_detectable):
            n_detectable = float(n_sub)
        if not np.isfinite(effective_n):
            effective_n = float(n_sub)
        if not np.isfinite(log_t_strongest):
            log_t_strongest = 1.0
        if not np.isfinite(strongest_t):
            strongest_t = 10.0

        timeline_effective = np.asarray(
            labels.attrs.get("timeline_effective_impacts", np.zeros(self.timeline_n_bins, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if len(timeline_effective) < self.timeline_n_bins:
            timeline_effective = np.pad(
                timeline_effective,
                (0, self.timeline_n_bins - len(timeline_effective)),
                mode="constant",
            )
        elif len(timeline_effective) > self.timeline_n_bins:
            timeline_effective = timeline_effective[: self.timeline_n_bins]
        timeline_effective = np.nan_to_num(timeline_effective, nan=0.0, posinf=0.0, neginf=0.0)

        timeline = [
            n_detectable,
            effective_n,
            log_t_strongest,
            strongest_t,
            *timeline_effective.astype(float).tolist(),
        ]
        return torch.tensor(base + timeline, dtype=torch.float32)

    def _downsample(self, x: np.ndarray, idx: Optional[int] = None) -> np.ndarray:
        if len(x) > self.max_stars:
            if self.downsample_seed is None or idx is None:
                choose_idx = np.random.choice(len(x), self.max_stars, replace=False)
            else:
                rng = np.random.default_rng(int(self.downsample_seed) + int(idx))
                choose_idx = rng.choice(len(x), self.max_stars, replace=False)
            return x[choose_idx]
        return x

    def _apply_error_dr(self, x: np.ndarray, idx: int) -> np.ndarray:
        """Draw realistic, varying per-star measurement errors (domain randomization).

        Sets the e_dist/e_pm1/e_pm2/e_vrad feature columns (indices 6-9) to values
        drawn log-uniformly over realistic Gaia DR3 ranges, adds matching
        observational noise to the corresponding observables, and masks the radial
        velocity for a fraction of simulations (mimicking streams without
        spectroscopy, e.g. GD-1). This gives the error features real spread so the
        normalizer does not clamp them, closing the sim-to-real gap that caused
        the detector to saturate to p_impact=1.0 on real data.
        """
        cfg = self.error_dr
        n = len(x)
        rng = np.random.default_rng(int(cfg.get("seed", 12345)) + int(idx))

        def _logu(rng_, rng_range):
            lo, hi = rng_range
            return np.exp(rng_.uniform(np.log(lo), np.log(hi), n)).astype(np.float32)

        e_dist = _logu(rng, cfg.get("e_dist_kpc", [0.3, 2.0]))
        e_pm1 = _logu(rng, cfg.get("e_pm_masyr", [0.05, 0.6]))
        e_pm2 = _logu(rng, cfg.get("e_pm_masyr", [0.05, 0.6]))
        e_vrad = _logu(rng, cfg.get("e_vrad_kms", [1.0, 8.0]))

        x = x.copy()
        if cfg.get("add_noise", True):
            x[:, 2] = x[:, 2] + rng.normal(0.0, e_dist)
            x[:, 3] = x[:, 3] + rng.normal(0.0, e_pm1)
            x[:, 4] = x[:, 4] + rng.normal(0.0, e_pm2)
        # Radial velocity: a fraction of streams have no spectroscopy at all.
        if rng.random() < float(cfg.get("rv_mask_prob", 0.5)):
            x[:, 5] = 0.0
        elif cfg.get("add_noise", True):
            x[:, 5] = x[:, 5] + rng.normal(0.0, e_vrad)
        x[:, 6] = e_dist
        x[:, 7] = e_pm1
        x[:, 8] = e_pm2
        x[:, 9] = e_vrad
        return x

    def _augment(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[:, 0] += np.random.uniform(-5.0, 5.0)         # phi1 rigid offset (physical)
        x[:, 1] += np.random.normal(0.0, 0.05, len(x))  # phi2 per-star noise
        x[:, 3] += np.random.normal(0.0, 0.02, len(x))  # pm1 noise
        x[:, 4] += np.random.normal(0.0, 0.02, len(x))  # pm2 noise
        # NOTE: phi1 flip (x[:,0] = -x[:,0]) is DISABLED because stream one-hot
        # features encode stream identity, and each stream has a specific phi1
        # range. Flipping phi1 creates physically impossible configurations
        # (e.g., GD-1 phi1 range [-30, 90] flipped to [-90, 30]).
        return x

    def _try_load_cache(self, key: str) -> Optional[Data]:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.pt"
        if path.exists():
            try:
                return torch.load(str(path), weights_only=False)
            except Exception:
                pass
        return None

    def _save_cache(self, key: str, data: Data) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, str(self.cache_dir / f"{key}.pt"))


# ---------------------------------------------------------------------------
# Real stream dataset
# ---------------------------------------------------------------------------

class RealStreamDataset(Dataset):
    """Dataset backed by processed real-stream HDF5 files (data/processed/)."""

    def __init__(
        self,
        processed_path: str | Path,
        stream_names: list[str],
        k_neighbors: int = 16,
        normalizer: Optional[FeatureNormalizer] = None,
        max_stars: int = 3000,
    ) -> None:
        self.processed_path = Path(processed_path)
        self.stream_names = stream_names
        self.k = k_neighbors
        self.normalizer = normalizer
        self.max_stars = max_stars

    def __len__(self) -> int:
        return len(self.stream_names)

    def __getitem__(self, idx: int) -> Data:
        name = self.stream_names[idx]
        with h5py.File(str(self.processed_path), "r") as f:
            mem = f[f"streams/{name}/members"]
            x = self._load_node_features(mem)

        if len(x) > self.max_stars:
            rng = np.random.default_rng(42)
            i = rng.choice(len(x), self.max_stars, replace=False)
            x = x[i]

        # Append stream one-hot encoding (7 dims) — must match simulation features
        x = np.column_stack([x, stream_one_hot(name, len(x))])

        x_t = torch.tensor(x, dtype=torch.float32)
        edge_index, edge_attr = build_knn_graph(x_t, self.k, self.normalizer)

        return Data(
            x=x_t,
            edge_index=edge_index,
            edge_attr=edge_attr,
            stream_name=name,
        )

    def _load_node_features(self, mem: h5py.Group) -> np.ndarray:
        def _g(name, fill=0.0):
            if name in mem:
                arr = mem[name][:].astype(np.float32)
                arr = np.where(np.isfinite(arr), arr, fill)
                return arr
            return np.full(mem["phi1"].shape, fill, dtype=np.float32)

        return np.column_stack([
            _g("phi1"), _g("phi2"), _g("dist"),
            _g("pm1"), _g("pm2"), _g("vrad", fill=0.0),  # 0 = no RV measurement
            _g("e_dist", fill=1.0), _g("e_pm1", fill=0.1),
            _g("e_pm2", fill=0.1), _g("e_vrad", fill=1.0),
            _g("membership_prob", fill=1.0),
        ])


# ---------------------------------------------------------------------------
# Normalizer helper: compute from simulation dataset
# ---------------------------------------------------------------------------

def compute_dataset_normalizer(dataset, n_samples: int = 5000) -> FeatureNormalizer:
    """Estimate per-feature mean/std from a random subset of the dataset (or Subset)."""
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    all_features = []
    for i in indices:
        data = dataset[i]
        all_features.append(data.x.numpy())
    arr = np.concatenate(all_features, axis=0)
    return FeatureNormalizer(arr.mean(axis=0), arr.std(axis=0))


# ---------------------------------------------------------------------------
# Multi-scale graph: segment-level summary (Feature 2)
# ---------------------------------------------------------------------------

def build_segment_graph(
    batch_data,
    n_segments: int = 20,
    k_seg: int = 4,
) -> None:
    """Build a coarse segment-level graph on top of the star-level graph IN-PLACE.

    Bins stars by phi1 into n_segments bins and computes per-segment features.
    Connects segments via a chain (adjacent) plus k_seg-nearest in PM-space.

    This captures gap *clustering* patterns (CDM signature: many weak impacts in
    nearby segments) vs individual gap *depth* (WDM: few strong isolated impacts)
    at the ~5 deg scale, complementing the star-level GNN which sees only local
    structure.

    Stores on batch_data:
        seg_x: [B*n_segments, 8] segment features
        seg_edge_index: [2, E_seg] segment graph edges
        seg_edge_attr: [E_seg, 4] segment edge features
        seg_batch: [B*n_segments] batch assignment

    Args:
        batch_data: PyG Batch (must have .x, .batch already set).
        n_segments: Number of phi1 bins per graph.
        k_seg: Edges per segment (chain + PM-nearest).
    """
    x = batch_data.x       # [N_total, >=11]
    batch = batch_data.batch  # [N_total]
    device = x.device
    n_graphs = int(batch.max().item()) + 1

    all_seg_x = []
    all_seg_edges = []
    all_seg_edge_attr = []
    seg_batch_list = []
    seg_offset = 0

    with torch.no_grad():
        for g in range(n_graphs):
            mask = (batch == g)
            x_g = x[mask]  # [N_g, F]
            n_stars_g = x_g.shape[0]

            phi1 = x_g[:, 0]  # raw phi1
            phi2 = x_g[:, 1]
            pm1 = x_g[:, 3]
            pm2 = x_g[:, 4]

            # Bin stars into segments by phi1
            phi1_min = phi1.min()
            phi1_max = phi1.max()
            phi1_range = phi1_max - phi1_min
            if phi1_range < 1e-6:
                phi1_range = torch.tensor(1.0, device=device)

            # Segment assignment per star
            seg_idx = ((phi1 - phi1_min) / phi1_range * n_segments).long().clamp(0, n_segments - 1)

            # Per-segment features (8): mean_phi1, mean_phi2, std_phi2,
            #   mean_pm1, mean_pm2, std_pm1, log_n_stars, density_contrast
            seg_features = torch.zeros(n_segments, 8, device=device)
            for s in range(n_segments):
                s_mask = (seg_idx == s)
                n_in_seg = s_mask.sum()
                if n_in_seg == 0:
                    # Empty segment: fill with interpolated position, zero std
                    seg_features[s, 0] = phi1_min + (s + 0.5) * phi1_range / n_segments
                    continue
                seg_features[s, 0] = phi1[s_mask].mean()      # mean_phi1
                seg_features[s, 1] = phi2[s_mask].mean()      # mean_phi2
                seg_features[s, 2] = phi2[s_mask].std() if n_in_seg > 1 else 0.0  # std_phi2
                seg_features[s, 3] = pm1[s_mask].mean()       # mean_pm1
                seg_features[s, 4] = pm2[s_mask].mean()       # mean_pm2
                seg_features[s, 5] = pm1[s_mask].std() if n_in_seg > 1 else 0.0  # std_pm1
                seg_features[s, 6] = torch.log1p(n_in_seg.float())  # log_n_stars
                # density_contrast: relative to expected uniform density
                expected = n_stars_g / n_segments
                seg_features[s, 7] = (n_in_seg.float() - expected) / max(expected, 1.0)

            all_seg_x.append(seg_features)

            # Segment edges: chain (adjacent) + PM-nearest
            edges_src = []
            edges_dst = []
            # Chain edges (adjacent segments)
            for s in range(n_segments - 1):
                edges_src.extend([s, s + 1])
                edges_dst.extend([s + 1, s])

            # PM-nearest: for each segment, find k_seg nearest by PM distance
            pm_coords = seg_features[:, 3:5]  # [n_seg, 2] mean_pm1, mean_pm2
            pm_dists = torch.cdist(pm_coords.unsqueeze(0), pm_coords.unsqueeze(0)).squeeze(0)
            # Set self-distance to inf
            pm_dists.fill_diagonal_(float("inf"))
            for s in range(n_segments):
                _, nearest = pm_dists[s].topk(min(k_seg, n_segments - 1), largest=False)
                for t in nearest:
                    edges_src.append(s)
                    edges_dst.append(t.item())

            seg_edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)
            # Remove duplicates
            seg_edge_index = torch.unique(seg_edge_index, dim=1)

            # Segment edge features (4): delta_mean_phi1, delta_mean_pm1, delta_mean_pm2, delta_density
            src_e, dst_e = seg_edge_index
            seg_edge_feat = torch.stack([
                seg_features[dst_e, 0] - seg_features[src_e, 0],  # delta_mean_phi1
                seg_features[dst_e, 3] - seg_features[src_e, 3],  # delta_mean_pm1
                seg_features[dst_e, 4] - seg_features[src_e, 4],  # delta_mean_pm2
                seg_features[dst_e, 7] - seg_features[src_e, 7],  # delta_density
            ], dim=1)

            # Offset edge indices for batching
            all_seg_edges.append(seg_edge_index + seg_offset)
            all_seg_edge_attr.append(seg_edge_feat)
            seg_batch_list.append(torch.full((n_segments,), g, dtype=torch.long, device=device))
            seg_offset += n_segments

    batch_data.seg_x = torch.cat(all_seg_x, dim=0)
    batch_data.seg_edge_index = torch.cat(all_seg_edges, dim=1)
    batch_data.seg_edge_attr = torch.cat(all_seg_edge_attr, dim=0)
    batch_data.seg_batch = torch.cat(seg_batch_list, dim=0)
