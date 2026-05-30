"""
Train / validation / test split management for simulation datasets.

Ensures reproducible splits that:
    1. Are stratified by DM model (equal representation in each split).
    2. Are deterministic given a seed (reproducible across runs).
    3. Never leak between training and evaluation.
    4. Support the SBC/coverage test set being held out from training entirely.

Split ratios (configurable):
    Train: 70%   — used for GNN + SBI training
    Val:   15%   — early stopping, hyperparameter selection
    Test:  15%   — final SBC, coverage, and reported metrics only

The test set is NEVER used during development. It's evaluated exactly once
before writing the paper. This prevents inadvertent overfitting to the test set
via iterative model selection.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def compute_split_indices(
    n_samples: int,
    dm_model_labels: np.ndarray | None = None,
    split_ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Compute stratified train/val/test split indices.

    Args:
        n_samples: Total number of simulations.
        dm_model_labels: [n_samples] integer labels for stratification.
            If None, splits are random (non-stratified).
        split_ratios: Dict with keys "train", "val", "test" summing to 1.0.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping split name -> integer index array.

    Raises:
        ValueError: If split ratios don't sum to ~1.0.
    """
    if split_ratios is None:
        split_ratios = DEFAULT_SPLITS.copy()

    # Validate ratios
    total = sum(split_ratios.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.3f}: {split_ratios}")

    rng = np.random.default_rng(seed)

    if dm_model_labels is None:
        # Simple random split
        indices = rng.permutation(n_samples)
        return _split_array(indices, split_ratios)

    # Stratified split: equal proportion from each model
    unique_labels = np.unique(dm_model_labels)
    split_indices = {name: [] for name in split_ratios}

    for label in unique_labels:
        label_idx = np.where(dm_model_labels == label)[0]
        shuffled = rng.permutation(label_idx)
        per_label_splits = _split_array(shuffled, split_ratios)
        for name in split_ratios:
            split_indices[name].append(per_label_splits[name])

    # Concatenate and shuffle within each split
    result = {}
    for name in split_ratios:
        combined = np.concatenate(split_indices[name])
        result[name] = rng.permutation(combined)

    log.info(
        "Split computed: %s (stratified=%s, seed=%d)",
        {k: len(v) for k, v in result.items()},
        dm_model_labels is not None,
        seed,
    )
    return result


def _split_array(arr: np.ndarray, ratios: dict[str, float]) -> dict[str, np.ndarray]:
    """Split an array according to ratios."""
    n = len(arr)
    result = {}
    start = 0
    names = list(ratios.keys())
    for i, name in enumerate(names):
        if i == len(names) - 1:
            # Last split gets the remainder (avoids off-by-one)
            result[name] = arr[start:]
        else:
            end = start + int(round(ratios[name] * n))
            result[name] = arr[start:end]
            start = end
    return result


def save_split_indices(
    split_indices: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    """Save split indices to disk for reproducibility.

    Args:
        split_indices: Dict from compute_split_indices().
        output_path: .npz file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **split_indices)
    log.info("Split indices saved to %s", output_path)


def load_split_indices(path: str | Path) -> dict[str, np.ndarray]:
    """Load previously saved split indices."""
    data = np.load(str(path))
    result = {key: data[key] for key in data.files}
    log.info("Split indices loaded from %s: %s", path, {k: len(v) for k, v in result.items()})
    return result


def get_split_hash(split_indices: dict[str, np.ndarray]) -> str:
    """Compute a deterministic hash of the split for integrity checking.

    Include this hash in experiment configs / results to verify the same
    split was used when comparing different model runs.
    """
    h = hashlib.sha256()
    for name in sorted(split_indices.keys()):
        h.update(name.encode())
        h.update(split_indices[name].tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# HDF5-aware splitting
# ---------------------------------------------------------------------------

def split_hdf5_dataset(
    h5_path: str | Path,
    stream_name: str,
    split_ratios: dict[str, float] | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Compute stratified splits for an HDF5 simulation dataset.

    Reads DM model labels from the HDF5 file and computes a stratified split.
    Optionally saves the split indices alongside the HDF5 file.

    Args:
        h5_path: Path to the simulation HDF5 file.
        stream_name: Stream name key in the HDF5 file.
        split_ratios: Optional custom ratios.
        seed: Random seed.
        output_dir: If provided, saves split indices to this directory.

    Returns:
        Dict of split name -> index arrays.
    """
    import h5py  # noqa: PLC0415

    h5_path = Path(h5_path)
    with h5py.File(str(h5_path), "r") as f:
        group = f[f"streams/{stream_name}/simulations"]
        n_sims = group.attrs.get("n_simulations", len(group))

        # Try to read DM model labels for stratification
        if "dm_model_idx" in group:
            labels = group["dm_model_idx"][:]
        elif "parameters" in group and "dm_model_idx" in group["parameters"]:
            labels = group["parameters/dm_model_idx"][:]
        else:
            labels = None
            log.warning("No dm_model_idx found in HDF5; using non-stratified split")

    split_indices = compute_split_indices(n_sims, labels, split_ratios, seed)

    if output_dir is not None:
        output_dir = Path(output_dir)
        save_split_indices(
            split_indices,
            output_dir / f"split_{stream_name}_seed{seed}.npz",
        )

    return split_indices


# ---------------------------------------------------------------------------
# Split-aware DataLoader construction
# ---------------------------------------------------------------------------

def get_subset_indices(
    split_indices: dict[str, np.ndarray],
    splits: str | list[str],
) -> np.ndarray:
    """Get combined indices for one or more splits.

    Args:
        split_indices: Full split dict.
        splits: Single split name or list (e.g. "train" or ["train", "val"]).

    Returns:
        Combined sorted index array.
    """
    if isinstance(splits, str):
        splits = [splits]
    indices = np.concatenate([split_indices[s] for s in splits])
    return np.sort(indices)
