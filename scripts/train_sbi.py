"""
Train the simulation-based inference (SBI) pipeline using pre-trained GNN embeddings.

Workflow per DM model:
  1. Load trained GNN encoder from checkpoint
  2. Load all simulations for this DM model
  3. Pre-compute GNN embeddings for all sims (GPU, batched)
  4. Extract theta (inference parameters) from HDF5 labels.attrs
  5. Align theta and embeddings (drop sims with NaN theta)
  6. Keep only the requested split (default: train+val, never test)
  7. Train SNPE-C density estimator (NSF) on (theta, embedding) pairs
  8. Save density estimator + posterior to checkpoints/

IMPORTANT: Run with PYTHONNOUSERSITE=1 to avoid Windows user-site-packages
           segfault (stale scikit-learn in AppData conflicts with conda env).

Usage:
    PYTHONNOUSERSITE=1 python -u scripts/train_sbi.py
    PYTHONNOUSERSITE=1 python -u scripts/train_sbi.py --dm-models CDM WDM
    PYTHONNOUSERSITE=1 python -u scripts/train_sbi.py --dm-models FDM --resume
"""

from __future__ import annotations

import argparse
import logging
import site
import sys
import time
from pathlib import Path

import numpy as np

# Keep stale user-site packages from shadowing the conda env's CUDA torch/PyG.
try:
    user_site = Path(site.getusersitepackages()).resolve()
    sys.path = [p for p in sys.path if not p or Path(p).resolve() != user_site]
except Exception:
    pass

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
(ROOT / "logs").mkdir(parents=True, exist_ok=True)

from src.data.dataset import (
    StreamSimDataset,
    FeatureNormalizer,
    build_knn_graph_batched,
    build_profile_features_batched,
    build_segment_graph,
    profile_feature_dim,
)
from src.data.splits import load_split_indices
from src.inference.sbi_pipeline import (
    build_prior,
    train_npe,
    precompute_embeddings,
    extract_theta_from_hdf5,
)
from src.models.gnn import StreamGNNMultiTask, StreamGNNMultiTaskV2
from src.models.utils import load_checkpoint, load_normalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "train_sbi.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

DM_MODELS = ["CDM", "WDM", "FDM", "SIDM"]


def _split_names(mode: str) -> list[str]:
    if mode == "trainval":
        return ["train", "val"]
    if mode in {"train", "val", "test"}:
        return [mode]
    raise ValueError(f"Unknown split mode {mode!r}")


def _resolve_split_file(split_file: str | None, checkpoint_path: Path) -> Path | None:
    """Resolve a split file, preferring the checkpoint directory."""
    if split_file:
        path = Path(split_file)
        return path if path.is_absolute() else ROOT / path

    candidates = sorted(
        checkpoint_path.parent.glob("split_*.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            (ROOT / "checkpoints").glob("split_*.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        return None
    if len(candidates) > 1:
        log.warning(
            "Multiple split files found; using newest %s. Pass --split-file to pin one.",
            candidates[0],
        )
    return candidates[0]


def _run_ids_for_split(cfg: dict, split_path: Path, split_mode: str) -> set[str]:
    """Map saved global split indices to HDF5 run_ids."""
    split_indices = load_split_indices(split_path)
    indices = np.concatenate([split_indices[name] for name in _split_names(split_mode)])
    sim_dir = ROOT / cfg["paths"]["simulations"]
    use_orbital = cfg["graph"].get("orbital_features", {}).get("enabled", False)
    ds = StreamSimDataset(
        sim_dir=sim_dir,
        k_neighbors=cfg["graph"]["k_neighbors"],
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False,
        preload_ram=False,
        use_orbital_features=use_orbital,
    )
    valid_indices = [int(i) for i in indices if int(i) < len(ds)]
    if len(valid_indices) != len(indices):
        log.warning(
            "Split file %s contains %d indices outside current dataset length %d",
            split_path,
            len(indices) - len(valid_indices),
            len(ds),
        )
    run_ids = {ds._index[i][1] for i in valid_indices}
    log.info(
        "Resolved split '%s' from %s: %d global rows -> %d run_ids",
        split_mode,
        split_path,
        len(valid_indices),
        len(run_ids),
    )
    return run_ids


def load_gnn_model(
    cfg: dict,
    checkpoint_path: Path,
    device: str,
    model_version: str = "v2",
    use_profile_branch: bool = False,
) -> tuple[torch.nn.Module, bool]:
    """Load a trained GNN model from a V1 or V2 multi-task checkpoint.

    Returns the full model (not just encoder) so that combined embeddings
    (GNN + profile branch) can be extracted via get_combined_embedding().

    Returns:
        model: The full multi-task model in eval mode.
        has_profile: Whether the model was built with a profile branch.
    """
    gcfg = cfg["model"]["gnn"]
    ms_cfg = cfg["graph"].get("multi_scale", {})
    profile_cfg = cfg["graph"].get("profile_branch", {})

    # Auto-detect profile branch from checkpoint if not explicitly set
    if not use_profile_branch:
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if "model_state_dict" in payload:
            use_profile_branch = any(
                k.startswith("profile_mlp.") for k in payload["model_state_dict"]
            )
        # Also check embedded config
        ckpt_cfg = payload.get("config", {})
        if ckpt_cfg:
            ckpt_profile = ckpt_cfg.get("graph", {}).get("profile_branch", {})
            if ckpt_profile.get("enabled", False):
                use_profile_branch = True
                profile_cfg = ckpt_profile
            ckpt_ms = ckpt_cfg.get("graph", {}).get("multi_scale", {})
            if ckpt_ms.get("enabled", False):
                ms_cfg = ckpt_ms

    if use_profile_branch:
        cfg["graph"]["profile_branch"] = profile_cfg
        cfg["graph"]["multi_scale"] = ms_cfg

    profile_dim = 0
    if use_profile_branch:
        profile_dim = profile_feature_dim(
            int(profile_cfg.get("n_bins", 48)),
            str(profile_cfg.get("feature_set", "compact")).lower(),
            bool(profile_cfg.get("include_stream_onehot", True)),
        )

    if model_version == "v2":
        v2_cfg = cfg.get("training_v2", {})
        multitask = StreamGNNMultiTaskV2(
            n_reg_targets=v2_cfg.get("n_reg_targets", 2),
            predict_uncertainty=v2_cfg.get("predict_uncertainty", False),
            profile_dim=profile_dim,
            profile_hidden_dim=int(profile_cfg.get("hidden_dim", 64)),
            profile_layer_norm=bool(profile_cfg.get("layer_norm", False)),
            n_node_features=cfg["graph"]["n_node_features"],
            n_edge_features=cfg["graph"]["n_edge_features"],
            hidden_dim=gcfg["hidden_dim"],
            n_layers=gcfg["n_layers"],
            embedding_dim=gcfg["embedding_dim"],
            dropout=gcfg["dropout"],
            use_attention_readout=gcfg.get("use_attention_readout", False),
            use_multi_scale=ms_cfg.get("enabled", False),
            seg_embedding_dim=ms_cfg.get("seg_embedding_dim", 64),
        )
    elif model_version == "v1":
        multitask = StreamGNNMultiTask(
            n_classes=3,
            n_reg_targets=cfg["model"].get("n_reg_targets", 2),
            n_node_features=cfg["graph"]["n_node_features"],
            n_edge_features=cfg["graph"]["n_edge_features"],
            hidden_dim=gcfg["hidden_dim"],
            n_layers=gcfg["n_layers"],
            embedding_dim=gcfg["embedding_dim"],
            dropout=gcfg["dropout"],
            use_attention_readout=gcfg.get("use_attention_readout", False),
            use_multi_scale=ms_cfg.get("enabled", False),
            seg_embedding_dim=ms_cfg.get("seg_embedding_dim", 64),
        )
    else:
        raise ValueError(f"Unknown model_version={model_version!r}")

    load_checkpoint(str(checkpoint_path), multitask, device=device)
    multitask = multitask.to(device).eval()
    n_params = sum(p.numel() for p in multitask.parameters())
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    log.info("GNN model loaded: %d params, profile=%s, checkpoint epoch=%s",
             n_params, use_profile_branch, payload.get("epoch", "?"))
    return multitask, use_profile_branch


# Backward compat alias
def load_gnn_encoder(
    cfg: dict,
    checkpoint_path: Path,
    device: str,
    model_version: str = "v2",
) -> torch.nn.Module:
    """Load just the GNN encoder (legacy interface, no profile branch)."""
    model, _ = load_gnn_model(cfg, checkpoint_path, device, model_version,
                              use_profile_branch=False)
    return model.encoder


def precompute_embeddings_for_model(
    dm_model: str,
    encoder: torch.nn.Module,
    normalizer: FeatureNormalizer,
    cfg: dict,
    device: str,
    batch_size: int = 64,
    max_sims: int | None = None,
    use_profile: bool = False,
    profile_features_path: str | Path | None = None,
    downsample_seed: int | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Load all sims for dm_model, build graphs on GPU, encode to embeddings.

    Supports two layouts:
        1. Mixed chunks: sim_dir/chunk_*.h5 (filter by dm_model attr)
        2. Per-model subdirs: sim_dir/{dm_model}/*.h5

    If use_profile=True and the encoder is a StreamGNNMultiTaskV2 with a
    profile branch, returns combined (GNN + profile) embeddings via
    get_combined_embedding(). Otherwise falls back to encoder-only embeddings.

    Returns:
        embeddings: [N, D] tensor (D=128 for encoder-only, D=192 with profile)
        run_ids:    list[str] of HDF5 run_id for each embedding row, in the
                    same order as embeddings. Used by train_sbi_for_model to
                    align theta and embeddings by key rather than by position.
    """
    from torch_geometric.loader import DataLoader  # noqa: PLC0415
    from torch.utils.data import Subset
    import h5py  # noqa: PLC0415

    sim_dir = ROOT / cfg["paths"]["simulations"]
    model_dir = sim_dir / dm_model
    use_orbital = cfg["graph"].get("orbital_features", {}).get("enabled", False)
    ms_cfg = cfg["graph"].get("multi_scale", {})
    use_multi_scale = ms_cfg.get("enabled", False)

    # Determine layout
    if model_dir.exists() and list(model_dir.glob("*.h5")):
        # Legacy per-model layout
        data_dir = model_dir
        dm_filter = None
        log.info("Loading %s simulations from per-model dir %s ...", dm_model, data_dir)
    else:
        # Mixed-chunk layout: load all, then filter by dm_model
        data_dir = sim_dir
        dm_filter = dm_model
        log.info("Loading %s simulations from mixed chunks in %s ...", dm_model, data_dir)

    # Build a full dataset index, then filter to only sims matching dm_model
    ds_full = StreamSimDataset(
        sim_dir=data_dir,
        k_neighbors=cfg["graph"]["k_neighbors"],
        normalizer=normalizer,
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False,
        preload_ram=False,  # don't preload yet; we need to filter first
        use_orbital_features=use_orbital,
        profile_features_path=profile_features_path,
        downsample_seed=downsample_seed,
    )

    if dm_filter is not None:
        # Identify which indices in the dataset belong to this DM model
        dm_model_bytes = dm_model.encode()
        matching_indices = []
        for i, (h5_path, run_id, _sname) in enumerate(ds_full._index):
            with h5py.File(str(h5_path), "r") as f:
                sim_dm = f[f"simulations/{run_id}"].attrs.get("dm_model", b"")
                if isinstance(sim_dm, bytes):
                    sim_dm = sim_dm.decode()
                if sim_dm == dm_model:
                    matching_indices.append(i)
        log.info("  Found %d/%d sims matching %s", len(matching_indices), len(ds_full), dm_model)
        if max_sims is not None and len(matching_indices) > max_sims:
            matching_indices = matching_indices[:max_sims]
            log.info("  Limiting to %d sims for smoke/dry run", len(matching_indices))

        # Build a filtered dataset using Subset
        ds_filtered = Subset(ds_full, matching_indices)
        # Now preload just the filtered indices into RAM
        if hasattr(ds_full, '_ram_x') and ds_full._ram_x is None:
            # Preload only the matching sims
            from src.data.dataset import stream_one_hot  # noqa: PLC0415
            ds_full._ram_x = [None] * len(ds_full._index)
            ds_full._ram_y = [None] * len(ds_full._index)
            from collections import defaultdict
            file_groups = defaultdict(list)
            for idx in matching_indices:
                h5_path, run_id, sname = ds_full._index[idx]
                file_groups[h5_path].append((idx, run_id, sname))
            for h5_path, items in file_groups.items():
                with h5py.File(str(h5_path), "r") as f:
                    for idx, run_id, sname in items:
                        grp = f[f"simulations/{run_id}"]
                        x = ds_full._load_node_features(grp["stream_data"])
                        y = ds_full._load_labels(grp["labels"])
                        x = ds_full._downsample(x, idx)
                        # Append stream one-hot (must match training pipeline)
                        x = np.column_stack([x, stream_one_hot(sname, len(x))])
                        ds_full._ram_x[idx] = x
                        ds_full._ram_y[idx] = y
            log.info("  Preloaded %d matching sims into RAM", len(matching_indices))
    else:
        # Per-model dir: all sims match
        matching_indices = list(range(len(ds_full)))
        ds_filtered = ds_full
        # Preload all
        if max_sims is None:
            ds_full._preload_all_into_ram()

    if dm_filter is None and max_sims is not None and len(matching_indices) > max_sims:
        matching_indices = matching_indices[:max_sims]
        ds_filtered = Subset(ds_full, matching_indices)
        log.info("  Limiting to %d sims for smoke/dry run", len(matching_indices))

    log.info("  %d simulations to embed", len(matching_indices))

    # Collect run_ids in DataLoader order (shuffle=False, so sequential by index).
    # These are used by the caller to align theta and embeddings by run_id key
    # instead of by fragile positional ordering.
    run_ids: list[str] = [ds_full._index[matching_indices[i]][1]
                          for i in range(len(matching_indices))]

    loader = DataLoader(ds_filtered, batch_size=batch_size, shuffle=False, num_workers=0)
    k = cfg["graph"]["k_neighbors"]

    # Cache normalizer stats on device for node feature normalization
    norm_mean_dev = normalizer.mean.to(device) if normalizer else None
    norm_std_dev = normalizer.std.to(device) if normalizer else None

    profile_cfg = cfg["graph"].get("profile_branch", {})
    profile_n_bins = int(profile_cfg.get("n_bins", 48))
    profile_feature_set = str(profile_cfg.get("feature_set", "compact")).lower()
    profile_include_stream_onehot = bool(profile_cfg.get("include_stream_onehot", True))
    # Determine whether to use combined embeddings (GNN + profile) or encoder-only
    use_combined = (
        use_profile
        and hasattr(encoder, "get_combined_embedding")
        and getattr(encoder, "profile_dim", 0) > 0
    )

    embeddings = []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            build_knn_graph_batched(
                batch,
                k=k,
                normalizer=normalizer,
                orbital_features=use_orbital,
            )
            if use_multi_scale:
                build_segment_graph(
                    batch,
                    n_segments=ms_cfg.get("n_segments", 20),
                    k_seg=ms_cfg.get("k_segment_neighbors", 4),
                )
            if use_combined:
                build_profile_features_batched(
                    batch,
                    n_bins=profile_n_bins,
                    feature_set=profile_feature_set,
                    include_stream_onehot=profile_include_stream_onehot,
                )
            # Normalize node features — MUST match training pipeline exactly
            if norm_mean_dev is not None:
                batch.x = (batch.x - norm_mean_dev) / norm_std_dev
            if use_combined:
                emb = encoder.get_combined_embedding(batch)  # [B, 128+64]
            elif hasattr(encoder, "encoder"):
                # Full model passed — use just the encoder sub-module
                emb = encoder.encoder(batch)  # [B, 128]
            else:
                emb = encoder(batch)  # [B, 128]
            embeddings.append(emb.cpu())

    all_emb = torch.cat(embeddings, dim=0)  # [N, D]
    log.info("  Embeddings computed: shape %s (combined=%s)", list(all_emb.shape), use_combined)
    return all_emb, run_ids


def train_sbi_for_model(
    dm_model: str,
    cfg: dict,
    encoder: torch.nn.Module,
    normalizer: FeatureNormalizer,
    device: str,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Full SBI training loop for one DM model."""
    t0 = time.time()

    # Skip if already trained and not re-running
    posterior_path = checkpoint_dir / f"posterior_{dm_model}.pkl"
    if posterior_path.exists() and not args.resume:
        log.info("Posterior already exists for %s at %s — skipping (use --resume to overwrite)",
                 dm_model, posterior_path)
        return

    sim_dir = ROOT / cfg["paths"]["simulations"]

    # --- Step 1: Extract theta ---
    # extract_theta_from_hdf5 auto-detects layout (mixed chunks vs per-model subdirs)
    log.info("[%s] Extracting theta from HDF5 ...", dm_model)
    theta_all, _valid_flat_indices, theta_run_ids = extract_theta_from_hdf5(
        sim_dir=sim_dir,
        dm_model=dm_model,
        config_path=str(ROOT / "config" / "dm_models.yaml"),
        theta_mode=args.theta_mode,
    )
    log.info("[%s] theta shape: %s", dm_model, list(theta_all.shape))

    # --- Step 2: Precompute embeddings ---
    has_profile = getattr(args, "use_profile", False)
    log.info("[%s] Precomputing GNN embeddings (profile=%s) ...", dm_model, has_profile)
    embeddings_all, emb_run_ids = precompute_embeddings_for_model(
        dm_model,
        encoder,
        normalizer,
        cfg,
        device,
        batch_size=args.batch_size,
        max_sims=args.max_sims_per_model,
        use_profile=has_profile,
        profile_features_path=args.profile_features_path,
        downsample_seed=args.downsample_seed,
    )

    # --- Step 3: Align theta and embeddings by run_id key ---
    # Previous approach relied on positional ordering, which breaks silently if
    # a file is added/removed between the two calls.  Aligning by run_id key
    # makes the pairing explicit and detectable — any orphaned run_id is logged.
    emb_run_id_to_row: dict[str, int] = {rid: row for row, rid in enumerate(emb_run_ids)}

    theta_rows: list[int] = []
    emb_rows: list[int] = []
    n_missing_emb = 0
    for theta_row, run_id in enumerate(theta_run_ids):
        emb_row = emb_run_id_to_row.get(run_id)
        if emb_row is None:
            n_missing_emb += 1
        else:
            theta_rows.append(theta_row)
            emb_rows.append(emb_row)

    if n_missing_emb:
        log.warning("[%s] %d theta rows have no matching embedding run_id — dropped",
                    dm_model, n_missing_emb)

    if not theta_rows:
        raise RuntimeError(
            f"[{dm_model}] No theta/embedding pairs could be aligned by run_id. "
            "Check that extract_theta_from_hdf5 and precompute_embeddings_for_model "
            "are reading from the same directory layout."
        )

    theta_all = theta_all[theta_rows]
    x_embeddings = embeddings_all[emb_rows]
    aligned_run_ids = [theta_run_ids[i] for i in theta_rows]
    log.info("[%s] Aligned by run_id: %d (theta, embedding) pairs (dropped %d unmatched)",
             dm_model, len(theta_rows), n_missing_emb)

    allowed_run_ids = getattr(args, "allowed_run_ids", None)
    if allowed_run_ids is not None:
        keep_np = np.asarray([rid in allowed_run_ids for rid in aligned_run_ids], dtype=bool)
        n_before = len(keep_np)
        if not keep_np.any():
            raise RuntimeError(
                f"[{dm_model}] Split '{args.sbi_split}' selected zero aligned simulations. "
                "Check --split-file and the simulation directory."
            )
        theta_all = theta_all[torch.from_numpy(keep_np)]
        x_embeddings = x_embeddings[torch.from_numpy(keep_np)]
        aligned_run_ids = [rid for rid, keep in zip(aligned_run_ids, keep_np) if keep]
        log.info(
            "[%s] Split filter '%s': kept %d/%d aligned pairs",
            dm_model,
            args.sbi_split,
            len(aligned_run_ids),
            n_before,
        )

    # Validate no NaN in either
    if torch.isnan(theta_all).any():
        bad = torch.isnan(theta_all).any(dim=1)
        log.warning("[%s] Dropping %d theta rows with NaN", dm_model, bad.sum().item())
        theta_all = theta_all[~bad]
        x_embeddings = x_embeddings[~bad]
        aligned_run_ids = [rid for rid, keep in zip(aligned_run_ids, (~bad).cpu().numpy()) if bool(keep)]

    if torch.isnan(x_embeddings).any():
        bad = torch.isnan(x_embeddings).any(dim=1)
        log.warning("[%s] Dropping %d embedding rows with NaN", dm_model, bad.sum().item())
        theta_all = theta_all[~bad]
        x_embeddings = x_embeddings[~bad]
        aligned_run_ids = [rid for rid, keep in zip(aligned_run_ids, (~bad).cpu().numpy()) if bool(keep)]

    n_total = len(theta_all)
    log.info("[%s] Final SBI training set: %d (theta, embedding) pairs", dm_model, n_total)

    # --- Step 4: Build prior ---
    prior = build_prior(
        dm_model=dm_model,
        config_path=str(ROOT / "config" / "dm_models.yaml"),
        device=device,
        theta_mode=args.theta_mode,
    )
    log.info("[%s] Prior built: %d parameters", dm_model, prior.event_shape[0])

    if args.dry_run:
        log.info("[%s] Dry run complete; skipping NPE training.", dm_model)
        return

    # --- Step 5: Validate theta is within prior bounds ---
    # sbi 0.24 BoxUniform stores bounds in .support; access via sample to check shape
    try:
        # Try common attribute names across sbi versions
        low = getattr(prior, "low", None)
        if low is None:
            low = getattr(prior, "_low", None)
        high = getattr(prior, "high", None)
        if high is None:
            high = getattr(prior, "_high", None)
        if low is None:  # sbi 0.24+: bounds in base_dist
            low  = prior.base_dist.low.cpu()
            high = prior.base_dist.high.cpu()
        else:
            low, high = low.cpu(), high.cpu()
        in_prior = ((theta_all >= low) & (theta_all <= high)).all(dim=1)
        n_out = (~in_prior).sum().item()
        if n_out > 0:
            log.warning("[%s] %d theta rows outside prior bounds — clipping to prior",
                        dm_model, n_out)
            theta_all = theta_all.clamp(min=low, max=high)
        log.info("[%s] Prior bounds: low=%s, high=%s",
                 dm_model, low.tolist(), high.tolist())
    except Exception as e:
        log.warning("[%s] Could not validate prior bounds (%s) — skipping clamp", dm_model, e)

    # --- Step 6: Train NPE ---
    log.info("[%s] Training SNPE-C on %d simulations ...", dm_model, n_total)
    density_estimator, posterior = train_npe(
        theta=theta_all,
        x_embeddings=x_embeddings,
        prior=prior,
        dm_model=dm_model,
        config_path=str(ROOT / "config" / "training.yaml"),
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    elapsed = time.time() - t0
    log.info("[%s] SBI training complete in %.1f min. Posterior saved to %s",
             dm_model, elapsed / 60, checkpoint_dir)


def main(args: argparse.Namespace) -> None:
    cfg_path = ROOT / "config" / "training.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if args.sim_dir:
        cfg["paths"]["simulations"] = args.sim_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("SBI training on device: %s", device)

    if device == "cuda":
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        frac = cfg["training"].get("cuda_memory_fraction", 0.75)
        torch.cuda.set_per_process_memory_fraction(frac)
        log.info("CUDA allocator capped at %.0f%% of VRAM (%.1f / %.1f GB)",
                 frac * 100, frac * vram_total, vram_total)

    checkpoint_dir = Path(args.sbi_output_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    norm_path = Path(args.normalizer)
    if not norm_path.is_absolute():
        norm_path = ROOT / norm_path

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"GNN checkpoint not found: {ckpt_path}. "
            "Train V2 first with scripts/train_v2.py --epochs 300."
        )
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalizer not found: {norm_path}")

    if args.sbi_split == "all":
        args.allowed_run_ids = None
        args.resolved_split_file = None
        log.warning("SBI split mode is 'all': the test split may leak into posterior training.")
    else:
        split_path = _resolve_split_file(args.split_file, ckpt_path)
        if split_path is None or not split_path.exists():
            raise FileNotFoundError(
                "No split file found for split-aware SBI training. "
                "Pass --split-file, or use --sbi-split all only for compatibility/debugging."
            )
        args.resolved_split_file = str(split_path)
        args.allowed_run_ids = _run_ids_for_split(cfg, split_path, args.sbi_split)

    # Load normalizer
    mean, std = load_normalizer(norm_path)
    normalizer = FeatureNormalizer(mean, std)
    log.info("Normalizer loaded from %s", norm_path)

    # Load GNN model once — shared across all DM models
    use_profile = getattr(args, "use_profile", False)
    model, has_profile = load_gnn_model(
        cfg, ckpt_path, device,
        model_version=args.model_version,
        use_profile_branch=use_profile,
    )
    args.use_profile = bool(use_profile or has_profile)
    profile_cfg = cfg["graph"].get("profile_branch", {})
    if args.profile_features_path is None:
        args.profile_features_path = profile_cfg.get("features_path")
    if args.downsample_seed is None and args.profile_features_path is not None:
        args.downsample_seed = int(profile_cfg.get("downsample_seed", 42))
    if args.profile_features_path is not None:
        profile_path = Path(args.profile_features_path)
        if not profile_path.is_absolute():
            profile_path = ROOT / profile_path
        args.profile_features_path = str(profile_path)
        log.info("Using profile feature cache for SBI embedding: %s", profile_path)
    # For backward compat: callers expecting an encoder get the full model
    # (precompute_embeddings_for_model handles both full model and bare encoder)
    encoder = model

    dm_models = args.dm_models
    log.info("Training SBI for models: %s (model_version=%s, theta_mode=%s, profile=%s)",
             dm_models, args.model_version, args.theta_mode, args.use_profile)

    for dm_model in dm_models:
        log.info("=" * 60)
        log.info("DM MODEL: %s", dm_model)
        log.info("=" * 60)
        try:
            train_sbi_for_model(
                dm_model=dm_model,
                cfg=cfg,
                encoder=encoder,
                normalizer=normalizer,
                device=device,
                checkpoint_dir=checkpoint_dir,
                args=args,
            )
        except Exception as e:
            log.error("[%s] Training FAILED: %s", dm_model, e, exc_info=True)
            log.error("[%s] Continuing with next model ...", dm_model)

    log.info("=" * 60)
    log.info("SBI training complete for all models: %s", dm_models)
    log.info("Posteriors saved to: %s", checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SBI posteriors for all DM models.")
    parser.add_argument(
        "--dm-models", nargs="+", default=DM_MODELS,
        choices=DM_MODELS, metavar="MODEL",
        help="DM models to train (default: all 4)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for GNN embedding precomputation (default: 64)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Overwrite existing posteriors (default: skip already-trained models)",
    )
    parser.add_argument(
        "--model-version", choices=["v1", "v2"], default="v2",
        help="GNN checkpoint architecture to load (default: v2)",
    )
    parser.add_argument(
        "--theta-mode", choices=["native", "suppression"], default="suppression",
        help="SBI parameterization: native model parameters or V2 [log10_M_hm, n_impacts]",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/gnn_v2_best.pt",
        help="GNN checkpoint path (default: checkpoints/gnn_v2_best.pt)",
    )
    parser.add_argument(
        "--normalizer", default="checkpoints/normalizer_v2.npz",
        help="Feature normalizer path (default: checkpoints/normalizer_v2.npz)",
    )
    parser.add_argument(
        "--sbi-output-dir", default="checkpoints/v2_sbi",
        help="Directory for trained SBI posteriors (default: checkpoints/v2_sbi)",
    )
    parser.add_argument(
        "--sim-dir", default=None,
        help="Override config paths.simulations for this SBI run.",
    )
    parser.add_argument(
        "--max-sims-per-model", type=int, default=None,
        help="Limit embedded simulations per model, useful for dry-run checks.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load V2 artifacts, extract theta, embed data, build prior, then skip NPE training.",
    )
    parser.add_argument(
        "--use-profile", action="store_true",
        help="Use combined GNN + profile branch embeddings (auto-detected from checkpoint if present).",
    )
    parser.add_argument(
        "--sbi-split", choices=["train", "trainval", "all"], default="trainval",
        help="Dataset split used for SBI training. Default trainval keeps the test split held out.",
    )
    parser.add_argument(
        "--split-file", default=None,
        help="Explicit split .npz file. Defaults to the newest split file next to the GNN checkpoint.",
    )
    parser.add_argument(
        "--profile-features-path", default=None,
        help="Optional precomputed profile-feature cache aligned to the simulation dataset.",
    )
    parser.add_argument(
        "--downsample-seed", type=int, default=None,
        help="Seed used for deterministic max_stars downsampling when profile caches are used.",
    )
    args = parser.parse_args()
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    main(args)
