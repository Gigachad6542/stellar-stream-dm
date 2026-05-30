"""
Run simulation-based calibration (SBC) using pre-computed GNN embeddings.

Uses the fast path (run_sbc_precomputed) on a held-out split of the simulation
dataset. For each DM model:
  1. Load trained posterior from checkpoints/posterior_{model}.pkl
  2. Extract theta and precompute embeddings for that model
  3. Use the saved test split by default (or an explicit split from the CLI)
  4. Run SBC: for each held-out (theta, embedding) pair, sample the posterior
     and compute the rank of the true theta
  5. K-S test rank uniformity (pass: p > 0.05)
  6. Save rank histograms to outputs/sbc/

Usage:
    python -u scripts/run_sbc.py
    python -u scripts/run_sbc.py --dm-models CDM WDM
    python -u scripts/run_sbc.py --n-sbc-trials 200 --n-posterior-samples 300
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import site
import sys
from pathlib import Path

import numpy as np

try:
    user_site = Path(site.getusersitepackages()).resolve()
    sys.path = [p for p in sys.path if not p or Path(p).resolve() != user_site]
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be before any plt import
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import (
    FeatureNormalizer,
    StreamSimDataset,
    build_knn_graph_batched,
    build_profile_features_batched,
    build_segment_graph,
    profile_feature_dim,
)
from src.data.splits import load_split_indices
from src.inference.calibration import run_sbc_precomputed, plot_rank_histograms
from src.inference.posteriors import get_param_names
from src.inference.sbi_pipeline import build_prior, extract_theta_from_hdf5
from src.models.gnn import StreamGNNMultiTask, StreamGNNMultiTaskV2
from src.models.utils import load_checkpoint, load_normalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
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


def load_gnn_model_for_sbc(
    cfg: dict,
    checkpoint_path: Path,
    device: str,
    model_version: str = "v2",
) -> tuple[torch.nn.Module, bool]:
    """Load trained GNN model from checkpoint, auto-detecting profile branch.

    Returns (model, has_profile).
    """
    gcfg = cfg["model"]["gnn"]
    ms_cfg = cfg["graph"].get("multi_scale", {})
    profile_cfg = cfg["graph"].get("profile_branch", {})

    # Auto-detect profile branch from checkpoint
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    has_profile = False
    if "model_state_dict" in payload:
        has_profile = any(k.startswith("profile_mlp.") for k in payload["model_state_dict"])
    ckpt_cfg = payload.get("config", {})
    if ckpt_cfg:
        ckpt_profile = ckpt_cfg.get("graph", {}).get("profile_branch", {})
        if ckpt_profile.get("enabled", False):
            has_profile = True
            profile_cfg = ckpt_profile
        ckpt_ms = ckpt_cfg.get("graph", {}).get("multi_scale", {})
        if ckpt_ms.get("enabled", False):
            ms_cfg = ckpt_ms

    if has_profile:
        cfg["graph"]["profile_branch"] = profile_cfg
        cfg["graph"]["multi_scale"] = ms_cfg

    profile_dim = (
        profile_feature_dim(
            int(profile_cfg.get("n_bins", 48)),
            str(profile_cfg.get("feature_set", "compact")).lower(),
            bool(profile_cfg.get("include_stream_onehot", True)),
        )
        if has_profile else 0
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
    log.info("GNN model loaded: profile=%s, epoch=%s", has_profile, payload.get("epoch", "?"))
    return multitask, has_profile


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
    """Load sims for dm_model and compute GNN embeddings.

    Returns (embeddings [N, D], run_ids [N]).
    D=128 for encoder-only, D=192 with profile branch.
    Reuses same logic as train_sbi.py.
    """
    from torch_geometric.loader import DataLoader
    from torch.utils.data import Subset
    import h5py
    from collections import defaultdict
    from src.data.dataset import stream_one_hot

    sim_dir = ROOT / cfg["paths"]["simulations"]
    model_dir = sim_dir / dm_model
    use_orbital = cfg["graph"].get("orbital_features", {}).get("enabled", False)
    ms_cfg = cfg["graph"].get("multi_scale", {})
    use_multi_scale = ms_cfg.get("enabled", False)

    if model_dir.exists() and list(model_dir.glob("*.h5")):
        data_dir = model_dir
        dm_filter = None
    else:
        data_dir = sim_dir
        dm_filter = dm_model

    ds_full = StreamSimDataset(
        sim_dir=data_dir,
        k_neighbors=cfg["graph"]["k_neighbors"],
        normalizer=normalizer,
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False,
        preload_ram=False,
        use_orbital_features=use_orbital,
        profile_features_path=profile_features_path,
        downsample_seed=downsample_seed,
    )

    if dm_filter is not None:
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

        ds_filtered = Subset(ds_full, matching_indices)

        # Preload matching sims
        if hasattr(ds_full, '_ram_x') and ds_full._ram_x is None:
            ds_full._ram_x = [None] * len(ds_full._index)
            ds_full._ram_y = [None] * len(ds_full._index)
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
                        x = np.column_stack([x, stream_one_hot(sname, len(x))])
                        ds_full._ram_x[idx] = x
                        ds_full._ram_y[idx] = y
    else:
        matching_indices = list(range(len(ds_full)))
        if max_sims is not None and len(matching_indices) > max_sims:
            matching_indices = matching_indices[:max_sims]
            log.info("  Limiting to %d sims for smoke/dry run", len(matching_indices))
            ds_filtered = Subset(ds_full, matching_indices)
        else:
            ds_filtered = ds_full
        if max_sims is None:
            ds_full._preload_all_into_ram()

    run_ids = [ds_full._index[matching_indices[i]][1] for i in range(len(matching_indices))]

    loader = DataLoader(ds_filtered, batch_size=batch_size, shuffle=False, num_workers=0)
    k = cfg["graph"]["k_neighbors"]

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
                emb = encoder.encoder(batch)  # [B, 128]
            else:
                emb = encoder(batch)  # [B, 128]
            embeddings.append(emb.cpu())

    all_emb = torch.cat(embeddings, dim=0)
    log.info("  Embeddings computed: shape %s (combined=%s)", list(all_emb.shape), use_combined)
    return all_emb, run_ids


def run_sbc_for_model(
    dm_model: str,
    cfg: dict,
    encoder: torch.nn.Module,
    normalizer: FeatureNormalizer,
    device: str,
    output_dir: Path,
    n_sbc_trials: int = 500,
    n_posterior_samples: int = 500,
    theta_mode: str = "suppression",
    posterior_dir: Path | None = None,
    use_profile: bool = False,
    profile_features_path: str | Path | None = None,
    downsample_seed: int | None = None,
    max_sims_per_model: int | None = None,
    sbc_split: str = "test",
    allowed_run_ids: set[str] | None = None,
    split_path: Path | None = None,
) -> dict:
    """Run SBC for one DM model using precomputed embeddings."""
    log.info("=" * 60)
    log.info("SBC for %s", dm_model)
    log.info("=" * 60)

    sim_dir = ROOT / cfg["paths"]["simulations"]
    ckpt_dir = posterior_dir or (ROOT / cfg["paths"]["checkpoints"])

    # 1. Load posterior
    posterior_path = ckpt_dir / f"posterior_{dm_model}.pkl"
    if not posterior_path.exists():
        log.error("No posterior found at %s — skipping %s", posterior_path, dm_model)
        return {}

    with open(posterior_path, "rb") as f:
        posterior = pickle.load(f)

    # 2. Extract theta
    theta_all, _, theta_run_ids = extract_theta_from_hdf5(
        sim_dir=sim_dir,
        dm_model=dm_model,
        config_path=str(ROOT / "config" / "dm_models.yaml"),
        theta_mode=theta_mode,
    )

    # 3. Precompute embeddings
    embeddings_all, emb_run_ids = precompute_embeddings_for_model(
        dm_model, encoder, normalizer, cfg, device,
        use_profile=use_profile,
        profile_features_path=profile_features_path,
        downsample_seed=downsample_seed,
        max_sims=max_sims_per_model,
    )

    # 4. Align by run_id
    emb_run_id_to_row = {rid: row for row, rid in enumerate(emb_run_ids)}
    theta_rows, emb_rows = [], []
    for theta_row, run_id in enumerate(theta_run_ids):
        emb_row = emb_run_id_to_row.get(run_id)
        if emb_row is not None:
            theta_rows.append(theta_row)
            emb_rows.append(emb_row)

    theta_aligned = theta_all[theta_rows]
    emb_aligned = embeddings_all[emb_rows]
    aligned_run_ids = np.asarray([theta_run_ids[i] for i in theta_rows], dtype=object)
    log.info("[%s] Aligned %d (theta, embedding) pairs", dm_model, len(theta_rows))

    # Drop NaN
    mask = ~(torch.isnan(theta_aligned).any(dim=1) | torch.isnan(emb_aligned).any(dim=1))
    mask_np = mask.cpu().numpy()
    theta_aligned = theta_aligned[mask]
    emb_aligned = emb_aligned[mask]
    aligned_run_ids = aligned_run_ids[mask_np]

    n_total = len(theta_aligned)
    if sbc_split == "last10":
        # Backward-compatible fallback only. Prefer the saved test split.
        n_holdout = max(100, int(n_total * 0.1))
        theta_test = theta_aligned[-n_holdout:]
        emb_test = emb_aligned[-n_holdout:]
        selected_run_ids = aligned_run_ids[-n_holdout:]
        split_label = "last10"
        log.warning("[%s] Using legacy last-10%% SBC split; this may overlap SBI training.", dm_model)
    else:
        if allowed_run_ids is None:
            raise RuntimeError(f"[{dm_model}] sbc_split={sbc_split!r} requires resolved split run IDs")
        keep_np = np.asarray([rid in allowed_run_ids for rid in aligned_run_ids], dtype=bool)
        if not keep_np.any():
            raise RuntimeError(
                f"[{dm_model}] Split '{sbc_split}' selected zero aligned simulations. "
                "Check --split-file and the simulation directory."
            )
        theta_test = theta_aligned[torch.from_numpy(keep_np)]
        emb_test = emb_aligned[torch.from_numpy(keep_np)]
        selected_run_ids = aligned_run_ids[keep_np]
        split_label = sbc_split

    n_selected_before_cap = len(theta_test)

    # Cap at n_sbc_trials
    if len(theta_test) > n_sbc_trials:
        theta_test = theta_test[:n_sbc_trials]
        emb_test = emb_test[:n_sbc_trials]
        selected_run_ids = selected_run_ids[:n_sbc_trials]

    log.info("[%s] SBC: %d trials from split '%s' (from %d aligned total)",
             dm_model, len(theta_test), split_label, n_total)

    # 6. Run SBC
    # Free GNN from VRAM before posterior sampling — Windows CUDA fragmentation
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = run_sbc_precomputed(
        posterior=posterior,
        theta=theta_test,
        embeddings=emb_test,
        n_posterior_samples=n_posterior_samples,
    )
    result["split_mode"] = split_label
    result["split_path"] = str(split_path) if split_path is not None else None
    result["n_aligned_total"] = int(n_total)
    result["n_selected_before_cap"] = int(n_selected_before_cap)

    # Get parameter names for plotting/reporting.
    param_names = get_param_names(
        dm_model,
        str(ROOT / "config" / "dm_models.yaml"),
        theta_mode=theta_mode,
    )
    result["param_names"] = param_names

    calibration_mask = np.ones(len(param_names), dtype=bool)
    excluded = []
    if theta_mode == "suppression" and dm_model in {"CDM", "SIDM"}:
        # CDM/SIDM are unsuppressed references. In suppression-mode theta,
        # log10_M_hm is mapped to the lower prior edge, so uniform-rank SBC is
        # not meaningful for that point-mass/boundary coordinate.
        for i, name in enumerate(param_names):
            if name == "log10_M_hm":
                calibration_mask[i] = False
                excluded.append(name)
    effective_pvalues = result["ks_pvalues"][calibration_mask]
    result["boundary_parameters_excluded_from_uniform_sbc"] = excluded
    result["calibration_mask"] = calibration_mask
    result["effective_ks_pvalues"] = effective_pvalues
    result["is_calibrated_effective"] = (
        bool(np.all(effective_pvalues > 0.05)) if len(effective_pvalues) else True
    )

    # 7. Save results
    model_out = output_dir / dm_model
    model_out.mkdir(parents=True, exist_ok=True)

    np.save(model_out / "sbc_ranks.npy", result["ranks"])
    np.save(model_out / "sbc_ks_pvalues.npy", result["ks_pvalues"])

    # Plot rank histograms
    try:
        plot_rank_histograms(
            result["ranks"],
            param_names,
            n_posterior_samples,
            output_path=str(model_out / "sbc_rank_histogram.pdf"),
        )
        log.info("[%s] Rank histogram saved to %s", dm_model, model_out / "sbc_rank_histogram.pdf")
    except Exception as e:
        log.warning("[%s] Rank histogram plot failed: %s", dm_model, e)

    # Report
    status = "PASSED" if result["is_calibrated_effective"] else "FAILED"
    if excluded:
        log.info("[%s] Excluding boundary parameters from uniform-rank SBC: %s",
                 dm_model, excluded)
    log.info("[%s] SBC %s - K-S p-values: %s",
             dm_model, status,
             {name: f"{p:.4f}" for name, p in zip(param_names, result["ks_pvalues"])})

    return result


def main(args: argparse.Namespace) -> None:
    cfg_path = ROOT / "config" / "training.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if args.sim_dir:
        cfg["paths"]["simulations"] = args.sim_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("SBC calibration on device: %s", device)

    if device == "cuda":
        frac = cfg["training"].get("cuda_memory_fraction", 0.75)
        torch.cuda.set_per_process_memory_fraction(frac)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    norm_path = Path(args.normalizer)
    if not norm_path.is_absolute():
        norm_path = ROOT / norm_path
    posterior_dir = Path(args.posterior_dir)
    if not posterior_dir.is_absolute():
        posterior_dir = ROOT / posterior_dir

    if args.sbc_split == "last10":
        allowed_run_ids = None
        split_path = None
    else:
        split_path = _resolve_split_file(args.split_file, ckpt_path)
        if split_path is None or not split_path.exists():
            raise FileNotFoundError(
                "No split file found for split-aware SBC. "
                "Pass --split-file, or use --sbc-split last10 only for legacy/debugging."
            )
        allowed_run_ids = _run_ids_for_split(cfg, split_path, args.sbc_split)

    # Load normalizer and GNN model (shared)
    mean, std = load_normalizer(norm_path)
    normalizer = FeatureNormalizer(mean, std)
    model, has_profile = load_gnn_model_for_sbc(cfg, ckpt_path, device, model_version=args.model_version)
    use_profile = args.use_profile or has_profile
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
        log.info("Using profile feature cache for SBC embedding: %s", profile_path)
    log.info("Profile branch: auto-detected=%s, cli=%s, using=%s", has_profile, args.use_profile, use_profile)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for dm_model in args.dm_models:
        try:
            result = run_sbc_for_model(
                dm_model=dm_model,
                cfg=cfg,
                encoder=model,
                normalizer=normalizer,
                device=device,
                output_dir=output_dir,
                n_sbc_trials=args.n_sbc_trials,
                n_posterior_samples=args.n_posterior_samples,
                theta_mode=args.theta_mode,
                posterior_dir=posterior_dir,
                use_profile=use_profile,
                profile_features_path=args.profile_features_path,
                downsample_seed=args.downsample_seed,
                max_sims_per_model=args.max_sims_per_model,
                sbc_split=args.sbc_split,
                allowed_run_ids=allowed_run_ids,
                split_path=split_path,
            )
            all_results[dm_model] = result
        except Exception as e:
            log.error("[%s] SBC FAILED: %s", dm_model, e, exc_info=True)

    # Summary
    log.info("=" * 60)
    log.info("SBC SUMMARY")
    log.info("=" * 60)
    all_pass = True
    for dm_model, result in all_results.items():
        if not result:
            log.info("  %s: SKIPPED (no posterior)", dm_model)
            all_pass = False
            continue
        effective_pass = bool(result.get("is_calibrated_effective", result["is_calibrated"]))
        status = "PASS" if effective_pass else "FAIL"
        pvals = result["ks_pvalues"]
        log.info("  %s: %s (K-S p-values: %s)", dm_model, status,
                 ", ".join(f"{p:.4f}" for p in pvals))
        if not effective_pass:
            all_pass = False

    summary = {
        "checkpoint": str(ckpt_path),
        "posterior_dir": str(posterior_dir),
        "output_dir": str(output_dir),
        "theta_mode": args.theta_mode,
        "sbc_split": args.sbc_split,
        "split_file": str(split_path) if split_path is not None else None,
        "max_sims_per_model": args.max_sims_per_model,
        "n_sbc_trials": args.n_sbc_trials,
        "n_posterior_samples": args.n_posterior_samples,
        "all_calibrated_effective": bool(all_pass),
        "models": {},
    }
    for dm_model, result in all_results.items():
        if not result:
            summary["models"][dm_model] = {"status": "skipped"}
            continue
        param_names = result.get("param_names", [])
        summary["models"][dm_model] = {
            "status": "pass" if result.get("is_calibrated_effective", False) else "fail",
            "param_names": list(param_names),
            "ks_pvalues": {
                name: float(value)
                for name, value in zip(param_names, result["ks_pvalues"])
            },
            "boundary_parameters_excluded_from_uniform_sbc": list(
                result.get("boundary_parameters_excluded_from_uniform_sbc", [])
            ),
            "n_trials": int(result.get("n_trials", 0)),
            "n_posterior_samples": int(result.get("n_posterior_samples", 0)),
            "rank_histogram_pdf": str(output_dir / dm_model / "sbc_rank_histogram.pdf"),
        }
    summary_path = output_dir / "sbc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("SBC summary written to %s", summary_path)

    if all_pass:
        log.info("ALL MODELS PASSED SBC CALIBRATION")
    else:
        log.warning("SOME MODELS FAILED SBC — see details above")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SBC calibration checks.")
    parser.add_argument(
        "--dm-models", nargs="+", default=DM_MODELS,
        choices=DM_MODELS, metavar="MODEL",
    )
    parser.add_argument(
        "--n-sbc-trials", type=int, default=500,
        help="Max SBC trials per model (default: 500)",
    )
    parser.add_argument(
        "--n-posterior-samples", type=int, default=500,
        help="Posterior samples per SBC trial (default: 500)",
    )
    parser.add_argument("--model-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--theta-mode", choices=["native", "suppression"], default="suppression")
    parser.add_argument("--checkpoint", default="checkpoints/gnn_v2_best.pt")
    parser.add_argument("--normalizer", default="checkpoints/normalizer_v2.npz")
    parser.add_argument("--posterior-dir", default="checkpoints/v2_sbi")
    parser.add_argument(
        "--sim-dir", default=None,
        help="Override config paths.simulations for this SBC run.",
    )
    parser.add_argument(
        "--sbc-split", choices=["test", "val", "train", "trainval", "last10"], default="test",
        help="Split used for SBC. Default test is held out from split-aware SBI training.",
    )
    parser.add_argument(
        "--split-file", default=None,
        help="Explicit split .npz file. Defaults to the newest split file next to the GNN checkpoint.",
    )
    parser.add_argument(
        "--use-profile", action="store_true",
        help="Use profile branch for combined embeddings (auto-detected from checkpoint if not set)",
    )
    parser.add_argument(
        "--profile-features-path", default=None,
        help="Optional precomputed profile-feature cache aligned to the simulation dataset.",
    )
    parser.add_argument(
        "--downsample-seed", type=int, default=None,
        help="Seed used for deterministic max_stars downsampling when profile caches are used.",
    )
    parser.add_argument(
        "--max-sims-per-model", type=int, default=None,
        help="Limit embedded simulations per model, useful for SBC smoke checks.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/sbc",
        help="Directory for SBC rank arrays and plots.",
    )
    args = parser.parse_args()
    main(args)
