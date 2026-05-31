"""Precompute graph-level profile features for V2 GNN profile-branch runs.

The profile branch is intentionally graph-level: density roughness, feature
quantiles, binned medians, and stream one-hot context. Building those features
inside every training batch is wasteful, so this script computes them once and
saves an index-aligned cache consumed by StreamSimDataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import site
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    user_site = Path(site.getusersitepackages()).resolve()
    sys.path = [p for p in sys.path if not p or Path(p).resolve() != user_site]
except Exception:
    pass

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.data.dataset import (
    StreamSimDataset,
    build_profile_features_batched,
    profile_feature_dim,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--feature-set", choices=["compact", "summary"], default="summary")
    parser.add_argument("--n-bins", type=int, default=48)
    parser.add_argument("--include-stream-onehot", action="store_true", default=True)
    parser.add_argument("--no-stream-onehot", dest="include_stream_onehot", action="store_false")
    parser.add_argument("--max-stars", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--downsample-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-preload-ram", action="store_true")
    parser.add_argument("--error-dr", action="store_true",
                        help="Build the cache from error-domain-randomized features, reading the "
                             "training_v2.error_domain_randomization block from --config so the cache "
                             "matches `train_v2.py --error-dr` exactly.")
    parser.add_argument("--config", default="config/training.yaml",
                        help="Config with the error_domain_randomization block (for --error-dr).")
    args = parser.parse_args()

    error_dr_cfg = None
    if args.error_dr:
        import yaml
        with open(args.config) as f:
            full_cfg = yaml.safe_load(f)
        error_dr_cfg = dict(full_cfg.get("training_v2", {}).get("error_domain_randomization", {}))
        error_dr_cfg["enabled"] = True
        log.info("Error domain randomization ENABLED for cache: %s", error_dr_cfg)

    device = _resolve_device(args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    expected_dim = profile_feature_dim(args.n_bins, args.feature_set, args.include_stream_onehot)

    log.info(
        "Precomputing profile features: sim_dir=%s feature_set=%s dim=%d device=%s",
        args.sim_dir,
        args.feature_set,
        expected_dim,
        device,
    )
    t0 = time.time()
    dataset = StreamSimDataset(
        args.sim_dir,
        max_stars=args.max_stars,
        augment=False,
        preload_ram=not args.no_preload_ram,
        downsample_seed=args.downsample_seed,
        error_dr=error_dr_cfg,
    )
    loader = PyGDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    features = np.empty((len(dataset), expected_dim), dtype=np.float32)
    offset = 0
    for batch_idx, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        build_profile_features_batched(
            batch,
            n_bins=args.n_bins,
            feature_set=args.feature_set,
            include_stream_onehot=args.include_stream_onehot,
        )
        chunk = batch.profile_x.detach().cpu().numpy().astype(np.float32, copy=False)
        n = len(chunk)
        features[offset:offset + n] = chunk
        offset += n
        if batch_idx == 1 or batch_idx % 10 == 0:
            log.info("Processed %d/%d simulations", offset, len(dataset))

    if offset != len(dataset):
        raise RuntimeError(f"Feature cache length mismatch: wrote {offset}, expected {len(dataset)}")

    metadata = {
        "sim_dir": str(args.sim_dir),
        "feature_set": args.feature_set,
        "n_bins": int(args.n_bins),
        "include_stream_onehot": bool(args.include_stream_onehot),
        "max_stars": int(args.max_stars),
        "downsample_seed": int(args.downsample_seed),
        "n_simulations": int(len(dataset)),
        "feature_dim": int(expected_dim),
        "elapsed_sec": float(time.time() - t0),
    }
    np.savez_compressed(out, features=features, metadata=json.dumps(metadata))
    (out.with_suffix(out.suffix + ".json")).write_text(json.dumps(metadata, indent=2))
    log.info("Wrote %s with shape %s in %.1fs", out, features.shape, metadata["elapsed_sec"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
