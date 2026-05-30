"""
Apply trained GNN + NPE posterior to real stellar stream data.

Workflow:
1. Load processed stream HDF5 (data/processed/streams.h5)
2. Build graph representation
3. Embed via trained GNN encoder
4. Sample from trained NPE posterior conditioned on the embedding
5. Save posterior samples and compute credible intervals
6. Generate corner plots

Usage:
    python scripts/run_inference.py --stream GD1 --dm-model CDM
    python scripts/run_inference.py --stream GD1 --all-models
    python scripts/run_inference.py --all-streams --all-models
"""

from __future__ import annotations

import argparse
import logging
import site
import sys
from pathlib import Path

try:
    user_site = Path(site.getusersitepackages()).resolve()
    sys.path = [p for p in sys.path if not p or Path(p).resolve() != user_site]
except Exception:
    pass

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import (
    FeatureNormalizer,
    RealStreamDataset,
    build_knn_graph_batched,
    build_segment_graph,
)
from src.inference.posteriors import (
    compute_credible_intervals,
    compute_log_evidence,
    get_param_names,
    sample_posterior,
)
from src.inference.sbi_pipeline import build_prior
from src.models.gnn import StreamGNNMultiTask, StreamGNNMultiTaskV2
from src.models.utils import load_checkpoint, load_normalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DM_MODELS = ["CDM", "WDM", "FDM", "SIDM"]


def run_inference_one_stream(
    stream_name: str,
    dm_model: str,
    processed_path: str,
    gnn_checkpoint: str,
    npe_checkpoint_dir: str,
    cfg: dict,
    output_dir: Path,
    device: str = "cuda",
    model_version: str = "v2",
    theta_mode: str = "suppression",
    normalizer_path: str | None = None,
) -> dict:
    """Run full inference pipeline on one stream under one DM model."""
    log.info("Running inference: stream=%s, model=%s", stream_name, dm_model)

    # Load normalizer (must match training)
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    norm_path = Path(normalizer_path) if normalizer_path else ckpt_dir / "normalizer_v2.npz"
    normalizer = None
    if norm_path.exists():
        mean, std = load_normalizer(norm_path)
        normalizer = FeatureNormalizer(mean, std)
        log.info("Loaded normalizer from %s", norm_path)
    else:
        log.warning("No normalizer found at %s — inference will use unnormalized features!", norm_path)

    # Load stream as graph
    dataset = RealStreamDataset(
        processed_path, [stream_name],
        k_neighbors=cfg["graph"]["k_neighbors"],
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        normalizer=normalizer,
    )
    data = dataset[0].to(device)
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
    # Build graph on device (deferred from __getitem__)
    use_orbital = cfg["graph"].get("orbital_features", {}).get("enabled", False)
    ms_cfg = cfg["graph"].get("multi_scale", {})
    use_multi_scale = ms_cfg.get("enabled", False)
    build_knn_graph_batched(
        data,
        cfg["graph"]["k_neighbors"],
        normalizer,
        orbital_features=use_orbital,
    )
    if use_multi_scale:
        build_segment_graph(
            data,
            n_segments=ms_cfg.get("n_segments", 20),
            k_seg=ms_cfg.get("k_segment_neighbors", 4),
        )
    # Normalize node features — MUST match training pipeline exactly
    if normalizer is not None:
        data.x = (data.x - normalizer.mean.to(device)) / normalizer.std.to(device)

    # Load GNN and embed
    gcfg = cfg["model"]["gnn"]
    if model_version == "v2":
        v2_cfg = cfg.get("training_v2", {})
        model = StreamGNNMultiTaskV2(
            n_reg_targets=v2_cfg.get("n_reg_targets", 2),
            predict_uncertainty=v2_cfg.get("predict_uncertainty", False),
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
        model = StreamGNNMultiTask(
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
    load_checkpoint(gnn_checkpoint, model, device=device)
    model = model.to(device).eval()

    with torch.no_grad():
        embedding = model.encoder(data).squeeze(0).cpu()

    log.info("Stream %s embedded to %d-d vector.", stream_name, len(embedding))

    # Free GNN from VRAM before loading the NSF posterior — CUDA fragmentation on Windows
    # causes the posterior sampler to fail silently (exit code 5) if the GNN stays loaded.
    import gc  # noqa: PLC0415
    del model, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load trained NPE posterior
    import pickle  # noqa: PLC0415
    npe_path = Path(npe_checkpoint_dir) / f"posterior_{dm_model}.pkl"
    if not npe_path.exists():
        log.error("NPE posterior not found at %s. Train first with sbi_pipeline.", npe_path)
        return {}

    with open(npe_path, "rb") as f:
        posterior = pickle.load(f)

    # Sample posterior
    n_samples = cfg["inference"]["n_posterior_samples"]
    samples_df = sample_posterior(
        posterior,
        embedding,
        n_samples,
        dm_model,
        theta_mode=theta_mode,
    )
    ci_df = compute_credible_intervals(samples_df, cfg["inference"]["credible_interval"])
    log_ev = compute_log_evidence(posterior, embedding, n_samples=5000)

    log.info("Posterior for %s/%s:\n%s", stream_name, dm_model, ci_df.to_string())
    log.info("Log evidence: %.3f", log_ev)

    # Save outputs
    stream_out = output_dir / stream_name
    stream_out.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(stream_out / f"samples_{dm_model}.csv", index=False)
    ci_df.to_csv(stream_out / f"credible_intervals_{dm_model}.csv")

    # Corner plot
    try:
        from src.analysis.visualization import plot_corner  # noqa: PLC0415
        fig = plot_corner(samples_df, title=f"{stream_name} / {dm_model}",
                          output_path=stream_out / f"corner_{dm_model}.pdf")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        plt.close(fig)
    except Exception as e:
        log.warning("Corner plot failed: %s", e)

    return {"samples": samples_df, "ci": ci_df, "log_evidence": log_ev}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on real stream data.")
    parser.add_argument("--stream", default="GD1", help="Stream name from streams.yaml")
    parser.add_argument("--dm-model", default="CDM", choices=DM_MODELS)
    parser.add_argument("--all-models", action="store_true", help="Run all 4 DM models")
    parser.add_argument("--all-streams", action="store_true", help="Run all streams")
    parser.add_argument("--processed-path", default="data/processed/streams.h5")
    parser.add_argument("--gnn-checkpoint", default="checkpoints/gnn_v2_best.pt")
    parser.add_argument("--npe-checkpoint-dir", default="checkpoints/v2_sbi")
    parser.add_argument("--normalizer", default="checkpoints/normalizer_v2.npz")
    parser.add_argument("--model-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--theta-mode", choices=["native", "suppression"], default="suppression")
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open("config/streams.yaml") as f:
        streams_cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    streams = list(streams_cfg["streams"].keys()) if args.all_streams else [args.stream]
    dm_models = DM_MODELS if args.all_models else [args.dm_model]

    all_evidences = {}
    for stream in streams:
        all_evidences[stream] = {}
        for dm_model in dm_models:
            result = run_inference_one_stream(
                stream, dm_model,
                args.processed_path, args.gnn_checkpoint, args.npe_checkpoint_dir,
                cfg, output_dir, device,
                model_version=args.model_version,
                theta_mode=args.theta_mode,
                normalizer_path=args.normalizer,
            )
            if result:
                all_evidences[stream][dm_model] = result.get("log_evidence", 0.0)

    # Save combined evidence matrix
    import pandas as pd  # noqa: PLC0415
    ev_df = pd.DataFrame(all_evidences).T
    ev_df.to_csv(output_dir / "log_evidences.csv")
    log.info("Log evidence matrix saved to %s/log_evidences.csv", output_dir)
