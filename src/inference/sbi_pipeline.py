"""
Simulation-based inference pipeline using the sbi package (SNPE-C / APT).

Two-level inference strategy:
    Level 1: Per-stream NPE — infer {log10_M_sub_mean, n_impacts} (+ model-specific param)
             from GNN embedding.
    Level 2: combine_posteriors.py — product of per-stream posteriors -> DM model parameters.

Critical design note:
    The NSF (neural spline flow) inside sbi cannot accept PyG Data objects directly.
    The GNN must be called once on the real observation to produce a fixed 128-d vector;
    that vector is then the observation for the NSF. Do NOT pass raw graphs to sbi.

References:
    Cranmer+2020 (sbi framework)
    Papamakarios+2021 (neural posterior estimation review)
    sbi >= 0.23: SNPE_C with custom embedding nets
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from torch_geometric.data import Data

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------------

def build_prior(
    dm_model: str,
    config_path: str = "config/dm_models.yaml",
    device: str = "cpu",
    theta_mode: str = "native",
) -> BoxUniform:
    """Build a uniform prior over the per-stream inference parameters.

    The prior is defined in dm_models.yaml under each model's
    'inferred_parameters' list.

    Returns:
        sbi BoxUniform prior over [log10_M_sub_mean, n_impacts, (model-specific)].
    """
    with open(config_path) as f:
        root_cfg = yaml.safe_load(f)

    if theta_mode == "suppression":
        sup_cfg = root_cfg.get("suppression_scale_inference", {})
        sup_prior = sup_cfg.get("prior", ["uniform", 4.0, 10.0])
        native_params = root_cfg["models"][dm_model]["inferred_parameters"]
        n_prior = next((p["prior"] for p in native_params if p["name"] == "n_impacts"),
                       ["poisson_rate_uniform", 0.0, 10.0])
        low = torch.tensor([float(sup_prior[1]), float(n_prior[1])], dtype=torch.float32)
        high = torch.tensor([float(sup_prior[2]), float(n_prior[2])], dtype=torch.float32)
        return BoxUniform(low=low, high=high, device=device)

    if theta_mode != "native":
        raise ValueError(f"Unknown theta_mode={theta_mode!r}; expected 'native' or 'suppression'.")

    cfg = root_cfg["models"][dm_model]

    params = cfg["inferred_parameters"]
    lows, highs = [], []
    for p in params:
        prior_spec = p["prior"]
        if prior_spec[0] in ("uniform", "log_uniform"):
            lows.append(float(prior_spec[1]))
            highs.append(float(prior_spec[2]))
        elif prior_spec[0] == "poisson_rate_uniform":
            lows.append(float(prior_spec[1]))
            highs.append(float(prior_spec[2]))
        else:
            raise ValueError(f"Unknown prior type: {prior_spec[0]}")

    low = torch.tensor(lows, dtype=torch.float32)
    high = torch.tensor(highs, dtype=torch.float32)
    return BoxUniform(low=low, high=high, device=device)


# ---------------------------------------------------------------------------
# SBI embedding wrapper
# ---------------------------------------------------------------------------

class SBIEmbeddingWrapper(nn.Module):
    """Thin wrapper that calls the GNN encoder and returns a flat 1D tensor.

    sbi's SNPE requires the observation to be a 1D float tensor.
    This wrapper is registered as the embedding_net so sbi can call it
    during training.

    IMPORTANT: at inference time, call this wrapper ONCE on the real stream Data
    object to get the fixed embedding, then pass the embedding tensor (not the Data)
    to posterior.sample().
    """

    def __init__(self, gnn_encoder: nn.Module) -> None:
        super().__init__()
        self.gnn = gnn_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """During SBI training x is a pre-computed embedding tensor [B, 128].

        We pre-compute embeddings from the simulation Data objects before
        passing to sbi (see train_npe). This wrapper is a pass-through
        used by sbi's density estimator.
        """
        return x  # embeddings already computed externally

    @torch.no_grad()
    def embed_observation(self, data: Data) -> torch.Tensor:
        """Embed a real or simulated stream graph into a 1D tensor for inference.

        Automatically moves `data` to the same device as the GNN encoder.
        """
        self.gnn.eval()
        # Infer device from GNN parameters
        try:
            device = next(self.gnn.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        data = data.to(device)
        return self.gnn(data).squeeze(0)  # [128]

    def embed_observation_mc(
        self, data: Data, T: int = 20
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout embedding of a real observation.

        Runs T stochastic forward passes through the GNN encoder with dropout
        active to estimate epistemic uncertainty in the embedding.

        Args:
            data: PyG Data object for one stream.
            T: Number of stochastic forward passes.

        Returns:
            mean_emb: [embedding_dim] mean embedding (use for NPE posterior sampling).
            std_emb: [embedding_dim] elementwise std (uncertainty signal for OOD detection).
        """
        try:
            device = next(self.gnn.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        data = data.to(device)

        # mc_embed handles eval + enable_mc_dropout internally
        mean_emb, std_emb = self.gnn.mc_embed(data, T=T)
        return mean_emb.squeeze(0), std_emb.squeeze(0)  # [128], [128]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def precompute_embeddings(
    gnn_encoder: nn.Module,
    data_list: list[Data],
    device: str = "cuda",
    batch_size: int = 32,
    k_neighbors: int = 16,
    normalizer=None,
    use_multi_scale: bool = False,
    multi_scale_cfg: Optional[dict] = None,
    use_orbital_features: bool = False,
) -> torch.Tensor:
    """Pre-compute GNN embeddings for all simulations before calling sbi.

    sbi trains its density estimator on (theta, x) pairs where x is a fixed
    1D tensor. Pre-computing embeddings avoids running the GNN inside sbi's
    training loop (which would be slow and break sbi's internal batching).

    Args:
        gnn_encoder: Trained GNN encoder (StreamGNNEncoder).
        data_list: List of PyG Data objects (one per simulation).
        device: CUDA or CPU.

    Returns:
        Tensor [N, embedding_dim].
    """
    from torch_geometric.loader import DataLoader  # noqa: PLC0415
    from src.data.dataset import build_knn_graph_batched, build_segment_graph  # noqa: PLC0415

    gnn_encoder = gnn_encoder.to(device).eval()
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    # Cache normalizer stats on device for node feature normalization
    norm_mean_dev = normalizer.mean.to(device) if normalizer else None
    norm_std_dev = normalizer.std.to(device) if normalizer else None
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Graph deferred to here (same as training loop) — build on GPU
            build_knn_graph_batched(
                batch,
                k_neighbors,
                normalizer,
                orbital_features=use_orbital_features,
            )
            if use_multi_scale:
                ms_cfg = multi_scale_cfg or {}
                build_segment_graph(
                    batch,
                    n_segments=ms_cfg.get("n_segments", 20),
                    k_seg=ms_cfg.get("k_segment_neighbors", 4),
                )
            # Normalize node features — MUST match training pipeline
            if norm_mean_dev is not None:
                batch.x = (batch.x - norm_mean_dev) / norm_std_dev
            emb = gnn_encoder(batch)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


def train_npe(
    theta: torch.Tensor,           # [N, n_params] parameter samples from prior
    x_embeddings: torch.Tensor,    # [N, 128] pre-computed GNN embeddings
    prior: BoxUniform,
    dm_model: str = "CDM",
    config_path: str = "config/training.yaml",
    checkpoint_dir: str | Path = "checkpoints",
    device: str = "cuda",
) -> tuple:
    """Train the neural posterior estimator (SNPE-C / APT).

    Args:
        theta: Parameter samples drawn from the prior.
        x_embeddings: Corresponding GNN embeddings of simulated streams.
        prior: sbi prior object.
        dm_model: DM model name (used for per-model checkpoint naming).
        config_path: Path to training.yaml.
        checkpoint_dir: Where to save trained density estimator.
        device: CUDA device string.

    Returns:
        (density_estimator, posterior) — both needed for sampling.
    """
    import pickle  # noqa: PLC0415

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["sbi"]

    de_cfg = cfg["density_estimator"]

    # sbi expects a simple nn.Module or string for embedding_net
    # Since we pre-computed embeddings, use an identity embedding net
    identity = nn.Identity()

    inference = SNPE(
        prior=prior,
        density_estimator=de_cfg["type"],  # "nsf"
        device=device,
    )

    inference.append_simulations(theta, x_embeddings)
    log.info("Training NPE (%s) with %d simulations...", dm_model, len(theta))
    density_estimator = inference.train(
        training_batch_size=512,
        learning_rate=5e-4,
        max_num_epochs=100,
        stop_after_epochs=20,
        show_train_summary=True,
    )

    posterior = inference.build_posterior(density_estimator)

    # Save density estimator state dict + full posterior
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(density_estimator.state_dict(), checkpoint_dir / f"density_estimator_{dm_model}.pt")

    # Save posterior via the reproducible format (torch.save + pickle fallback)
    from src.models.utils import save_posterior  # noqa: PLC0415
    save_posterior(
        checkpoint_dir / f"posterior_{dm_model}.pt",
        posterior,
        dm_model=dm_model,
        density_estimator_state_dict=density_estimator.state_dict(),
        prior_bounds={"low": prior.base_dist.low, "high": prior.base_dist.high},
    )
    # Legacy pickle format (backward compatibility with existing scripts)
    import pickle  # noqa: PLC0415
    with open(checkpoint_dir / f"posterior_{dm_model}.pkl", "wb") as f:
        pickle.dump(posterior, f)
    log.info("NPE training complete (%s). Density estimator + posterior saved to %s",
             dm_model, checkpoint_dir)

    return density_estimator, posterior


def run_sequential_rounds(
    simulator_fn: Callable,
    prior: BoxUniform,
    gnn_encoder: nn.Module,
    n_rounds: int = 5,
    n_sims_first: int = 5000,
    n_sims_per_round: int = 2000,
    device: str = "cuda",
    config_path: str = "config/training.yaml",
    n_jobs: int = 4,
) -> tuple:
    """Sequential NPE rounds (SNPE-C algorithm) with parallel simulation.

    Each round:
    1. Sample theta from prior (round 1) or proposal posterior (round 2+).
    2. Simulate streams in parallel using joblib (loky backend for Windows).
    3. Pre-compute GNN embeddings for all simulated streams.
    4. Append (theta, embeddings) to the SNPE training set.
    5. Train the density estimator and build a new proposal posterior.

    This is the full SNPE-C / APT loop from Greenberg+2019. It focuses
    simulations on high-posterior-density regions, improving sample efficiency
    vs. the amortised single-round approach in train_npe().

    For initial training with a pre-existing 40K simulation budget, use
    train_npe() directly (scripts/train_sbi.py). Use this function when you
    want to refine posteriors with targeted simulations.

    Args:
        simulator_fn: Callable(theta_i: np.ndarray, seed: int) -> Data.
            Must return a PyG Data object with x, edge_index, edge_attr.
            Must be a top-level importable function (not a lambda/closure)
            for joblib's loky backend on Windows.
        prior: sbi BoxUniform prior.
        gnn_encoder: Trained GNN encoder module.
        n_rounds: Number of sequential rounds.
        n_sims_first: Number of simulations in round 1.
        n_sims_per_round: Number of simulations in rounds 2+.
        device: CUDA device string.
        config_path: Path to training.yaml.
        n_jobs: Number of parallel simulation workers.

    Returns:
        (density_estimator, posterior) after the final round.
    """
    from joblib import Parallel, delayed  # noqa: PLC0415

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["sbi"]

    inference = SNPE(
        prior=prior,
        density_estimator=cfg["density_estimator"]["type"],
        device=device,
    )
    proposal = prior
    posterior = None

    for round_i in range(1, n_rounds + 1):
        n_sims = n_sims_first if round_i == 1 else n_sims_per_round
        log.info("=== Sequential round %d/%d: %d simulations ===", round_i, n_rounds, n_sims)

        # 1. Sample theta from proposal
        theta = proposal.sample((n_sims,))
        theta_np = theta.cpu().numpy()

        # 2. Simulate in parallel
        log.info("Simulating %d streams in parallel (n_jobs=%d)...", n_sims, n_jobs)
        data_list = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(simulator_fn)(theta_np[i], seed=round_i * 100000 + i)
            for i in range(n_sims)
        )

        # 3. Pre-compute embeddings
        log.info("Pre-computing GNN embeddings...")
        embeddings = precompute_embeddings(
            gnn_encoder, data_list, device=device,
            k_neighbors=cfg.get("k_neighbors", 16),
        )

        # 4. Append to training set
        inference.append_simulations(theta, embeddings, proposal=proposal)

        # 5. Train
        log.info("Training density estimator (round %d)...", round_i)
        density_estimator = inference.train(
            training_batch_size=512,
            learning_rate=5e-4,
            max_num_epochs=100,
            stop_after_epochs=20,
            show_train_summary=True,
        )
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior  # next round samples from the sharpened posterior

    log.info("Sequential rounds complete (%d rounds).", n_rounds)
    return density_estimator, posterior


# ---------------------------------------------------------------------------
# Theta extraction from HDF5 dataset
# ---------------------------------------------------------------------------

# Explicit mapping: (dm_model, param_name_in_yaml) → (hdf5_attr_key, transform_fn)
# The transform_fn converts the stored value to the prior parameter space.
# e.g. WDM stores wdm_mass_kev in linear space; inference uses log10 space.
#
# DEQUANTIZATION: n_impacts is an integer (from Poisson draw) but the NSF density
# estimator is continuous. Feeding raw integers causes the NSF to learn point masses
# it cannot represent, leading to systematic SBC miscalibration (rank pileup at 0).
# Fix: add Uniform(-0.5, 0.5) noise ("dequantization", Hoogeboom+2021) so the NSF
# sees a smooth distribution it CAN learn. At inference time, round posterior samples
# for n_impacts back to nearest integer.
_DEQUANT_RNG = np.random.default_rng(12345)

# Prior bounds for n_impacts clamping during dequantization.
# Must match the bounds in dm_models.yaml. Updated at runtime by extract_theta_from_hdf5.
_N_IMPACTS_LO, _N_IMPACTS_HI = 0.0, 10.0


def _dequant_n_impacts(x: float) -> float:
    """Dequantize integer n_impacts with truncated uniform noise.

    Adds U(-0.5, 0.5) noise but clips to prior bounds [lo, hi] to avoid
    point masses at the boundaries after downstream clamping.
    """
    return float(np.clip(x + _DEQUANT_RNG.uniform(-0.5, 0.5),
                         _N_IMPACTS_LO, _N_IMPACTS_HI))


_PARAM_EXTRACTION = {
    # Shared across all models
    ("*", "log10_M_hm"):      ("log10_M_hm",     lambda x: x),
    ("*", "log10_M_sub_mean"): ("log_m_sub_mean", lambda x: x),   # already log10
    ("*", "n_impacts"):        ("n_subhalos",      _dequant_n_impacts),
    # WDM-specific
    ("WDM", "log10_m_wdm_kev"): ("wdm_mass_kev",  lambda x: np.log10(x) if x > 0 else np.nan),
    # FDM-specific
    ("FDM", "log10_m_axion_ev"): ("fdm_mass_ev",  lambda x: np.log10(x) if x > 0 else np.nan),
    # SIDM-specific
    ("SIDM", "log10_sigma_SIDM"): ("sidm_cross_sec", lambda x: np.log10(x) if x > 0 else np.nan),
}


def _get_param_extractor(dm_model: str, param_name: str):
    """Return (hdf5_attr_key, transform_fn) for a given model + param name."""
    # Try model-specific first, then wildcard
    key = (dm_model, param_name)
    if key in _PARAM_EXTRACTION:
        return _PARAM_EXTRACTION[key]
    key_wild = ("*", param_name)
    if key_wild in _PARAM_EXTRACTION:
        return _PARAM_EXTRACTION[key_wild]
    raise KeyError(
        f"No extraction rule for ({dm_model}, {param_name}). "
        f"Add it to _PARAM_EXTRACTION in sbi_pipeline.py."
    )


def extract_theta_from_hdf5(
    sim_dir: str | Path,
    dm_model: str,
    config_path: str = "config/dm_models.yaml",
    theta_mode: str = "native",
) -> tuple[torch.Tensor, list[int], list[str]]:
    """Extract theta (inference parameter) tensors from HDF5 simulation files.

    Reads the HDF5 labels.attrs for every simulation of the given DM model and
    assembles a [N, n_params] float32 tensor matching the prior in dm_models.yaml.

    Supports two directory layouts:
        1. Mixed chunks: sim_dir/chunk_*.h5 (each chunk contains all DM models;
           filter by attrs['dm_model']).
        2. Per-model subdirs: sim_dir/{dm_model}/*.h5 (legacy layout).

    Transformations applied (stored -> prior space):
        log10_M_sub_mean : log_m_sub_mean attr  (already log10, no transform)
        n_impacts        : n_subhalos attr       (dequantized with Uniform(-0.5, 0.5))
        log10_m_wdm_kev  : log10(wdm_mass_kev)
        log10_m_axion_ev : log10(fdm_mass_ev)
        log10_sigma_SIDM : log10(sidm_cross_sec)

    Special handling for n_impacts=0 sims:
        When n_impacts=0, log_m_sub_mean is undefined (no impactors). Legacy sims
        stored 0.0 which is outside the prior [5.0, 9.0]. We replace these with a
        random draw from the prior so the NSF learns posterior = prior when n_impacts=0.

    Args:
        sim_dir: Root simulation directory (contains chunk_*.h5 or {model}/ subdirs).
        dm_model: One of 'CDM', 'WDM', 'FDM', 'SIDM'.
        config_path: Path to dm_models.yaml.

    Returns:
        theta: [N_valid, n_params] float32 tensor.
        valid_indices: Flat dataset indices (0-based, HDF5 iteration order for
                       sims of this dm_model only) — legacy; prefer run_id alignment.
        valid_run_ids: HDF5 run_id string for each accepted theta row. Used by
                       train_sbi.py for explicit run_id-keyed alignment so that
                       theta[i] and embedding[j] are matched by key rather than by
                       position (position is fragile if files change between calls).
    """
    import h5py  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    sim_dir = _Path(sim_dir)

    # Determine layout: per-model subdir vs mixed chunks
    model_dir = sim_dir / dm_model
    if model_dir.exists() and list(model_dir.glob("*.h5")):
        # Legacy per-model layout
        h5_files = sorted(model_dir.glob("*.h5"))
        filter_by_attr = False
        log.info("Using per-model directory layout: %s", model_dir)
    else:
        # Mixed chunk layout: all models in sim_dir/*.h5
        h5_files = sorted(sim_dir.glob("chunk_*.h5"))
        if not h5_files:
            h5_files = sorted(sim_dir.glob("**/*.h5"))
        filter_by_attr = True
        log.info("Using mixed-chunk layout: %d files in %s (filtering for %s)",
                 len(h5_files), sim_dir, dm_model)

    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found for {dm_model} in {sim_dir}")

    with open(config_path) as f:
        root_cfg = yaml.safe_load(f)
    cfg_models = root_cfg["models"][dm_model]
    if theta_mode == "suppression":
        sup_cfg = root_cfg.get("suppression_scale_inference", {})
        sup_prior = sup_cfg.get("prior", ["uniform", 4.0, 10.0])
        native_n_prior = next(
            (p["prior"] for p in cfg_models["inferred_parameters"] if p["name"] == "n_impacts"),
            ["poisson_rate_uniform", 0.0, 10.0],
        )
        param_specs = [
            {"name": "log10_M_hm", "prior": sup_prior},
            {"name": "n_impacts", "prior": native_n_prior},
        ]
    elif theta_mode == "native":
        param_specs = cfg_models["inferred_parameters"]
    else:
        raise ValueError(f"Unknown theta_mode={theta_mode!r}; expected 'native' or 'suppression'.")
    param_names = [p["name"] for p in param_specs]
    log.info("Extracting theta for %s (%s): params=%s", dm_model, theta_mode, param_names)

    # Build prior bounds lookup for replacement draws
    _prior_bounds = {}
    for p in param_specs:
        _prior_bounds[p["name"]] = (float(p["prior"][1]), float(p["prior"][2]))

    # Update module-level n_impacts dequantization bounds to match this model's prior
    global _N_IMPACTS_LO, _N_IMPACTS_HI
    if "n_impacts" in _prior_bounds:
        _N_IMPACTS_LO, _N_IMPACTS_HI = _prior_bounds["n_impacts"]

    # RNG for replacing undefined parameters (seeded for reproducibility)
    _replace_rng = np.random.default_rng(99999)

    # Pre-build extractors list — fail fast if any param has no rule
    extractors = [_get_param_extractor(dm_model, pname) for pname in param_names]

    rows: list[list[float]] = []
    valid_indices: list[int] = []
    valid_run_ids: list[str] = []   # run_id for each accepted theta row — used by
                                    # train_sbi.py for explicit run_id-keyed alignment
                                    # instead of fragile positional ordering.
    flat_idx = 0
    n_nan = 0
    n_replaced = 0
    n_unsuppressed_mhm_remapped = 0
    n_skipped_model = 0

    dm_model_bytes = dm_model.encode()  # for comparing with h5py bytes attrs

    for h5_file in h5_files:
        with h5py.File(str(h5_file), "r") as f:
            if "simulations" not in f:
                continue
            for run_id in sorted(f["simulations"].keys()):
                grp = f[f"simulations/{run_id}"]

                # Filter by DM model if using mixed chunks
                if filter_by_attr:
                    sim_dm = grp.attrs.get("dm_model", b"")
                    if isinstance(sim_dm, bytes):
                        sim_dm_str = sim_dm.decode()
                    else:
                        sim_dm_str = str(sim_dm)
                    if sim_dm_str != dm_model:
                        n_skipped_model += 1
                        continue

                attrs = dict(grp["labels"].attrs)
                n_sub = int(attrs.get("n_subhalos", 0))
                row = []
                ok = True
                for (attr_key, transform), pname in zip(extractors, param_names):
                    if (
                        theta_mode == "suppression"
                        and pname == "log10_M_hm"
                        and dm_model in {"CDM", "SIDM"}
                    ):
                        # CDM/SIDM are the unsuppressed reference family in this
                        # parameterization.  Older Plan 2 files store 10.0 as a
                        # sentinel for "no suppression", but physically a high
                        # M_hm means strong suppression.  Map the reference case
                        # to the lower prior edge so SBC/posterior calibration
                        # tests do not treat the sentinel as a real half-mode
                        # mass.
                        val = float(_prior_bounds[pname][0])
                        n_unsuppressed_mhm_remapped += 1
                    else:
                        raw = attrs.get(attr_key)
                        if raw is None:
                            ok = False
                            break
                        val = float(transform(float(raw)))

                    # Handle undefined parameters when n_impacts=0:
                    # log_m_sub_mean is stored as 0.0 (outside prior [5,9]) when
                    # no impacts occurred. Replace with uniform draw from prior.
                    if pname == "log10_M_sub_mean" and n_sub == 0:
                        lo, hi = _prior_bounds["log10_M_sub_mean"]
                        val = float(_replace_rng.uniform(lo, hi))
                        n_replaced += 1

                    if not np.isfinite(val):
                        ok = False
                        n_nan += 1
                        break
                    row.append(val)

                if ok:
                    rows.append(row)
                    valid_indices.append(flat_idx)
                    valid_run_ids.append(run_id)
                flat_idx += 1

    if not rows:
        raise RuntimeError(f"No valid theta rows found for {dm_model} in {sim_dir}")

    if n_skipped_model:
        log.info("[%s] Skipped %d sims belonging to other DM models", dm_model, n_skipped_model)
    if n_nan:
        log.warning("[%s] %d simulations dropped — NaN in theta (e.g. model-specific "
                    "param not sampled for this DM model)", dm_model, n_nan)
    if n_replaced:
        log.info("[%s] %d sims with n_impacts=0: log10_M_sub_mean replaced with prior draws",
                 dm_model, n_replaced)
    if n_unsuppressed_mhm_remapped:
        log.info(
            "[%s] %d unsuppressed reference sims mapped to lower log10_M_hm prior edge",
            dm_model,
            n_unsuppressed_mhm_remapped,
        )

    theta = torch.tensor(rows, dtype=torch.float32)
    log.info("Extracted %d valid theta vectors for %s (out of %d total, %d NaN dropped)",
             len(rows), dm_model, flat_idx, flat_idx - len(rows))
    return theta, valid_indices, valid_run_ids
