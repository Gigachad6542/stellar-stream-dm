"""
Parallelised training data generation: 40K simulations across 4 DM models.

Uses joblib loky backend (not multiprocessing.Pool) for stability on Windows
— the loky backend respawns workers cleanly without fork(), which is essential
with gala/astropy's complex import chains.

Usage:
    python scripts/generate_training_data.py --n-sims 10000 --n-jobs -2
    python scripts/generate_training_data.py --benchmark 100  # estimate runtime

Output:
    data/simulations/chunk_<chunk_id>.h5 files
    data/simulations/simulations.h5  (merged master file)
"""

from __future__ import annotations

# Pre-import sys / Path FIRST, then immediately patch sys.path before heavy imports.
# On Python 3.11 / Windows, importing numpy before src.* modules causes a zoneinfo
# circular-import deadlock (stack overflow) in the Python import lock machinery.
# Patching sys.path here — before numpy — prevents the deadlock.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Windows/conda galpy C-extension fix (must run BEFORE any galpy import).
#
# Problem: galpy's libgalpy.cp311-win_amd64.pyd cannot find its runtime DLL
#   dependencies (GSL, MinGW runtimes) because conda's Library/bin is not on
#   the Windows DLL search PATH when Python is invoked directly (not via
#   "conda activate").  galpy falls back to pure-Python scipy odeint, which is
#   100-200x slower than the C dop853_c integrator.
#
# Fix:
#   1. Add all conda DLL directories to os.environ["PATH"] so child processes
#      inherit them (critical for multiprocessing "spawn" workers).
#   2. Call os.add_dll_directory() so the current process's loader finds them.
#   3. Pre-load libgalpy via ctypes — this puts the DLL into the Windows
#      per-process loaded-module cache before galpy's import machinery tries
#      to load it, which guarantees the fast C integrator is available.
# ---------------------------------------------------------------------------

def _fix_galpy_dll_path() -> None:
    """Ensure galpy's C extension can find its Windows DLL dependencies."""
    import os
    import ctypes
    conda_root = Path(sys.executable).parent
    dll_dirs = [
        conda_root / "Library" / "bin",
        conda_root / "Library" / "mingw-w64" / "bin",
        conda_root,
    ]
    for d in dll_dirs:
        if d.exists():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(d))
                except OSError:
                    pass
    # Pre-load the PYD so it's in the process loaded-module cache
    pyd = (conda_root / "Lib" / "site-packages"
           / "libgalpy.cp311-win_amd64.pyd")
    if pyd.exists():
        try:
            ctypes.CDLL(str(pyd))
        except OSError:
            pass


_fix_galpy_dll_path()

import argparse
import logging
import time
import uuid

import h5py
import numpy as np
import yaml

from src.simulation.baryonic import (
    apply_bar_perturbation,
    apply_giant_molecular_cloud_encounter,
)
from src.simulation.mass_functions import (
    compute_encounter_rate,
    fdm_jeans_mass,
    mass_function_suppression_factor,
    wdm_half_mode_mass,
)
from src.simulation.noise import (
    add_foreground_contamination,
    add_gaia_noise_randomized,
    build_membership_probabilities,
)
from src.simulation.potentials import get_mw_potential, vary_potential
from src.simulation.stream_gen import (
    StreamParticles,
    generate_stream,
    resample_to_observational_selection,
    set_progenitor_ic,
)
from src.simulation.subhalo import apply_n_subhalo_encounters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DM_MODELS = ["CDM", "WDM", "FDM", "SIDM"]
SIGNAL_LADDER_STAGES = [
    "clean_impact",
    "gaia_noise",
    "foreground",
    "baryonic",
    "detector_balanced",
]


def _sample_config_value(spec, rng: np.random.Generator, default):
    """Sample scalar values from compact YAML specs.

    Supported forms:
      - ["uniform", low, high]
      - [low, high]
      - scalar
    """
    if spec is None:
        return default
    if isinstance(spec, (list, tuple)):
        if len(spec) == 3 and str(spec[0]).lower() == "uniform":
            return float(rng.uniform(float(spec[1]), float(spec[2])))
        if len(spec) == 2:
            return float(rng.uniform(float(spec[0]), float(spec[1])))
    return spec


def _log10_half_mode_mass(dm_model: str, dm_params: dict) -> float:
    """Return model-independent log10(M_hm/Msun) label for training."""
    if dm_model == "WDM":
        return float(np.log10(wdm_half_mode_mass(float(dm_params["m_wdm_kev"]))))
    if dm_model == "FDM":
        return float(np.log10(fdm_jeans_mass(float(dm_params["m_axion_ev"]))))
    # CDM and the current SIDM configuration are unsuppressed in the mass
    # function.  A high M_hm would mean strong suppression, so use a lower-edge
    # CDM-like value rather than the legacy 10.0 sentinel.
    return 4.0


def _apply_signal_ladder_preset(training_cfg: dict, stage: str) -> dict:
    """Return a generation config for signal-first curriculum datasets.

    These presets intentionally keep the existing Plan 2 configuration intact.
    They are written to an effective YAML file in the output directory so child
    worker processes see the same overrides as the parent process.
    """
    import copy

    cfg = copy.deepcopy(training_cfg)
    sim_cfg = cfg.setdefault("simulation", {})
    prep_cfg = cfg.setdefault("preprocessing", {})
    ladder_cfg = sim_cfg.setdefault("signal_ladder", {})

    stage = stage.lower()
    if stage not in set(SIGNAL_LADDER_STAGES):
        raise ValueError(
            "signal ladder stage must be one of: " + ", ".join(SIGNAL_LADDER_STAGES)
        )

    sim_cfg["output_dir"] = f"data/simulations_signal_ladder/{stage}"
    sim_cfg["streams_for_training"] = ["GD1", "Pal5", "Orphan"]
    sim_cfg["n_steps_back"] = min(int(sim_cfg.get("n_steps_back", 60)), 60)
    sim_cfg["min_source_n_stars"] = 1200
    sim_cfg["max_source_n_stars"] = 2400
    sim_cfg["source_n_stars_factor"] = 1.5
    prep_cfg["max_stars_per_sim"] = min(int(prep_cfg.get("max_stars_per_sim", 1200)), 1200)

    # Keep WDM/FDM parameter labels available, but the first curriculum target
    # is observable impact/no-impact, not model family.
    domain_cfg = sim_cfg.setdefault("domain_randomization", {})
    domain_cfg.update({
        "enabled": False,
        "potential_variation": False,
        "n_stars_per_sim": 1200,
    })

    ladder_cfg.update({
        "enabled": True,
        "stage": stage,
        "encounter_count_mode": "alternating_zero_vs_impacted",
        "impacted_n_encounters": ["uniform_int", 1, 3],
        "disable_baryonic": True,
        "noise_scale_factor": 0.0,
        "foreground_contamination_fraction": 0.0,
        "membership_corruption_fraction": 0.0,
        "membership_noise_sigma": 0.0,
        "fixed_n_stars_per_sim": 1200,
        "log10_mass_range": [7.0, 7.8],
        "impact_parameter_pc_range": [80.0, 300.0],
        "flyby_velocity_kms_range": [150.0, 250.0],
        "t_since_impact_gyr_range": [2.0, 10.0],
    })

    if stage in {"gaia_noise", "foreground", "baryonic"}:
        ladder_cfg["noise_scale_factor"] = 1.0
    if stage in {"foreground", "baryonic"}:
        ladder_cfg["foreground_contamination_fraction"] = 0.10
        ladder_cfg["membership_corruption_fraction"] = 0.05
        ladder_cfg["membership_noise_sigma"] = 0.03
    if stage == "baryonic":
        ladder_cfg["disable_baryonic"] = False
        ladder_cfg["baryonic_perturbation_probability"] = 0.8
        ladder_cfg["bar_perturbation_probability"] = 0.3

    if stage == "detector_balanced":
        # Bridge dataset for the V2 detector.  Unlike Plan 2, impacts are
        # balanced independently of DM family so the binary head learns
        # observable morphology before returning to suppressed-vs-unsuppressed
        # inference.  Keep potential variation off for this stage; it is a
        # later robustness ablation once impact recovery is stable.
        sim_cfg["output_dir"] = "data/simulations_v2_detector_balanced"
        sim_cfg["streams_for_training"] = [
            "GD1", "Pal5", "Orphan", "ATLAS", "Jhelum", "Fjorm", "Sylgr",
        ]
        sim_cfg["n_steps_back"] = min(int(sim_cfg.get("n_steps_back", 60)), 60)
        sim_cfg["min_source_n_stars"] = 800
        sim_cfg["max_source_n_stars"] = 2600
        sim_cfg["source_n_stars_factor"] = 1.5
        prep_cfg["max_stars_per_sim"] = min(int(prep_cfg.get("max_stars_per_sim", 1200)), 1200)

        ladder_cfg.update({
            "noise_scale_factor": ["uniform", 0.5, 2.0],
            "foreground_contamination_fraction": ["uniform", 0.05, 0.30],
            "membership_corruption_fraction": ["uniform", 0.05, 0.20],
            "membership_noise_sigma": ["uniform", 0.03, 0.12],
            "n_stars_per_sim": ["uniform", 400, 1200],
            "disable_baryonic": False,
            "baryonic_perturbation_probability": 0.8,
            "bar_perturbation_probability": 0.3,
            "potential_variation": False,
            "log10_mass_range": [7.0, 7.8],
            "impact_parameter_pc_range": [80.0, 300.0],
            "flyby_velocity_kms_range": [150.0, 250.0],
            "t_since_impact_gyr_range": [2.0, 10.0],
        })

    return cfg


def _sample_ladder_encounter_count(
    ladder_cfg: dict,
    rng: np.random.Generator,
    seed: int,
    default: int,
    job_index: int | None = None,
) -> int:
    """Override encounter counts for signal-ladder datasets when requested."""
    if not ladder_cfg.get("enabled", False):
        return default

    mode = str(ladder_cfg.get("encounter_count_mode", "")).lower()
    if mode == "alternating_zero_vs_impacted":
        discriminator = seed if job_index is None else job_index
        if discriminator % 2 == 0:
            return 0
        spec = ladder_cfg.get("impacted_n_encounters", ["uniform_int", 1, 3])
        if isinstance(spec, (list, tuple)) and len(spec) >= 3 and str(spec[0]).lower() == "uniform_int":
            return int(rng.integers(int(spec[1]), int(spec[2]) + 1))
        return int(spec)
    if mode == "uniform_int":
        lo = int(ladder_cfg.get("min_n_encounters", 0))
        hi = int(ladder_cfg.get("max_n_encounters", 4))
        return int(rng.integers(lo, hi + 1))
    if mode == "fixed":
        return int(ladder_cfg.get("n_encounters", default))
    return default


# ---------------------------------------------------------------------------
# Module-level galstreams cache — loaded once per worker process (~13 s)
# ---------------------------------------------------------------------------

_MWS_CACHE = None
_CONFIG_CACHE: dict[str, dict] = {}


def _get_mws():
    """Return a cached galstreams.MWStreams instance (loaded on first call)."""
    global _MWS_CACHE
    if _MWS_CACHE is None:
        import galstreams  # noqa: PLC0415
        _MWS_CACHE = galstreams.MWStreams(verbose=False)
    return _MWS_CACHE


def _load_yaml_cached(path: str) -> dict:
    """Load YAML once per worker process."""
    resolved = str(Path(path).resolve())
    if resolved not in _CONFIG_CACHE:
        with open(resolved) as f:
            _CONFIG_CACHE[resolved] = yaml.safe_load(f)
    return _CONFIG_CACHE[resolved]


# ---------------------------------------------------------------------------
# Single simulation worker (must be top-level for joblib loky)
# ---------------------------------------------------------------------------

def generate_one_simulation(args_tuple) -> dict | None:
    """Generate one simulated stream with the given parameters.

    Args:
        args_tuple: (run_id, stream_name, dm_model, seed, cfg_path_streams,
            cfg_path_dm, cfg_path_training)

    Returns:
        Dict with 'run_id', 'data', 'labels' or None on failure.
    """
    if len(args_tuple) == 8:
        run_id, stream_name, dm_model, seed, cfg_streams, cfg_dm, cfg_training, job_index = args_tuple
    else:
        run_id, stream_name, dm_model, seed, cfg_streams, cfg_dm, cfg_training = args_tuple
        job_index = None

    try:
        rng = np.random.default_rng(seed)

        # Load configs
        stream_cfg = _load_yaml_cached(cfg_streams)["streams"][stream_name]
        dm_cfg_all = _load_yaml_cached(cfg_dm)
        training_cfg = _load_yaml_cached(cfg_training)
        dm_model_cfg = dict(dm_cfg_all["models"][dm_model])
        impact_cfg = dm_cfg_all["impact_physics"]
        dm_model_cfg["impact_physics"] = impact_cfg
        sim_cfg = training_cfg.get("simulation", {})
        domain_cfg = sim_cfg.get("domain_randomization", {})
        ladder_cfg = sim_cfg.get("signal_ladder", {})
        ladder_enabled = bool(ladder_cfg.get("enabled", False))
        domain_enabled = bool(domain_cfg.get("enabled", False)) and not ladder_enabled

        noise_scale_factor = float(_sample_config_value(
            domain_cfg.get("noise_scale_factor"), rng, 1.0,
        )) if domain_enabled else float(_sample_config_value(ladder_cfg.get("noise_scale_factor"), rng, 1.0))
        contamination_fraction = float(_sample_config_value(
            domain_cfg.get("foreground_contamination_fraction"), rng, 0.10,
        )) if domain_enabled else float(_sample_config_value(ladder_cfg.get("foreground_contamination_fraction"), rng, 0.10))
        membership_corruption_fraction = float(_sample_config_value(
            domain_cfg.get("membership_corruption_fraction"), rng, 0.0,
        )) if domain_enabled else float(_sample_config_value(ladder_cfg.get("membership_corruption_fraction"), rng, 0.0))
        membership_noise_sigma = float(_sample_config_value(
            domain_cfg.get("membership_noise_sigma"), rng, 0.0,
        )) if domain_enabled else float(_sample_config_value(ladder_cfg.get("membership_noise_sigma"), rng, 0.0))
        baryonic_prob = float(_sample_config_value(domain_cfg.get(
            "baryonic_perturbation_probability",
            sim_cfg.get("baryonic_perturbation_prob", 0.5),
        ), rng, sim_cfg.get("baryonic_perturbation_prob", 0.5))) if domain_enabled else float(_sample_config_value(
            ladder_cfg.get("baryonic_perturbation_probability"),
            rng,
            sim_cfg.get("baryonic_perturbation_prob", 0.5),
        ))
        bar_prob = float(_sample_config_value(domain_cfg.get("bar_perturbation_probability"), rng, 0.3)) if domain_enabled else float(_sample_config_value(ladder_cfg.get("bar_perturbation_probability"), rng, 0.3))
        use_potential_variation = (
            bool(ladder_cfg.get("potential_variation", False))
            if ladder_enabled
            else bool(domain_cfg.get("potential_variation", True))
        )

        if domain_enabled and "n_stars_per_sim" in domain_cfg:
            target_n = int(round(_sample_config_value(domain_cfg.get("n_stars_per_sim"), rng, 1000)))
        elif ladder_enabled and "n_stars_per_sim" in ladder_cfg:
            target_n = int(round(_sample_config_value(ladder_cfg.get("n_stars_per_sim"), rng, 1000)))
        elif ladder_enabled and "fixed_n_stars_per_sim" in ladder_cfg:
            target_n = int(ladder_cfg["fixed_n_stars_per_sim"])
        else:
            target_n = int(stream_cfg.get("expected_n_members", 500) * rng.uniform(0.7, 1.3))
        max_stars_target = int(training_cfg.get("preprocessing", {}).get(
            "max_stars_per_sim", sim_cfg.get("max_stars_per_sim", 3000),
        ))
        target_n = min(max(target_n, 50), max_stars_target)

        source_factor = float(sim_cfg.get("source_n_stars_factor", 1.8))
        min_source = int(sim_cfg.get("min_source_n_stars", 800))
        max_source = int(sim_cfg.get("max_source_n_stars", sim_cfg.get("n_stars_per_sim", 5000)))
        source_n_stars = min(max(int(np.ceil(target_n * source_factor)), min_source), max_source)
        n_steps_back = int(sim_cfg.get("n_steps_back", 100))

        # MW potential: in JAX mode use base potential (force table is built once
        # per worker and cached), otherwise perturb within observational uncertainties.
        import os as _os  # noqa: PLC0415
        if _os.environ.get("USE_JAX_INTEGRATOR", "0") == "1" or not use_potential_variation:
            potential = get_mw_potential()
        else:
            potential = vary_potential(seed, config_path=cfg_streams)

        # Generate unperturbed stream. The final graph is downsampled after
        # noise/foreground injection, so avoid integrating far more source
        # particles than the training example can use.
        # Pass pre-loaded mws so galstreams is not reloaded on every call (~13s).
        stream_age_gyr = float(rng.uniform(8.0, min(stream_cfg.get("isochrone_age_gyr", 12.0) + 1, 13.5)))
        # Historical high-fidelity setting was n_stars=5000, n_steps_back=100.
        # Plan 2 training uses domain-randomized target sizes, so source_n_stars
        # and n_steps_back are configurable speed/fidelity knobs.
        # The C extension fix (_fix_galpy_dll_path) must be applied before any
        # galpy import — without it, galpy falls back to Python odeint which is
        # 100x slower and produced the old segfault at n_stars=5000.
        particles = generate_stream(
            stream_name, potential, n_stars=source_n_stars,
            seed=seed, stream_age_gyr=stream_age_gyr,
            config_path=cfg_streams,
            mws=_get_mws(),
            n_steps_back=n_steps_back,
        )

        if len(particles.phi1) < 50:
            log.warning("Stream %s generated too few particles (%d); skipping.", stream_name, len(particles.phi1))
            return None

        # Draw model parameters once per simulation. These parameters define both
        # the M_hm label and the encounter mass function used below.
        dm_params = {}
        alpha = float(dm_model_cfg["mass_function"].get("alpha", -1.9))
        if domain_enabled and dm_model in {"CDM", "SIDM"}:
            alpha = float(rng.uniform(-2.1, -1.7))
        dm_params["alpha"] = alpha
        if dm_model == "WDM":
            dm_params["m_wdm_kev"] = float(rng.uniform(0.5, 10.0))
        elif dm_model == "FDM":
            dm_params["m_axion_ev"] = 10.0 ** float(rng.uniform(-23, -20))
        elif dm_model == "SIDM":
            dm_params["log10_sigma"] = float(rng.uniform(-1.0, 2.0))

        # Sample number of subhalo encounters
        log10_m_min = float(dm_model_cfg["mass_function"].get("log10_M_min_solar", 5.0))
        log10_m_max = float(dm_model_cfg["mass_function"].get("log10_M_max_solar", 9.0))
        if ladder_enabled and "log10_mass_range" in ladder_cfg:
            log10_m_min = float(ladder_cfg["log10_mass_range"][0])
            log10_m_max = float(ladder_cfg["log10_mass_range"][1])
            dm_model_cfg["mass_function"] = dict(dm_model_cfg["mass_function"])
            dm_model_cfg["mass_function"]["log10_M_min_solar"] = log10_m_min
            dm_model_cfg["mass_function"]["log10_M_max_solar"] = log10_m_max
        if ladder_enabled:
            impact_cfg_ladder = dict(dm_model_cfg["impact_physics"])
            if "impact_parameter_pc_range" in ladder_cfg:
                lo, hi = ladder_cfg["impact_parameter_pc_range"]
                impact_cfg_ladder["impact_parameter_pc_prior"] = ["uniform", float(lo), float(hi)]
            if "flyby_velocity_kms_range" in ladder_cfg:
                lo, hi = ladder_cfg["flyby_velocity_kms_range"]
                impact_cfg_ladder["flyby_velocity_kms_prior"] = ["uniform", float(lo), float(hi)]
            if "t_since_impact_gyr_range" in ladder_cfg:
                lo, hi = ladder_cfg["t_since_impact_gyr_range"]
                impact_cfg_ladder["t_since_impact_gyr_prior"] = ["uniform", float(lo), float(hi)]
            dm_model_cfg["impact_physics"] = impact_cfg_ladder

        stream_len_kpc = float(np.abs(stream_cfg["phi1_range_deg"][1] - stream_cfg["phi1_range_deg"][0])) \
                         * np.pi / 180.0 * float(np.mean(stream_cfg["distance_kpc"]))
        rate = compute_encounter_rate(
            stream_age_gyr, stream_len_kpc, stream_cfg.get("width_pc", 100.0),
            log10_m_min, log10_m_max,
            alpha=alpha,
        )
        rate *= mass_function_suppression_factor(
            dm_model,
            dm_params,
            log10_m_min,
            log10_m_max,
            alpha=alpha,
        )
        n_encounters = int(rng.poisson(rate))
        n_encounters = min(n_encounters, 10)  # cap to avoid runaway computation
        n_encounters = _sample_ladder_encounter_count(
            ladder_cfg, rng, seed, n_encounters, job_index=job_index,
        )

        # Apply subhalo perturbations
        if n_encounters > 0:
            particles, encounters = apply_n_subhalo_encounters(
                particles,
                n_encounters,
                dm_model,
                dm_model_cfg,
                stream_cfg,
                dm_params=dm_params,
                seed=seed + 100,
            )
            enc_masses = np.array([e.mass_solar for e in encounters], dtype=np.float32)
            enc_bkpc = np.array([e.impact_param_kpc for e in encounters], dtype=np.float32)
            enc_vkms = np.array([e.flyby_vel_kms for e in encounters], dtype=np.float32)
            enc_t_since = np.array([e.t_since_impact_gyr for e in encounters], dtype=np.float32)
            log_m_mean = float(np.log10(np.mean(enc_masses)))
        else:
            enc_masses = np.array([], dtype=np.float32)
            enc_bkpc = np.array([], dtype=np.float32)
            enc_vkms = np.array([], dtype=np.float32)
            enc_t_since = np.array([], dtype=np.float32)
            # No impacts: data carries no information about impactor mass.
            # Draw from prior so the NSF learns that posterior = prior when n_impacts=0.
            # Using a fixed value (e.g. 0.0) caused a delta function at the prior boundary
            # after clamping, leading to SBC miscalibration (rank pileup at 0).
            log_m_mean = float(rng.uniform(log10_m_min, log10_m_max))

        # Baryonic perturbations: 50% probability for ALL DM models; always for disk-crossers.
        # (Must be model-independent to avoid information leakage — the network should not
        # learn to distinguish CDM by the presence of baryonic signatures.)
        apply_baryonic = stream_cfg.get("disk_crossing", False) or (rng.random() < baryonic_prob)
        if ladder_enabled and bool(ladder_cfg.get("disable_baryonic", False)):
            apply_baryonic = False
        if apply_baryonic:
            particles = apply_giant_molecular_cloud_encounter(particles, seed=seed + 200)
            if rng.random() < bar_prob:
                particles = apply_bar_perturbation(particles, seed=seed + 300)

        # Add observational noise + foreground
        n_pre_cont = len(particles.phi1)
        if noise_scale_factor <= 0.0:
            sigma_pm1 = np.zeros(n_pre_cont, dtype=np.float32)
            sigma_pm2 = np.zeros(n_pre_cont, dtype=np.float32)
            sigma_dist = np.zeros(n_pre_cont, dtype=np.float32)
            sigma_vrad = np.zeros(n_pre_cont, dtype=np.float32)
        else:
            particles, sigma_pm1, sigma_pm2, sigma_dist, sigma_vrad = add_gaia_noise_randomized(
                particles,
                stream_cfg,
                noise_scale_factor=noise_scale_factor,
                seed=seed + 400,
            )
        particles = add_foreground_contamination(
            particles,
            stream_cfg,
            contamination_fraction=contamination_fraction,
            seed=seed + 500,
        )

        # Extend sigma arrays for contamination stars (use generous defaults — foreground
        # has no meaningful Gaia noise model, but the feature must be present).
        n_cont = len(particles.phi1) - n_pre_cont
        membership_prob = build_membership_probabilities(
            n_stream=n_pre_cont,
            n_foreground=n_cont,
            corruption_fraction=membership_corruption_fraction,
            low_prob_range=tuple(domain_cfg.get("membership_low_prob_range", [0.05, 0.49])),
            noise_sigma=membership_noise_sigma,
            seed=seed + 501,
        )
        if n_cont > 0:
            sigma_pm1 = np.concatenate([sigma_pm1, np.full(n_cont, 0.5)])
            sigma_pm2 = np.concatenate([sigma_pm2, np.full(n_cont, 0.5)])
            sigma_dist = np.concatenate([sigma_dist, np.full(n_cont, 5.0)])
            sigma_vrad = np.concatenate([sigma_vrad, np.full(n_cont, np.nan)])

        # Resample to observed count — apply same index to sigmas
        # Keep the pre-generation target_n so source_n_stars is matched to the
        # final graph size instead of integrating a large stream and discarding it.
        if len(particles.phi1) > target_n:
            resample_rng = np.random.default_rng(seed + 600)
            idx = resample_rng.choice(len(particles.phi1), target_n, replace=False)
            particles = StreamParticles(
                phi1=particles.phi1[idx], phi2=particles.phi2[idx],
                dist=particles.dist[idx], pm1=particles.pm1[idx],
                pm2=particles.pm2[idx], vrad=particles.vrad[idx],
                xyz_kpc=particles.xyz_kpc[:, idx], vxyz_kms=particles.vxyz_kms[:, idx],
            )
            sigma_pm1 = sigma_pm1[idx]
            sigma_pm2 = sigma_pm2[idx]
            sigma_dist = sigma_dist[idx]
            sigma_vrad = sigma_vrad[idx]
            membership_prob = membership_prob[idx]

        # Build labels
        dm_idx = DM_MODELS.index(dm_model)
        # Perturbation age: log10 of the most recent impact time.
        # For no-impact sims, use stream age (uninformative prior-like value).
        if n_encounters > 0:
            log_t_since_last = float(np.log10(np.clip(enc_t_since.min(), 0.01, None)))
        else:
            log_t_since_last = float(np.log10(stream_age_gyr))

        labels = {
            "dm_model_idx": dm_idx,
            "dm_model": dm_model,
            "log_m_sub_mean": log_m_mean,
            "log10_M_hm": _log10_half_mode_mass(dm_model, dm_params),
            "n_subhalos": n_encounters,
            "encounter_rate": float(rate),
            "mass_function_alpha": alpha,
            "stream_age_gyr": stream_age_gyr,
            "log_t_since_last_impact_gyr": log_t_since_last,
            "wdm_mass_kev": dm_params.get("m_wdm_kev", float("nan")),
            "fdm_mass_ev": dm_params.get("m_axion_ev", float("nan")),
            "sidm_cross_sec": 10.0 ** dm_params.get("log10_sigma", float("nan")) if "log10_sigma" in dm_params else float("nan"),
            "noise_scale_factor": noise_scale_factor,
            "contamination_fraction": contamination_fraction,
            "membership_corruption_fraction": membership_corruption_fraction,
            "baryonic_applied": int(apply_baryonic),
            "impact_detectable": int(n_encounters > 0),
            "signal_ladder_stage": ladder_cfg.get("stage", ""),
        }

        return {
            "run_id": run_id,
            "stream_name": stream_name,
            "particles": particles,
            "labels": labels,
            "enc_masses": enc_masses,
            "enc_bkpc": enc_bkpc,
            "enc_vkms": enc_vkms,
            "enc_t_since": enc_t_since,
            "sigma_pm1": sigma_pm1,
            "sigma_pm2": sigma_pm2,
            "sigma_dist": sigma_dist,
            "sigma_vrad": sigma_vrad,
            "membership_prob": membership_prob,
        }

    except Exception as exc:
        log.error("Worker failed for run_id=%s stream=%s dm=%s seed=%d: %s",
                  run_id, stream_name, dm_model, seed, exc)
        return None


# ---------------------------------------------------------------------------
# HDF5 write helper
# ---------------------------------------------------------------------------

def write_simulation(f: h5py.File, result: dict) -> None:
    run_id = result["run_id"]
    grp = f.create_group(f"simulations/{run_id}")
    # h5py requires bytes for string attributes (not Python str / numpy unicode)
    grp.attrs["dm_model"] = np.bytes_(result["labels"]["dm_model"])
    grp.attrs["stream_name"] = np.bytes_(result["stream_name"])
    grp.attrs["seed"] = 0  # already encoded in run_id

    p = result["particles"]
    n = len(p.phi1)
    sd = grp.create_group("stream_data")
    for name, arr in [
        ("phi1", p.phi1), ("phi2", p.phi2), ("dist", p.dist),
        ("pm1", p.pm1), ("pm2", p.pm2), ("vrad", p.vrad),
        ("e_pm1", result["sigma_pm1"]), ("e_pm2", result["sigma_pm2"]),
        ("e_dist", result["sigma_dist"]), ("e_vrad", result["sigma_vrad"]),
        ("membership_prob", result["membership_prob"]),
    ]:
        chunk = (min(n, 1024),)
        sd.create_dataset(name, data=arr.astype(np.float32), compression="gzip",
                          compression_opts=4, chunks=chunk)

    sub = grp.create_group("subhalos")
    for name, arr in [
        ("mass", result["enc_masses"]),
        ("impact_param", result["enc_bkpc"]),
        ("flyby_vel", result["enc_vkms"]),
        ("t_since_impact_gyr", result["enc_t_since"]),
    ]:
        sub.create_dataset(name, data=arr.astype(np.float32))

    # Store cylindrical galactocentric coordinates for orbit-phase features
    xyz = p.xyz_kpc  # [3, N]
    R_cyl = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2).astype(np.float32)
    z_cyl = xyz[2].astype(np.float32)
    chunk = (min(n, 1024),)
    sd.create_dataset("R_cyl", data=R_cyl, compression="gzip", compression_opts=4, chunks=chunk)
    sd.create_dataset("z_cyl", data=z_cyl, compression="gzip", compression_opts=4, chunks=chunk)

    lab = grp.create_group("labels")
    for k, v in result["labels"].items():
        if k != "dm_model":
            lab.attrs[k] = np.bytes_(v) if isinstance(v, str) else v
    lab["dm_model_idx"] = result["labels"]["dm_model_idx"]
    lab["log_m_sub_mean"] = result["labels"]["log_m_sub_mean"]
    lab["log10_M_hm"] = result["labels"]["log10_M_hm"]
    lab["n_subhalos"] = result["labels"]["n_subhalos"]
    lab["log_t_since_last_impact_gyr"] = result["labels"]["log_t_since_last_impact_gyr"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_job_list(
    n_sims_per_model: int,
    streams: list[str],
    cfg_streams: str,
    cfg_dm: str,
    cfg_training: str,
    dm_model_filter: str | None = None,
    seed: int = 42,
) -> list:
    """Build the list of (run_id, stream, dm_model, seed, ...) job tuples.

    If dm_model_filter is given (e.g. "CDM"), only jobs for that DM model are returned.
    Seeds are derived deterministically from the DM-model order so that results
    are reproducible and compatible with a merged master dataset regardless of
    which dm_model_filter was used.
    """
    jobs = []
    rng = np.random.default_rng(seed)
    job_index = 0
    for dm_model in DM_MODELS:
        for _ in range(n_sims_per_model):
            stream_name = rng.choice(streams)
            seed = int(rng.integers(0, 2**31))
            run_id = str(uuid.uuid4())[:8]
            if dm_model_filter is None or dm_model == dm_model_filter:
                jobs.append((run_id, stream_name, dm_model, seed, cfg_streams, cfg_dm, cfg_training, job_index))
            job_index += 1
    return jobs


def _pool_worker_init() -> None:
    """Pre-load galstreams once per worker process at pool startup.

    Runs inside the spawned worker before any tasks are dispatched.
    Populates _MWS_CACHE so generate_one_simulation() never pays the 14-second
    galstreams load penalty during a simulation call.

    Also calls _fix_galpy_dll_path() to ensure the C extension loads in the
    fresh spawned process (conda Library/bin is not on PATH by default).

    In JAX mode, pre-builds the force table and warms up the JIT-compiled
    integrator so the first real simulation doesn't pay the 30-second JIT cost.
    """
    _fix_galpy_dll_path()  # must be before any galpy import in this process
    global _MWS_CACHE
    import galstreams  # noqa: PLC0415
    _MWS_CACHE = galstreams.MWStreams(verbose=False)

    # Pre-warm JAX force table + JIT compilation in JAX mode
    import os  # noqa: PLC0415
    if os.environ.get("USE_JAX_INTEGRATOR", "0") == "1":
        try:
            from src.simulation.jax_integrator import (  # noqa: PLC0415
                JAX_AVAILABLE,
                get_force_table,
                integrate_particles_jax,
            )
            if JAX_AVAILABLE:
                # Build and cache force table from base potential
                base_pot = get_mw_potential()
                ft = get_force_table(base_pot)
                log.info("Worker: JAX force table built and cached")
                # Warm up JIT with a tiny dummy integration (triggers compilation)
                dummy_pos = np.array([[8.0, 0.0, 0.0]])
                dummy_vel = np.array([[0.0, 220.0, 0.0]])
                dummy_t = np.array([-0.001])
                integrate_particles_jax(dummy_pos, dummy_vel, dummy_t, base_pot, dt_myr=0.5)
                log.info("Worker: JAX JIT warm-up complete")
        except Exception as exc:
            log.warning("Worker: JAX warm-up failed (%s), falling back to galpy", exc)


def run(args) -> None:
    cfg_streams = args.config_streams
    cfg_dm = args.config_dm
    cfg_training = args.config_training
    with open(cfg_training) as f:
        training_cfg = yaml.safe_load(f)

    if args.signal_ladder_stage:
        training_cfg = _apply_signal_ladder_preset(training_cfg, args.signal_ladder_stage)

    sim_cfg = training_cfg.get("simulation", {})

    output_dir = Path(args.output_dir or sim_cfg.get("output_dir", "data/simulations"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.signal_ladder_stage:
        effective_cfg = output_dir / "_effective_training.yaml"
        with open(effective_cfg, "w") as f:
            yaml.safe_dump(training_cfg, f, sort_keys=False)
        cfg_training = str(effective_cfg)
        sim_cfg = training_cfg.get("simulation", {})
        log.info("Signal ladder stage %s enabled; effective config written to %s",
                 args.signal_ladder_stage, effective_cfg)

    with open(cfg_streams) as f:
        cfg = yaml.safe_load(f)
    streams = list(sim_cfg.get("streams_for_training") or cfg["streams"].keys())

    dm_filter = getattr(args, "dm_model", None)
    n_sims = args.n_sims if args.n_sims is not None else int(sim_cfg.get("n_sims_per_model", 10000))
    jobs = build_job_list(
        n_sims,
        streams,
        cfg_streams,
        cfg_dm,
        cfg_training,
        dm_model_filter=dm_filter,
        seed=int(sim_cfg.get("random_seed", 42)),
    )
    if dm_filter:
        log.info("DM model filter: %s  (%d jobs)", dm_filter, len(jobs))

    chunk_size = args.chunk_size if args.chunk_size is not None else int(sim_cfg.get("chunk_size", 500))
    n_done_total = 0

    # Determine next free chunk_id so new chunks don't overwrite existing ones.
    existing = sorted(output_dir.glob("chunk_*.h5"))
    chunk_id = len(existing)
    log.info("Starting at chunk_id %d (found %d existing chunk files)", chunk_id, len(existing))

    if args.benchmark:
        log.info("Benchmark mode: running %d jobs", args.benchmark)
        jobs = jobs[: args.benchmark]
    elif existing:
        skip_jobs = min(len(jobs), len(existing) * chunk_size)
        if skip_jobs:
            log.info(
                "Resume mode: skipping %d completed jobs from %d existing chunks",
                skip_jobs,
                len(existing),
            )
            jobs = jobs[skip_jobs:]

    log.info("Total remaining jobs: %d", len(jobs))
    if not jobs:
        log.info("No remaining jobs to run.")
        return

    start = time.time()

    try:
        from tqdm import tqdm  # noqa: PLC0415
        pbar = tqdm(desc="Simulations", total=len(jobs))
    except ImportError:
        pbar = None

    chunk_results: list = []

    def flush_chunk(results: list, cid: int) -> int:
        """Write results to chunk file; return count of valid sims."""
        path = output_dir / f"chunk_{cid:05d}.h5"
        valid = [r for r in results if r is not None]
        if valid:
            with h5py.File(str(path), "w") as f:
                for r in valid:
                    write_simulation(f, r)
        log.info("Wrote chunk %d → %s  (%d/%d valid)", cid, path, len(valid), len(results))
        return len(valid)

    # ── Crash-isolated parallel execution ───────────────────────────────────
    # Use multiprocessing.Pool (NOT joblib loky) with chunksize=1.
    #
    # Why not joblib?
    #   joblib's loky backend raises TerminatedWorkerError for the ENTIRE
    #   mini-batch when one worker crashes (C-level access violation in
    #   galpy's dop853_c).  With 1000-sim batches and a ~1/400 crash rate,
    #   virtually every batch crashes and no data is ever written.
    #
    # Why multiprocessing.Pool with chunksize=1?
    #   Python's standard Pool detects a dead worker and stores WorkerLostError
    #   in exactly that task's result slot.  The pool spawns a replacement worker
    #   and all other tasks continue unaffected.  With chunksize=1, each task is
    #   dispatched individually so one crash == one lost sim, not one lost batch.
    #
    # initializer=_pool_worker_init pre-loads galstreams once per worker at
    # startup so the 14-second load is not paid per simulation.
    #
    # maxtasksperchild=500 restarts workers every ~30 min to clear any
    # accumulated C-extension state (0.8% overhead per restart).

    import multiprocessing as mp  # noqa: PLC0415

    n_jobs = args.n_jobs if args.n_jobs is not None else int(sim_cfg.get("n_jobs", 4))
    if n_jobs < 0:
        n_jobs = max(1, mp.cpu_count() + n_jobs + 1)
    n_jobs = max(1, n_jobs)

    log.info("Starting pool with n_jobs=%d, maxtasksperchild=500", n_jobs)

    ctx = mp.get_context("spawn")   # Windows-safe (same as loky)
    pool = ctx.Pool(
        processes=n_jobs,
        initializer=_pool_worker_init,
        maxtasksperchild=500,
    )

    try:
        results_iter = pool.imap_unordered(
            generate_one_simulation, jobs, chunksize=1
        )

        n_completed = 0
        n_jobs_total = len(jobs)

        while n_completed < n_jobs_total:
            try:
                result = next(results_iter)
            except StopIteration:
                # Iterator exhausted (should coincide with n_completed == n_jobs_total)
                break
            except Exception as exc:
                # WorkerLostError (C crash in dop853_c) — pool continues normally.
                log.error("Sim crashed, skipping (pool continues): %s", type(exc).__name__)
                result = None

            n_completed += 1
            chunk_results.append(result)
            if pbar is not None:
                pbar.update(1)
            if len(chunk_results) >= chunk_size:
                n_done_total += flush_chunk(chunk_results, chunk_id)
                chunk_id += 1
                chunk_results = []

    finally:
        pool.close()
        pool.join()

    if chunk_results:
        n_done_total += flush_chunk(chunk_results, chunk_id)

    if pbar is not None:
        pbar.close()

    elapsed = time.time() - start
    log.info("Done: %d simulations in %.1f seconds (%.2f s/sim)",
             n_done_total, elapsed, elapsed / max(n_done_total, 1))

    if args.benchmark:
        rate = n_jobs_total / elapsed  # sims/s including parallelism
        n_full = n_sims * (1 if args.dm_model else len(DM_MODELS))
        estimated_full = n_full / rate / 3600.0
        log.info(
            "Benchmark: %.3f sims/s wall-clock (n_jobs=%d) → estimated full run (%d sims): %.1f hours",
            rate, n_jobs, n_full, estimated_full,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stellar stream simulation dataset.")
    parser.add_argument("--n-sims", type=int, default=None, help="Simulations per DM model (default: config simulation.n_sims_per_model)")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel workers (default: config simulation.n_jobs)")
    parser.add_argument("--chunk-size", type=int, default=None, help="Sims per HDF5 chunk (default: config simulation.chunk_size)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: config simulation.output_dir)")
    parser.add_argument("--config-streams", default="config/streams.yaml")
    parser.add_argument("--config-dm", default="config/dm_models.yaml")
    parser.add_argument("--config-training", default="config/training.yaml")
    parser.add_argument("--benchmark", type=int, default=0,
                        help="Run only N jobs for benchmarking (0 = full run)")
    parser.add_argument("--dm-model", choices=DM_MODELS, default=None,
                        help="Generate only this DM model (default: all 4). "
                             "Use to run 4 separate processes in parallel for stability.")
    parser.add_argument("--signal-ladder-stage",
                        choices=SIGNAL_LADDER_STAGES,
                        default=None,
                        help="Generate a signal-first curriculum dataset instead of the full randomized Plan 2 set.")
    run(parser.parse_args())
