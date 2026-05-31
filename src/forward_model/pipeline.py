"""
Timeline Forward Model pipeline.

Three stages:
    1. Real-data preparation: load observed stream, compute density profile + gaps.
    2. Candidate simulation: for each parameter combination, generate a forced
       encounter via full orbit integration (spray -> evolve to impact epoch ->
       kick -> evolve to present) and compute the resulting stream profile.
    3. Scoring: compare each candidate to the observed data using the scoring module.

The pipeline operates on a single stream at a time and evaluates a grid (or
sampled set) of encounter parameters.

Design decisions:
    - Uses full orbit-integrated evolution (evolve.py) for physical correctness
    - Falls back to impulse-approximation mode for fast grid scans (--fast flag)
    - Scoring is modular: individual scores + configurable weighted combination
    - Output is a ranked list of candidates with per-score breakdown
    - Supports multiprocessing: n_workers > 1 evaluates candidates in parallel
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import yaml

from ..simulation.potentials import get_mw_potential
from ..simulation.stream_gen import StreamParticles, generate_stream
from ..simulation.subhalo import (
    DM_MODELS,
    EncounterParams,
    apply_impulse_approximation,
    scale_radius_from_mass,
    subhalo_profile_for_model,
)
from .evolve import generate_perturbed_stream_evolved, generate_perturbed_stream_multi_evolved
from .scoring import (
    DensityProfile,
    GapFeature,
    ScoreResult,
    ScoreWeights,
    combined_score,
    compute_density_profile,
    detect_gaps,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multiprocessing worker state (module-level for pickling)
# ---------------------------------------------------------------------------

# Per-worker state initialised once per process via pool initializer.
_worker_state: dict = {}


def _worker_init(
    stream_name: str,
    config_path: str,
    use_fast_mode: bool,
    base_stream_dict: Optional[dict],
    obs_particles: dict,
    obs_profile_dict: dict,
    obs_gaps_list: list,
    phi1_range: tuple,
    cfg_dict: dict,
) -> None:
    """Initializer for multiprocessing Pool workers.

    Each worker creates its own galpy potential and galstreams instance
    (these are not pickle-safe). Observed data is shared via serialised dicts.
    """
    import galstreams

    _worker_state["stream_name"] = stream_name
    _worker_state["config_path"] = config_path
    _worker_state["use_fast_mode"] = use_fast_mode
    _worker_state["potential"] = get_mw_potential(config_path)
    _worker_state["mws"] = galstreams.MWStreams(verbose=False)
    _worker_state["phi1_range"] = phi1_range
    # Convert lists back to numpy arrays
    _worker_state["obs_particles"] = {k: np.array(v) for k, v in obs_particles.items()}
    _worker_state["cfg"] = cfg_dict

    # Reconstruct observed profile
    bin_centers = np.array(obs_profile_dict["bin_centers"])
    bw = obs_profile_dict["bin_width_deg"]
    bin_edges = np.concatenate([bin_centers - bw / 2, [bin_centers[-1] + bw / 2]])
    _worker_state["obs_profile"] = DensityProfile(
        bin_centers=bin_centers,
        bin_edges=bin_edges,
        density=np.array(obs_profile_dict["density"]),
        counts=np.array(obs_profile_dict["counts"]),
        bin_width_deg=bw,
    )

    # Reconstruct observed gaps
    _worker_state["obs_gaps"] = [
        GapFeature(**g) for g in obs_gaps_list
    ]

    # Reconstruct base stream (only needed in fast mode)
    if base_stream_dict is not None:
        _worker_state["base_stream"] = StreamParticles(
            phi1=np.array(base_stream_dict["phi1"]),
            phi2=np.array(base_stream_dict["phi2"]),
            dist=np.array(base_stream_dict["dist"]),
            pm1=np.array(base_stream_dict["pm1"]),
            pm2=np.array(base_stream_dict["pm2"]),
            vrad=np.array(base_stream_dict["vrad"]),
            xyz_kpc=np.array(base_stream_dict["xyz_kpc"]),
            vxyz_kms=np.array(base_stream_dict["vxyz_kms"]),
        )
    else:
        _worker_state["base_stream"] = None


def _worker_evaluate(params: dict) -> dict:
    """Worker function that evaluates a single candidate.

    Returns a dict (not CandidateResult) because it crosses process boundaries.
    """
    t0 = time.perf_counter()

    cfg = _worker_state["cfg"]
    log10_m = params["log10_mass"]
    t_since = params["t_since_gyr"]
    phi1_enc = params["impact_phi1"]
    mass = 10.0 ** log10_m
    a_kpc = scale_radius_from_mass(mass)

    encounter = EncounterParams(
        mass_solar=mass,
        scale_radius_kpc=a_kpc,
        impact_param_kpc=cfg["impact_param_kpc"],
        flyby_vel_kms=cfg["flyby_vel_kms"],
        encounter_phi1=phi1_enc,
        t_since_impact_gyr=t_since,
        is_valid=True,
        is_massive=(mass > 1e8),
    )

    if _worker_state["use_fast_mode"]:
        perturbed = apply_impulse_approximation(
            _worker_state["base_stream"], encounter
        )
    else:
        perturbed = generate_perturbed_stream_evolved(
            stream_name=_worker_state["stream_name"],
            potential=_worker_state["potential"],
            encounter=encounter,
            n_stars=cfg["n_stars_sim"],
            seed=cfg["base_seed"],
            config_path=_worker_state["config_path"],
            mws=_worker_state["mws"],
        )

    # Compute simulated profile
    sim_profile = compute_density_profile(
        perturbed.phi1,
        _worker_state["phi1_range"],
        bin_width_deg=cfg["density_bin_width_deg"],
    )
    sim_gaps = detect_gaps(
        sim_profile,
        min_depth=cfg["gap_detection_min_depth"],
        min_significance=cfg["gap_detection_min_significance"],
    )

    obs = _worker_state["obs_particles"]
    score = combined_score(
        sim_profile=sim_profile,
        obs_profile=_worker_state["obs_profile"],
        sim_gaps=sim_gaps,
        obs_gaps=_worker_state["obs_gaps"],
        sim_phi1=perturbed.phi1,
        sim_pm1=perturbed.pm1,
        sim_pm2=perturbed.pm2,
        obs_phi1=obs["phi1"],
        obs_pm1=obs["pm1"],
        obs_pm2=obs["pm2"],
        phi1_range=_worker_state["phi1_range"],
        weights=ScoreWeights(**cfg["score_weights"]),
        density_bin_width=cfg["density_bin_width_deg"],
        kinematic_bin_width=cfg["kinematic_bin_width_deg"],
        sim_vrad=perturbed.vrad,
        obs_vrad=obs.get("vrad"),
    )

    elapsed = time.perf_counter() - t0

    return {
        "log10_mass": log10_m,
        "t_since_gyr": t_since,
        "impact_phi1": phi1_enc,
        "flyby_vel_kms": cfg["flyby_vel_kms"],
        "impact_param_kpc": cfg["impact_param_kpc"],
        "scale_radius_kpc": a_kpc,
        "density_residual": score.density_residual,
        "gap_agreement": score.gap_agreement,
        "kinematic_perturbation": score.kinematic_perturbation,
        "radial_velocity": score.radial_velocity,
        "combined": score.combined,
        "n_stars_sim": len(perturbed.phi1),
        "runtime_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ForwardModelConfig:
    """Configuration for one forward-model run."""
    stream_name: str = "GD1"
    config_path: str = "config/streams.yaml"
    processed_h5_path: str = "data/processed/streams.h5"

    # Grid parameters
    log10_mass_range: tuple[float, float] = (6.0, 8.5)
    log10_mass_step: float = 0.25
    t_since_range: tuple[float, float] = (0.5, 10.0)
    t_since_step: float = 0.5
    impact_phi1_values: list[float] = field(default_factory=lambda: [-40.0])

    # Encounter geometry defaults
    flyby_vel_kms: float = 200.0
    impact_param_kpc: float = 0.1

    # Simulation parameters
    n_stars_sim: int = 5000
    base_seed: int = 42

    # Observation filtering
    phi2_cut_deg: float = 1.0         # half-width in phi2 to select stream stars
    membership_prob_min: float = 0.5  # minimum membership probability

    # Scoring
    density_bin_width_deg: float = 1.0
    kinematic_bin_width_deg: float = 2.0
    gap_detection_min_depth: float = 0.3
    gap_detection_min_significance: float = 2.0
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)

    # GNN profile scorer
    gnn_checkpoint: str = "checkpoints/gnn_best.pt"  # path to trained GNN checkpoint
    use_gnn_scorer: bool = True  # enable GNN embedding distance in combined score

    # Evolution mode
    use_fast_mode: bool = False  # True = impulse approx (quick scan), False = full orbit evolution
    n_workers: int = 1           # parallel workers; uses multiprocessing (process-safe for orbit integration)

    # Output
    output_dir: str = "outputs/forward_model"
    top_k: int = 20  # number of best candidates to save in detail


@dataclass
class CandidateResult:
    """Result for one encounter-parameter candidate."""
    # Parameters
    log10_mass: float
    t_since_gyr: float
    impact_phi1: float
    flyby_vel_kms: float
    impact_param_kpc: float
    scale_radius_kpc: float

    # Scores
    score: ScoreResult = field(default_factory=ScoreResult)

    # Metadata
    n_stars_sim: int = 0
    runtime_s: float = 0.0

    def to_dict(self) -> dict:
        """Serialisable dictionary."""
        return {
            "log10_mass": self.log10_mass,
            "t_since_gyr": self.t_since_gyr,
            "impact_phi1": self.impact_phi1,
            "flyby_vel_kms": self.flyby_vel_kms,
            "impact_param_kpc": self.impact_param_kpc,
            "scale_radius_kpc": self.scale_radius_kpc,
            "combined_score": self.score.combined,
            "density_residual": self.score.density_residual,
            "gap_agreement": self.score.gap_agreement,
            "kinematic_perturbation": self.score.kinematic_perturbation,
            "radial_velocity": self.score.radial_velocity,
            "profile_distance": self.score.profile_distance,
            "n_stars_sim": self.n_stars_sim,
            "runtime_s": self.runtime_s,
        }


@dataclass
class MultiEncounterResult:
    """Result for a multi-encounter (sequential impacts) candidate."""
    # Encounter list (each is a dict with log10_mass, t_since_gyr, impact_phi1, etc.)
    encounters: list[dict] = field(default_factory=list)

    # Scores
    score: ScoreResult = field(default_factory=ScoreResult)

    # Metadata
    n_encounters: int = 0
    n_stars_sim: int = 0
    runtime_s: float = 0.0

    def to_dict(self) -> dict:
        """Serialisable dictionary."""
        return {
            "n_encounters": self.n_encounters,
            "encounters": self.encounters,
            "combined_score": self.score.combined,
            "density_residual": self.score.density_residual,
            "gap_agreement": self.score.gap_agreement,
            "kinematic_perturbation": self.score.kinematic_perturbation,
            "profile_distance": self.score.profile_distance,
            "n_stars_sim": self.n_stars_sim,
            "runtime_s": self.runtime_s,
        }


@dataclass
class ModelComparisonResult:
    """Result of comparing all four DM models for one encounter configuration."""
    # Encounter parameters (shared across all models)
    log10_mass: float = 0.0
    t_since_gyr: float = 0.0
    impact_phi1: float = 0.0

    # Per-model scores (keyed by model name: CDM, WDM, FDM, SIDM)
    model_scores: dict[str, ScoreResult] = field(default_factory=dict)

    # Per-model subhalo profile info
    model_profiles: dict[str, dict] = field(default_factory=dict)

    # Best model
    best_model: str = ""
    best_score: float = 0.0

    # Null reference
    null_combined: float = 0.0

    runtime_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "log10_mass": self.log10_mass,
            "t_since_gyr": self.t_since_gyr,
            "impact_phi1": self.impact_phi1,
            "best_model": self.best_model,
            "best_score": self.best_score,
            "null_combined": self.null_combined,
            "models": {
                name: {
                    "combined_score": sr.combined,
                    "density_residual": sr.density_residual,
                    "gap_agreement": sr.gap_agreement,
                    "kinematic_perturbation": sr.kinematic_perturbation,
                    "profile_distance": sr.profile_distance,
                    "scale_radius_kpc": self.model_profiles.get(name, {}).get("scale_radius_kpc", 0),
                    "core_radius_kpc": self.model_profiles.get(name, {}).get("core_radius_kpc", 0),
                    "kick_suppression": self.model_profiles.get(name, {}).get("kick_suppression", 1),
                    "profile_type": self.model_profiles.get(name, {}).get("profile_type", ""),
                }
                for name, sr in self.model_scores.items()
            },
            "runtime_s": self.runtime_s,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TimelineForwardModel:
    """Orchestrator for the timeline forward-modelling pipeline.

    Usage:
        cfg = ForwardModelConfig(stream_name="GD1")
        model = TimelineForwardModel(cfg)
        model.prepare()             # Stage 1: load real data
        results = model.run_grid()  # Stage 2+3: simulate + score
        model.save_results(results) # Write JSON output
    """

    def __init__(self, config: ForwardModelConfig) -> None:
        self.cfg = config
        self.potential = None
        self.stream_config = None
        self.obs_particles = None    # real-data star arrays
        self.obs_profile = None      # DensityProfile
        self.obs_gaps = None         # list[GapFeature]
        self.base_stream = None      # unperturbed simulated stream
        self.null_score = None       # ScoreResult for unperturbed baseline
        self.gnn_scorer = None       # GNNProfileScorer (optional)
        self._mws = None             # cached galstreams.MWStreams

    # ------------------------------------------------------------------
    # Stage 1: Real-data preparation
    # ------------------------------------------------------------------

    def prepare(self, observed_override: Optional[dict] = None) -> None:
        """Load observed stream data and compute baseline observables.

        Args:
            observed_override: Optional dict of star arrays (phi1, phi2, pm1,
                pm2, dist, vrad, and optionally membership_prob) to use as the
                observed stream instead of loading from HDF5. This is the entry
                point for injection-recovery experiments, where a synthetic
                stream with a known encounter plays the role of the real data.
        """
        log.info("Stage 1: Preparing observed data for %s", self.cfg.stream_name)

        # Load stream configuration
        with open(self.cfg.config_path) as f:
            all_config = yaml.safe_load(f)
        self.stream_config = all_config["streams"][self.cfg.stream_name]
        self.potential = get_mw_potential(self.cfg.config_path)

        # Observed data: injected synthetic stream, or real data from HDF5.
        if observed_override is not None:
            self.obs_particles = {k: np.asarray(v) for k, v in observed_override.items()}
            log.info("  Using injected synthetic observed stream (%d stars)",
                     len(self.obs_particles["phi1"]))
        else:
            self._load_observed_data()

        # Use the actual data extent for profiling (not the config's simulation
        # range which may be broader). Trim 2% off each end to avoid edge effects.
        obs_phi1 = self.obs_particles["phi1"]
        p2, p98 = np.percentile(obs_phi1, [2, 98])
        self.phi1_range = (float(p2), float(p98))
        log.info("  Data phi1 extent: [%.1f, %.1f] (2nd-98th percentile)",
                 self.phi1_range[0], self.phi1_range[1])

        # Compute observed density profile
        self.obs_profile = compute_density_profile(
            obs_phi1,
            self.phi1_range,
            bin_width_deg=self.cfg.density_bin_width_deg,
            weights=self.obs_particles.get("membership_prob"),
        )
        log.info("  Observed profile: %d bins, %d total stars",
                 self.obs_profile.n_bins, int(self.obs_profile.counts.sum()))

        # Detect gaps in observed data
        self.obs_gaps = detect_gaps(
            self.obs_profile,
            min_depth=self.cfg.gap_detection_min_depth,
            min_significance=self.cfg.gap_detection_min_significance,
        )
        log.info("  Detected %d gaps in observed stream", len(self.obs_gaps))
        for g in self.obs_gaps:
            log.info("    phi1=%.1f, depth=%.2f, width=%.1f deg, sig=%.1f",
                     g.phi1_center, g.depth, g.phi1_width, g.significance)

        # Generate unperturbed baseline stream (reusable for all candidates)
        log.info("  Generating unperturbed baseline stream (%d stars)...",
                 self.cfg.n_stars_sim)
        self.base_stream = self._generate_base_stream()
        log.info("  Baseline stream: %d stars in phi1 range",
                 len(self.base_stream.phi1))

        # GNN profile scorer (optional) — must be loaded before null hypothesis
        # so the null score includes the profile distance component.
        if self.cfg.use_gnn_scorer:
            try:
                from .gnn_scorer import GNNProfileScorer
                self.gnn_scorer = GNNProfileScorer(
                    checkpoint_path=self.cfg.gnn_checkpoint,
                    device="cpu",
                )
                if self.gnn_scorer.is_available:
                    self.gnn_scorer.set_observed_embedding(
                        self.obs_particles, self.cfg.stream_name
                    )
                else:
                    log.info("  GNN scorer: checkpoint not found, disabled")
                    self.gnn_scorer = None
            except Exception as e:
                log.warning("  GNN scorer failed to load: %s", e)
                self.gnn_scorer = None

        # Null hypothesis: score the unperturbed stream against observations.
        # Any encounter candidate that scores worse than this adds no explanatory power.
        # This uses evaluate_candidate logic (including GNN if available).
        self.null_score = self._evaluate_null_hypothesis()
        log.info("  Null hypothesis (unperturbed) score: combined=%.4f",
                 self.null_score.combined)
        log.info("    density_residual=%.4f, gap_agreement=%.4f, kinematic=%.4f, profile_dist=%.4f",
                 self.null_score.density_residual, self.null_score.gap_agreement,
                 self.null_score.kinematic_perturbation, self.null_score.profile_distance)

    def _load_observed_data(self) -> None:
        """Load real stream members from the processed HDF5 with quality filtering.

        Applies:
            - phi2 cut: keep only stars within phi2_cut_deg of the stream track
            - membership probability cut: keep stars above membership_prob_min
            - PM range cut: use stream config pm1/pm2 ranges to reject outliers
        """
        h5_path = Path(self.cfg.processed_h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(
                f"Processed stream file not found: {h5_path}\n"
                "Run scripts/process_gaia_streams.py first."
            )

        name = self.cfg.stream_name
        with h5py.File(str(h5_path), "r") as f:
            grp = f[f"streams/{name}/members"]
            raw = {
                "phi1": grp["phi1"][:].astype(np.float64),
                "phi2": grp["phi2"][:].astype(np.float64),
                "pm1": grp["pm1"][:].astype(np.float64) if "pm1" in grp else np.zeros_like(grp["phi1"][:]),
                "pm2": grp["pm2"][:].astype(np.float64) if "pm2" in grp else np.zeros_like(grp["phi1"][:]),
                "dist": grp["dist"][:].astype(np.float64) if "dist" in grp else np.full(len(grp["phi1"][:]), 8.0),
                "vrad": grp["vrad"][:].astype(np.float64) if "vrad" in grp else np.zeros_like(grp["phi1"][:]),
            }
            if "membership_prob" in grp:
                raw["membership_prob"] = grp["membership_prob"][:].astype(np.float64)
            # Real per-star measurement errors (used by the detector to impute /
            # weight features accurately instead of fabricated constants).
            for ecol in ("e_dist", "e_pm1", "e_pm2", "e_vrad"):
                if ecol in grp:
                    raw[ecol] = grp[ecol][:].astype(np.float64)

        n_raw = len(raw["phi1"])

        # Build quality mask
        mask = np.ones(n_raw, dtype=bool)

        # phi2 cut: keep stars close to the stream track
        mask &= np.abs(raw["phi2"]) < self.cfg.phi2_cut_deg

        # Membership probability cut
        if "membership_prob" in raw:
            mask &= raw["membership_prob"] >= self.cfg.membership_prob_min

        # PM range cut from stream config (reject background stars with wrong PMs)
        pm1_range = self.stream_config.get("pm1_range_masyr")
        pm2_range = self.stream_config.get("pm2_range_masyr")
        if pm1_range is not None:
            mask &= (raw["pm1"] >= pm1_range[0]) & (raw["pm1"] <= pm1_range[1])
        if pm2_range is not None:
            mask &= (raw["pm2"] >= pm2_range[0]) & (raw["pm2"] <= pm2_range[1])

        # Apply mask
        self.obs_particles = {k: v[mask] for k, v in raw.items()}

        n_filtered = int(mask.sum())
        log.info("  Loaded %d observed members from %s (%d raw -> %d after filtering)",
                 n_filtered, h5_path, n_raw, n_filtered)
        log.info("    Filters: |phi2| < %.1f deg, membership >= %.2f, PM cuts",
                 self.cfg.phi2_cut_deg, self.cfg.membership_prob_min)

    def _generate_base_stream(self, seed: Optional[int] = None) -> StreamParticles:
        """Generate one unperturbed stream simulation as the encounter substrate."""
        if self._mws is None:
            import galstreams
            self._mws = galstreams.MWStreams(verbose=False)

        return generate_stream(
            stream_name=self.cfg.stream_name,
            potential=self.potential,
            n_stars=self.cfg.n_stars_sim,
            seed=self.cfg.base_seed if seed is None else seed,
            config_path=self.cfg.config_path,
            mws=self._mws,
        )

    def _score_stream_vs_obs(self, stream: StreamParticles) -> ScoreResult:
        """Score any simulated stream against the observed data (density+gap+kin+RV)."""
        sim_profile = compute_density_profile(
            stream.phi1, self.phi1_range, bin_width_deg=self.cfg.density_bin_width_deg,
        )
        sim_gaps = detect_gaps(
            sim_profile, min_depth=self.cfg.gap_detection_min_depth,
            min_significance=self.cfg.gap_detection_min_significance,
        )
        return combined_score(
            sim_profile=sim_profile,
            obs_profile=self.obs_profile,
            sim_gaps=sim_gaps,
            obs_gaps=self.obs_gaps,
            sim_phi1=stream.phi1, sim_pm1=stream.pm1, sim_pm2=stream.pm2,
            obs_phi1=self.obs_particles["phi1"],
            obs_pm1=self.obs_particles["pm1"],
            obs_pm2=self.obs_particles["pm2"],
            phi1_range=self.phi1_range,
            weights=self.cfg.score_weights,
            density_bin_width=self.cfg.density_bin_width_deg,
            kinematic_bin_width=self.cfg.kinematic_bin_width_deg,
            sim_vrad=stream.vrad,
            obs_vrad=self.obs_particles.get("vrad"),
        )

    def build_null_distribution(
        self, n_realizations: int = 20, seed0: int = 1000,
    ) -> list[float]:
        """Score many *unperturbed* stream realizations (varied seeds) vs observations.

        This is the no-impact baseline distribution: its scatter sets the scale
        against which a candidate encounter's improvement is judged. A candidate
        is only meaningful if it scores better than this distribution by a
        margin large compared to the null scatter (see compute_significance).
        """
        scores = []
        for i in range(n_realizations):
            stream = self._generate_base_stream(seed=seed0 + i)
            scores.append(float(self._score_stream_vs_obs(stream).combined))
        log.info("Null distribution (%d realizations): mean=%.4f std=%.4f",
                 n_realizations, float(np.mean(scores)), float(np.std(scores, ddof=1)))
        return scores

    def build_lookelsewhere_null(
        self, n_realizations: int = 10, seed0: int = 3000,
    ) -> list[float]:
        """Null distribution of the *best-of-grid* score (look-elsewhere correction).

        For each no-impact realization, an unperturbed stream plays the role of
        the observed data and the FULL candidate grid is scored against it; the
        best score is recorded. Because we always report the best of many
        candidates, the real best score must be compared to this distribution of
        best scores (not to a single-candidate null) for a correctly calibrated
        p-value. Expensive (n x grid); intended for fast mode.
        """
        grid = self.build_parameter_grid()
        saved = (self.obs_particles, self.obs_profile, self.obs_gaps)
        best_scores = []
        try:
            for i in range(n_realizations):
                stream = self._generate_base_stream(seed=seed0 + 100 * i)
                self.obs_particles = {
                    "phi1": stream.phi1, "phi2": stream.phi2,
                    "pm1": stream.pm1, "pm2": stream.pm2,
                    "dist": stream.dist, "vrad": stream.vrad,
                    "membership_prob": np.ones(len(stream.phi1)),
                }
                self.obs_profile = compute_density_profile(
                    stream.phi1, self.phi1_range, bin_width_deg=self.cfg.density_bin_width_deg)
                self.obs_gaps = detect_gaps(
                    self.obs_profile, min_depth=self.cfg.gap_detection_min_depth,
                    min_significance=self.cfg.gap_detection_min_significance)
                results = self._run_grid_sequential(grid)
                best_scores.append(float(min(r.score.combined for r in results)))
        finally:
            self.obs_particles, self.obs_profile, self.obs_gaps = saved
        log.info("Look-elsewhere null (%d realizations): best-score mean=%.4f std=%.4f",
                 n_realizations, float(np.mean(best_scores)), float(np.std(best_scores, ddof=1)))
        return best_scores

    def monte_carlo_impact_time(
        self, n_realizations: int = 30, error_scale: float = 1.0, seed0: int = 5000,
    ):
        """Monte-Carlo posterior over the best-fit impact time from measurement errors.

        Each realization perturbs the observed kinematics (pm1, pm2, vrad, dist)
        within their per-star errors (scaled by ``error_scale``), re-runs the
        candidate grid, and records the best-fit (t_since, mass, phi1). The spread
        of best-fit t_since is the impact-time posterior driven by the present-day
        measurement precision: tighter errors (e.g. from multi-epoch fusion, or a
        smaller ``error_scale``) sharpen it. ``error_scale=0`` reduces to the
        deterministic best fit (zero spread).

        Returns an ``ImpactTimePosterior``.
        """
        from .significance import summarize_impact_time

        grid = self.build_parameter_grid()
        obs0 = {k: np.array(v) for k, v in self.obs_particles.items()}
        saved = (self.obs_particles, self.obs_profile, self.obs_gaps)
        err_map = {"pm1": "e_pm1", "pm2": "e_pm2", "vrad": "e_vrad", "dist": "e_dist"}
        rng = np.random.default_rng(seed0)
        t_s, m_s, p_s = [], [], []
        try:
            for _ in range(n_realizations):
                pert = {k: np.array(v) for k, v in obs0.items()}
                for field, ecol in err_map.items():
                    if field in pert and ecol in obs0:
                        e = np.asarray(obs0[ecol], dtype=float)
                        sig = np.where(np.isfinite(e), e, 0.0) * error_scale
                        noise = rng.normal(0.0, np.maximum(sig, 0.0))
                        pert[field] = np.where(np.isfinite(pert[field]),
                                               pert[field] + noise, pert[field])
                self.obs_particles = pert
                # phi1 is unperturbed (astrometric position errors are negligible
                # vs the bin width), so the density profile/gaps are unchanged.
                results = self._run_grid_sequential(grid)
                best = min(results, key=lambda r: r.score.combined)
                t_s.append(best.t_since_gyr); m_s.append(best.log10_mass); p_s.append(best.impact_phi1)
        finally:
            self.obs_particles, self.obs_profile, self.obs_gaps = saved
        post = summarize_impact_time(t_s, m_s, p_s, error_scale=error_scale)
        log.info("MC impact time (n=%d, error_scale=%.2f): t_since=%.2f [%.2f, %.2f] Gyr (std %.2f)",
                 n_realizations, error_scale, post.t_since_median,
                 post.t_since_p16, post.t_since_p84, post.t_since_std)
        return post

    def evaluate_candidate_multiseed(
        self, params: dict, seeds: list[int],
    ) -> tuple[float, float, list[float]]:
        """Evaluate one candidate over several base-stream seeds.

        Returns (mean_combined, std_combined, per_seed_scores). Averaging over
        seeds removes the sampling-noise component so candidate ranking reflects
        the physical encounter rather than a particular spray realization.
        """
        scores = []
        for s in seeds:
            scores.append(float(self.evaluate_candidate(params, seed=s).score.combined))
        return float(np.mean(scores)), float(np.std(scores, ddof=1) if len(scores) > 1 else 0.0), scores

    def _evaluate_null_hypothesis(self) -> ScoreResult:
        """Score the unperturbed baseline stream against observations.

        This establishes the null hypothesis: a smooth stream with no subhalo
        encounter. Candidates that score worse than this provide no evidence
        for a subhalo impact.

        Returns:
            ScoreResult for the unperturbed stream.
        """
        # Compute density profile of the unperturbed stream
        null_profile = compute_density_profile(
            self.base_stream.phi1,
            self.phi1_range,
            bin_width_deg=self.cfg.density_bin_width_deg,
        )

        # Detect gaps in unperturbed stream (should be few/none)
        null_gaps = detect_gaps(
            null_profile,
            min_depth=self.cfg.gap_detection_min_depth,
            min_significance=self.cfg.gap_detection_min_significance,
        )

        # Score against observations
        score = combined_score(
            sim_profile=null_profile,
            obs_profile=self.obs_profile,
            sim_gaps=null_gaps,
            obs_gaps=self.obs_gaps,
            sim_phi1=self.base_stream.phi1,
            sim_pm1=self.base_stream.pm1,
            sim_pm2=self.base_stream.pm2,
            obs_phi1=self.obs_particles["phi1"],
            obs_pm1=self.obs_particles["pm1"],
            obs_pm2=self.obs_particles["pm2"],
            phi1_range=self.phi1_range,
            weights=self.cfg.score_weights,
            density_bin_width=self.cfg.density_bin_width_deg,
            kinematic_bin_width=self.cfg.kinematic_bin_width_deg,
            sim_vrad=self.base_stream.vrad,
            obs_vrad=self.obs_particles.get("vrad"),
        )

        # Add GNN profile distance if available
        if self.gnn_scorer is not None:
            profile_dist = self.gnn_scorer.score(self.base_stream, self.cfg.stream_name)
            score.profile_distance = profile_dist
            w = self.cfg.score_weights
            w_total = w.density + w.gap + w.kinematic + w.profile
            score.combined = (
                w.density * score.density_residual
                + w.gap * score.gap_agreement
                + w.kinematic * score.kinematic_perturbation
                + w.profile * profile_dist
            ) / max(w_total, 1e-6)

        return score

    # ------------------------------------------------------------------
    # Stage 2+3: Grid evaluation (simulate + score)
    # ------------------------------------------------------------------

    def build_parameter_grid(self) -> list[dict]:
        """Build the Cartesian product grid of encounter parameters."""
        masses = np.arange(
            self.cfg.log10_mass_range[0],
            self.cfg.log10_mass_range[1] + 0.01,
            self.cfg.log10_mass_step,
        )
        times = np.arange(
            self.cfg.t_since_range[0],
            self.cfg.t_since_range[1] + 0.01,
            self.cfg.t_since_step,
        )
        phi1s = self.cfg.impact_phi1_values

        grid = []
        for m in masses:
            for t in times:
                for p in phi1s:
                    grid.append({
                        "log10_mass": float(m),
                        "t_since_gyr": float(t),
                        "impact_phi1": float(p),
                    })

        log.info("  Parameter grid: %d candidates (%d masses x %d times x %d phi1s)",
                 len(grid), len(masses), len(times), len(phi1s))
        return grid

    def evaluate_candidate(self, params: dict, seed: Optional[int] = None) -> CandidateResult:
        """Evaluate a single encounter candidate: simulate + score.

        In full mode (default): generates a fresh stream, applies the encounter
        at the correct past epoch, and integrates all particles forward through
        the MW potential to the present day. This is physically correct but
        expensive (~10-30s per candidate).

        In fast mode (use_fast_mode=True): applies the impulse approximation
        to a pre-generated base stream. Fast (~1ms) but uses an analytic
        drift formula instead of real orbit integration.

        Args:
            params: dict with keys log10_mass, t_since_gyr, impact_phi1.

        Returns:
            CandidateResult with all scores populated.
        """
        t0 = time.perf_counter()

        log10_m = params["log10_mass"]
        t_since = params["t_since_gyr"]
        phi1_enc = params["impact_phi1"]
        mass = 10.0 ** log10_m
        a_kpc = scale_radius_from_mass(mass)

        # Build forced encounter
        encounter = EncounterParams(
            mass_solar=mass,
            scale_radius_kpc=a_kpc,
            impact_param_kpc=self.cfg.impact_param_kpc,
            flyby_vel_kms=self.cfg.flyby_vel_kms,
            encounter_phi1=phi1_enc,
            t_since_impact_gyr=t_since,
            is_valid=True,
            is_massive=(mass > 1e8),
        )

        if self.cfg.use_fast_mode:
            # Fast mode: impulse approximation on the base stream. With a seed
            # override (multi-seed significance), use a fresh base realization.
            base = self.base_stream if seed is None else self._generate_base_stream(seed=seed)
            perturbed = apply_impulse_approximation(base, encounter)
        else:
            # Full mode: orbit-integrated evolution
            # Each candidate generates a fresh perturbed stream from scratch.
            # This is expensive but physically correct.
            if self._mws is None:
                import galstreams
                self._mws = galstreams.MWStreams(verbose=False)

            perturbed = generate_perturbed_stream_evolved(
                stream_name=self.cfg.stream_name,
                potential=self.potential,
                encounter=encounter,
                n_stars=self.cfg.n_stars_sim,
                seed=self.cfg.base_seed if seed is None else seed,
                config_path=self.cfg.config_path,
                mws=self._mws,
            )

        # Compute simulated density profile (same phi1_range as observed)
        sim_profile = compute_density_profile(
            perturbed.phi1,
            self.phi1_range,
            bin_width_deg=self.cfg.density_bin_width_deg,
        )

        # Detect gaps in simulated stream
        sim_gaps = detect_gaps(
            sim_profile,
            min_depth=self.cfg.gap_detection_min_depth,
            min_significance=self.cfg.gap_detection_min_significance,
        )

        # Score
        score = combined_score(
            sim_profile=sim_profile,
            obs_profile=self.obs_profile,
            sim_gaps=sim_gaps,
            obs_gaps=self.obs_gaps,
            sim_phi1=perturbed.phi1,
            sim_pm1=perturbed.pm1,
            sim_pm2=perturbed.pm2,
            obs_phi1=self.obs_particles["phi1"],
            obs_pm1=self.obs_particles["pm1"],
            obs_pm2=self.obs_particles["pm2"],
            phi1_range=self.phi1_range,
            weights=self.cfg.score_weights,
            density_bin_width=self.cfg.density_bin_width_deg,
            kinematic_bin_width=self.cfg.kinematic_bin_width_deg,
            sim_vrad=perturbed.vrad,
            obs_vrad=self.obs_particles.get("vrad"),
        )

        # GNN profile distance (if scorer available)
        if self.gnn_scorer is not None:
            profile_dist = self.gnn_scorer.score(perturbed, self.cfg.stream_name)
            score.profile_distance = profile_dist
            # Re-compute combined score including profile distance
            w = self.cfg.score_weights
            w_total = w.density + w.gap + w.kinematic + w.profile
            score.combined = (
                w.density * score.density_residual
                + w.gap * score.gap_agreement
                + w.kinematic * score.kinematic_perturbation
                + w.profile * profile_dist
            ) / max(w_total, 1e-6)

        elapsed = time.perf_counter() - t0

        return CandidateResult(
            log10_mass=log10_m,
            t_since_gyr=t_since,
            impact_phi1=phi1_enc,
            flyby_vel_kms=self.cfg.flyby_vel_kms,
            impact_param_kpc=self.cfg.impact_param_kpc,
            scale_radius_kpc=a_kpc,
            score=score,
            n_stars_sim=len(perturbed.phi1),
            runtime_s=elapsed,
        )

    def run_grid(self) -> list[CandidateResult]:
        """Evaluate all candidates in the parameter grid.

        Uses multiprocessing when cfg.n_workers > 1. Each worker process gets
        its own galpy potential and galstreams instance (these are not picklable)
        via a pool initializer.

        Returns:
            List of CandidateResult, sorted by combined_score (ascending = best first).
        """
        grid = self.build_parameter_grid()

        log.info("Stage 2+3: Evaluating %d candidates (n_workers=%d)...",
                 len(grid), self.cfg.n_workers)
        t0 = time.perf_counter()

        if self.cfg.n_workers > 1:
            results = self._run_grid_parallel(grid)
        else:
            results = self._run_grid_sequential(grid)

        # Sort by combined score (lower is better)
        results.sort(key=lambda r: r.score.combined)

        total_time = time.perf_counter() - t0
        log.info("  Grid complete: %d candidates in %.1f s (%.1f cand/s)",
                 len(results), total_time, len(results) / total_time)
        log.info("  Best candidate: log10_M=%.2f, t=%.1f Gyr, phi1=%.1f, score=%.4f",
                 results[0].log10_mass, results[0].t_since_gyr,
                 results[0].impact_phi1, results[0].score.combined)

        # Report improvement over null hypothesis
        if self.null_score is not None:
            null_combined = self.null_score.combined
            best_combined = results[0].score.combined
            n_better_than_null = sum(1 for r in results if r.score.combined < null_combined)
            log.info("  Null hypothesis score: %.4f", null_combined)
            log.info("  Improvement over null: %.4f (%.1f%%)",
                     null_combined - best_combined,
                     100.0 * (null_combined - best_combined) / max(null_combined, 1e-8))
            log.info("  Candidates better than null: %d / %d",
                     n_better_than_null, len(results))

        return results

    def _run_grid_sequential(self, grid: list[dict]) -> list[CandidateResult]:
        """Sequential grid evaluation (single process)."""
        results = []
        t0 = time.perf_counter()

        for i, params in enumerate(grid):
            result = self.evaluate_candidate(params)
            results.append(result)

            if (i + 1) % 50 == 0 or (i + 1) == len(grid):
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                log.info("  [%d/%d] %.1f candidates/sec, best combined=%.4f",
                         i + 1, len(grid), rate, min(r.score.combined for r in results))

        return results

    def _run_grid_parallel(self, grid: list[dict]) -> list[CandidateResult]:
        """Parallel grid evaluation using multiprocessing.Pool.

        Each worker process gets its own galpy potential and galstreams via
        a pool initializer. Observed data and config are serialised as plain
        dicts/arrays (pickle-safe).
        """
        # Serialise observed profile for workers
        obs_profile_dict = {
            "bin_centers": self.obs_profile.bin_centers.tolist(),
            "density": self.obs_profile.density.tolist(),
            "counts": self.obs_profile.counts.tolist(),
            "bin_width_deg": self.obs_profile.bin_width_deg,
            "n_bins": self.obs_profile.n_bins,
        }

        # Serialise observed gaps
        obs_gaps_list = [
            {"phi1_center": g.phi1_center, "phi1_width": g.phi1_width,
             "depth": g.depth, "significance": g.significance}
            for g in self.obs_gaps
        ]

        # Serialise base stream (for fast mode)
        if self.base_stream is not None and self.cfg.use_fast_mode:
            base_stream_dict = {
                "phi1": self.base_stream.phi1.tolist(),
                "phi2": self.base_stream.phi2.tolist(),
                "dist": self.base_stream.dist.tolist(),
                "pm1": self.base_stream.pm1.tolist(),
                "pm2": self.base_stream.pm2.tolist(),
                "vrad": self.base_stream.vrad.tolist(),
                "xyz_kpc": self.base_stream.xyz_kpc.tolist(),
                "vxyz_kms": self.base_stream.vxyz_kms.tolist(),
            }
        else:
            base_stream_dict = None

        # Serialise config subset needed by workers
        cfg_dict = {
            "flyby_vel_kms": self.cfg.flyby_vel_kms,
            "impact_param_kpc": self.cfg.impact_param_kpc,
            "n_stars_sim": self.cfg.n_stars_sim,
            "base_seed": self.cfg.base_seed,
            "density_bin_width_deg": self.cfg.density_bin_width_deg,
            "kinematic_bin_width_deg": self.cfg.kinematic_bin_width_deg,
            "gap_detection_min_depth": self.cfg.gap_detection_min_depth,
            "gap_detection_min_significance": self.cfg.gap_detection_min_significance,
            "score_weights": asdict(self.cfg.score_weights),
        }

        # Serialise observed particles (numpy arrays -> lists for pickling)
        obs_particles_ser = {k: v.tolist() for k, v in self.obs_particles.items()}

        n_workers = min(self.cfg.n_workers, len(grid))
        log.info("  Starting multiprocessing pool with %d workers...", n_workers)

        init_args = (
            self.cfg.stream_name,
            self.cfg.config_path,
            self.cfg.use_fast_mode,
            base_stream_dict,
            obs_particles_ser,
            obs_profile_dict,
            obs_gaps_list,
            self.phi1_range,
            cfg_dict,
        )

        results = []
        t0 = time.perf_counter()
        completed = 0

        with mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=init_args,
        ) as pool:
            # Use imap_unordered for best throughput + progress reporting
            for result_dict in pool.imap_unordered(_worker_evaluate, grid, chunksize=1):
                candidate = CandidateResult(
                    log10_mass=result_dict["log10_mass"],
                    t_since_gyr=result_dict["t_since_gyr"],
                    impact_phi1=result_dict["impact_phi1"],
                    flyby_vel_kms=result_dict["flyby_vel_kms"],
                    impact_param_kpc=result_dict["impact_param_kpc"],
                    scale_radius_kpc=result_dict["scale_radius_kpc"],
                    score=ScoreResult(
                        density_residual=result_dict["density_residual"],
                        gap_agreement=result_dict["gap_agreement"],
                        kinematic_perturbation=result_dict["kinematic_perturbation"],
                        radial_velocity=result_dict.get("radial_velocity", 0.0),
                        combined=result_dict["combined"],
                    ),
                    n_stars_sim=result_dict["n_stars_sim"],
                    runtime_s=result_dict["runtime_s"],
                )
                results.append(candidate)
                completed += 1

                if completed % 10 == 0 or completed == len(grid):
                    elapsed = time.perf_counter() - t0
                    rate = completed / elapsed
                    best_so_far = min(r.score.combined for r in results)
                    log.info("  [%d/%d] %.1f candidates/sec, best combined=%.4f",
                             completed, len(grid), rate, best_so_far)

        return results

    # ------------------------------------------------------------------
    # Grid refinement (zoom-in)
    # ------------------------------------------------------------------

    def refine_top_candidates(
        self,
        coarse_results: list[CandidateResult],
        n_top: int = 5,
        mass_half_range: float = 0.25,
        mass_step: float = 0.05,
        time_half_range: float = 1.0,
        time_step: float = 0.25,
        phi1_half_range: float = 3.0,
        phi1_step: float = 1.0,
    ) -> list[CandidateResult]:
        """Refine the top candidates with a finer parameter grid.

        For each of the top-N candidates from the coarse grid, builds a fine
        sub-grid centered on that candidate's parameters and evaluates all
        points. Returns the refined results merged and sorted.

        This implements a zoom-in strategy: coarse sweep -> identify peaks ->
        fine grid around each peak -> precise parameter constraints.

        Args:
            coarse_results: Results from run_grid(), sorted best-first.
            n_top: Number of top candidates to refine around.
            mass_half_range: Half-range in log10(M) for fine grid.
            mass_step: Step size in log10(M) for fine grid.
            time_half_range: Half-range in t_since [Gyr] for fine grid.
            time_step: Step size in t_since [Gyr] for fine grid.
            phi1_half_range: Half-range in phi1 [deg] for fine grid.
            phi1_step: Step size in phi1 [deg] for fine grid.

        Returns:
            List of CandidateResult from all refinement sub-grids, sorted by
            combined score (best first). Includes the original coarse results.
        """
        top = coarse_results[:n_top]

        # De-duplicate: if multiple top candidates are within one coarse step
        # of each other, only refine around the unique peaks.
        unique_peaks = []
        for cand in top:
            is_dup = False
            for peak in unique_peaks:
                if (abs(cand.log10_mass - peak.log10_mass) < self.cfg.log10_mass_step
                        and abs(cand.t_since_gyr - peak.t_since_gyr) < self.cfg.t_since_step
                        and abs(cand.impact_phi1 - peak.impact_phi1) < 2.0):
                    is_dup = True
                    break
            if not is_dup:
                unique_peaks.append(cand)

        log.info("Refinement: %d unique peaks from top-%d candidates", len(unique_peaks), n_top)

        # Build fine sub-grids
        fine_grid = []
        for peak in unique_peaks:
            m_min = max(self.cfg.log10_mass_range[0], peak.log10_mass - mass_half_range)
            m_max = min(self.cfg.log10_mass_range[1], peak.log10_mass + mass_half_range)
            t_min = max(self.cfg.t_since_range[0], peak.t_since_gyr - time_half_range)
            t_max = min(self.cfg.t_since_range[1], peak.t_since_gyr + time_half_range)
            p_min = peak.impact_phi1 - phi1_half_range
            p_max = peak.impact_phi1 + phi1_half_range

            masses_fine = np.arange(m_min, m_max + 0.001, mass_step)
            times_fine = np.arange(t_min, t_max + 0.001, time_step)
            phi1s_fine = np.arange(p_min, p_max + 0.001, phi1_step)

            # Clip phi1 to data extent
            phi1s_fine = phi1s_fine[
                (phi1s_fine >= self.phi1_range[0]) & (phi1s_fine <= self.phi1_range[1])
            ]

            for m in masses_fine:
                for t in times_fine:
                    for p in phi1s_fine:
                        fine_grid.append({
                            "log10_mass": float(m),
                            "t_since_gyr": float(t),
                            "impact_phi1": float(p),
                        })

            log.info("  Peak: log10_M=%.2f, t=%.1f, phi1=%.1f -> %d fine-grid points "
                     "(M:[%.2f,%.2f], t:[%.1f,%.1f], phi1:[%.1f,%.1f])",
                     peak.log10_mass, peak.t_since_gyr, peak.impact_phi1,
                     len(masses_fine) * len(times_fine) * len(phi1s_fine),
                     m_min, m_max, t_min, t_max, p_min, p_max)

        # Remove duplicates with coarse grid (don't re-evaluate)
        coarse_set = {
            (round(r.log10_mass, 4), round(r.t_since_gyr, 4), round(r.impact_phi1, 4))
            for r in coarse_results
        }
        fine_grid = [
            p for p in fine_grid
            if (round(p["log10_mass"], 4), round(p["t_since_gyr"], 4),
                round(p["impact_phi1"], 4)) not in coarse_set
        ]

        log.info("  Total refinement candidates: %d (after removing coarse duplicates)",
                 len(fine_grid))

        if not fine_grid:
            log.info("  No new points to evaluate; returning coarse results")
            return coarse_results

        # Evaluate fine grid
        t0 = time.perf_counter()
        log.info("  Evaluating %d refinement candidates (n_workers=%d)...",
                 len(fine_grid), self.cfg.n_workers)

        if self.cfg.n_workers > 1:
            fine_results = self._run_grid_parallel(fine_grid)
        else:
            fine_results = self._run_grid_sequential(fine_grid)

        elapsed = time.perf_counter() - t0
        log.info("  Refinement complete: %d candidates in %.1f s (%.1f cand/s)",
                 len(fine_results), elapsed, len(fine_results) / max(elapsed, 0.001))

        # Merge with coarse results and re-sort
        all_results = coarse_results + fine_results
        all_results.sort(key=lambda r: r.score.combined)

        # Report improvement
        if coarse_results:
            old_best = coarse_results[0].score.combined
            new_best = all_results[0].score.combined
            if new_best < old_best:
                log.info("  Refinement improved best score: %.4f -> %.4f (%.2f%% improvement)",
                         old_best, new_best, 100.0 * (old_best - new_best) / max(old_best, 1e-8))
                log.info("  New best: log10_M=%.3f, t=%.2f Gyr, phi1=%.2f",
                         all_results[0].log10_mass, all_results[0].t_since_gyr,
                         all_results[0].impact_phi1)
            else:
                log.info("  Refinement did not improve best score (coarse optimum is sharp)")

        return all_results

    # ------------------------------------------------------------------
    # DM model comparison (CDM vs WDM vs FDM vs SIDM)
    # ------------------------------------------------------------------

    def evaluate_candidate_all_models(
        self,
        params: dict,
        m_wdm_kev: float = 3.0,
        m_axion_ev: float = 1e-22,
        sigma_sidm_cm2g: float = 1.0,
    ) -> ModelComparisonResult:
        """Evaluate one encounter under all four DM models and compare.

        For the same mass, time, and phi1, each DM model predicts different
        subhalo internal structure (density profile, concentration, core size).
        This changes the velocity kick morphology — broader/shallower for cored
        profiles (SIDM, FDM), sharper/deeper for cuspy profiles (CDM, WDM).

        The method simulates the encounter four times (one per model), scores
        each against observations, and identifies which model's kick pattern
        best reproduces the observed stream morphology.

        Args:
            params: dict with log10_mass, t_since_gyr, impact_phi1.
            m_wdm_kev: WDM thermal relic mass [keV] (default 3.0).
            m_axion_ev: FDM axion mass [eV] (default 1e-22).
            sigma_sidm_cm2g: SIDM cross-section [cm^2/g] (default 1.0).

        Returns:
            ModelComparisonResult with per-model scores and best model identified.
        """
        t0 = time.perf_counter()

        log10_m = params["log10_mass"]
        t_since = params["t_since_gyr"]
        phi1_enc = params["impact_phi1"]
        mass = 10.0 ** log10_m

        model_scores = {}
        model_profiles = {}

        for dm_model in DM_MODELS:
            # Get model-specific subhalo structure
            profile = subhalo_profile_for_model(
                dm_model, mass,
                m_wdm_kev=m_wdm_kev,
                m_axion_ev=m_axion_ev,
                sigma_sidm_cm2g=sigma_sidm_cm2g,
            )
            model_profiles[dm_model] = profile

            # Build encounter with model-specific scale radius
            encounter = EncounterParams(
                mass_solar=mass,
                scale_radius_kpc=profile["scale_radius_kpc"],
                impact_param_kpc=self.cfg.impact_param_kpc,
                flyby_vel_kms=self.cfg.flyby_vel_kms,
                encounter_phi1=phi1_enc,
                t_since_impact_gyr=t_since,
                is_valid=True,
                is_massive=(mass > 1e8),
            )

            # Simulate the encounter
            if self.cfg.use_fast_mode:
                perturbed = apply_impulse_approximation(self.base_stream, encounter)
            else:
                if self._mws is None:
                    import galstreams
                    self._mws = galstreams.MWStreams(verbose=False)
                perturbed = generate_perturbed_stream_evolved(
                    stream_name=self.cfg.stream_name,
                    potential=self.potential,
                    encounter=encounter,
                    n_stars=self.cfg.n_stars_sim,
                    seed=self.cfg.base_seed,
                    config_path=self.cfg.config_path,
                    mws=self._mws,
                )

            # Apply kick suppression from the density profile
            # Cored profiles (FDM, SIDM) produce weaker kicks than NFW —
            # we model this by scaling the kinematic perturbation component.
            kick_supp = profile["kick_suppression"]

            # Score
            sim_profile = compute_density_profile(
                perturbed.phi1, self.phi1_range,
                bin_width_deg=self.cfg.density_bin_width_deg,
            )
            sim_gaps = detect_gaps(
                sim_profile,
                min_depth=self.cfg.gap_detection_min_depth,
                min_significance=self.cfg.gap_detection_min_significance,
            )

            score = combined_score(
                sim_profile=sim_profile,
                obs_profile=self.obs_profile,
                sim_gaps=sim_gaps,
                obs_gaps=self.obs_gaps,
                sim_phi1=perturbed.phi1,
                sim_pm1=perturbed.pm1,
                sim_pm2=perturbed.pm2,
                obs_phi1=self.obs_particles["phi1"],
                obs_pm1=self.obs_particles["pm1"],
                obs_pm2=self.obs_particles["pm2"],
                phi1_range=self.phi1_range,
                weights=self.cfg.score_weights,
                density_bin_width=self.cfg.density_bin_width_deg,
                kinematic_bin_width=self.cfg.kinematic_bin_width_deg,
                sim_vrad=perturbed.vrad,
                obs_vrad=self.obs_particles.get("vrad"),
            )

            # GNN profile distance
            if self.gnn_scorer is not None:
                profile_dist = self.gnn_scorer.score(perturbed, self.cfg.stream_name)
                score.profile_distance = profile_dist
                w = self.cfg.score_weights
                w_total = w.density + w.gap + w.kinematic + w.profile
                score.combined = (
                    w.density * score.density_residual
                    + w.gap * score.gap_agreement
                    + w.kinematic * score.kinematic_perturbation
                    + w.profile * profile_dist
                ) / max(w_total, 1e-6)

            model_scores[dm_model] = score

        # Identify best model (lowest combined score)
        best_model = min(model_scores, key=lambda m: model_scores[m].combined)
        elapsed = time.perf_counter() - t0

        null_combined = self.null_score.combined if self.null_score else 0.0

        return ModelComparisonResult(
            log10_mass=log10_m,
            t_since_gyr=t_since,
            impact_phi1=phi1_enc,
            model_scores=model_scores,
            model_profiles=model_profiles,
            best_model=best_model,
            best_score=model_scores[best_model].combined,
            null_combined=null_combined,
            runtime_s=elapsed,
        )

    def run_model_comparison(
        self,
        candidates: list[dict] | None = None,
        top_k: int = 10,
        m_wdm_kev: float = 3.0,
        m_axion_ev: float = 1e-22,
        sigma_sidm_cm2g: float = 1.0,
    ) -> list[ModelComparisonResult]:
        """Run all four DM models on a set of encounter candidates.

        If no candidates are provided, uses the top-k from a coarse grid run.

        Args:
            candidates: List of param dicts (log10_mass, t_since_gyr, impact_phi1).
                If None, runs a coarse grid first and takes the top-k.
            top_k: Number of top candidates to compare (if candidates is None).
            m_wdm_kev: WDM particle mass.
            m_axion_ev: FDM axion mass.
            sigma_sidm_cm2g: SIDM cross-section.

        Returns:
            List of ModelComparisonResult sorted by best model score.
        """
        if candidates is None:
            log.info("Running coarse grid to find top-%d candidates for model comparison...", top_k)
            grid_results = self.run_grid()
            candidates = [
                {
                    "log10_mass": r.log10_mass,
                    "t_since_gyr": r.t_since_gyr,
                    "impact_phi1": r.impact_phi1,
                }
                for r in grid_results[:top_k]
            ]

        log.info("Model comparison: evaluating %d candidates × 4 DM models", len(candidates))
        log.info("  DM model params: m_wdm=%.1f keV, m_axion=%.1e eV, sigma_sidm=%.1f cm^2/g",
                 m_wdm_kev, m_axion_ev, sigma_sidm_cm2g)

        results = []
        t0 = time.perf_counter()

        for i, params in enumerate(candidates):
            result = self.evaluate_candidate_all_models(
                params,
                m_wdm_kev=m_wdm_kev,
                m_axion_ev=m_axion_ev,
                sigma_sidm_cm2g=sigma_sidm_cm2g,
            )
            results.append(result)

            if (i + 1) % 5 == 0 or (i + 1) == len(candidates):
                elapsed = time.perf_counter() - t0
                log.info("  [%d/%d] best so far: %s (%.4f)",
                         i + 1, len(candidates), result.best_model, result.best_score)

        results.sort(key=lambda r: r.best_score)
        total_time = time.perf_counter() - t0

        # Tally which model wins across all candidates
        model_wins = {m: 0 for m in DM_MODELS}
        for r in results:
            model_wins[r.best_model] += 1

        log.info("")
        log.info("  Model comparison complete: %d candidates in %.1f s", len(results), total_time)
        log.info("  Model win tally:")
        for m in DM_MODELS:
            pct = 100.0 * model_wins[m] / max(len(results), 1)
            log.info("    %s: %d / %d (%.0f%%)", m, model_wins[m], len(results), pct)

        # Per-model average score
        log.info("  Average combined score by model:")
        for m in DM_MODELS:
            avg = np.mean([r.model_scores[m].combined for r in results])
            log.info("    %s: %.4f", m, avg)

        overall_best = results[0]
        log.info("  Overall best: %s at M=10^%.2f, t=%.1f Gyr, phi1=%.1f (score=%.4f)",
                 overall_best.best_model, overall_best.log10_mass,
                 overall_best.t_since_gyr, overall_best.impact_phi1,
                 overall_best.best_score)
        if self.null_score:
            improvement = self.null_score.combined - overall_best.best_score
            log.info("  vs null: %+.4f (%.2f%% improvement)",
                     improvement,
                     100.0 * improvement / max(self.null_score.combined, 1e-8))

        return results

    def save_model_comparison_results(
        self, results: list[ModelComparisonResult]
    ) -> Path:
        """Save model comparison results to JSON."""
        out_dir = Path(self.cfg.output_dir) / self.cfg.stream_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Tally wins
        model_wins = {m: 0 for m in DM_MODELS}
        for r in results:
            model_wins[r.best_model] += 1

        # Average scores
        avg_scores = {}
        for m in DM_MODELS:
            avg_scores[m] = float(np.mean([r.model_scores[m].combined for r in results]))

        output = {
            "stream_name": self.cfg.stream_name,
            "mode": "model_comparison",
            "dm_models": list(DM_MODELS),
            "n_candidates": len(results),
            "null_hypothesis_combined": self.null_score.combined if self.null_score else None,
            "model_win_tally": model_wins,
            "model_average_scores": avg_scores,
            "top_results": [r.to_dict() for r in results[:self.cfg.top_k]],
        }

        out_file = out_dir / "model_comparison_results.json"
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)

        log.info("Model comparison results saved to %s", out_file)
        return out_file

    # ------------------------------------------------------------------
    # Multi-encounter evaluation (sequential impacts)
    # ------------------------------------------------------------------

    def evaluate_multi_encounter(
        self, encounter_params_list: list[dict]
    ) -> MultiEncounterResult:
        """Evaluate multiple sequential subhalo encounters on the same stream.

        Each encounter dict must have: log10_mass, t_since_gyr, impact_phi1.
        Optional keys: flyby_vel_kms, impact_param_kpc.

        Physics:
            Multiple subhalos impact the same stream at different past epochs.
            The encounters are applied chronologically (oldest first): earlier
            encounters create gaps that subsequent encounters further modify.
            Full orbit integration between encounters captures gap widening,
            phase mixing, and non-linear cumulative effects.

        In fast mode: sequential impulse approximations on the base stream.
        In full orbit mode: orbit-integrated evolution between each encounter epoch.

        Args:
            encounter_params_list: List of dicts, each specifying one encounter.

        Returns:
            MultiEncounterResult with cumulative scores.
        """
        t0 = time.perf_counter()

        # Build EncounterParams objects
        encounters = []
        encounter_dicts = []
        for ep in encounter_params_list:
            mass = 10.0 ** ep["log10_mass"]
            a_kpc = scale_radius_from_mass(mass)
            enc = EncounterParams(
                mass_solar=mass,
                scale_radius_kpc=a_kpc,
                impact_param_kpc=ep.get("impact_param_kpc", self.cfg.impact_param_kpc),
                flyby_vel_kms=ep.get("flyby_vel_kms", self.cfg.flyby_vel_kms),
                encounter_phi1=ep["impact_phi1"],
                t_since_impact_gyr=ep["t_since_gyr"],
                is_valid=True,
                is_massive=(mass > 1e8),
            )
            encounters.append(enc)
            encounter_dicts.append({
                "log10_mass": ep["log10_mass"],
                "t_since_gyr": ep["t_since_gyr"],
                "impact_phi1": ep["impact_phi1"],
                "flyby_vel_kms": enc.flyby_vel_kms,
                "impact_param_kpc": enc.impact_param_kpc,
                "scale_radius_kpc": a_kpc,
            })

        if self.cfg.use_fast_mode:
            # Fast mode: apply impulse approximations sequentially
            # Order by t_since (oldest first) for physically correct gap widening
            sorted_indices = sorted(
                range(len(encounters)),
                key=lambda i: encounters[i].t_since_impact_gyr,
                reverse=True,
            )
            perturbed = self.base_stream
            for idx in sorted_indices:
                perturbed = apply_impulse_approximation(perturbed, encounters[idx])
        else:
            # Full orbit mode: orbit-integrated multi-encounter evolution
            if self._mws is None:
                import galstreams
                self._mws = galstreams.MWStreams(verbose=False)

            perturbed = generate_perturbed_stream_multi_evolved(
                stream_name=self.cfg.stream_name,
                potential=self.potential,
                encounters=encounters,
                n_stars=self.cfg.n_stars_sim,
                seed=self.cfg.base_seed,
                config_path=self.cfg.config_path,
                mws=self._mws,
            )

        # Compute simulated density profile
        sim_profile = compute_density_profile(
            perturbed.phi1,
            self.phi1_range,
            bin_width_deg=self.cfg.density_bin_width_deg,
        )
        sim_gaps = detect_gaps(
            sim_profile,
            min_depth=self.cfg.gap_detection_min_depth,
            min_significance=self.cfg.gap_detection_min_significance,
        )

        # Score
        score = combined_score(
            sim_profile=sim_profile,
            obs_profile=self.obs_profile,
            sim_gaps=sim_gaps,
            obs_gaps=self.obs_gaps,
            sim_phi1=perturbed.phi1,
            sim_pm1=perturbed.pm1,
            sim_pm2=perturbed.pm2,
            obs_phi1=self.obs_particles["phi1"],
            obs_pm1=self.obs_particles["pm1"],
            obs_pm2=self.obs_particles["pm2"],
            phi1_range=self.phi1_range,
            weights=self.cfg.score_weights,
            density_bin_width=self.cfg.density_bin_width_deg,
            kinematic_bin_width=self.cfg.kinematic_bin_width_deg,
            sim_vrad=perturbed.vrad,
            obs_vrad=self.obs_particles.get("vrad"),
        )

        # GNN profile distance
        if self.gnn_scorer is not None:
            profile_dist = self.gnn_scorer.score(perturbed, self.cfg.stream_name)
            score.profile_distance = profile_dist
            w = self.cfg.score_weights
            w_total = w.density + w.gap + w.kinematic + w.profile
            score.combined = (
                w.density * score.density_residual
                + w.gap * score.gap_agreement
                + w.kinematic * score.kinematic_perturbation
                + w.profile * profile_dist
            ) / max(w_total, 1e-6)

        elapsed = time.perf_counter() - t0

        return MultiEncounterResult(
            encounters=encounter_dicts,
            score=score,
            n_encounters=len(encounters),
            n_stars_sim=len(perturbed.phi1),
            runtime_s=elapsed,
        )

    def run_multi_encounter_grid(
        self,
        n_encounters: int = 2,
        n_random_samples: int = 100,
        seed: int = 42,
    ) -> list[MultiEncounterResult]:
        """Evaluate a random sample of multi-encounter configurations.

        Because the multi-encounter parameter space is combinatorially explosive
        (each additional encounter multiplies the grid size), we use random
        sampling rather than a full Cartesian grid. Each sample draws N independent
        encounters from the single-encounter parameter ranges.

        Strategy:
            - Draw log10_mass uniformly from the configured range
            - Draw t_since uniformly from the configured range
            - Draw impact_phi1 uniformly from the observed phi1 extent
            - Ensure encounters are well-separated in time (>0.5 Gyr apart)
            - Ensure encounters are well-separated in phi1 (>5 deg apart)

        Args:
            n_encounters: Number of sequential encounters per sample.
            n_random_samples: How many multi-encounter configurations to evaluate.
            seed: Random seed for sampling.

        Returns:
            List of MultiEncounterResult, sorted by combined score (ascending).
        """
        rng = np.random.default_rng(seed)

        # Parameter ranges
        m_lo, m_hi = self.cfg.log10_mass_range
        t_lo, t_hi = self.cfg.t_since_range
        phi1_lo, phi1_hi = self.phi1_range

        log.info("Multi-encounter grid: %d samples, %d encounters each",
                 n_random_samples, n_encounters)
        log.info("  Ranges: log10_M=[%.1f, %.1f], t=[%.1f, %.1f] Gyr, "
                 "phi1=[%.1f, %.1f] deg",
                 m_lo, m_hi, t_lo, t_hi, phi1_lo, phi1_hi)

        samples = []
        attempts = 0
        max_attempts = n_random_samples * 20  # allow retries for separation constraints

        while len(samples) < n_random_samples and attempts < max_attempts:
            attempts += 1

            # Draw N encounters
            masses = rng.uniform(m_lo, m_hi, n_encounters)
            times = np.sort(rng.uniform(t_lo, t_hi, n_encounters))[::-1]  # sorted oldest-first
            phi1s = rng.uniform(phi1_lo, phi1_hi, n_encounters)

            # Separation constraints
            time_diffs = np.diff(np.sort(times))
            phi1_diffs = np.abs(np.diff(np.sort(phi1s)))

            # Require >0.5 Gyr separation in time and >5 deg in phi1
            if n_encounters > 1:
                if np.any(time_diffs < 0.5) or np.any(phi1_diffs < 5.0):
                    continue

            encounter_list = []
            for i in range(n_encounters):
                encounter_list.append({
                    "log10_mass": float(masses[i]),
                    "t_since_gyr": float(times[i]),
                    "impact_phi1": float(phi1s[i]),
                })
            samples.append(encounter_list)

        log.info("  Generated %d valid samples from %d attempts", len(samples), attempts)

        # Evaluate all samples
        results = []
        t0 = time.perf_counter()

        for i, encounter_list in enumerate(samples):
            result = self.evaluate_multi_encounter(encounter_list)
            results.append(result)

            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                best_so_far = min(r.score.combined for r in results)
                log.info("  [%d/%d] %.2f samples/sec, best combined=%.4f",
                         i + 1, len(samples), rate, best_so_far)

        # Sort by combined score (ascending = best first)
        results.sort(key=lambda r: r.score.combined)

        # Report vs null
        if self.null_score is not None:
            null_combined = self.null_score.combined
            best = results[0]
            n_better = sum(1 for r in results if r.score.combined < null_combined)
            improvement = null_combined - best.score.combined
            log.info("  Multi-encounter best: combined=%.4f (%.1f%% better than null)",
                     best.score.combined,
                     100.0 * improvement / max(null_combined, 1e-8))
            log.info("  Best config: %d encounters", best.n_encounters)
            for enc in best.encounters:
                log.info("    M=10^%.2f, t=%.1f Gyr, phi1=%.1f deg",
                         enc["log10_mass"], enc["t_since_gyr"], enc["impact_phi1"])
            log.info("  Samples better than null: %d / %d", n_better, len(results))

            # Compare to single-encounter best (if available)
            log.info("  (Compare: single-encounter null score = %.4f)", null_combined)

        total_time = time.perf_counter() - t0
        log.info("  Multi-encounter complete: %d samples in %.1f s (%.2f samples/s)",
                 len(results), total_time, len(results) / max(total_time, 0.001))

        return results

    def save_multi_encounter_results(
        self, results: list[MultiEncounterResult]
    ) -> Path:
        """Save multi-encounter results to JSON.

        Returns:
            Path to the output JSON file.
        """
        out_dir = Path(self.cfg.output_dir) / self.cfg.stream_name
        out_dir.mkdir(parents=True, exist_ok=True)

        null_score_dict = None
        if self.null_score is not None:
            null_score_dict = {
                "combined": self.null_score.combined,
                "density_residual": self.null_score.density_residual,
                "gap_agreement": self.null_score.gap_agreement,
                "kinematic_perturbation": self.null_score.kinematic_perturbation,
            }

        n_better = sum(
            1 for r in results
            if self.null_score and r.score.combined < self.null_score.combined
        )

        output = {
            "stream_name": self.cfg.stream_name,
            "mode": "multi_encounter",
            "n_samples_total": len(results),
            "n_encounters_per_sample": results[0].n_encounters if results else 0,
            "n_observed_stars": len(self.obs_particles["phi1"]),
            "null_hypothesis": null_score_dict,
            "n_samples_better_than_null": n_better,
            "top_results": [r.to_dict() for r in results[: self.cfg.top_k]],
            "all_combined_scores": [r.score.combined for r in results],
        }

        out_file = out_dir / "multi_encounter_results.json"
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)

        log.info("Multi-encounter results saved to %s", out_file)
        return out_file

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_results(self, results: list[CandidateResult]) -> Path:
        """Save results to JSON in the output directory.

        Returns:
            Path to the output JSON file.
        """
        out_dir = Path(self.cfg.output_dir) / self.cfg.stream_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Null hypothesis score
        null_score_dict = None
        if self.null_score is not None:
            null_score_dict = {
                "combined": self.null_score.combined,
                "density_residual": self.null_score.density_residual,
                "gap_agreement": self.null_score.gap_agreement,
                "kinematic_perturbation": self.null_score.kinematic_perturbation,
            }

        # Summary: top-k results + metadata
        n_better = sum(1 for r in results if self.null_score and r.score.combined < self.null_score.combined)
        output = {
            "stream_name": self.cfg.stream_name,
            "n_candidates_total": len(results),
            "n_observed_stars": len(self.obs_particles["phi1"]),
            "n_observed_gaps": len(self.obs_gaps),
            "observed_gaps": [
                {"phi1": g.phi1_center, "depth": g.depth, "width": g.phi1_width}
                for g in self.obs_gaps
            ],
            "null_hypothesis": null_score_dict,
            "n_candidates_better_than_null": n_better,
            "config": {
                "log10_mass_range": list(self.cfg.log10_mass_range),
                "t_since_range": list(self.cfg.t_since_range),
                "impact_phi1_values": self.cfg.impact_phi1_values,
                "flyby_vel_kms": self.cfg.flyby_vel_kms,
                "impact_param_kpc": self.cfg.impact_param_kpc,
                "n_stars_sim": self.cfg.n_stars_sim,
                "n_workers": self.cfg.n_workers,
                "density_bin_width_deg": self.cfg.density_bin_width_deg,
            },
            "top_candidates": [r.to_dict() for r in results[: self.cfg.top_k]],
            "all_combined_scores": [r.score.combined for r in results],
        }

        out_file = out_dir / "forward_model_results.json"
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)

        log.info("Results saved to %s", out_file)
        return out_file
