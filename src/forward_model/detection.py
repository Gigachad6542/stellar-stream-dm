"""
Detection -> timeline handoff for the forward model.

This module realises the *front end* of the timeline analysis:

    1. Look at the real stream and decide whether there is a probable-enough
       subhalo impact ("is something there?").
    2. Estimate *where* along the stream the impact sits (phi1) and *when* it
       happened (time since impact).
    3. Translate those estimates into a narrowed forward-model search grid, so
       the expensive orbit-integrated candidate sweep is centred on the region
       of parameter space the data actually points to, instead of a blind scan.

Two complementary detectors are combined:

``detect_impacts_modelfree``
    Always-available, model-free gap finder. It builds the observed density
    profile and runs the same ``detect_gaps`` routine used elsewhere, returning
    the gap longitudes (phi1) and their Poisson significances. This is the
    robust backbone: it gives the *localisation* that seeds the grid even with
    no trained network present.

``StreamImpactDetector``
    Optional, learned detector. It loads a trained ``StreamGNNMultiTaskV2``
    checkpoint (the timeline variant, ``binary_target=impact_timeline_detectable``,
    ``regression_target=timeline_effective``) and runs the real stream through it
    to obtain:
        - ``p_impact``  = sigmoid(binary logit)         -> probability of impact
        - ``t_since``   = 10 ** regression_output[1]     -> Gyr since strongest impact
        - ``effective_n_impacts`` = regression_output[0]
    The regression targets were trained in raw physical units (no
    standardisation), so the decode is a direct power of ten.

    Because the network is trained on simulations, its real-data predictions are
    subject to the known sim-to-real gap. It therefore degrades gracefully: if
    the checkpoint, its sibling ``normalizer_v2.npz``, or the feature build fail,
    the GNN path is simply marked unavailable and the model-free localisation is
    used on its own.

``seed_config_from_detection``
    Turns a ``DetectionResult`` into a narrowed ``ForwardModelConfig``: phi1
    candidates come from the detected gaps, and the time grid is focused around
    the GNN time estimate (when available).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from .scoring import GapFeature, compute_density_profile, detect_gaps, find_density_minima

if TYPE_CHECKING:  # avoid a hard import cycle at module load time
    from .pipeline import ForwardModelConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Outcome of running the detection front-end on one observed stream."""

    stream_name: str
    phi1_range: tuple[float, float]

    # --- Model-free gap detection (always populated) ---
    gaps: list[GapFeature] = field(default_factory=list)
    gap_phi1_locations: list[float] = field(default_factory=list)
    max_gap_significance: float = 0.0
    n_significant_gaps: int = 0
    gap_detected: bool = False
    # Data-driven phi1 positions to scan when no significant gap is found.
    # Always in-frame (drawn from the observed phi1 distribution).
    fallback_phi1_values: list[float] = field(default_factory=list)

    # --- Learned GNN detector (optional) ---
    gnn_available: bool = False
    p_impact: Optional[float] = None
    gnn_impact_flag: Optional[bool] = None
    t_since_gyr_estimate: Optional[float] = None
    t_since_gyr_log10: Optional[float] = None
    effective_n_impacts: Optional[float] = None
    # Out-of-distribution diagnostics for the GNN inputs (real data vs training).
    # High values mean the network is extrapolating and its p_impact is untrustworthy.
    ood_max_sigma: Optional[float] = None
    ood_frac_clipped: Optional[float] = None

    # --- Decision ---
    # impact_detected is True if EITHER the model-free gap finder fires (the
    # trustworthy, data-only signal) OR the GNN flags an impact. The two sources
    # are kept separate (gap_detected / gnn_impact_flag) because the GNN is
    # trained on simulations and is known to be over-confident on real data.
    impact_detected: bool = False
    detection_reason: str = ""

    def to_dict(self) -> dict:
        """JSON-serialisable summary."""
        return {
            "stream_name": self.stream_name,
            "phi1_range": list(self.phi1_range),
            "gaps": [
                {
                    "phi1_center": g.phi1_center,
                    "phi1_width": g.phi1_width,
                    "depth": g.depth,
                    "significance": g.significance,
                }
                for g in self.gaps
            ],
            "gap_phi1_locations": self.gap_phi1_locations,
            "max_gap_significance": self.max_gap_significance,
            "n_significant_gaps": self.n_significant_gaps,
            "gap_detected": self.gap_detected,
            "fallback_phi1_values": self.fallback_phi1_values,
            "gnn_available": self.gnn_available,
            "p_impact": self.p_impact,
            "gnn_impact_flag": self.gnn_impact_flag,
            "ood_max_sigma": self.ood_max_sigma,
            "ood_frac_clipped": self.ood_frac_clipped,
            "t_since_gyr_estimate": self.t_since_gyr_estimate,
            "t_since_gyr_log10": self.t_since_gyr_log10,
            "effective_n_impacts": self.effective_n_impacts,
            "impact_detected": self.impact_detected,
            "detection_reason": self.detection_reason,
        }


# ---------------------------------------------------------------------------
# Model-free gap detection (no network required)
# ---------------------------------------------------------------------------

def detect_impacts_modelfree(
    obs_particles: dict,
    phi1_range: tuple[float, float],
    stream_name: str = "GD1",
    density_bin_width_deg: float = 1.0,
    gap_min_depth: float = 0.3,
    gap_min_significance: float = 2.0,
    significant_threshold: float = 3.0,
) -> DetectionResult:
    """Detect candidate impacts from the observed density profile alone.

    Builds the membership-weighted density profile and finds gaps. Gap centres
    become the localisation used to seed the forward-model phi1 grid.

    Args:
        obs_particles: dict with at least ``phi1``; optional ``membership_prob``.
        phi1_range: (min, max) phi1 extent for binning.
        stream_name: name (for reporting only).
        density_bin_width_deg: profile bin width.
        gap_min_depth / gap_min_significance: passed to ``detect_gaps``.
        significant_threshold: gaps at/above this significance count as
            "significant" and drive the impact-detected decision.

    Returns:
        DetectionResult with the model-free fields populated.
    """
    phi1 = np.asarray(obs_particles["phi1"], dtype=np.float64)
    weights = obs_particles.get("membership_prob")

    profile = compute_density_profile(
        phi1, phi1_range, bin_width_deg=density_bin_width_deg, weights=weights,
    )
    # Strict gaps (clean, deep) drive the detection DECISION.
    gaps = detect_gaps(
        profile, min_depth=gap_min_depth, min_significance=gap_min_significance,
    )
    # Prominence-ranked minima drive LOCALISATION — these always exist, so the
    # forward-model phi1 grid is seeded from the real most-depleted regions even
    # when gaps are shallow or contamination-diluted (e.g. the GD-1 catalog).
    minima = find_density_minima(profile, top_k=5)

    gap_locations = [float(g.phi1_center) for g in minima]
    max_sig = float(max((g.significance for g in gaps), default=0.0))
    n_sig = int(sum(1 for g in gaps if g.significance >= significant_threshold))

    # In-frame fallback (only if even minima-finding returns nothing).
    finite_phi1 = phi1[np.isfinite(phi1)]
    fallback = ([float(x) for x in np.percentile(finite_phi1, [25, 50, 75])]
                if finite_phi1.size else [])

    detected = n_sig > 0
    if detected:
        reason = (f"{n_sig} significant gap(s) (max sig {max_sig:.1f}); "
                  f"deepest minima at {[f'{g.phi1_center:.0f}' for g in minima[:3]]}")
    elif minima:
        best = minima[0]
        reason = (f"no >={significant_threshold:.0f}-sigma gap; deepest minimum at "
                  f"phi1={best.phi1_center:.0f} (depth {best.depth:.2f}, sig {best.significance:.1f})")
    else:
        reason = "no density minima found"

    return DetectionResult(
        stream_name=stream_name,
        phi1_range=(float(phi1_range[0]), float(phi1_range[1])),
        gaps=minima,                       # localisation candidates (ranked)
        gap_phi1_locations=gap_locations,
        max_gap_significance=max_sig,
        n_significant_gaps=n_sig,
        gap_detected=detected,
        fallback_phi1_values=fallback,
        impact_detected=detected,
        detection_reason=reason,
    )


# ---------------------------------------------------------------------------
# Learned GNN detector
# ---------------------------------------------------------------------------

class StreamImpactDetector:
    """Runs the trained timeline GNN on a real stream to estimate P(impact) and time.

    The checkpoint must be a ``StreamGNNMultiTaskV2`` trained with the timeline
    schema (regression target ``timeline_effective``), so that regression output
    index 1 is ``log10(t_since_strongest_impact / Gyr)``.

    Mirrors the inference recipe used in training exactly:
        build kNN graph (k from config, with the saved normalizer)
        -> build profile features from raw node features
        -> standardise node features with the saved normalizer
        -> forward pass.

    Usage:
        det = StreamImpactDetector("checkpoints/gnn_v2_timeline_.../gnn_v2_best.pt")
        if det.is_available:
            result = det.detect(obs_particles, phi1_range, stream_name="GD1")
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        max_stars: int = 1200,
        subsample_seed: int = 42,
        p_impact_threshold: float = 0.5,
        clip_sigma: float = 5.0,
        temperature: Optional[float] = None,
    ) -> None:
        import torch

        self.device = torch.device(device)
        self.max_stars = max_stars
        self.subsample_seed = subsample_seed
        self.p_impact_threshold = p_impact_threshold
        # Clip standardized features to +/- clip_sigma so a single out-of-
        # distribution feature cannot explode the logit (the cause of the
        # p_impact=1.0 saturation on real data). temperature softens the
        # probabilities; loaded from calibration.json next to the checkpoint
        # if present, else 1.0.
        self.clip_sigma = clip_sigma
        self.temperature = temperature
        self.last_ood_max_sigma = None
        self.last_ood_frac_clipped = None

        self.model = None
        self.normalizer = None
        self.k_neighbors = 8
        self.profile_n_bins = 48
        self.profile_feature_set = "summary"
        self.profile_include_onehot = True
        self.predict_uncertainty = False
        self._checkpoint_path = checkpoint_path

        self._load_checkpoint(checkpoint_path)

    @property
    def is_available(self) -> bool:
        return self.model is not None and self.normalizer is not None

    def _load_checkpoint(self, path: str) -> None:
        import torch

        from ..data.dataset import FeatureNormalizer, profile_feature_dim
        from ..models.gnn import StreamGNNMultiTaskV2
        from ..models.utils import load_normalizer

        ckpt_path = Path(path)
        if not ckpt_path.exists():
            log.warning("Detector checkpoint not found at %s; GNN detection disabled", path)
            return

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {}) or {}
        graph_cfg = cfg.get("graph", {})
        gnn_cfg = cfg.get("model", {}).get("gnn", {})
        prof_cfg = graph_cfg.get("profile_branch", {})
        tv2 = cfg.get("training_v2", {})

        # Graph + profile configuration (must match training to reproduce inputs)
        self.k_neighbors = int(graph_cfg.get("k_neighbors", 8))
        self.profile_n_bins = int(prof_cfg.get("n_bins", 48))
        self.profile_feature_set = str(prof_cfg.get("feature_set", "summary"))
        self.profile_include_onehot = bool(prof_cfg.get("include_stream_onehot", True))
        self.predict_uncertainty = bool(tv2.get("predict_uncertainty", False))

        reg_target = tv2.get("regression_target")
        if reg_target not in ("timeline_effective", "timeline_detectable", "strength_time"):
            log.warning(
                "Detector checkpoint regression_target=%r is not a timeline schema; "
                "the time-since-impact estimate may be meaningless.", reg_target,
            )

        profile_dim = 0
        if prof_cfg.get("enabled", False):
            profile_dim = profile_feature_dim(
                self.profile_n_bins, self.profile_feature_set, self.profile_include_onehot,
            )

        model = StreamGNNMultiTaskV2(
            n_reg_targets=int(tv2.get("n_reg_targets", 2)),
            predict_uncertainty=self.predict_uncertainty,
            profile_dim=profile_dim,
            profile_hidden_dim=int(prof_cfg.get("hidden_dim", 64)),
            profile_layer_norm=bool(prof_cfg.get("layer_norm", False)),
            n_node_features=int(graph_cfg.get("n_node_features", 18)),
            n_edge_features=int(graph_cfg.get("n_edge_features", 5)),
            hidden_dim=int(gnn_cfg.get("hidden_dim", 256)),
            embedding_dim=int(gnn_cfg.get("embedding_dim", 128)),
            n_layers=int(gnn_cfg.get("n_layers", 6)),
            dropout=float(gnn_cfg.get("dropout", 0.1)),
            use_attention_readout=bool(gnn_cfg.get("use_attention_readout", False)),
        )

        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            log.warning("Detector load: %d missing keys (e.g. %s)", len(missing), missing[:3])
        if unexpected:
            log.warning("Detector load: %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])

        model.to(self.device)
        model.eval()
        self.model = model

        # The normalizer is saved next to the checkpoint as normalizer_v2.npz.
        norm_path = ckpt_path.parent / "normalizer_v2.npz"
        if norm_path.exists():
            mean, std = load_normalizer(norm_path)
            self.normalizer = FeatureNormalizer(mean, std)
            # Temperature for probability calibration (calibration.json sidecar).
            if self.temperature is None:
                calib_path = ckpt_path.parent / "calibration.json"
                if calib_path.exists():
                    import json
                    try:
                        self.temperature = float(json.loads(calib_path.read_text())["temperature"])
                        log.info("  Loaded calibration temperature T=%.3f", self.temperature)
                    except Exception as e:
                        log.warning("  Failed to read calibration.json: %s", e)
                        self.temperature = 1.0
                else:
                    self.temperature = 1.0
            log.info("Detector loaded from %s (normalizer %s, k=%d, profile_dim=%d, T=%.2f)",
                     path, norm_path.name, self.k_neighbors, profile_dim, self.temperature)
        else:
            log.warning(
                "Normalizer %s not found; GNN detection disabled (cannot reproduce "
                "training-time feature scaling).", norm_path,
            )
            self.model = None

    def _predict(self, obs_particles: dict, stream_name: str) -> Optional[dict]:
        """Run the GNN forward pass. Returns dict of predictions or None on failure."""
        import torch

        from ..data.dataset import build_knn_graph_batched, build_profile_features_batched
        from .gnn_scorer import obs_particles_to_data

        try:
            data = obs_particles_to_data(obs_particles, stream_name)

            # Match the training cap of max_stars per stream (deterministic).
            n = data.x.shape[0]
            if n > self.max_stars:
                g = torch.Generator().manual_seed(self.subsample_seed)
                idx = torch.randperm(n, generator=g)[: self.max_stars]
                data.x = data.x[idx]
                data.batch = torch.zeros(self.max_stars, dtype=torch.long)

            # Impute unmeasured features (NaN, e.g. missing RV for a stream with
            # no spectroscopy) to the training mean so they contribute ~0 after
            # standardisation, instead of feeding fake constants that explode
            # through the std-clamped normalizer.
            means = self.normalizer.mean.to(dtype=data.x.dtype)
            nan_mask = torch.isnan(data.x)
            if nan_mask.any():
                data.x = torch.where(nan_mask, means.unsqueeze(0).expand_as(data.x), data.x)
            data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)

            # 1) kNN graph (uses normalizer internally for construction features)
            build_knn_graph_batched(data, self.k_neighbors, self.normalizer)
            # 2) profile features from RAW node features (before standardisation)
            build_profile_features_batched(
                data,
                n_bins=self.profile_n_bins,
                feature_set=self.profile_feature_set,
                include_stream_onehot=self.profile_include_onehot,
            )
            # 3) standardise node features (matches train_v2 order)
            data.x = self.normalizer(data.x)

            # Out-of-distribution guard: record how far real features land from
            # the training distribution, then clip so no single feature can
            # saturate the logit.
            abs_x = data.x.abs()
            self.last_ood_max_sigma = float(abs_x.max().item())
            self.last_ood_frac_clipped = float((abs_x > self.clip_sigma).float().mean().item())
            data.x = torch.clamp(data.x, -self.clip_sigma, self.clip_sigma)

            data = data.to(self.device)
            with torch.no_grad():
                _, binary_logit, _mhm_out, reg_out = self.model(data)

            logit = float(binary_logit.flatten()[0].item())
            T = self.temperature or 1.0
            p_impact = float(1.0 / (1.0 + np.exp(-logit / T)))
            reg = reg_out.flatten().cpu().numpy()
            effective_n = float(reg[0]) if reg.size >= 1 else None
            log_t = float(reg[1]) if reg.size >= 2 else None

            if log_t is None or not np.isfinite(log_t):
                t_since = None
                log_t = None
            else:
                # Clamp to a sane physical window before exponentiating.
                log_t = float(np.clip(log_t, -1.0, 1.3))  # 0.1 .. ~20 Gyr
                t_since = float(10.0 ** log_t)

            return {
                "p_impact": p_impact,
                "t_since_gyr_estimate": t_since,
                "t_since_gyr_log10": log_t,
                "effective_n_impacts": effective_n,
            }
        except Exception as e:  # graceful degradation
            log.warning("GNN detection failed (%s); falling back to model-free only", e)
            return None

    def detect(
        self,
        obs_particles: dict,
        phi1_range: tuple[float, float],
        stream_name: str = "GD1",
        density_bin_width_deg: float = 1.0,
        gap_min_depth: float = 0.3,
        gap_min_significance: float = 2.0,
        significant_threshold: float = 3.0,
    ) -> DetectionResult:
        """Full detection: model-free gaps + (if available) the learned GNN."""
        result = detect_impacts_modelfree(
            obs_particles, phi1_range, stream_name,
            density_bin_width_deg=density_bin_width_deg,
            gap_min_depth=gap_min_depth,
            gap_min_significance=gap_min_significance,
            significant_threshold=significant_threshold,
        )

        if not self.is_available:
            return result

        preds = self._predict(obs_particles, stream_name)
        if preds is None:
            return result

        result.gnn_available = True
        result.p_impact = preds["p_impact"]
        result.t_since_gyr_estimate = preds["t_since_gyr_estimate"]
        result.t_since_gyr_log10 = preds["t_since_gyr_log10"]
        result.effective_n_impacts = preds["effective_n_impacts"]
        result.ood_max_sigma = self.last_ood_max_sigma
        result.ood_frac_clipped = self.last_ood_frac_clipped
        if result.ood_max_sigma is not None and result.ood_max_sigma >= self.clip_sigma:
            log.warning("  GNN inputs are out-of-distribution (max %.1f sigma, %.1f%% clipped); "
                        "p_impact is unreliable.", result.ood_max_sigma,
                        100.0 * (result.ood_frac_clipped or 0.0))

        # The GNN flag is kept separate from the model-free gap flag because the
        # network is trained on simulations and tends to be over-confident on
        # real data (it frequently reports p_impact ~ 1). A probable impact is
        # declared if EITHER source fires, but downstream code can inspect the
        # two flags independently and weight them accordingly.
        result.gnn_impact_flag = (
            result.p_impact is not None and result.p_impact >= self.p_impact_threshold
        )
        result.impact_detected = bool(result.gap_detected or result.gnn_impact_flag)
        result.detection_reason += (
            f"; GNN p_impact={result.p_impact:.2f}"
            + (f", t_since~{result.t_since_gyr_estimate:.1f} Gyr"
               if result.t_since_gyr_estimate is not None else "")
        )
        return result


# ---------------------------------------------------------------------------
# Detection -> forward-model grid
# ---------------------------------------------------------------------------

def seed_config_from_detection(
    detection: DetectionResult,
    base_config: "ForwardModelConfig",
    t_window_gyr: float = 2.0,
    max_phi1_seeds: int = 3,
    min_t_since_gyr: float = 0.5,
    max_t_since_gyr: float = 10.0,
    stream_age_gyr: Optional[float] = None,
) -> "ForwardModelConfig":
    """Build a narrowed ForwardModelConfig from a DetectionResult.

    Two knobs are tightened relative to a blind sweep:

    * ``impact_phi1_values`` is set to the detected gap longitudes (the most
      significant first, up to ``max_phi1_seeds``). If no gaps were detected the
      base configuration's phi1 values are kept unchanged.

    * ``t_since_range`` is focused to ``estimate +/- t_window_gyr`` when the GNN
      provided a finite time estimate; otherwise the base range is kept.

    The mass range is left untouched: the detector estimates an impact *count*
    and time, not the perturber mass, so mass is still scanned in full.

    Returns a new config (the input is not mutated).
    """
    updates: dict = {}

    # --- phi1 localisation ---
    # Prefer detected gap longitudes; otherwise fall back to in-frame quartile
    # positions from the data (never the base config's possibly out-of-frame
    # default, which for GD-1 is in a different phi1 convention).
    if detection.gap_phi1_locations:
        seeds = detection.gap_phi1_locations[:max_phi1_seeds]
        updates["impact_phi1_values"] = [float(p) for p in seeds]
        log.info("Seeded phi1 grid from %d detected gap(s): %s",
                 len(seeds), [f"{p:.1f}" for p in seeds])
    elif detection.fallback_phi1_values:
        seeds = detection.fallback_phi1_values[:max_phi1_seeds]
        updates["impact_phi1_values"] = [float(p) for p in seeds]
        log.info("No significant gap; seeding phi1 grid from in-frame quartiles: %s",
                 [f"{p:.1f}" for p in seeds])
    else:
        log.info("No gaps or fallback positions; keeping base phi1 values %s",
                 base_config.impact_phi1_values)

    # --- time window from the GNN estimate ---
    # A subhalo cannot have struck before the stream began forming, so the time
    # since impact is physically capped at the stream's disruption age. The GNN
    # head (trained across streams of different ages, and unreliable on OOD real
    # data) can exceed this; cap it so the seeded window stays physical.
    t_max = max_t_since_gyr
    if stream_age_gyr is not None:
        t_max = min(t_max, float(stream_age_gyr))

    t_est = detection.t_since_gyr_estimate
    if t_est is not None and np.isfinite(t_est):
        t_est_capped = min(t_est, t_max)
        if stream_age_gyr is not None and t_est > stream_age_gyr:
            log.info("GNN t_since estimate %.2f Gyr exceeds %s disruption age %.2f Gyr; "
                     "capping at the stream age.", t_est, base_config.stream_name, stream_age_gyr)
        t_lo = max(min_t_since_gyr, t_est_capped - t_window_gyr)
        t_hi = min(t_max, t_est_capped + t_window_gyr)
        if t_hi <= t_lo:  # degenerate window guard
            t_lo, t_hi = min_t_since_gyr, t_max
        updates["t_since_range"] = (float(t_lo), float(t_hi))
        log.info("Focused time grid on GNN estimate %.2f Gyr (capped %.2f) -> range [%.2f, %.2f] Gyr",
                 t_est, t_est_capped, t_lo, t_hi)
    else:
        log.info("No GNN time estimate; keeping base t_since_range %s",
                 base_config.t_since_range)

    return dataclasses.replace(base_config, **updates)
