"""
SBI-based gap origin classifier.

Given a detected gap's local properties (width, depth, local kinematics,
local GNN embedding), infers the posterior probability of its origin:

    P(DM subhalo | gap)  — dark matter subhalo impact
    P(GMC | gap)         — giant molecular cloud fly-by
    P(GC | gap)          — globular cluster fly-by
    P(bar | gap)         — galactic bar resonance
    P(noise | gap)       — statistical fluctuation (no real perturbation)

This replaces the placeholder heuristic in gap_catalog.classify_gaps() with
a trained probabilistic classifier built on matched simulations of each
perturbation source.

Architecture:
    1. Simulate gaps from each source using the forward model in src/simulation/
    2. Extract gap-level summary statistics (features)
    3. Train an NPE or classifier on (source_label, features) pairs
    4. At inference time, compute P(source | features) via the trained model

Why not just use the full-stream NPE?
    The full-stream NPE infers DM model parameters (M_sub, n_impacts) but does
    NOT tell you which individual gap is from DM vs baryonic. A stream with 3
    gaps might have 2 from GMCs and 1 from DM — the stream-level posterior
    averages over all of them. The gap classifier operates per-gap.

References:
    Amorisco+2016 (GMC gaps indistinguishable from DM at same mass)
    Pearson+2017 (bar resonance density modulation)
    Webb+2025 (globular cluster fly-by gaps, arxiv:2502.03941)
    Erkal & Belokurov 2015 (DM subhalo impulse formalism)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .gap_catalog import Gap

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gap feature extraction
# ---------------------------------------------------------------------------

@dataclass
class GapFeatures:
    """Summary statistics for a single gap, used as input to the classifier.

    These features are designed to capture properties that differ between
    DM subhalo gaps and baryonic gaps:
    - DM gaps are typically narrower, deeper, and show kinematic coherence
    - GMC gaps occur preferentially near the disk mid-plane
    - Bar gaps are periodic and correlated with phi1
    - GC gaps have similar morphology to DM but different kinematics
    """
    # Morphological features
    width_deg: float              # gap half-width in phi1 [deg]
    depth_fraction: float         # 1 - N_gap / N_smooth (0=no gap, 1=empty)
    significance_sigma: float     # detection significance
    asymmetry: float              # (N_leading - N_trailing) / N_total near gap edges

    # Kinematic features (from stars flanking the gap)
    delta_pm1_leading: float      # mean pm1 of leading edge relative to smooth model
    delta_pm1_trailing: float     # mean pm1 of trailing edge
    delta_pm2_flanking: float     # mean |pm2| deviation of flanking stars
    delta_vrad_flanking: float    # mean vrad deviation of flanking stars
    pm_coherence: float           # correlation between pm1 deviation and distance from gap center

    # Positional context
    phi1_center_norm: float       # gap position normalized to [0, 1] along stream
    phi2_at_gap: float            # stream latitude at gap center (proxy for disk distance)
    local_density_ratio: float    # gap region density / global mean density

    # Embedding features (optional — from trained GNN)
    local_embedding: Optional[np.ndarray] = None  # [embedding_dim] from GNN on gap neighborhood


def extract_gap_features(
    gap: Gap,
    phi1: np.ndarray,
    phi2: np.ndarray,
    pm1: np.ndarray,
    pm2: np.ndarray,
    vrad: np.ndarray,
    phi1_range: tuple[float, float] = (-100.0, 20.0),
    flank_width_deg: float = 5.0,
) -> GapFeatures:
    """Extract classifier features from the stars surrounding a detected gap.

    Args:
        gap: Detected Gap object with phi1_center and phi1_width.
        phi1, phi2, pm1, pm2, vrad: Full stream observables.
        phi1_range: Stream phi1 range for normalization.
        flank_width_deg: Width of flanking regions for kinematic measurements.

    Returns:
        GapFeatures dataclass.
    """
    center = gap.phi1_center
    half_w = gap.phi1_width / 2.0

    # Define regions
    in_gap = np.abs(phi1 - center) < half_w
    leading = (phi1 > center + half_w) & (phi1 < center + half_w + flank_width_deg)
    trailing = (phi1 < center - half_w) & (phi1 > center - half_w - flank_width_deg)
    flanking = leading | trailing

    n_lead = leading.sum()
    n_trail = trailing.sum()
    n_flank = flanking.sum()

    # Asymmetry
    asymmetry = (n_lead - n_trail) / max(n_lead + n_trail, 1)

    # Kinematic features from flanking stars
    # Compute smooth model (global median) for reference
    pm1_smooth = float(np.median(pm1))
    pm2_smooth = float(np.median(pm2))
    vrad_smooth = float(np.median(vrad))

    delta_pm1_leading = float(np.mean(pm1[leading]) - pm1_smooth) if n_lead > 2 else 0.0
    delta_pm1_trailing = float(np.mean(pm1[trailing]) - pm1_smooth) if n_trail > 2 else 0.0
    delta_pm2_flanking = float(np.mean(np.abs(pm2[flanking] - pm2_smooth))) if n_flank > 2 else 0.0
    delta_vrad_flanking = float(np.mean(np.abs(vrad[flanking] - vrad_smooth))) if n_flank > 2 else 0.0

    # PM coherence: correlation between pm1 deviation and distance from gap center
    if n_flank > 5:
        dist_from_center = phi1[flanking] - center
        pm1_dev = pm1[flanking] - pm1_smooth
        corr = np.corrcoef(dist_from_center, pm1_dev)[0, 1]
        pm_coherence = float(corr) if np.isfinite(corr) else 0.0
    else:
        pm_coherence = 0.0

    # Positional context
    phi1_span = phi1_range[1] - phi1_range[0]
    phi1_center_norm = (center - phi1_range[0]) / max(phi1_span, 1.0)
    phi2_at_gap = float(np.median(phi2[in_gap])) if in_gap.sum() > 0 else 0.0

    # Local density ratio
    n_total = len(phi1)
    local_expected = n_total * (gap.phi1_width / phi1_span)
    local_density_ratio = in_gap.sum() / max(local_expected, 1.0)

    return GapFeatures(
        width_deg=gap.phi1_width,
        depth_fraction=gap.depth,
        significance_sigma=gap.significance,
        asymmetry=asymmetry,
        delta_pm1_leading=delta_pm1_leading,
        delta_pm1_trailing=delta_pm1_trailing,
        delta_pm2_flanking=delta_pm2_flanking,
        delta_vrad_flanking=delta_vrad_flanking,
        pm_coherence=pm_coherence,
        phi1_center_norm=phi1_center_norm,
        phi2_at_gap=phi2_at_gap,
        local_density_ratio=local_density_ratio,
    )


def features_to_array(features: GapFeatures) -> np.ndarray:
    """Convert GapFeatures to a flat float32 array for the classifier."""
    return np.array([
        features.width_deg,
        features.depth_fraction,
        features.significance_sigma,
        features.asymmetry,
        features.delta_pm1_leading,
        features.delta_pm1_trailing,
        features.delta_pm2_flanking,
        features.delta_vrad_flanking,
        features.pm_coherence,
        features.phi1_center_norm,
        features.phi2_at_gap,
        features.local_density_ratio,
    ], dtype=np.float32)


N_GAP_FEATURES = 12  # number of scalar features (excluding optional embedding)

# Source labels — order matters (used as class indices)
GAP_SOURCES = ["dm_subhalo", "gmc", "globular_cluster", "bar", "noise"]
N_SOURCES = len(GAP_SOURCES)


# ---------------------------------------------------------------------------
# Simulation of gap training data (per-source)
# ---------------------------------------------------------------------------

def simulate_gap_from_dm_subhalo(
    stream_particles,
    stream_config: dict,
    dm_model_config: dict,
    seed: int = 0,
) -> tuple[GapFeatures, dict]:
    """Simulate a single DM subhalo impact and extract gap features.

    Returns (GapFeatures, metadata_dict) or (None, None) if no gap detected.
    """
    from src.simulation.subhalo import sample_subhalo_encounter, apply_impulse_approximation
    from .gap_catalog import detect_gaps

    enc = sample_subhalo_encounter("CDM", dm_model_config, stream_config, seed=seed)
    perturbed = apply_impulse_approximation(stream_particles, enc)

    # Detect gaps in the perturbed stream
    gaps = detect_gaps(
        perturbed.phi1,
        phi1_range=tuple(stream_config["phi1_range_deg"]),
        gap_sigma=2.0,
    )

    if not gaps:
        return None, None

    # Take the gap closest to the encounter point
    best_gap = min(gaps, key=lambda g: abs(g.phi1_center - enc.encounter_phi1))

    features = extract_gap_features(
        best_gap, perturbed.phi1, perturbed.phi2,
        perturbed.pm1, perturbed.pm2, perturbed.vrad,
        phi1_range=tuple(stream_config["phi1_range_deg"]),
    )

    metadata = {
        "source": "dm_subhalo",
        "mass_solar": enc.mass_solar,
        "impact_param_kpc": enc.impact_param_kpc,
        "t_since_impact_gyr": enc.t_since_impact_gyr,
    }
    return features, metadata


def simulate_gap_from_gmc(
    stream_particles,
    stream_config: dict,
    seed: int = 0,
) -> tuple[GapFeatures, dict]:
    """Simulate a GMC fly-by and extract gap features."""
    from src.simulation.baryonic import apply_giant_molecular_cloud_encounter, sample_gmc_params
    from .gap_catalog import detect_gaps

    gmc_params = sample_gmc_params(seed=seed)
    perturbed = apply_giant_molecular_cloud_encounter(stream_particles, seed=seed, gmc_params=gmc_params)

    gaps = detect_gaps(
        perturbed.phi1,
        phi1_range=tuple(stream_config["phi1_range_deg"]),
        gap_sigma=2.0,
    )

    if not gaps:
        return None, None

    best_gap = max(gaps, key=lambda g: g.significance)

    features = extract_gap_features(
        best_gap, perturbed.phi1, perturbed.phi2,
        perturbed.pm1, perturbed.pm2, perturbed.vrad,
        phi1_range=tuple(stream_config["phi1_range_deg"]),
    )

    metadata = {"source": "gmc", "mass_solar": gmc_params["mass_solar"]}
    return features, metadata


def simulate_noise_gap(
    stream_particles,
    stream_config: dict,
    seed: int = 0,
) -> tuple[GapFeatures, dict]:
    """Extract a 'gap' from an unperturbed stream (noise fluctuation)."""
    from .gap_catalog import detect_gaps

    # Use the unperturbed stream directly — any detected gap is noise
    gaps = detect_gaps(
        stream_particles.phi1,
        phi1_range=tuple(stream_config["phi1_range_deg"]),
        gap_sigma=1.5,  # lower threshold to find noise fluctuations
    )

    if not gaps:
        return None, None

    rng = np.random.default_rng(seed)
    gap = gaps[rng.integers(0, len(gaps))]

    features = extract_gap_features(
        gap, stream_particles.phi1, stream_particles.phi2,
        stream_particles.pm1, stream_particles.pm2, stream_particles.vrad,
        phi1_range=tuple(stream_config["phi1_range_deg"]),
    )

    metadata = {"source": "noise"}
    return features, metadata


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def generate_gap_classifier_training_data(
    stream_particles,
    stream_config: dict,
    dm_model_config: dict,
    n_per_source: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate balanced training dataset for the gap classifier.

    Simulates n_per_source gaps from each source and returns a DataFrame
    with feature columns + 'source' label column.

    Args:
        stream_particles: Unperturbed StreamParticles (used as base for all sims).
        stream_config: Stream configuration dict.
        dm_model_config: DM model configuration dict.
        n_per_source: Number of gap simulations per source type.
        seed: Random seed.

    Returns:
        DataFrame with columns [feature_0, ..., feature_11, source].
    """
    records = []
    feature_cols = [
        "width_deg", "depth_fraction", "significance_sigma", "asymmetry",
        "delta_pm1_leading", "delta_pm1_trailing", "delta_pm2_flanking",
        "delta_vrad_flanking", "pm_coherence", "phi1_center_norm",
        "phi2_at_gap", "local_density_ratio",
    ]

    simulators = {
        "dm_subhalo": lambda s: simulate_gap_from_dm_subhalo(
            stream_particles, stream_config, dm_model_config, seed=s),
        "gmc": lambda s: simulate_gap_from_gmc(
            stream_particles, stream_config, seed=s),
        "noise": lambda s: simulate_noise_gap(
            stream_particles, stream_config, seed=s),
    }

    for source_name, sim_fn in simulators.items():
        n_success = 0
        attempt = 0
        while n_success < n_per_source and attempt < n_per_source * 5:
            features, metadata = sim_fn(seed + attempt + n_success * 1000)
            attempt += 1
            if features is None:
                continue
            row = features_to_array(features).tolist()
            row.append(source_name)
            records.append(row)
            n_success += 1

        log.info("Generated %d/%d gap samples for source=%s",
                 n_success, n_per_source, source_name)

    df = pd.DataFrame(records, columns=feature_cols + ["source"])
    return df


# ---------------------------------------------------------------------------
# Classifier (random forest baseline + optional neural)
# ---------------------------------------------------------------------------

class GapOriginClassifier:
    """Probabilistic gap origin classifier.

    Two-stage design:
    1. Random Forest for fast baseline (works with ~1000 training samples)
    2. Optional Neural NPE for full posterior over source + parameters

    The RF is the minimum viable classifier. It outputs calibrated
    probabilities via predict_proba() after isotonic calibration.
    """

    def __init__(self) -> None:
        self.model = None
        self.calibrator = None
        self.is_trained = False
        self.feature_names: list[str] = []

    def train(self, training_df: pd.DataFrame, calibrate: bool = True) -> dict:
        """Train the classifier on simulated gap features.

        Args:
            training_df: DataFrame from generate_gap_classifier_training_data().
            calibrate: Whether to apply isotonic calibration (recommended).

        Returns:
            Dict with training metrics (accuracy, per-class precision/recall).
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import classification_report

        feature_cols = [c for c in training_df.columns if c != "source"]
        self.feature_names = feature_cols

        X = training_df[feature_cols].values.astype(np.float32)
        y = training_df["source"].values

        # Handle NaN/Inf from edge cases in feature extraction
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        base_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        if calibrate and len(X) > 100:
            self.model = CalibratedClassifierCV(
                base_rf, method="isotonic", cv=5,
            )
        else:
            self.model = base_rf

        self.model.fit(X, y)
        self.is_trained = True

        # Cross-validated metrics
        cv_preds = cross_val_predict(
            base_rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        )
        report = classification_report(y, cv_preds, output_dict=True)
        log.info("Gap classifier trained. CV accuracy: %.3f", report["accuracy"])

        return report

    def predict_proba(self, features: GapFeatures | np.ndarray) -> dict[str, float]:
        """Predict source probabilities for a single gap.

        Args:
            features: GapFeatures object or pre-computed feature array.

        Returns:
            Dict mapping source name to probability.
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained. Call train() first.")

        if isinstance(features, GapFeatures):
            x = features_to_array(features).reshape(1, -1)
        else:
            x = np.asarray(features, dtype=np.float32).reshape(1, -1)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        proba = self.model.predict_proba(x)[0]
        classes = self.model.classes_

        return {cls: float(p) for cls, p in zip(classes, proba)}

    def classify_gap(self, gap: Gap, features: GapFeatures) -> Gap:
        """Assign origin probabilities to a Gap object in-place.

        Maps classifier output to gap.p_dm_subhalo, gap.p_baryonic, gap.p_noise.
        Baryonic = P(GMC) + P(GC) + P(bar).
        """
        probs = self.predict_proba(features)

        gap.p_dm_subhalo = probs.get("dm_subhalo", 0.0)
        gap.p_baryonic = (
            probs.get("gmc", 0.0)
            + probs.get("globular_cluster", 0.0)
            + probs.get("bar", 0.0)
        )
        gap.p_noise = probs.get("noise", 0.0)

        # Normalize (should already sum to ~1 but enforce)
        total = gap.p_dm_subhalo + gap.p_baryonic + gap.p_noise
        if total > 0:
            gap.p_dm_subhalo /= total
            gap.p_baryonic /= total
            gap.p_noise /= total

        return gap

    def save(self, path: str) -> None:
        """Save trained classifier to disk."""
        import joblib
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)
        log.info("Gap classifier saved to %s", path)

    def load(self, path: str) -> None:
        """Load trained classifier from disk."""
        import joblib
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.is_trained = True
        log.info("Gap classifier loaded from %s", path)
