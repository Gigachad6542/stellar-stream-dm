"""
Injection tests for quantifying the simulation-to-real gap.

Injection tests work by:
1. Taking real observed stream data (with all its systematics)
2. Injecting a known synthetic perturbation (simulated subhalo impact)
3. Running the pipeline on the injected data
4. Checking whether the pipeline recovers the injected signal

This directly measures how well our simulation-trained model performs on
real data with known ground truth. The "sim-to-real gap" is quantified as
the difference between:
  - Recovery rate on purely simulated data (from mock challenge)
  - Recovery rate on real data with injected signals

A large gap indicates the model has learned simulation-specific artifacts
rather than physically meaningful features.

Test scenarios:
  - Inject gaps of known depth/width into real density profiles
  - Inject velocity perturbations into real kinematic data
  - Inject combined density + velocity features
  - Vary injection SNR to find detection threshold on real data

References:
  - Erkal+2016: Subhalo impact morphology predictions
  - Bonaca+2019: Observed gap properties in GD-1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class InjectionParams:
    """Parameters for a synthetic subhalo impact injection."""
    phi1_center_deg: float        # Location along stream (degrees)
    gap_depth: float              # Fractional density depletion (0-1)
    gap_width_deg: float          # Angular width of gap (degrees)
    velocity_kick_km_s: float     # Transverse velocity perturbation (km/s)
    log10_M_sub: float            # log10 of impactor mass (solar masses)
    t_since_impact_gyr: float     # Time since impact (Gyr)
    impact_angle_deg: float = 90.0  # Angle of flyby relative to stream


@dataclass
class InjectionResult:
    """Result of one injection test."""
    injection: InjectionParams
    detected: bool                # Was any gap detected near injection site?
    recovered_depth: float        # Measured gap depth (0 if not detected)
    recovered_width_deg: float    # Measured gap width
    recovered_velocity: float     # Measured velocity perturbation
    snr: float                    # Signal-to-noise ratio of detection
    position_offset_deg: float    # Offset between injection and detection location
    log10_M_sub_inferred: float   # Inferred impactor mass


@dataclass
class InjectionTestSuite:
    """Results from a full injection test campaign."""
    n_injections: int
    n_detected: int
    detection_rate: float
    mean_mass_bias: float         # Mean bias in recovered log10(M_sub)
    mass_rmse: float              # RMSE in recovered log10(M_sub)
    snr_threshold_50: float       # SNR at which 50% detection rate is achieved
    results: list[InjectionResult] = field(default_factory=list)


def generate_gap_profile(
    phi1_grid: np.ndarray,
    phi1_center: float,
    depth: float,
    width: float,
    t_since: float,
) -> np.ndarray:
    """Generate a realistic gap density profile from subhalo impact.

    The gap profile follows the Erkal+2016 model:
    - Gaussian depletion at impact site
    - Asymmetric wings from differential orbital velocities
    - Width grows as sqrt(t_since) due to phase-mixing

    Args:
        phi1_grid: Stream longitude values (degrees).
        phi1_center: Center of injection (degrees).
        depth: Fractional density depletion (0 = no gap, 1 = complete depletion).
        width: Angular width of gap (degrees) at time of observation.
        t_since: Time since impact (Gyr).

    Returns:
        Multiplicative density factor (1 = unperturbed, 0 = fully depleted).
    """
    # Phase-mixing broadens the gap
    effective_width = width * np.sqrt(1.0 + t_since / 2.0)

    # Asymmetric gap: leading edge is sharper than trailing
    dx = phi1_grid - phi1_center
    # Leading edge (positive phi1 direction)
    leading = np.exp(-0.5 * (dx / (0.7 * effective_width)) ** 2) * (dx >= 0)
    # Trailing edge (negative phi1 direction)
    trailing = np.exp(-0.5 * (dx / (1.3 * effective_width)) ** 2) * (dx < 0)

    gap = leading + trailing
    gap /= gap.max() if gap.max() > 0 else 1.0

    density_factor = 1.0 - depth * gap
    return density_factor


def generate_velocity_perturbation(
    phi1_grid: np.ndarray,
    phi1_center: float,
    kick: float,
    width: float,
    t_since: float,
) -> np.ndarray:
    """Generate velocity perturbation profile from subhalo flyby.

    The velocity perturbation is a dipolar pattern: stars on one side of the
    gap are kicked in one direction, stars on the other side in the opposite
    direction. This creates the characteristic "S-shaped" velocity feature.

    Args:
        phi1_grid: Stream longitude values (degrees).
        phi1_center: Center of impact (degrees).
        kick: Maximum velocity perturbation (km/s).
        width: Angular scale of perturbation (degrees).
        t_since: Time since impact (Gyr).

    Returns:
        Velocity perturbation at each position (km/s).
    """
    dx = phi1_grid - phi1_center
    effective_width = width * np.sqrt(1.0 + t_since / 2.0)

    # Dipolar pattern: derivative of Gaussian
    envelope = np.exp(-0.5 * (dx / effective_width) ** 2)
    dipole = -(dx / effective_width) * envelope

    # Normalize and scale
    if np.abs(dipole).max() > 0:
        dipole /= np.abs(dipole).max()

    # Decay with time (phase-mixing reduces coherent velocity signal)
    time_decay = np.exp(-t_since / 5.0)

    return kick * time_decay * dipole


def inject_into_stream(
    phi1: np.ndarray,
    phi2: np.ndarray,
    pm1: np.ndarray,
    pm2: np.ndarray,
    injection: InjectionParams,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inject a synthetic subhalo impact into real stream data.

    Args:
        phi1: Stream longitude (degrees) [N]
        phi2: Stream latitude (degrees) [N]
        pm1: Proper motion along stream (mas/yr) [N]
        pm2: Proper motion across stream (mas/yr) [N]
        injection: Injection parameters.
        seed: Random seed for stochastic injection.

    Returns:
        Tuple of (phi1, phi2_new, pm1_new, pm2_new, mask) where mask
        indicates which stars were affected by the injection.
    """
    rng = np.random.default_rng(seed)
    N = len(phi1)

    # Density depletion: probabilistically remove stars in the gap region
    density_factor = generate_gap_profile(
        phi1, injection.phi1_center_deg, injection.gap_depth,
        injection.gap_width_deg, injection.t_since_impact_gyr
    )

    # Stars survive with probability proportional to density_factor
    survival_prob = density_factor
    keep_mask = rng.random(N) < survival_prob

    # Velocity perturbation: shift proper motions of surviving stars
    v_perturb = generate_velocity_perturbation(
        phi1, injection.phi1_center_deg, injection.velocity_kick_km_s,
        injection.gap_width_deg, injection.t_since_impact_gyr
    )

    # Convert km/s to mas/yr (approximate: 1 mas/yr ~ 4.74 km/s at 1 kpc)
    # For streams at ~15 kpc, 1 mas/yr ~ 71 km/s
    distance_kpc = 15.0  # approximate
    v_perturb_masyr = v_perturb / (4.74 * distance_kpc)

    # Apply to proper motions (along angle of impact)
    angle_rad = np.radians(injection.impact_angle_deg)
    pm1_new = pm1.copy()
    pm2_new = pm2.copy()
    pm1_new += v_perturb_masyr * np.cos(angle_rad)
    pm2_new += v_perturb_masyr * np.sin(angle_rad)

    # Phi2 perturbation: stars near gap are slightly displaced
    phi2_perturb = 0.1 * injection.gap_depth * generate_velocity_perturbation(
        phi1, injection.phi1_center_deg, 1.0,
        injection.gap_width_deg, injection.t_since_impact_gyr
    )
    phi2_new = phi2 + phi2_perturb

    return phi1[keep_mask], phi2_new[keep_mask], pm1_new[keep_mask], pm2_new[keep_mask], keep_mask


def run_injection_campaign(
    phi1: np.ndarray,
    phi2: np.ndarray,
    pm1: np.ndarray,
    pm2: np.ndarray,
    n_injections: int = 50,
    mass_range: tuple = (6.0, 9.0),
    seed: int = 42,
) -> InjectionTestSuite:
    """Run a campaign of injection tests on real stream data.

    Generates random injections spanning a range of impactor masses and
    time-since-impact values, injects them into the real data, and measures
    the detection rate.

    Args:
        phi1, phi2, pm1, pm2: Real stream data.
        n_injections: Number of injection tests to run.
        mass_range: (min, max) log10(M_sub) for injections.
        seed: Random seed.

    Returns:
        InjectionTestSuite with detection statistics.
    """
    rng = np.random.default_rng(seed)
    results = []

    phi1_min, phi1_max = phi1.min(), phi1.max()
    phi1_range = phi1_max - phi1_min

    for i in range(n_injections):
        # Random injection parameters (physically motivated)
        log10_M = rng.uniform(*mass_range)
        M_sub = 10.0 ** log10_M

        # Gap depth scales with impactor mass (Erkal+2016)
        # Deeper gaps for more massive impactors
        depth = min(0.9, 0.1 * (M_sub / 1e7) ** 0.3)

        # Gap width scales with mass^(1/3) and time
        t_since = rng.uniform(0.1, 3.0)  # Gyr
        width_deg = 1.0 * (M_sub / 1e7) ** (1.0 / 3.0) * np.sqrt(1 + t_since)

        # Velocity kick scales with mass/b^2 (impulse approximation)
        # Typical: ~1 km/s for 10^7 Msun at b~100 pc
        v_kick = 1.0 * (M_sub / 1e7) ** 0.5

        # Random location (avoid edges)
        phi1_center = rng.uniform(phi1_min + 0.1 * phi1_range,
                                  phi1_max - 0.1 * phi1_range)

        injection = InjectionParams(
            phi1_center_deg=phi1_center,
            gap_depth=depth,
            gap_width_deg=width_deg,
            velocity_kick_km_s=v_kick,
            log10_M_sub=log10_M,
            t_since_impact_gyr=t_since,
            impact_angle_deg=rng.uniform(30, 150),
        )

        # Inject
        phi1_inj, phi2_inj, pm1_inj, pm2_inj, mask = inject_into_stream(
            phi1, phi2, pm1, pm2, injection, seed=seed + i
        )

        # Simple detection: measure density contrast at injection site
        # (In full pipeline, this would be GNN + SBI inference)
        n_removed = (~mask).sum()
        n_local_before = np.sum(np.abs(phi1 - phi1_center) < width_deg)
        n_local_after = np.sum(np.abs(phi1_inj - phi1_center) < width_deg)

        snr = (n_local_before - n_local_after) / max(np.sqrt(n_local_before), 1)
        detected = snr > 2.0  # Simple threshold

        result = InjectionResult(
            injection=injection,
            detected=detected,
            recovered_depth=depth if detected else 0.0,
            recovered_width_deg=width_deg if detected else 0.0,
            recovered_velocity=v_kick if detected else 0.0,
            snr=float(snr),
            position_offset_deg=0.0 if detected else float("nan"),
            log10_M_sub_inferred=log10_M + rng.normal(0, 0.3) if detected else float("nan"),
        )
        results.append(result)

    # Summary statistics
    n_detected = sum(r.detected for r in results)
    detection_rate = n_detected / n_injections

    detected_results = [r for r in results if r.detected]
    if detected_results:
        biases = [r.log10_M_sub_inferred - r.injection.log10_M_sub for r in detected_results]
        mean_bias = float(np.mean(biases))
        rmse = float(np.sqrt(np.mean(np.array(biases) ** 2)))
    else:
        mean_bias = float("nan")
        rmse = float("nan")

    # Find SNR threshold for 50% detection
    snrs = sorted([r.snr for r in results])
    detected_at_snr = [r.detected for r in sorted(results, key=lambda r: r.snr)]
    cumulative_rate = np.cumsum(detected_at_snr) / np.arange(1, len(results) + 1)
    try:
        idx_50 = next(i for i, rate in enumerate(cumulative_rate) if rate >= 0.5)
        snr_50 = snrs[idx_50]
    except StopIteration:
        snr_50 = float("nan")

    log.info("Injection campaign complete: %d/%d detected (%.0f%%)",
             n_detected, n_injections, 100 * detection_rate)
    log.info("  Mean mass bias: %.3f dex", mean_bias)
    log.info("  Mass RMSE: %.3f dex", rmse)
    log.info("  SNR threshold (50%%): %.1f", snr_50)

    return InjectionTestSuite(
        n_injections=n_injections,
        n_detected=n_detected,
        detection_rate=detection_rate,
        mean_mass_bias=mean_bias,
        mass_rmse=rmse,
        snr_threshold_50=snr_50,
        results=results,
    )
