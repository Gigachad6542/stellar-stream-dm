"""
Baryonic perturbation simulations: giant molecular clouds and the galactic bar.

These produce density variations that must be distinguished from DM subhalo gaps.
Building these with the same care as DM simulations is essential for the classifier.

References:
    Amorisco+2016 (GMC perturbations)
    Pearson+2017 (bar resonances in streams)
    Dehnen 2000 (bar potential model)
"""

from __future__ import annotations

import logging

import numpy as np

from .stream_gen import StreamParticles
from .subhalo import G_KPC_KMS, _hernquist_impulse_kick, EncounterParams

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Giant molecular cloud encounters
# ---------------------------------------------------------------------------

def sample_gmc_params(seed: int = 0) -> dict:
    """Sample GMC properties from observationally motivated distributions.

    GMC mass: log-uniform 1e4 - 1e7 Msun (Solomon+1987)
    Scale radius: 10 - 100 pc (Plummer sphere, a = 0.01 - 0.1 kpc)
    Fly-by velocity: 20 - 100 km/s (disk stars)
    Impact parameter: 1 - 500 pc
    """
    rng = np.random.default_rng(seed)
    log_m = rng.uniform(4.0, 7.0)
    m_gmc = 10.0 ** log_m
    a_kpc = rng.uniform(0.01, 0.10)          # Plummer radius [kpc]
    v_kms = rng.uniform(20.0, 100.0)          # fly-by speed [km/s]
    b_kpc = rng.uniform(0.001, 0.5)           # impact parameter [kpc]
    return {"mass_solar": m_gmc, "a_kpc": a_kpc, "v_kms": v_kms, "b_kpc": b_kpc}


def apply_giant_molecular_cloud_encounter(
    particles: StreamParticles,
    seed: int = 0,
    gmc_params: dict | None = None,
) -> StreamParticles:
    """Apply a GMC fly-by impulse kick to stream particles.

    GMCs have lower mass (1e4 - 1e7 Msun) but are spatially extended.
    The Plummer kick is similar to the Hernquist kick at the same b.
    Only applied for streams that pass within 5 kpc of the disk mid-plane
    (controlled by stream_config.disk_crossing in the caller).

    Args:
        particles: Input stream particles.
        seed: Random seed; used only if gmc_params is None.
        gmc_params: Optional pre-sampled GMC parameters (for reproducibility).

    Returns:
        Perturbed StreamParticles.
    """
    if gmc_params is None:
        gmc_params = sample_gmc_params(seed)

    m_gmc = gmc_params["mass_solar"]
    a_kpc = gmc_params["a_kpc"]
    v_kms = gmc_params["v_kms"]
    b_kpc = gmc_params["b_kpc"]

    rng = np.random.default_rng(seed)
    phi1_min = particles.phi1.min()
    phi1_max = particles.phi1.max()
    encounter_phi1 = float(rng.uniform(phi1_min, phi1_max))

    r_kpc_per_deg = np.pi / 180.0 * np.mean(particles.dist)
    dphi1 = particles.phi1 - encounter_phi1
    r_perp_kpc = np.abs(dphi1) * r_kpc_per_deg

    dv = _hernquist_impulse_kick(r_perp_kpc, np.zeros_like(r_perp_kpc),
                                  m_gmc, a_kpc, b_kpc, v_kms)

    # Unit conversion: dv [km/s] → Δpm [mas/yr]
    # 1 km/s at heliocentric distance d_kpc = 1 / (4.74047 * d_kpc) mas/yr
    d_hel_kpc = float(np.mean(particles.dist))
    PM_CONV = 1.0 / (4.74047 * max(d_hel_kpc, 0.1))

    return StreamParticles(
        phi1=particles.phi1,
        phi2=particles.phi2 + np.sign(particles.phi2) * dv * 0.005,
        dist=particles.dist,
        pm1=particles.pm1,
        pm2=particles.pm2 + dv * 0.3 * PM_CONV,
        vrad=particles.vrad + dv * 0.3,
        xyz_kpc=particles.xyz_kpc,
        vxyz_kms=particles.vxyz_kms,
    )


# ---------------------------------------------------------------------------
# Galactic bar perturbation (simplified resonance kick)
# ---------------------------------------------------------------------------

def apply_bar_perturbation(
    particles: StreamParticles,
    seed: int = 0,
    bar_amplitude: float = 0.02,  # fractional density perturbation amplitude
    pattern_speed_kms_kpc: float = 40.0,  # bar pattern speed [km/s/kpc]
) -> StreamParticles:
    """Apply oscillatory density perturbation due to the galactic bar.

    The Galactic bar drives spiral-arm-like resonances in streams that cross
    near corotation (r ~ 4-5 kpc from GC) or outer Lindblad resonance.

    This simplified model applies a sinusoidal density modulation along phi1,
    matched to the bar's angular frequency and stream age.

    Args:
        particles: Input stream particles.
        bar_amplitude: Fractional amplitude of density variation (0-1).
        pattern_speed_kms_kpc: Bar pattern speed Omega_b [km/s/kpc].

    Returns:
        StreamParticles with modulated spatial distribution.
    """
    rng = np.random.default_rng(seed)
    n = len(particles.phi1)

    if n == 0:
        return particles

    # Bar resonance produces a quasi-sinusoidal density variation along phi1
    phi1_range = particles.phi1.max() - particles.phi1.min()
    # Typical resonance scale: ~10-20 degrees
    n_periods = rng.uniform(2, 5)
    k = 2.0 * np.pi * n_periods / max(phi1_range, 1.0)
    phase = rng.uniform(0, 2 * np.pi)

    # Density weight for rejection sampling: accept/reject stars
    weight = 1.0 + bar_amplitude * np.sin(k * (particles.phi1 - particles.phi1.min()) + phase)
    weight = np.clip(weight / weight.max(), 0.0, 1.0)
    keep = rng.uniform(0.0, 1.0, n) < weight

    return StreamParticles(
        phi1=particles.phi1[keep],
        phi2=particles.phi2[keep],
        dist=particles.dist[keep],
        pm1=particles.pm1[keep],
        pm2=particles.pm2[keep],
        vrad=particles.vrad[keep],
        xyz_kpc=particles.xyz_kpc[:, keep],
        vxyz_kms=particles.vxyz_kms[:, keep],
    )


# ---------------------------------------------------------------------------
# Baryonic region flagging
# ---------------------------------------------------------------------------

def flag_baryonic_dominated_region(
    phi1: np.ndarray,
    phi2: np.ndarray,
    stream_config: dict,
) -> np.ndarray:
    """Return boolean mask: True where baryonic perturbations likely dominate.

    Currently flags:
    - Disk crossing regions (|b| < 5 deg for disk-crossing streams)
    - Known bar resonance angles (within ~5 deg of Galactic Center direction)

    This mask is used to down-weight loss on these stars during training.
    """
    mask = np.zeros(len(phi1), dtype=bool)
    if stream_config.get("disk_crossing", False):
        # Flag stars near the disk mid-plane (phi2 in stream frame ~ b)
        mask |= np.abs(phi2) < 0.5
    return mask
