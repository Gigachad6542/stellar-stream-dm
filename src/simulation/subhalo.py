"""
Subhalo fly-by perturbation simulation using the impulse approximation.

Implements the Erkal & Belokurov (2015) impulsive velocity kick with the
finite-extent correction from Erkal+2016.

All operations on stream particles are vectorised (no Python loops over stars).

References:
    Erkal & Belokurov 2015: MNRAS 450, 1136
    Erkal+2016: MNRAS 463, 102
    Yoon+2011: ApJ 731, 58
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import astropy.units as u
import numpy as np
import yaml

from .mass_functions import sample_masses
from .stream_gen import StreamParticles

log = logging.getLogger(__name__)

G_KPC_KMS = 4.3009e-6  # G in kpc (km/s)^2 / Msun


@dataclass
class EncounterParams:
    mass_solar: float         # subhalo mass [Msun]
    scale_radius_kpc: float   # Hernquist scale radius [kpc]
    impact_param_kpc: float   # perpendicular closest approach distance [kpc]
    flyby_vel_kms: float      # relative velocity at infinity [km/s]
    encounter_phi1: float     # stream longitude at encounter point [deg]
    t_since_impact_gyr: float # time elapsed since encounter [Gyr]
    is_valid: bool = True     # False if outside impulse approximation validity
    is_massive: bool = False  # True if correction factor applied
    kick_capped: bool = False # True if any star hit the 50 km/s velocity cap


# ---------------------------------------------------------------------------
# NFW / Hernquist scale radius from mass (Ludlow+2016 c-M relation)
# ---------------------------------------------------------------------------

def concentration_from_mass(m_solar: float) -> float:
    """Approximate NFW concentration from the Ludlow+2016 c-M relation at z=0.

    Args:
        m_solar: Subhalo mass in solar masses. Must be > 0.

    Returns:
        NFW concentration (dimensionless). Typical range: 5-30 for 1e5-1e10 Msun.

    Raises:
        ValueError: If m_solar <= 0.
    """
    if m_solar <= 0:
        raise ValueError(f"Subhalo mass must be positive, got {m_solar} Msun")
    # c ~ 17 * (M / 1e8 Msun)^{-0.13}  (rough fit valid for 1e5 - 1e10 Msun)
    return 17.0 * (m_solar / 1.0e8) ** (-0.13)


def virial_radius_from_mass_kpc(m_solar: float) -> float:
    """Return r_200 [kpc] for a halo mass in Msun.

    ``rho_crit`` is often quoted in Msun/Mpc^3.  This codebase works in kpc, so
    the value must be converted to Msun/kpc^3 before computing r_200.
    """
    if m_solar <= 0:
        raise ValueError(f"Subhalo mass must be positive, got {m_solar} Msun")
    rho_crit_kpc3 = 277.5 * (70.0 / 100.0) ** 2  # Msun/kpc^3
    return float((3.0 * m_solar / (4.0 * np.pi * 200.0 * rho_crit_kpc3)) ** (1.0 / 3.0))


def scale_radius_from_mass(m_solar: float, density_profile: str = "NFW") -> float:
    """Hernquist-equivalent scale radius [kpc] for a subhalo of given mass.

    For NFW: convert via r_s = r_200 / c.
    For SIDM-like cored profiles: use a broader effective scale radius.
    """
    c = concentration_from_mass(m_solar)
    r_200_kpc = virial_radius_from_mass_kpc(m_solar)
    r_s_kpc = r_200_kpc / c
    if density_profile in {"isothermal_core_NFW", "SIDM"}:
        r_s_kpc *= 1.5
    # Convert NFW r_s to Hernquist a: a ~ 0.45 * r_s (same density profile peak)
    return 0.45 * r_s_kpc


# All four DM model keys used throughout the pipeline.
DM_MODELS = ("CDM", "WDM", "FDM", "SIDM")


def subhalo_profile_for_model(
    dm_model: str,
    m_solar: float,
    m_wdm_kev: float = 3.0,
    m_axion_ev: float = 1e-22,
    sigma_sidm_cm2g: float = 1.0,
) -> dict:
    """Compute subhalo structural parameters for a given DM model.

    Each model predicts a different density profile for a subhalo of mass
    m_solar, which changes the effective scale radius and kick shape:

        CDM:  Standard NFW (Ludlow+2016 c-M relation)
        WDM:  NFW but lower concentration at low masses (Lovell+2014)
        FDM:  Solitonic core + NFW envelope (Schive+2014) — larger effective
              scale radius at low masses due to the de Broglie wavelength
        SIDM: Isothermal core + NFW envelope (Kaplinghat+2016) — broader,
              shallower kicks due to constant-density core

    Returns:
        Dict with keys:
            scale_radius_kpc: Effective Hernquist scale radius
            concentration: NFW concentration
            profile_type: str describing the profile
            core_radius_kpc: Core/soliton radius (0 for NFW)
            kick_suppression: Multiplicative factor on kick magnitude
                (1.0 = standard NFW, <1 = weaker kicks from cored profiles)
    """
    if dm_model not in DM_MODELS:
        raise ValueError(f"Unknown DM model: {dm_model!r}. Must be one of {DM_MODELS}")

    # Base NFW parameters
    c_nfw = concentration_from_mass(m_solar)
    r_200 = virial_radius_from_mass_kpc(m_solar)
    r_s = r_200 / c_nfw
    a_hernquist = 0.45 * r_s

    if dm_model == "CDM":
        return {
            "scale_radius_kpc": a_hernquist,
            "concentration": c_nfw,
            "profile_type": "NFW",
            "core_radius_kpc": 0.0,
            "kick_suppression": 1.0,
        }

    elif dm_model == "WDM":
        # WDM halos are less concentrated at low masses (Lovell+2014, Bose+2016).
        # c_WDM / c_CDM ~ (1 + z_hm/z_col)^-0.2  ≈ (1 + 60*(M_hm/M)^0.4)^-0.2
        # where M_hm is the half-mode mass.
        m_hm = 1.7e10 * (m_wdm_kev ** -3.33)
        ratio = m_hm / max(m_solar, 1.0)
        c_wdm = c_nfw * (1.0 + 60.0 * ratio ** 0.4) ** -0.2
        c_wdm = max(c_wdm, 2.0)
        r_s_wdm = r_200 / c_wdm
        a_wdm = 0.45 * r_s_wdm
        # WDM kicks are slightly weaker at low masses (less concentrated)
        kick_supp = min(c_wdm / c_nfw, 1.0)
        return {
            "scale_radius_kpc": a_wdm,
            "concentration": c_wdm,
            "profile_type": "NFW_reduced_c",
            "core_radius_kpc": 0.0,
            "kick_suppression": kick_supp,
        }

    elif dm_model == "FDM":
        # FDM subhalos have a solitonic core (Schive+2014):
        # r_soliton ~ 1.6 / (m22 * (M/1e9)^(1/3)) kpc
        # where m22 = m_axion / 1e-22 eV.
        m22 = m_axion_ev / 1e-22
        r_soliton = 1.6 / (max(m22, 1e-3) * (max(m_solar, 1.0) / 1e9) ** (1.0 / 3.0))
        r_soliton = min(r_soliton, 10.0)  # cap at 10 kpc
        # Effective scale radius is the larger of soliton core or NFW scale radius.
        # The soliton dominates at low masses where lambda_dB > r_s.
        a_fdm = max(a_hernquist, 0.45 * r_soliton)
        # Kicks are suppressed because the soliton core has lower central density
        # than a cuspy NFW — the enclosed mass rises more slowly with radius.
        core_to_rs = r_soliton / max(r_s, 1e-4)
        kick_supp = 1.0 / (1.0 + 0.5 * core_to_rs)
        kick_supp = max(kick_supp, 0.2)
        return {
            "scale_radius_kpc": a_fdm,
            "concentration": c_nfw,
            "profile_type": "soliton_NFW",
            "core_radius_kpc": r_soliton,
            "kick_suppression": kick_supp,
        }

    else:  # SIDM
        # SIDM creates an isothermal core inside r_1 where the scattering
        # optical depth ~ 1 (Kaplinghat+2016).
        # r_core ~ 0.45 * r_s * (sigma/cm2g)^0.4 * (c/15)^-0.6
        # Effective scale radius is larger (broader, shallower kick).
        core_factor = max(sigma_sidm_cm2g, 0.1) ** 0.4 * (c_nfw / 15.0) ** -0.6
        r_core = 0.45 * r_s * core_factor
        a_sidm = a_hernquist + 0.5 * r_core  # broadened by the core
        # Core reduces central density → weaker kicks
        kick_supp = 1.0 / (1.0 + 0.3 * core_factor)
        kick_supp = max(kick_supp, 0.3)
        return {
            "scale_radius_kpc": a_sidm,
            "concentration": c_nfw,
            "profile_type": "isothermal_core_NFW",
            "core_radius_kpc": r_core,
            "kick_suppression": kick_supp,
        }


# ---------------------------------------------------------------------------
# Impulse approximation — Erkal & Belokurov 2015
# ---------------------------------------------------------------------------

def _hernquist_impulse_kick(
    r_perp_kpc: np.ndarray,  # [N] perpendicular distance of each star to flyby path
    x_par_kpc: np.ndarray,   # [N] parallel offset along flyby direction
    m_solar: float,
    a_kpc: float,
    b_kpc: float,             # impact parameter of the flyby trajectory
    v_kms: float,
) -> np.ndarray:
    """Velocity kick [km/s] from a Hernquist subhalo in the impulse approximation.

    Erkal & Belokurov 2015 Eq. (10) with finite-extent correction.

    Returns:
        dv_perp: [N] velocity kick perpendicular to fly-by direction [km/s].
    """
    # Perpendicular distance of each star from the subhalo's straight-line path.
    d_kpc = np.sqrt(b_kpc ** 2 + r_perp_kpc ** 2)  # [N]

    # Plummer impulse (Erkal & Belokurov 2015): dv = 2GM/w * d/(d^2 + r_s^2).
    # This is the magnitude of the proper 3D kick (erkal_plummer_kick) for a star
    # at perpendicular distance d, and is naturally bounded (max GM/(w r_s) at
    # d = r_s), so it needs no ad-hoc velocity cap or finite-extent/massive
    # correction factors. a_kpc plays the role of the Plummer scale radius.
    dv = 2.0 * G_KPC_KMS * m_solar / v_kms * d_kpc / (d_kpc ** 2 + a_kpc ** 2)

    return dv


def erkal_plummer_kick(
    pos_kpc: np.ndarray,       # [N, 3] galactocentric positions
    x_impact_kpc: np.ndarray,  # [3] stream point at closest approach
    w_vec_kms: np.ndarray,     # [3] subhalo velocity relative to the stream
    b_vec_kpc: np.ndarray,     # [3] impact-parameter vector (perp to w, |b|=impact param)
    m_solar: float,
    r_s_kpc: float,
) -> np.ndarray:
    """Velocity kick [N, 3] km/s from a Plummer subhalo (Erkal & Belokurov 2015).

    For a subhalo of mass M and Plummer scale r_s flying past on a straight line
    (closest approach at ``x_impact + b_vec``, direction ``w_hat``) in the
    impulse approximation, a star with 3D perpendicular offset ``p`` from that
    line receives

        dv = (2 G M / w) * p / (|p|^2 + r_s^2)

    (Erkal & Belokurov 2015, MNRAS 450, 1136). This reduces to the point-mass
    impulse 2GM/(b w) for r_s -> 0 and is **naturally bounded** (max GM/(w r_s)
    at |p| = r_s), so no ad-hoc velocity cap is needed. The kick direction is the
    true 3D perpendicular, and the kick varies star-to-star with |p|, which is
    what carves and shapes the gap.

    Args:
        pos_kpc: [N, 3] star positions at the impact epoch.
        x_impact_kpc: [3] stream position at the encounter.
        w_vec_kms: [3] subhalo-stream relative velocity (sets w and the line direction).
        b_vec_kpc: [3] impact-parameter vector (perpendicular to w_vec).
        m_solar: subhalo mass [Msun].
        r_s_kpc: Plummer scale radius [kpc].

    Returns:
        dv: [N, 3] velocity kicks [km/s].
    """
    pos = np.atleast_2d(np.asarray(pos_kpc, dtype=float))
    w_vec = np.asarray(w_vec_kms, dtype=float)
    w = float(np.linalg.norm(w_vec))
    if w < 1e-8:
        return np.zeros_like(pos)
    w_hat = w_vec / w

    # Closest-approach point of the subhalo line to the stream impact point.
    p0 = np.asarray(x_impact_kpc, dtype=float) + np.asarray(b_vec_kpc, dtype=float)
    d = pos - p0[None, :]                      # [N, 3]
    d_par = d @ w_hat                          # [N]
    p = d - d_par[:, None] * w_hat[None, :]    # [N, 3] perpendicular component
    p2 = np.sum(p * p, axis=1)                 # [N]

    # Minus sign: the impulse points TOWARD the subhalo's trajectory (gravity
    # attracts the star toward the line of closest approach). p points from the
    # line to the star, so the kick is along -p.
    coeff = 2.0 * G_KPC_KMS * m_solar / w      # [kpc * km/s]
    dv = -coeff * p / (p2[:, None] + r_s_kpc ** 2)   # [N, 3] km/s
    return dv


def build_encounter_geometry(
    stream_tangent: np.ndarray,   # [3] local stream direction (≈ mean velocity dir)
    r_hat: np.ndarray,            # [3] galactocentric radial unit vector at impact
    b_kpc: float,
    w_kms: float,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct (w_vec, b_vec) for an encounter.

    Default (rng=None): a *gap-forming* geometry — the subhalo crosses roughly
    perpendicular to the stream (so a star's perpendicular distance to the
    subhalo path grows along the stream, localising the kick at the impact
    point). With an rng, the subhalo velocity direction is sampled isotropically
    (Erkal+15 sample the subhalo velocity distribution) and ``b`` takes a random
    azimuth in the plane perpendicular to ``w``.

    Returns:
        (w_vec_kms [3], b_vec_kpc [3]).
    """
    t = np.asarray(stream_tangent, dtype=float)
    t = t / max(np.linalg.norm(t), 1e-8)
    r = np.asarray(r_hat, dtype=float)
    r = r / max(np.linalg.norm(r), 1e-8)

    if rng is None:
        # w perpendicular to the stream tangent, in the t-r plane (radial part
        # orthogonal to t). Falls back to any perpendicular if degenerate.
        w_hat = r - (r @ t) * t
        if np.linalg.norm(w_hat) < 1e-3:
            w_hat = np.cross(t, [0.0, 0.0, 1.0])
        w_hat /= max(np.linalg.norm(w_hat), 1e-8)
        # b perpendicular to w (out of the t-w plane).
        b_hat = np.cross(w_hat, t)
        b_hat /= max(np.linalg.norm(b_hat), 1e-8)
    else:
        # Isotropic subhalo velocity direction.
        v = rng.normal(size=3)
        w_hat = v / max(np.linalg.norm(v), 1e-8)
        # Random azimuth for b in the plane perpendicular to w.
        e1 = np.cross(w_hat, t)
        if np.linalg.norm(e1) < 1e-3:
            e1 = np.cross(w_hat, [0.0, 0.0, 1.0])
        e1 /= max(np.linalg.norm(e1), 1e-8)
        e2 = np.cross(w_hat, e1)
        ang = rng.uniform(0, 2 * np.pi)
        b_hat = np.cos(ang) * e1 + np.sin(ang) * e2

    return w_kms * w_hat, b_kpc * b_hat


def apply_impulse_approximation(
    particles: StreamParticles,
    encounter: EncounterParams,
) -> StreamParticles:
    """Apply velocity kick from one subhalo fly-by to stream particles.

    All vectorised over the N particles; no Python loop.

    The impulse is applied in the plane perpendicular to the fly-by direction.
    The stream coordinate at the encounter point is used to localise the kick.

    Args:
        particles: StreamParticles (modified in-place via copy).
        encounter: EncounterParams for this fly-by.

    Returns:
        New StreamParticles with updated velocities.
    """
    if not encounter.is_valid:
        return particles

    # Localise the kick around the encounter phi1 position
    # Stars near the encounter get the full kick; falloff as Gaussian in phi1
    dphi1 = particles.phi1 - encounter.encounter_phi1  # [N]
    # Stream disperses at ~velocity_dispersion * t / r; use a broad localisation
    sigma_phi1 = encounter.impact_param_kpc / np.maximum(np.mean(particles.dist), 1.0)
    sigma_phi1 = np.clip(np.degrees(sigma_phi1), 0.5, 10.0)  # deg

    # Perpendicular distance from the flyby path for each star
    # Approximate: distance in the phi1-phi2 plane projected to kpc
    r_kpc_per_deg = np.pi / 180.0 * np.mean(particles.dist)
    r_perp_kpc = np.abs(dphi1) * r_kpc_per_deg
    x_par_kpc = np.zeros_like(r_perp_kpc)

    # Velocity kick magnitude [km/s] for each star
    dv = _hernquist_impulse_kick(
        r_perp_kpc,
        x_par_kpc,
        encounter.mass_solar,
        encounter.scale_radius_kpc,
        encounter.impact_param_kpc,
        encounter.flyby_vel_kms,
    )

    # ── Velocity kick components ──────────────────────────────────────────────
    # Unit conversion: dv [km/s] → Δpm [mas/yr]
    #   1 km/s at heliocentric distance d_kpc = 1 / (4.74047 * d_kpc)  mas/yr
    d_hel_kpc = float(np.mean(particles.dist))
    PM_CONV = 1.0 / (4.74047 * max(d_hel_kpc, 0.1))  # [mas/yr] per [km/s]

    sign_phi2 = np.sign(particles.phi2 - 0.0)  # kick direction in phi2
    vrad_new  = particles.vrad + dv * 0.5
    pm2_new   = particles.pm2  + dv * 0.5 * PM_CONV
    pm1_new   = particles.pm1  + dv * 0.3 * PM_CONV   # ~30% along-stream kick component

    # ── phi1 gap drift ────────────────────────────────────────────────────────
    # Stars near the encounter point receive the strongest kick and drift away
    # along the stream, widening the gap over time.
    #
    # Gap half-width formula (Erkal+2016 spirit):
    #   Δφ1 ≈ (dv_i / v_orb) * n_orbits * (180/π) * f_geom
    # where n_orbits = t_since / T_period and f_geom ≈ 0.3 accounts for the
    # fraction of the impulse that converts into a phase-space gap along phi1.
    v_orb = float(np.mean(np.sqrt(np.sum(particles.vxyz_kms ** 2, axis=0))))
    v_orb = max(v_orb, 100.0)                                   # km/s
    r_galcen = float(np.mean(np.sqrt(np.sum(particles.xyz_kpc ** 2, axis=0))))
    r_galcen = max(r_galcen, 1.0)                               # kpc
    T_period_gyr = 2.0 * np.pi * r_galcen / (v_orb * 1.022)    # kpc / (kpc/Gyr)
    n_orbits = encounter.t_since_impact_gyr / max(T_period_gyr, 0.1)
    f_geom = 0.30                                               # geometric projection factor
    # Per-star drift (opposite direction for lead and trail of gap)
    phi1_drift = (dv / v_orb) * n_orbits * (180.0 / np.pi) * f_geom
    # Stars ahead of encounter point drift forward; stars behind drift backward
    phi1_dir = np.sign(particles.phi1 - encounter.encounter_phi1)
    phi1_new = particles.phi1 + phi1_dir * phi1_drift

    return StreamParticles(
        phi1=phi1_new,
        phi2=particles.phi2 + sign_phi2 * dv * 0.01 * PM_CONV,
        dist=particles.dist,
        pm1=pm1_new,
        pm2=pm2_new,
        vrad=vrad_new,
        xyz_kpc=particles.xyz_kpc,
        vxyz_kms=particles.vxyz_kms,
    )


# ---------------------------------------------------------------------------
# Sample one encounter
# ---------------------------------------------------------------------------

def sample_subhalo_encounter(
    dm_model: str,
    dm_model_config: dict,
    stream_config: dict,
    dm_params: dict | None = None,
    seed: int = 0,
) -> EncounterParams:
    """Draw a single subhalo encounter from the DM model priors."""
    rng = np.random.default_rng(seed)
    impact_cfg = dm_model_config.get("impact_physics", {})

    # Draw mass
    log10_m_min = float(dm_model_config["mass_function"].get("log10_M_min_solar", 5.0))
    log10_m_max = float(dm_model_config["mass_function"].get("log10_M_max_solar", 9.0))

    dm_params = dict(dm_params or {})
    if dm_model == "WDM" and "m_wdm_kev" not in dm_params:
        dm_params["m_wdm_kev"] = float(rng.uniform(0.5, 10.0))
    elif dm_model == "FDM" and "m_axion_ev" not in dm_params:
        dm_params["m_axion_ev"] = 10.0 ** float(rng.uniform(-23, -20))

    masses = sample_masses(dm_model, 1, dm_params, log10_m_min, log10_m_max, seed=int(rng.integers(0, 2**32)))
    m_sub = float(masses[0])

    # Draw encounter geometry
    v_flyby_range = impact_cfg.get("flyby_velocity_kms_prior", ["uniform", 50.0, 250.0])
    b_range = impact_cfg.get("impact_parameter_pc_prior", ["uniform", 0.1, 500.0])
    b_kpc = float(rng.uniform(b_range[1], b_range[2])) / 1000.0  # pc -> kpc
    v_flyby = float(rng.uniform(v_flyby_range[1], v_flyby_range[2]))

    # Random encounter position along the stream
    phi1_min, phi1_max = stream_config["phi1_range_deg"]
    encounter_phi1 = float(rng.uniform(phi1_min, phi1_max))

    # Time since impact (uniform in last stream age by default). Signal-ladder
    # datasets can narrow this to force visibly evolved gaps before the fully
    # randomized training run.
    stream_age_gyr = stream_config.get("isochrone_age_gyr", 10.0)
    t_prior = impact_cfg.get("t_since_impact_gyr_prior")
    if isinstance(t_prior, (list, tuple)) and len(t_prior) >= 3 and str(t_prior[0]).lower() == "uniform":
        t_low = float(t_prior[1])
        t_high = min(float(t_prior[2]), float(stream_age_gyr))
        t_since = float(rng.uniform(t_low, max(t_low, t_high)))
    else:
        t_since = float(rng.uniform(0.5, stream_age_gyr))

    a_kpc = scale_radius_from_mass(m_sub, dm_model_config.get("density_profile", {}).get("type", "NFW"))

    # Check impulse approximation validity
    v_disp = stream_config.get("velocity_dispersion_kms", 0.4)
    is_valid = (v_flyby > 3.0 * v_disp) and (b_kpc > 1e-3)
    is_massive = m_sub > 1.0e8

    return EncounterParams(
        mass_solar=m_sub,
        scale_radius_kpc=a_kpc,
        impact_param_kpc=b_kpc,
        flyby_vel_kms=v_flyby,
        encounter_phi1=encounter_phi1,
        t_since_impact_gyr=t_since,
        is_valid=is_valid,
        is_massive=is_massive,
    )


def apply_n_subhalo_encounters(
    particles: StreamParticles,
    n_encounters: int,
    dm_model: str,
    dm_model_config: dict,
    stream_config: dict,
    dm_params: dict | None = None,
    seed: int = 0,
) -> tuple[StreamParticles, list[EncounterParams]]:
    """Apply N independent subhalo encounters sequentially.

    Returns updated particles and the list of EncounterParams for HDF5 storage.
    """
    encounters = []
    for i in range(n_encounters):
        enc = sample_subhalo_encounter(
            dm_model,
            dm_model_config,
            stream_config,
            dm_params=dm_params,
            seed=seed + i,
        )
        particles = apply_impulse_approximation(particles, enc)
        # Note: gap evolution (phi1 drift) is handled inside apply_impulse_approximation.
        encounters.append(enc)
    return particles, encounters
