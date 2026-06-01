"""
Full orbit-integrated timeline evolution for the forward model.

This module implements the physically correct approach:
    1. Generate stream particles (Fardal spray) up to the impact epoch
    2. Integrate particles forward to the impact epoch (t = -t_impact)
    3. Apply the subhalo velocity kick in galactocentric coordinates
    4. Integrate all particles forward from the impact epoch to t = 0
    5. Convert to stream-frame observables

This is computationally expensive (~10-30s per candidate depending on n_stars
and t_impact) but produces the correct phase-space evolution: gap widening,
caustic formation, and stream fanning are all captured via real orbit integration
rather than analytic approximations.

Key differences from apply_impulse_approximation:
    - Gap morphology evolves via actual orbit integration, not a drift formula
    - Stars on different energies phase-mix correctly over Gyr timescales
    - The velocity kick is applied in galactocentric 3D (not stream-aligned)
    - The stream structure at t=0 reflects the full non-linear evolution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import astropy.units as u
import numpy as np
import yaml
from galpy.orbit import Orbit
from galpy.potential import vcirc

from ..simulation.stream_gen import (
    StreamParticles,
    _galactocentric_to_stream_coords,
    _load_stream_config,
    _pos_vel_to_orbit,
    set_progenitor_ic_track6d,
)
from ..simulation.subhalo import (
    EncounterParams,
    G_KPC_KMS,
    build_encounter_geometry,
    erkal_plummer_kick,
    scale_radius_from_mass,
)

log = logging.getLogger(__name__)

_RO: float = 8.0     # kpc
_VO: float = 220.0   # km/s


# ---------------------------------------------------------------------------
# Core: generate stream at impact epoch, apply kick, evolve to present
# ---------------------------------------------------------------------------

def generate_perturbed_stream_evolved(
    stream_name: str,
    potential: list,
    encounter: EncounterParams,
    n_stars: int = 2000,
    seed: int = 0,
    config_path: str = "config/streams.yaml",
    progenitor_ic: Optional[dict] = None,
    cache_dir: str = "data/processed",
    mws=None,
    n_steps_back: int = 80,
) -> StreamParticles:
    """Generate a stream with a forced encounter, evolving through the MW potential.

    Algorithm
    ---------
    1. Integrate progenitor orbit backward for the full disruption age.
    2. Release test particles (Fardal spray) at uniformly-spaced times.
    3. Integrate particles forward from their release time to t = -t_impact.
    4. Apply the subhalo velocity kick at t = -t_impact.
    5. Integrate all particles forward from t = -t_impact to t = 0.
    6. Convert to stream-frame observables.

    This captures the full non-linear phase-space evolution of the gap:
    differential orbital precession, gap widening, caustic formation.

    Args:
        stream_name: Key in config/streams.yaml (e.g. "GD1").
        potential: galpy MWPotential list.
        encounter: Forced encounter parameters (mass, position, time, etc).
        n_stars: Number of stream particles to simulate.
        seed: Random seed for reproducibility.
        config_path: Path to streams.yaml.
        progenitor_ic: Optional pre-computed progenitor ICs.
        cache_dir: Directory for cached ICs.
        mws: Pre-loaded galstreams.MWStreams instance.
        n_steps_back: Number of release-time bins for particle spray.

    Returns:
        StreamParticles at t=0 with the encounter gap evolved.
    """
    if mws is None:
        import galstreams
        mws = galstreams.MWStreams(verbose=False)

    rng = np.random.default_rng(seed)
    sc = _load_stream_config(stream_name, config_path)

    stream_age_gyr = sc.get("disruption_age_gyr", sc.get("isochrone_age_gyr", 10.0))
    t_impact = encounter.t_since_impact_gyr

    if t_impact > stream_age_gyr:
        log.warning("t_impact (%.1f Gyr) > stream age (%.1f Gyr); clamping",
                    t_impact, stream_age_gyr)
        t_impact = stream_age_gyr * 0.95

    if progenitor_ic is None:
        progenitor_ic = set_progenitor_ic_track6d(
            stream_name, config_path, cache_dir, mws=mws,
        )

    pos0 = np.array(progenitor_ic["pos_kpc"])   # [3]
    vel0 = np.array(progenitor_ic["vel_kms"])   # [3]

    # ---- Step 1: Integrate progenitor orbit backward ----
    t_back_gyr = np.linspace(0.0, -stream_age_gyr, n_steps_back)
    t_back = t_back_gyr * u.Gyr

    prog_orbit = _pos_vel_to_orbit(pos0, vel0)
    prog_orbit.integrate(t_back, potential, method="leapfrog_c", progressbar=False)

    prog_R   = prog_orbit.R(t_back, use_physical=True)
    prog_vR  = prog_orbit.vR(t_back, use_physical=True)
    prog_vT  = prog_orbit.vT(t_back, use_physical=True)
    prog_z   = prog_orbit.z(t_back, use_physical=True)
    prog_vz  = prog_orbit.vz(t_back, use_physical=True)
    prog_phi = prog_orbit.phi(t_back, use_physical=False)

    prog_x = prog_R * np.cos(prog_phi)
    prog_y = prog_R * np.sin(prog_phi)
    prog_vx = prog_vR * np.cos(prog_phi) - prog_vT * np.sin(prog_phi)
    prog_vy = prog_vR * np.sin(prog_phi) + prog_vT * np.cos(prog_phi)

    prog_pos = np.stack([prog_x, prog_y, prog_z], axis=-1)       # [N_steps, 3]
    prog_vel = np.stack([prog_vx, prog_vy, prog_vz], axis=-1)     # [N_steps, 3]
    r_mag = np.linalg.norm(prog_pos, axis=-1, keepdims=True)      # [N_steps, 1]
    r_hat = prog_pos / np.clip(r_mag, 1e-6, None)

    v_c_orbit = np.atleast_1d(np.asarray(
        vcirc(potential, r_mag[:, 0] * u.kpc, use_physical=True, ro=_RO, vo=_VO)
    ))

    # ---- Step 2: Sample release times and build spray ICs ----
    n_lead = n_stars // 2
    n_trail = n_stars - n_lead
    k_v = 0.3
    m_prog_msun = sc.get("prog_mass_solar", 2e4)

    idx_lead = rng.integers(0, n_steps_back, n_lead)
    idx_trail = rng.integers(0, n_steps_back, n_trail)

    def _make_perturbed_ic(idx_arr, sign):
        p_pos = prog_pos[idx_arr]
        p_vel = prog_vel[idx_arr]
        p_r = r_mag[idx_arr, 0]
        p_rh = r_hat[idx_arr]
        G_kpc = 4.3009e-6
        v_c_arr = v_c_orbit[idx_arr]
        m_enc = v_c_arr**2 * p_r / G_kpc
        r_tidal = p_r * (m_prog_msun / (3.0 * np.clip(m_enc, 1.0, None))) ** (1.0 / 3.0)
        pos_new = p_pos + sign * r_tidal[:, np.newaxis] * p_rh
        omega = v_c_arr / p_r
        sigma_v = k_v * omega * r_tidal
        dv = rng.normal(0, sigma_v[:, np.newaxis], size=(len(idx_arr), 3))
        vel_new = p_vel + sign * dv
        t_release = t_back_gyr[idx_arr]
        return pos_new, vel_new, t_release

    pos_lead, vel_lead, t_lead = _make_perturbed_ic(idx_lead, +1.0)
    pos_trail, vel_trail, t_trail = _make_perturbed_ic(idx_trail, -1.0)

    pos_all_spray = np.concatenate([pos_lead, pos_trail], axis=0)    # [N, 3]
    vel_all_spray = np.concatenate([vel_lead, vel_trail], axis=0)    # [N, 3]
    t_release_all = np.concatenate([t_lead, t_trail])                # [N], negative Gyr

    # ---- Step 3-5: Split particles by impact timing ----
    # Critical physics: particles released AFTER the impact epoch (t_release > -t_impact)
    # did not exist when the subhalo flew by. They must NOT receive a velocity kick.
    #
    # Split into two groups:
    #   pre_impact:  released before impact (t_release <= -t_impact) -> kick + evolve
    #   post_impact: released after impact (t_release > -t_impact)  -> evolve directly
    pre_impact_mask = t_release_all <= -t_impact + 1e-6  # small tolerance
    n_pre = pre_impact_mask.sum()
    n_post = (~pre_impact_mask).sum()
    log.info("  Particle split: %d pre-impact (will be kicked), %d post-impact (no kick)",
             n_pre, n_post)

    track = mws[sc["galstreams_key"]]
    frame = track.stream_frame

    # --- Pre-impact particles: integrate to impact epoch, kick, then to present ---
    if n_pre > 0:
        pos_pre = pos_all_spray[pre_impact_mask]
        vel_pre = vel_all_spray[pre_impact_mask]
        t_pre = t_release_all[pre_impact_mask]

        log.info("  Integrating %d pre-impact particles to impact epoch (t=%.2f Gyr ago)...",
                 n_pre, t_impact)
        pos_at_impact, vel_at_impact = _integrate_particles_to_epoch(
            pos_pre, vel_pre, t_pre,
            target_epoch_gyr=-t_impact,
            potential=potential,
        )

        # Convert to stream frame to determine proximity to encounter phi1
        pos_at_impact_T = pos_at_impact.T  # [3, N]
        vel_at_impact_T = vel_at_impact.T  # [3, N]
        phi1_at_impact, _, _, _, _, _ = _galactocentric_to_stream_coords(
            pos_at_impact_T, vel_at_impact_T, frame,
        )

        # Apply the 3D velocity kick
        log.info("  Applying velocity kick (M=%.1e Msun, phi1=%.1f deg)...",
                 encounter.mass_solar, encounter.encounter_phi1)
        vel_kicked = _apply_3d_velocity_kick(
            pos_at_impact, vel_at_impact,
            phi1_at_impact, encounter,
        )

        # Integrate kicked particles from impact epoch to present
        log.info("  Integrating kicked particles forward to present...")
        t_start_pre = np.full(n_pre, -t_impact)
        pos_final_pre, vel_final_pre = _integrate_particles_to_epoch(
            pos_at_impact, vel_kicked, t_start_pre,
            target_epoch_gyr=0.0,
            potential=potential,
        )
    else:
        pos_final_pre = np.empty((0, 3))
        vel_final_pre = np.empty((0, 3))

    # --- Post-impact particles: integrate directly from release to present ---
    if n_post > 0:
        pos_post = pos_all_spray[~pre_impact_mask]
        vel_post = vel_all_spray[~pre_impact_mask]
        t_post = t_release_all[~pre_impact_mask]

        log.info("  Integrating %d post-impact particles directly to present...", n_post)
        pos_final_post, vel_final_post = _integrate_particles_to_epoch(
            pos_post, vel_post, t_post,
            target_epoch_gyr=0.0,
            potential=potential,
        )
    else:
        pos_final_post = np.empty((0, 3))
        vel_final_post = np.empty((0, 3))

    # Combine pre-impact (kicked) and post-impact (unkicked) particles
    pos_final = np.concatenate([pos_final_pre, pos_final_post], axis=0)
    vel_final = np.concatenate([vel_final_pre, vel_final_post], axis=0)

    # ---- Step 6: Convert to stream-frame observables ----
    pos_final_T = pos_final.T   # [3, N]
    vel_final_T = vel_final.T   # [3, N]

    phi1, phi2, dist, pm1, pm2, vrad = _galactocentric_to_stream_coords(
        pos_final_T, vel_final_T, frame,
    )

    # Selection cuts
    phi1_min, phi1_max = sc["phi1_range_deg"]
    mask = (phi1 >= phi1_min) & (phi1 <= phi1_max)
    mask &= np.abs(phi2) < sc["phi2_selection_deg"]
    # Also reject non-finite results (particles that went to R=0, etc)
    mask &= np.isfinite(phi1) & np.isfinite(phi2)

    log.info("  Final stream: %d / %d particles in phi1/phi2 selection",
             mask.sum(), len(mask))

    return StreamParticles(
        phi1=phi1[mask],
        phi2=phi2[mask],
        dist=dist[mask],
        pm1=pm1[mask],
        pm2=pm2[mask],
        vrad=vrad[mask],
        xyz_kpc=pos_final_T[:, mask],
        vxyz_kms=vel_final_T[:, mask],
    )


# ---------------------------------------------------------------------------
# Orbit integration helpers
# ---------------------------------------------------------------------------

def _integrate_particles_to_epoch(
    pos_arr: np.ndarray,       # [N, 3] galactocentric xyz [kpc]
    vel_arr: np.ndarray,       # [N, 3] galactocentric vxvyvz [km/s]
    t_start_arr: np.ndarray,   # [N] start epoch [Gyr, negative = past]
    target_epoch_gyr: float,   # target epoch [Gyr], e.g. -2.0 or 0.0
    potential: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate particles from individual start times to a common target epoch.

    Uses galpy vectorised Orbit grouped by unique start time for efficiency.
    Handles the same safety guards as stream_gen._integrate_particles.

    Args:
        pos_arr: [N, 3] positions at start times.
        vel_arr: [N, 3] velocities at start times.
        t_start_arr: [N] start epochs (negative = past, 0 = present).
        target_epoch_gyr: The epoch to integrate to.
        potential: galpy potential list.

    Returns:
        (pos_final, vel_final): [N, 3] arrays at target_epoch_gyr.
    """
    _R_MIN_KPC = 0.5
    _V_MAX_KMS = 2000.0

    n = len(pos_arr)
    results_pos = np.empty((n, 3))
    results_vel = np.empty((n, 3))

    # Group by start time for batched integration
    # Round to 4 decimal places to group nearby times
    t_rounded = np.round(t_start_arr, 4)
    unique_t, inverse = np.unique(t_rounded, return_inverse=True)

    for gi, t_start in enumerate(unique_t):
        grp = np.where(inverse == gi)[0]
        dt = target_epoch_gyr - t_start  # duration to integrate (positive = forward)

        # Skip if already at target
        if abs(dt) < 1e-6:
            results_pos[grp] = pos_arr[grp]
            results_vel[grp] = vel_arr[grp]
            continue

        x = pos_arr[grp, 0]; y = pos_arr[grp, 1]; z = pos_arr[grp, 2]
        vx = vel_arr[grp, 0]; vy = vel_arr[grp, 1]; vz = vel_arr[grp, 2]

        R = np.sqrt(x**2 + y**2)
        R = np.maximum(R, _R_MIN_KPC)

        phi = np.arctan2(y, x)
        vR = (x * vx + y * vy) / R
        vT = (x * vy - y * vx) / R

        # Non-finite guard (CRITICAL): NaN/Inf phase-space coords crash galpy's
        # dop853_c C integrator with a SIGSEGV *before* any Python exception or
        # the finite-check below can fire (and np.maximum / `> V_MAX` silently
        # pass NaN through). A few particles can acquire NaN from a divergent
        # kick or upstream step. Park any non-finite particle at R=500 kpc, v=0:
        # it integrates safely and is removed by the phi1/phi2 selection cut.
        bad = ~(np.isfinite(R) & np.isfinite(phi) & np.isfinite(vR)
                & np.isfinite(vT) & np.isfinite(z) & np.isfinite(vz))
        if bad.any():
            R = np.where(bad, 500.0, R)
            phi = np.where(bad, 0.0, phi)
            vR = np.where(bad, 0.0, vR)
            vT = np.where(bad, 0.0, vT)
            z = np.where(bad, 0.0, z)
            vz = np.where(bad, 0.0, vz)

        # Speed guard
        speed = np.sqrt(vR**2 + vT**2 + vz**2)
        too_fast = speed > _V_MAX_KMS
        if too_fast.any():
            scale = np.where(too_fast, _V_MAX_KMS / np.maximum(speed, 1.0), 1.0)
            vR = vR * scale; vT = vT * scale; vz = vz * scale

        # Time grid: from t_start to target_epoch
        n_int = max(8, int(abs(dt) * 40))  # ~25 Myr resolution
        t_grid = np.linspace(float(t_start), float(target_epoch_gyr), n_int) * u.Gyr

        orb = Orbit(
            vxvv=[R * u.kpc, vR * u.km / u.s, vT * u.km / u.s,
                  z * u.kpc, vz * u.km / u.s, phi * u.rad],
            ro=_RO, vo=_VO,
        )

        try:
            orb.integrate(t_grid, potential, method="dop853_c", progressbar=False)
        except Exception:
            orb = Orbit(
                vxvv=[R * u.kpc, vR * u.km / u.s, vT * u.km / u.s,
                      z * u.kpc, vz * u.km / u.s, phi * u.rad],
                ro=_RO, vo=_VO,
            )
            orb.integrate(t_grid, potential, method="leapfrog_c", progressbar=False)

        if not np.isfinite(orb.orbit).all():
            # Fallback to leapfrog
            orb = Orbit(
                vxvv=[R * u.kpc, vR * u.km / u.s, vT * u.km / u.s,
                      z * u.kpc, vz * u.km / u.s, phi * u.rad],
                ro=_RO, vo=_VO,
            )
            orb.integrate(t_grid, potential, method="leapfrog_c", progressbar=False)
            if not np.isfinite(orb.orbit).all():
                results_pos[grp] = np.array([500.0, 0.0, 0.0])
                results_vel[grp] = 0.0
                continue

        t_end = t_grid[-1]
        results_pos[grp, 0] = np.atleast_1d(np.asarray(orb.x(t_end, use_physical=True)))
        results_pos[grp, 1] = np.atleast_1d(np.asarray(orb.y(t_end, use_physical=True)))
        results_pos[grp, 2] = np.atleast_1d(np.asarray(orb.z(t_end, use_physical=True)))
        results_vel[grp, 0] = np.atleast_1d(np.asarray(orb.vx(t_end, use_physical=True)))
        results_vel[grp, 1] = np.atleast_1d(np.asarray(orb.vy(t_end, use_physical=True)))
        results_vel[grp, 2] = np.atleast_1d(np.asarray(orb.vz(t_end, use_physical=True)))

    return results_pos, results_vel


# ---------------------------------------------------------------------------
# 3D velocity kick in galactocentric coordinates
# ---------------------------------------------------------------------------

def _apply_3d_velocity_kick(
    pos: np.ndarray,           # [N, 3] galactocentric positions at impact epoch [kpc]
    vel: np.ndarray,           # [N, 3] galactocentric velocities at impact epoch [km/s]
    phi1_stream: np.ndarray,   # [N] stream longitude at impact epoch [deg]
    encounter: EncounterParams,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply the Erkal & Belokurov 2015 Plummer impulse in galactocentric 3D.

    The subhalo flies past on a straight line at closest-approach point
    ``x_impact + b_vec`` with relative velocity ``w_vec``; each star receives
    ``dv = (2 G M / w) p / (|p|^2 + r_s^2)`` where ``p`` is its true 3D
    perpendicular offset to that line (see ``subhalo.erkal_plummer_kick``). The
    kick direction and per-star magnitude come straight from the geometry — no
    Gaussian-in-phi1 localisation, no fixed kick axis, no 50 km/s cap, and no
    ad-hoc along-stream fraction (all of which the previous heuristic used).

    Args:
        pos/vel: [N, 3] positions/velocities at the impact epoch.
        phi1_stream: [N] stream phi1 (to locate the impact point on the stream).
        encounter: EncounterParams (mass, scale_radius, impact_param, flyby_vel, phi1).
        rng: optional generator; if given, the encounter geometry (subhalo
            velocity direction + b azimuth) is sampled isotropically, else a
            gap-forming perpendicular geometry is used.

    Returns:
        vel_kicked: [N, 3] velocities with the impulse applied.
    """
    # Locate the impact point on the stream: centroid of stars near encounter phi1.
    dphi1 = phi1_stream - encounter.encounter_phi1
    near = np.abs(dphi1) < 5.0
    if near.sum() < 5:
        near = np.abs(dphi1) < 15.0
    if near.sum() < 5:
        near = np.ones(len(pos), dtype=bool)

    x_impact = pos[near].mean(axis=0)
    # Stream spatial tangent ~ mean velocity direction of nearby stars.
    v_mean = vel[near].mean(axis=0)
    tangent = v_mean / max(np.linalg.norm(v_mean), 1e-8)
    r_hat = x_impact / max(np.linalg.norm(x_impact), 1e-8)

    w_vec, b_vec = build_encounter_geometry(
        tangent, r_hat, encounter.impact_param_kpc, encounter.flyby_vel_kms, rng=rng,
    )
    dv = erkal_plummer_kick(
        pos, x_impact, w_vec, b_vec, encounter.mass_solar, encounter.scale_radius_kpc,
    )

    dv_mag = np.linalg.norm(dv, axis=1)
    log.debug("  Erkal kick: %d/%d stars > 1 km/s (max %.1f km/s)",
              int((dv_mag > 1.0).sum()), len(pos), float(dv_mag.max()))
    return vel + dv


# ---------------------------------------------------------------------------
# Multi-encounter: sequential subhalo impacts
# ---------------------------------------------------------------------------

def generate_perturbed_stream_multi_evolved(
    stream_name: str,
    potential: list,
    encounters: list[EncounterParams],
    n_stars: int = 2000,
    seed: int = 0,
    config_path: str = "config/streams.yaml",
    progenitor_ic: Optional[dict] = None,
    cache_dir: str = "data/processed",
    mws=None,
    n_steps_back: int = 80,
) -> StreamParticles:
    """Generate a stream with MULTIPLE sequential encounters, fully orbit-integrated.

    Algorithm
    ---------
    1. Generate stream particles (Fardal spray) for the full disruption age.
    2. Sort encounters chronologically (oldest first, i.e. largest t_since first).
    3. For each encounter epoch (oldest → newest):
       a. Identify particles that existed at this epoch (released before it).
       b. Integrate those particles from their current epoch to this encounter epoch.
       c. Apply the subhalo velocity kick.
       d. Update each particle's "current epoch" to this encounter time.
    4. After all encounters: integrate all particles from their current epoch to t=0.
    5. Convert to stream-frame observables.

    This captures the cumulative effect of multiple subhalo impacts: earlier
    encounters create gaps that subsequent impacts further modify, and the full
    non-linear evolution (gap widening, phase mixing) is captured between each
    encounter epoch.

    Args:
        stream_name: Key in config/streams.yaml (e.g. "GD1").
        potential: galpy MWPotential list.
        encounters: List of EncounterParams (one per subhalo impact).
        n_stars: Number of stream particles to simulate.
        seed: Random seed for reproducibility.
        config_path: Path to streams.yaml.
        progenitor_ic: Optional pre-computed progenitor ICs.
        cache_dir: Directory for cached ICs.
        mws: Pre-loaded galstreams.MWStreams instance.
        n_steps_back: Number of release-time bins for particle spray.

    Returns:
        StreamParticles at t=0 with all encounter gaps evolved.
    """
    if not encounters:
        raise ValueError("encounters list must contain at least one encounter")

    if mws is None:
        import galstreams
        mws = galstreams.MWStreams(verbose=False)

    rng = np.random.default_rng(seed)
    sc = _load_stream_config(stream_name, config_path)

    stream_age_gyr = sc.get("disruption_age_gyr", sc.get("isochrone_age_gyr", 10.0))

    # Sort encounters by t_since_impact_gyr DESCENDING (oldest/earliest first)
    encounters_sorted = sorted(encounters, key=lambda e: e.t_since_impact_gyr, reverse=True)

    # Validate: no encounter older than the stream
    for enc in encounters_sorted:
        if enc.t_since_impact_gyr > stream_age_gyr:
            log.warning("Encounter t_since=%.1f Gyr > stream age=%.1f; clamping",
                        enc.t_since_impact_gyr, stream_age_gyr)
            enc.t_since_impact_gyr = stream_age_gyr * 0.95

    log.info("Multi-encounter: %d impacts on %s (oldest %.1f Gyr, newest %.1f Gyr ago)",
             len(encounters_sorted), stream_name,
             encounters_sorted[0].t_since_impact_gyr,
             encounters_sorted[-1].t_since_impact_gyr)

    if progenitor_ic is None:
        progenitor_ic = set_progenitor_ic_track6d(
            stream_name, config_path, cache_dir, mws=mws,
        )

    pos0 = np.array(progenitor_ic["pos_kpc"])
    vel0 = np.array(progenitor_ic["vel_kms"])

    # ---- Step 1: Integrate progenitor orbit backward & build spray ICs ----
    t_back_gyr = np.linspace(0.0, -stream_age_gyr, n_steps_back)
    t_back = t_back_gyr * u.Gyr

    prog_orbit = _pos_vel_to_orbit(pos0, vel0)
    prog_orbit.integrate(t_back, potential, method="leapfrog_c", progressbar=False)

    prog_R   = prog_orbit.R(t_back, use_physical=True)
    prog_vR  = prog_orbit.vR(t_back, use_physical=True)
    prog_vT  = prog_orbit.vT(t_back, use_physical=True)
    prog_z   = prog_orbit.z(t_back, use_physical=True)
    prog_vz  = prog_orbit.vz(t_back, use_physical=True)
    prog_phi = prog_orbit.phi(t_back, use_physical=False)

    prog_x = prog_R * np.cos(prog_phi)
    prog_y = prog_R * np.sin(prog_phi)
    prog_vx = prog_vR * np.cos(prog_phi) - prog_vT * np.sin(prog_phi)
    prog_vy = prog_vR * np.sin(prog_phi) + prog_vT * np.cos(prog_phi)

    prog_pos = np.stack([prog_x, prog_y, prog_z], axis=-1)
    prog_vel = np.stack([prog_vx, prog_vy, prog_vz], axis=-1)
    r_mag = np.linalg.norm(prog_pos, axis=-1, keepdims=True)
    r_hat = prog_pos / np.clip(r_mag, 1e-6, None)

    v_c_orbit = np.atleast_1d(np.asarray(
        vcirc(potential, r_mag[:, 0] * u.kpc, use_physical=True, ro=_RO, vo=_VO)
    ))

    # Spray particles
    n_lead = n_stars // 2
    n_trail = n_stars - n_lead
    k_v = 0.3
    m_prog_msun = sc.get("prog_mass_solar", 2e4)

    idx_lead = rng.integers(0, n_steps_back, n_lead)
    idx_trail = rng.integers(0, n_steps_back, n_trail)

    def _make_perturbed_ic(idx_arr, sign):
        p_pos = prog_pos[idx_arr]
        p_vel = prog_vel[idx_arr]
        p_r = r_mag[idx_arr, 0]
        p_rh = r_hat[idx_arr]
        G_kpc = 4.3009e-6
        v_c_arr = v_c_orbit[idx_arr]
        m_enc = v_c_arr**2 * p_r / G_kpc
        r_tidal = p_r * (m_prog_msun / (3.0 * np.clip(m_enc, 1.0, None))) ** (1.0 / 3.0)
        pos_new = p_pos + sign * r_tidal[:, np.newaxis] * p_rh
        omega = v_c_arr / p_r
        sigma_v = k_v * omega * r_tidal
        dv = rng.normal(0, sigma_v[:, np.newaxis], size=(len(idx_arr), 3))
        vel_new = p_vel + sign * dv
        t_release = t_back_gyr[idx_arr]
        return pos_new, vel_new, t_release

    pos_lead, vel_lead, t_lead = _make_perturbed_ic(idx_lead, +1.0)
    pos_trail, vel_trail, t_trail = _make_perturbed_ic(idx_trail, -1.0)

    pos_all = np.concatenate([pos_lead, pos_trail], axis=0)    # [N, 3]
    vel_all = np.concatenate([vel_lead, vel_trail], axis=0)    # [N, 3]
    t_release_all = np.concatenate([t_lead, t_trail])          # [N], negative Gyr

    # ---- Step 2-4: Sequential encounters ----
    # Track the "current epoch" for each particle (where it has been integrated to)
    # Initially, each particle is at its release epoch.
    current_epoch = t_release_all.copy()  # [N], negative values
    pos_current = pos_all.copy()
    vel_current = vel_all.copy()

    track = mws[sc["galstreams_key"]]
    frame = track.stream_frame

    for enc_idx, encounter in enumerate(encounters_sorted):
        t_enc = -encounter.t_since_impact_gyr  # negative epoch (past)

        # Which particles existed at this epoch? (released before or at this time)
        existed_mask = t_release_all <= t_enc + 1e-6
        n_exist = existed_mask.sum()

        if n_exist == 0:
            log.info("  Encounter %d/%d (M=%.1e, t=%.1f Gyr): no particles existed yet, skipping",
                     enc_idx + 1, len(encounters_sorted),
                     encounter.mass_solar, encounter.t_since_impact_gyr)
            continue

        log.info("  Encounter %d/%d: M=%.1e Msun, t=%.1f Gyr ago, phi1=%.1f "
                 "(%d/%d particles pre-exist)",
                 enc_idx + 1, len(encounters_sorted),
                 encounter.mass_solar, encounter.t_since_impact_gyr,
                 encounter.encounter_phi1, n_exist, len(pos_current))

        # Integrate existed particles from their current epoch to this encounter epoch
        pos_existed = pos_current[existed_mask]
        vel_existed = vel_current[existed_mask]
        epoch_existed = current_epoch[existed_mask]

        # Only integrate particles not already at this epoch
        needs_integration = np.abs(epoch_existed - t_enc) > 1e-6
        if needs_integration.any():
            idx_need = np.where(existed_mask)[0][needs_integration]
            pos_at_enc, vel_at_enc = _integrate_particles_to_epoch(
                pos_current[idx_need],
                vel_current[idx_need],
                current_epoch[idx_need],
                target_epoch_gyr=t_enc,
                potential=potential,
            )
            pos_current[idx_need] = pos_at_enc
            vel_current[idx_need] = vel_at_enc
            current_epoch[idx_need] = t_enc

        # Also update the rest that were already at this epoch
        current_epoch[existed_mask] = t_enc

        # Get stream-frame phi1 for proximity determination
        pos_enc_T = pos_current[existed_mask].T
        vel_enc_T = vel_current[existed_mask].T
        phi1_at_enc, _, _, _, _, _ = _galactocentric_to_stream_coords(
            pos_enc_T, vel_enc_T, frame,
        )

        # Apply 3D velocity kick
        vel_kicked = _apply_3d_velocity_kick(
            pos_current[existed_mask],
            vel_current[existed_mask],
            phi1_at_enc,
            encounter,
        )
        vel_current[existed_mask] = vel_kicked

        log.info("    Kick applied to %d particles", n_exist)

    # ---- Step 5: Integrate all particles from their current epoch to present ----
    log.info("  Integrating all %d particles to present (t=0)...", len(pos_current))
    needs_final = np.abs(current_epoch - 0.0) > 1e-6
    if needs_final.any():
        idx_final = np.where(needs_final)[0]
        pos_final_part, vel_final_part = _integrate_particles_to_epoch(
            pos_current[idx_final],
            vel_current[idx_final],
            current_epoch[idx_final],
            target_epoch_gyr=0.0,
            potential=potential,
        )
        pos_current[idx_final] = pos_final_part
        vel_current[idx_final] = vel_final_part

    # ---- Step 6: Convert to stream-frame observables ----
    pos_final_T = pos_current.T
    vel_final_T = vel_current.T

    phi1, phi2, dist, pm1, pm2, vrad = _galactocentric_to_stream_coords(
        pos_final_T, vel_final_T, frame,
    )

    # Selection cuts
    phi1_min, phi1_max = sc["phi1_range_deg"]
    mask = (phi1 >= phi1_min) & (phi1 <= phi1_max)
    mask &= np.abs(phi2) < sc["phi2_selection_deg"]
    mask &= np.isfinite(phi1) & np.isfinite(phi2)

    log.info("  Final stream: %d / %d particles in phi1/phi2 selection",
             mask.sum(), len(mask))

    return StreamParticles(
        phi1=phi1[mask],
        phi2=phi2[mask],
        dist=dist[mask],
        pm1=pm1[mask],
        pm2=pm2[mask],
        vrad=vrad[mask],
        xyz_kpc=pos_final_T[:, mask],
        vxyz_kms=vel_final_T[:, mask],
    )
