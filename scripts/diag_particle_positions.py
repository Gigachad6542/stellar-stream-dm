"""
Diagnostic: show WHERE spray particles end up (before selection cuts).

Determines whether particles are lost to phi1 range, phi2 range, or both.
"""
from __future__ import annotations
import sys, logging, json
from pathlib import Path
import numpy as np
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.potentials import _RO, _VO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# Fix DLL paths
if sys.platform == "win32":
    import os
    env_dir = Path(sys.executable).parent
    lib_bin = env_dir / "Library" / "bin"
    if lib_bin.exists():
        os.add_dll_directory(str(lib_bin))
    import galpy
    for subdir in ["", "potential", "orbit", "actionAngle", "util"]:
        d = Path(galpy.__file__).parent / subdir if subdir else Path(galpy.__file__).parent
        if d.exists():
            os.add_dll_directory(str(d))
    os.add_dll_directory(str(Path(galpy.__file__).parent.parent))

import yaml, galstreams
import astropy.coordinates as coord
from galpy.orbit import Orbit
from galpy.potential import vcirc
from src.simulation.potentials import get_mw_potential
from src.simulation.stream_gen import (
    _pos_vel_to_orbit, _galactocentric_to_stream_phi12,
    _galactocentric_to_stream_coords,
)

config_path = str(ROOT / "config" / "streams.yaml")
with open(config_path) as f:
    sc = yaml.safe_load(f)["streams"]["GD1"]

potential = get_mw_potential(config_path)
mws = galstreams.MWStreams(verbose=False)
track = mws[sc["galstreams_key"]]
sf = track.stream_frame

# Load the cached IC
ic_file = ROOT / "data" / "processed" / "GD1_progenitor_ic.json"
with open(ic_file) as f:
    ic = json.load(f)
pos0 = np.array(ic["pos_kpc"])
vel0 = np.array(ic["vel_kms"])

log.info("IC: pos=%s  vel=%s  rms=%.3f", pos0, vel0, ic.get("_rms_deg", -1))
log.info("|pos|=%.1f kpc  |vel|=%.1f km/s", np.linalg.norm(pos0), np.linalg.norm(vel0))

# -- 1. Check progenitor orbit phi1/phi2 at t=0 --
p1_0, p2_0 = _galactocentric_to_stream_phi12(pos0.reshape(3, 1), sf)
log.info("Progenitor at t=0: phi1=%.2f  phi2=%.2f deg", p1_0[0], p2_0[0])

# -- 2. Integrate progenitor backward for disruption_age --
stream_age_gyr = sc.get("disruption_age_gyr", 3.0)
n_steps = 100
t_back = np.linspace(0.0, -stream_age_gyr, n_steps) * u.Gyr

prog_orbit = _pos_vel_to_orbit(pos0, vel0)
prog_orbit.integrate(t_back, potential, method="leapfrog_c", progressbar=False)

# Progenitor orbit in stream frame
xs = np.atleast_1d(prog_orbit.x(t_back, use_physical=True))
ys = np.atleast_1d(prog_orbit.y(t_back, use_physical=True))
zs = np.atleast_1d(prog_orbit.z(t_back, use_physical=True))
orbit_pos = np.stack([xs, ys, zs])  # [3, N]
orb_phi1, orb_phi2 = _galactocentric_to_stream_phi12(orbit_pos, sf)

phi1_min, phi1_max = sc["phi1_range_deg"]
in_phi1 = (orb_phi1 >= phi1_min) & (orb_phi1 <= phi1_max)
log.info("\nProgenitor orbit (n_steps=%d, T=%.1f Gyr):", n_steps, stream_age_gyr)
log.info("  phi1 range: [%.1f, %.1f]", orb_phi1.min(), orb_phi1.max())
log.info("  phi2 range: [%.1f, %.1f]", orb_phi2.min(), orb_phi2.max())
log.info("  Points in phi1 range [%d,%d]: %d/%d", phi1_min, phi1_max, in_phi1.sum(), len(orb_phi1))
if in_phi1.sum() > 0:
    log.info("  phi2 in phi1 range: [%.1f, %.1f]", orb_phi2[in_phi1].min(), orb_phi2[in_phi1].max())

# Print orbit at key times
for i in [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]:
    t_gyr = float(t_back[i].to(u.Gyr).value)
    R = np.sqrt(xs[i]**2 + ys[i]**2 + zs[i]**2)
    log.info("  t=%.2f Gyr: phi1=%.1f phi2=%.1f R=%.1f kpc", t_gyr, orb_phi1[i], orb_phi2[i], R)

# -- 3. Now do a mini spray and check raw positions --
log.info("\n--- Mini particle spray (500 particles, no selection) ---")
rng = np.random.default_rng(0)
n_stars = 500
n_lead = n_stars // 2
n_trail = n_stars - n_lead

t_back_gyr = np.linspace(0.0, -stream_age_gyr, n_steps)

# Get orbit quantities
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

# Circular velocity along orbit
v_c_orbit = np.atleast_1d(np.asarray(
    vcirc(potential, r_mag[:, 0] * u.kpc, use_physical=True, ro=_RO, vo=_VO)
))

m_prog = sc.get("prog_mass_solar", 5000)
G_kpc = 4.3009e-6
k_v = 0.3

idx_lead = rng.integers(0, n_steps, n_lead)
idx_trail = rng.integers(0, n_steps, n_trail)

def make_perturbed_ic(idx_arr, sign):
    p_pos = prog_pos[idx_arr]
    p_vel = prog_vel[idx_arr]
    p_r = r_mag[idx_arr, 0]
    p_rh = r_hat[idx_arr]
    v_c_arr = v_c_orbit[idx_arr]
    m_enc = v_c_arr**2 * p_r / G_kpc
    r_tidal = p_r * (m_prog / (3.0 * np.clip(m_enc, 1.0, None))) ** (1.0 / 3.0)
    pos_new = p_pos + sign * r_tidal[:, np.newaxis] * p_rh
    omega = v_c_arr / p_r
    sigma_v = k_v * omega * r_tidal
    dv = rng.normal(0, sigma_v[:, np.newaxis], size=(len(idx_arr), 3))
    vel_new = p_vel + sign * dv
    t_release = t_back_gyr[idx_arr]
    return pos_new, vel_new, t_release

pos_lead, vel_lead, t_lead = make_perturbed_ic(idx_lead, +1.0)
pos_trail, vel_trail, t_trail = make_perturbed_ic(idx_trail, -1.0)

# Check tidal radius and velocity kicks
log.info("Tidal radius stats:")
for idx_arr, label in [(idx_lead, "lead"), (idx_trail, "trail")]:
    p_r = r_mag[idx_arr, 0]
    v_c_arr = v_c_orbit[idx_arr]
    m_enc = v_c_arr**2 * p_r / G_kpc
    r_tidal = p_r * (m_prog / (3.0 * np.clip(m_enc, 1.0, None))) ** (1.0 / 3.0)
    omega = v_c_arr / p_r
    sigma_v = k_v * omega * r_tidal
    log.info("  %s: r_tidal=[%.4f, %.4f] kpc  sigma_v=[%.4f, %.4f] km/s",
             label, r_tidal.min(), r_tidal.max(), sigma_v.min(), sigma_v.max())

# Integrate forward to t=0
log.info("Integrating particles forward...")

def integrate_particles_simple(pos_arr, vel_arr, t_release_arr):
    n = len(pos_arr)
    results_pos = np.empty((n, 3))
    results_vel = np.empty((n, 3))
    unique_t, inverse = np.unique(t_release_arr, return_inverse=True)
    n_nan = 0
    n_escape = 0
    for gi, t_rel in enumerate(unique_t):
        grp = np.where(inverse == gi)[0]
        if abs(float(t_rel)) < 1e-9:
            results_pos[grp] = pos_arr[grp]
            results_vel[grp] = vel_arr[grp]
            continue
        x = pos_arr[grp, 0]; y = pos_arr[grp, 1]; z = pos_arr[grp, 2]
        vx = vel_arr[grp, 0]; vy = vel_arr[grp, 1]; vz = vel_arr[grp, 2]
        R = np.sqrt(x**2 + y**2)
        R = np.maximum(R, 0.5)
        phi = np.arctan2(y, x)
        vR = (x * vx + y * vy) / R
        vT = (x * vy - y * vx) / R
        n_int = max(8, int(abs(float(t_rel)) * 40))
        t_fwd = np.linspace(float(t_rel), 0.0, n_int) * u.Gyr
        orb = Orbit(
            vxvv=[R * u.kpc, vR * u.km/u.s, vT * u.km/u.s,
                  z * u.kpc, vz * u.km/u.s, phi * u.rad],
            ro=_RO, vo=_VO,
        )
        orb.integrate(t_fwd, potential, method="dop853_c", progressbar=False)
        if not np.isfinite(orb.orbit).all():
            orb = Orbit(
                vxvv=[R * u.kpc, vR * u.km/u.s, vT * u.km/u.s,
                      z * u.kpc, vz * u.km/u.s, phi * u.rad],
                ro=_RO, vo=_VO,
            )
            orb.integrate(t_fwd, potential, method="leapfrog_c", progressbar=False)
            if not np.isfinite(orb.orbit).all():
                results_pos[grp] = 500.0
                results_vel[grp] = 0.0
                n_nan += len(grp)
                continue
        t0 = t_fwd[-1]
        results_pos[grp, 0] = np.atleast_1d(np.asarray(orb.x(t0, use_physical=True)))
        results_pos[grp, 1] = np.atleast_1d(np.asarray(orb.y(t0, use_physical=True)))
        results_pos[grp, 2] = np.atleast_1d(np.asarray(orb.z(t0, use_physical=True)))
        results_vel[grp, 0] = np.atleast_1d(np.asarray(orb.vx(t0, use_physical=True)))
        results_vel[grp, 1] = np.atleast_1d(np.asarray(orb.vy(t0, use_physical=True)))
        results_vel[grp, 2] = np.atleast_1d(np.asarray(orb.vz(t0, use_physical=True)))
        # Check for escapees
        r_final = np.sqrt(results_pos[grp, 0]**2 + results_pos[grp, 1]**2 + results_pos[grp, 2]**2)
        n_escape += (r_final > 200).sum()
    log.info("  NaN orbits: %d  Escaped (r>200 kpc): %d", n_nan, n_escape)
    return results_pos, results_vel

pos_lead_f, vel_lead_f = integrate_particles_simple(pos_lead, vel_lead, t_lead)
pos_trail_f, vel_trail_f = integrate_particles_simple(pos_trail, vel_trail, t_trail)

pos_all = np.concatenate([pos_lead_f, pos_trail_f], axis=0).T  # [3, N]
vel_all = np.concatenate([vel_lead_f, vel_trail_f], axis=0).T  # [3, N]

# Convert to stream frame
phi1, phi2, dist, pm1, pm2, vrad = _galactocentric_to_stream_coords(pos_all, vel_all, sf)

# Print raw distributions
log.info("\n--- RAW particle distributions (no cuts) ---")
log.info("phi1: min=%.1f  max=%.1f  median=%.1f", np.nanmin(phi1), np.nanmax(phi1), np.nanmedian(phi1))
log.info("phi2: min=%.1f  max=%.1f  median=%.1f", np.nanmin(phi2), np.nanmax(phi2), np.nanmedian(phi2))
log.info("dist: min=%.1f  max=%.1f  median=%.1f kpc", np.nanmin(dist), np.nanmax(dist), np.nanmedian(dist))

r_all = np.sqrt(pos_all[0]**2 + pos_all[1]**2 + pos_all[2]**2)
log.info("r_gc: min=%.1f  max=%.1f  median=%.1f kpc", r_all.min(), r_all.max(), np.median(r_all))

# Check cuts independently
phi1_ok = (phi1 >= phi1_min) & (phi1 <= phi1_max)
phi2_ok = np.abs(phi2) < sc["phi2_selection_deg"]
both_ok = phi1_ok & phi2_ok

log.info("\n--- Selection cut breakdown ---")
log.info("phi1 in [%.0f, %.0f]: %d/%d (%.1f%%)", phi1_min, phi1_max, phi1_ok.sum(), len(phi1), 100*phi1_ok.sum()/len(phi1))
log.info("|phi2| < %.1f: %d/%d (%.1f%%)", sc["phi2_selection_deg"], phi2_ok.sum(), len(phi2), 100*phi2_ok.sum()/len(phi2))
log.info("Both: %d/%d (%.1f%%)", both_ok.sum(), len(phi1), 100*both_ok.sum()/len(phi1))

if phi1_ok.sum() > 0:
    log.info("\nFor particles IN phi1 range:")
    log.info("  phi2: min=%.1f  max=%.1f  median=%.1f", phi2[phi1_ok].min(), phi2[phi1_ok].max(), np.median(phi2[phi1_ok]))
    log.info("  |phi2| < 5: %d/%d", (phi1_ok & phi2_ok).sum(), phi1_ok.sum())

# Histogram of phi2 for particles in phi1 range
if phi1_ok.sum() > 0:
    log.info("\nphi2 histogram (phi1 in range):")
    edges = np.arange(-90, 91, 10)
    counts, _ = np.histogram(phi2[phi1_ok], bins=edges)
    for i, c in enumerate(counts):
        if c > 0:
            log.info("  [%4d, %4d): %d", edges[i], edges[i+1], c)

# phi1 histogram for all particles
log.info("\nphi1 histogram (all particles):")
edges = np.arange(-180, 181, 30)
counts, _ = np.histogram(phi1, bins=edges)
for i, c in enumerate(counts):
    if c > 0:
        log.info("  [%4d, %4d): %d", edges[i], edges[i+1], c)

log.info("\nDone.")
