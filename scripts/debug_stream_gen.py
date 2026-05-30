"""Debug why generate_stream produces 0 particles despite good orbit RMS.

Prints phi1/phi2 statistics before and after selection cuts.
"""
from __future__ import annotations
import faulthandler, sys, json, logging
from pathlib import Path
import numpy as np

faulthandler.enable()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.potentials import _RO, _VO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

import astropy.coordinates as coord
import astropy.units as u
import galstreams
import yaml
from galpy.orbit import Orbit
from galpy.potential import vcirc

from src.simulation.potentials import get_mw_potential
from src.simulation.stream_gen import (
    _galactocentric_to_stream_coords,
    _galactocentric_to_stream_phi12,
    _pos_vel_to_orbit,
    _load_stream_config,
)

log.info("ro=%.3f kpc, vo=%.3f km/s", _RO, _VO)

potential = get_mw_potential(str(ROOT / "config" / "streams.yaml"))
log.info("Potential type: %s", type(potential))

mws = galstreams.MWStreams(verbose=False)
sc = _load_stream_config("GD1", str(ROOT / "config" / "streams.yaml"))

# Load cached IC
ic_file = ROOT / "data" / "processed" / "GD1_progenitor_ic.json"
with open(ic_file) as f:
    ic = json.load(f)
log.info("IC: pos=%s vel=%s rms=%.3f", ic["pos_kpc"], ic["vel_kms"], ic.get("_rms_deg", -1))

pos0 = np.array(ic["pos_kpc"])
vel0 = np.array(ic["vel_kms"])

# ---- 1. Progenitor orbit (same as optimizer) ----
stream_age = sc.get("isochrone_age_gyr", 10.0)
log.info("Stream age: %.1f Gyr", stream_age)

n_orb = 200
t_back = np.linspace(0.0, -stream_age, n_orb) * u.Gyr
prog = _pos_vel_to_orbit(pos0, vel0)
prog.integrate(t_back, potential, method="leapfrog_c", progressbar=False)

xs = np.atleast_1d(prog.x(t_back, use_physical=True))
ys = np.atleast_1d(prog.y(t_back, use_physical=True))
zs = np.atleast_1d(prog.z(t_back, use_physical=True))

# Check orbit is finite
log.info("Orbit x range: [%.1f, %.1f] kpc", xs.min(), xs.max())
log.info("Orbit y range: [%.1f, %.1f] kpc", ys.min(), ys.max())
log.info("Orbit z range: [%.1f, %.1f] kpc", zs.min(), zs.max())
log.info("Orbit R range: [%.1f, %.1f] kpc",
         np.sqrt(xs**2 + ys**2).min(), np.sqrt(xs**2 + ys**2).max())

# Convert orbit to stream frame
track = mws[sc["galstreams_key"]]
sf = track.stream_frame
phi1_o, phi2_o = _galactocentric_to_stream_phi12(np.stack([xs, ys, zs]), sf)

log.info("Orbit phi1 range: [%.1f, %.1f] deg", phi1_o.min(), phi1_o.max())
log.info("Orbit phi2 range: [%.1f, %.1f] deg", phi2_o.min(), phi2_o.max())

phi1_min, phi1_max = sc["phi1_range_deg"]
in_range = (phi1_o >= phi1_min) & (phi1_o <= phi1_max)
log.info("Orbit points in phi1 range [%.0f, %.0f]: %d/%d",
         phi1_min, phi1_max, in_range.sum(), len(phi1_o))
if in_range.sum() > 0:
    log.info("  phi2 in range: [%.2f, %.2f] deg", phi2_o[in_range].min(), phi2_o[in_range].max())

# ---- 2. Now run full stream generation WITHOUT the selection cut ----
log.info("\n--- Full stream generation (no selection) ---")

from src.simulation.stream_gen import generate_stream, StreamParticles
import src.simulation.stream_gen as sg

# Temporarily patch to skip selection cut
_orig_func = sg.generate_stream

# We'll just generate and look at pre-cut stats
# Run the actual generation
particles_full = generate_stream(
    "GD1", potential, n_stars=5000, seed=0,
    config_path=str(ROOT / "config" / "streams.yaml"),
    progenitor_ic=ic, mws=mws, n_steps_back=100,
)
log.info("Particles after selection: %d", len(particles_full.phi1))
if len(particles_full.phi1) > 0:
    log.info("  phi1: [%.2f, %.2f]", particles_full.phi1.min(), particles_full.phi1.max())
    log.info("  phi2: [%.2f, %.2f]", particles_full.phi2.min(), particles_full.phi2.max())

# Also generate WITHOUT selection cut by directly accessing the internal arrays
# Re-run but intercept before cut
log.info("\n--- Generating stream and inspecting pre-cut ---")
from src.simulation.stream_gen import _galactocentric_to_stream_coords

# Re-do the generation manually to see pre-cut phi1/phi2
rng = np.random.default_rng(0)
n_steps_back = 100
t_back2 = np.linspace(0.0, -stream_age, n_steps_back) * u.Gyr
prog2 = _pos_vel_to_orbit(pos0, vel0)
prog2.integrate(t_back2, potential, method="leapfrog_c", progressbar=False)

prog_R   = prog2.R(t_back2, use_physical=True)
prog_vR  = prog2.vR(t_back2, use_physical=True)
prog_vT  = prog2.vT(t_back2, use_physical=True)
prog_z   = prog2.z(t_back2, use_physical=True)
prog_vz  = prog2.vz(t_back2, use_physical=True)
prog_phi = prog2.phi(t_back2, use_physical=False)

prog_x = prog_R * np.cos(prog_phi)
prog_y = prog_R * np.sin(prog_phi)
prog_vx = prog_vR * np.cos(prog_phi) - prog_vT * np.sin(prog_phi)
prog_vy = prog_vR * np.sin(prog_phi) + prog_vT * np.cos(prog_phi)
prog_pos = np.stack([prog_x, prog_y, prog_z], axis=-1)
prog_vel = np.stack([prog_vx, prog_vy, prog_vz], axis=-1)

# Check progenitor orbit in stream frame
phi1_prog, phi2_prog = _galactocentric_to_stream_phi12(
    np.stack([prog_x, prog_y, prog_z]), sf
)
log.info("Progenitor orbit (100 steps):")
log.info("  phi1 range: [%.1f, %.1f] deg", phi1_prog.min(), phi1_prog.max())
log.info("  phi2 range: [%.1f, %.1f] deg", phi2_prog.min(), phi2_prog.max())
in_range2 = (phi1_prog >= phi1_min) & (phi1_prog <= phi1_max)
log.info("  Points in phi1 range: %d/%d", in_range2.sum(), len(phi1_prog))

# Check what galpy's R, phi look like
log.info("\nProgentitor orbit galpy coords:")
log.info("  R range: [%.2f, %.2f] kpc", prog_R.min(), prog_R.max())
log.info("  z range: [%.2f, %.2f] kpc", prog_z.min(), prog_z.max())
log.info("  phi range: [%.2f, %.2f] rad", prog_phi.min(), prog_phi.max())
log.info("  vR range: [%.1f, %.1f] km/s", prog_vR.min(), prog_vR.max())
log.info("  vT range: [%.1f, %.1f] km/s", prog_vT.min(), prog_vT.max())

# Check the first and last 5 orbit points in stream frame
for i in [0, 1, 2, n_steps_back//4, n_steps_back//2, 3*n_steps_back//4, n_steps_back-2, n_steps_back-1]:
    log.info("  t=%.2f Gyr: phi1=%.1f phi2=%.1f R=%.1f kpc",
             float(t_back2[i].value), phi1_prog[i], phi2_prog[i],
             float(np.sqrt(prog_x[i]**2 + prog_y[i]**2)))
