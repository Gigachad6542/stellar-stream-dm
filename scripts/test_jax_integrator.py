"""
Validate JAX orbit integrator against galpy's C integrator.

Tests:
1. Single orbit: compare JAX vs galpy leapfrog_c for a circular-ish orbit
2. Single orbit: compare JAX vs galpy for an eccentric halo orbit
3. Batch integration: 5000 particles (stream generation scenario)
4. Timing benchmark: measure speedup

Usage:
    python -u scripts/test_jax_integrator.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Fix galpy DLLs (must be before galpy import)
from src.simulation.potentials import _RO, _VO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _fix_galpy_dll_path():
    if sys.platform != "win32":
        return
    import os
    try:
        env_dir = Path(sys.executable).parent
        lib_bin = env_dir / "Library" / "bin"
        if lib_bin.exists():
            os.add_dll_directory(str(lib_bin))
        import galpy
        galpy_dir = Path(galpy.__file__).parent
        for subdir in ["", "potential", "orbit", "actionAngle", "util"]:
            dll_dir = galpy_dir / subdir if subdir else galpy_dir
            if dll_dir.exists():
                os.add_dll_directory(str(dll_dir))
        site_pkg = galpy_dir.parent
        os.add_dll_directory(str(site_pkg))
    except Exception:
        pass


_fix_galpy_dll_path()

import astropy.units as u
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

from src.simulation.jax_integrator import (
    JAX_AVAILABLE,
    build_force_table,
    integrate_particles_jax,
)


def test_single_orbit_circular():
    """Test: near-circular orbit at R=8 kpc. JAX vs galpy."""
    log.info("=" * 60)
    log.info("Test 1: Near-circular orbit at R=8 kpc")
    log.info("=" * 60)

    pot = MWPotential2014

    # IC: Sun-like circular orbit
    R0 = 8.0  # kpc
    z0 = 0.02  # kpc (slightly above plane)
    vR0 = 0.0  # km/s
    vT0 = 220.0  # km/s (circular)
    vz0 = 5.0  # km/s (small vertical motion)
    phi0 = 0.0  # rad

    # galpy orbit
    orb = Orbit(
        vxvv=[R0 * u.kpc, vR0 * u.km/u.s, vT0 * u.km/u.s,
              z0 * u.kpc, vz0 * u.km/u.s, phi0 * u.rad],
        ro=_RO, vo=_VO,
    )
    t_gyr = 1.0  # integrate for 1 Gyr
    n_steps = 500
    ts = np.linspace(0, t_gyr, n_steps) * u.Gyr
    orb.integrate(ts, pot, method="leapfrog_c", progressbar=False)

    x_galpy = orb.x(ts[-1], use_physical=True)
    y_galpy = orb.y(ts[-1], use_physical=True)
    z_galpy = orb.z(ts[-1], use_physical=True)
    vx_galpy = orb.vx(ts[-1], use_physical=True)
    vy_galpy = orb.vy(ts[-1], use_physical=True)
    vz_galpy = orb.vz(ts[-1], use_physical=True)

    # JAX orbit: same IC in Cartesian
    x0 = R0  # phi=0 so x=R, y=0
    y0 = 0.0
    pos = np.array([[x0, y0, z0]])
    vel = np.array([[vR0, vT0, vz0]])  # vx=vR=0, vy=vT=220, vz=5
    t_release = np.array([-t_gyr])  # released 1 Gyr ago

    t0 = time.time()
    pos_jax, vel_jax = integrate_particles_jax(
        pos, vel, t_release, pot, ro=_RO, vo=_VO, dt_myr=1.0,
    )
    jax_time = time.time() - t0

    dx = pos_jax[0, 0] - x_galpy
    dy = pos_jax[0, 1] - y_galpy
    dz = pos_jax[0, 2] - z_galpy
    pos_err = np.sqrt(dx**2 + dy**2 + dz**2)

    dvx = vel_jax[0, 0] - vx_galpy
    dvy = vel_jax[0, 1] - vy_galpy
    dvz = vel_jax[0, 2] - vz_galpy
    vel_err = np.sqrt(dvx**2 + dvy**2 + dvz**2)

    log.info("galpy final: pos=(%.4f, %.4f, %.4f) kpc  vel=(%.2f, %.2f, %.2f) km/s",
             x_galpy, y_galpy, z_galpy, vx_galpy, vy_galpy, vz_galpy)
    log.info("JAX   final: pos=(%.4f, %.4f, %.4f) kpc  vel=(%.2f, %.2f, %.2f) km/s",
             pos_jax[0, 0], pos_jax[0, 1], pos_jax[0, 2],
             vel_jax[0, 0], vel_jax[0, 1], vel_jax[0, 2])
    log.info("Position error: %.6f kpc", pos_err)
    log.info("Velocity error: %.4f km/s", vel_err)
    log.info("JAX time (incl JIT compile + force table): %.2f s", jax_time)

    # Accept if position error < 0.5 kpc (leapfrog vs leapfrog with different dt)
    status = "PASS" if pos_err < 0.5 else "FAIL"
    log.info("Result: %s (threshold: 0.5 kpc)", status)
    return pos_err < 0.5


def test_single_orbit_eccentric():
    """Test: eccentric halo orbit. More demanding test."""
    log.info("=" * 60)
    log.info("Test 2: Eccentric halo orbit")
    log.info("=" * 60)

    pot = MWPotential2014

    # Eccentric orbit: apo~25 kpc, peri~5 kpc
    R0 = 15.0
    z0 = 5.0
    vR0 = -80.0
    vT0 = 100.0
    vz0 = 30.0
    phi0 = 0.5

    orb = Orbit(
        vxvv=[R0 * u.kpc, vR0 * u.km/u.s, vT0 * u.km/u.s,
              z0 * u.kpc, vz0 * u.km/u.s, phi0 * u.rad],
        ro=_RO, vo=_VO,
    )
    t_gyr = 2.0
    n_steps = 1000
    ts = np.linspace(0, t_gyr, n_steps) * u.Gyr
    orb.integrate(ts, pot, method="leapfrog_c", progressbar=False)

    x_galpy = orb.x(ts[-1], use_physical=True)
    y_galpy = orb.y(ts[-1], use_physical=True)
    z_galpy = orb.z(ts[-1], use_physical=True)

    # JAX: convert cylindrical IC to Cartesian
    x0_cart = R0 * np.cos(phi0)
    y0_cart = R0 * np.sin(phi0)
    vx0_cart = vR0 * np.cos(phi0) - vT0 * np.sin(phi0)
    vy0_cart = vR0 * np.sin(phi0) + vT0 * np.cos(phi0)

    pos = np.array([[x0_cart, y0_cart, z0]])
    vel = np.array([[vx0_cart, vy0_cart, vz0]])
    t_release = np.array([-t_gyr])

    t0 = time.time()
    pos_jax, vel_jax = integrate_particles_jax(
        pos, vel, t_release, pot, ro=_RO, vo=_VO, dt_myr=1.0,
    )
    jax_time = time.time() - t0

    pos_err = np.sqrt(
        (pos_jax[0, 0] - x_galpy)**2 +
        (pos_jax[0, 1] - y_galpy)**2 +
        (pos_jax[0, 2] - z_galpy)**2
    )

    log.info("galpy final: (%.4f, %.4f, %.4f) kpc", x_galpy, y_galpy, z_galpy)
    log.info("JAX   final: (%.4f, %.4f, %.4f) kpc",
             pos_jax[0, 0], pos_jax[0, 1], pos_jax[0, 2])
    log.info("Position error: %.6f kpc", pos_err)
    log.info("JAX time (cached force table): %.2f s", jax_time)

    status = "PASS" if pos_err < 1.0 else "FAIL"
    log.info("Result: %s (threshold: 1.0 kpc)", status)
    return pos_err < 1.0


def test_batch_timing():
    """Benchmark: 5000 particles with varying release times."""
    log.info("=" * 60)
    log.info("Test 3: Batch integration timing (5000 particles)")
    log.info("=" * 60)

    pot = MWPotential2014
    rng = np.random.default_rng(42)

    n_particles = 5000

    # Generate random particles near a typical stream orbit
    R_base = 10.0 + rng.normal(0, 2, n_particles)
    phi_base = rng.uniform(0, 2 * np.pi, n_particles)
    z_base = rng.normal(0, 3, n_particles)

    x = R_base * np.cos(phi_base)
    y = R_base * np.sin(phi_base)
    z = z_base

    vR = rng.normal(0, 50, n_particles)
    vT = 180.0 + rng.normal(0, 30, n_particles)
    vz = rng.normal(0, 20, n_particles)

    vx = vR * np.cos(phi_base) - vT * np.sin(phi_base)
    vy = vR * np.sin(phi_base) + vT * np.cos(phi_base)

    pos = np.stack([x, y, z], axis=-1)
    vel = np.stack([vx, vy, vz], axis=-1)

    # Release times: 100 unique times over 3 Gyr (like stream_gen)
    n_unique = 100
    unique_times = np.linspace(-3.0, -0.03, n_unique)
    t_release = rng.choice(unique_times, n_particles)

    # -- JAX integration --
    log.info("Running JAX integration...")
    t0 = time.time()
    pos_jax, vel_jax = integrate_particles_jax(
        pos, vel, t_release, pot, ro=_RO, vo=_VO, dt_myr=2.0,
    )
    jax_time = time.time() - t0
    log.info("JAX: %.2f s for %d particles", jax_time, n_particles)

    # -- galpy integration (same loop as stream_gen._integrate_particles) --
    log.info("Running galpy integration (batched by release time)...")
    t0 = time.time()
    results_pos_galpy = np.empty((n_particles, 3))
    results_vel_galpy = np.empty((n_particles, 3))

    unique_t, inverse = np.unique(t_release, return_inverse=True)
    for gi, t_rel in enumerate(unique_t):
        grp = np.where(inverse == gi)[0]
        if abs(float(t_rel)) < 1e-9:
            results_pos_galpy[grp] = pos[grp]
            results_vel_galpy[grp] = vel[grp]
            continue

        x_g = pos[grp, 0]; y_g = pos[grp, 1]; z_g = pos[grp, 2]
        vx_g = vel[grp, 0]; vy_g = vel[grp, 1]; vz_g = vel[grp, 2]

        R_g = np.sqrt(x_g**2 + y_g**2)
        R_g = np.maximum(R_g, 0.5)
        phi_g = np.arctan2(y_g, x_g)
        vR_g = (x_g * vx_g + y_g * vy_g) / R_g
        vT_g = (x_g * vy_g - y_g * vx_g) / R_g

        n_int = max(8, int(abs(float(t_rel)) * 40))
        t_fwd = np.linspace(float(t_rel), 0.0, n_int) * u.Gyr

        orb = Orbit(
            vxvv=[R_g * u.kpc, vR_g * u.km/u.s, vT_g * u.km/u.s,
                  z_g * u.kpc, vz_g * u.km/u.s, phi_g * u.rad],
            ro=_RO, vo=_VO,
        )
        orb.integrate(t_fwd, pot, method="leapfrog_c", progressbar=False)
        t_end = t_fwd[-1]

        results_pos_galpy[grp, 0] = np.atleast_1d(np.asarray(orb.x(t_end, use_physical=True)))
        results_pos_galpy[grp, 1] = np.atleast_1d(np.asarray(orb.y(t_end, use_physical=True)))
        results_pos_galpy[grp, 2] = np.atleast_1d(np.asarray(orb.z(t_end, use_physical=True)))
        results_vel_galpy[grp, 0] = np.atleast_1d(np.asarray(orb.vx(t_end, use_physical=True)))
        results_vel_galpy[grp, 1] = np.atleast_1d(np.asarray(orb.vy(t_end, use_physical=True)))
        results_vel_galpy[grp, 2] = np.atleast_1d(np.asarray(orb.vz(t_end, use_physical=True)))

    galpy_time = time.time() - t0
    log.info("galpy: %.2f s for %d particles", galpy_time, n_particles)

    speedup = galpy_time / jax_time if jax_time > 0 else float('inf')
    log.info("Speedup: %.1fx", speedup)

    # Compare results
    pos_diff = np.sqrt(np.sum((pos_jax - results_pos_galpy)**2, axis=1))
    log.info("Position difference: median=%.4f kpc, 95th=%.4f kpc, max=%.4f kpc",
             np.median(pos_diff), np.percentile(pos_diff, 95), np.max(pos_diff))

    # For stream work, <1 kpc agreement is fine (noise and spray spread dominate)
    median_ok = np.median(pos_diff) < 2.0
    status = "PASS" if median_ok else "FAIL"
    log.info("Result: %s (median threshold: 2.0 kpc)", status)

    return median_ok, jax_time, galpy_time


def test_jax_second_call_timing():
    """Benchmark: second call (JIT cached). Shows actual per-simulation cost."""
    log.info("=" * 60)
    log.info("Test 4: Second-call timing (JIT cached)")
    log.info("=" * 60)

    pot = MWPotential2014
    rng = np.random.default_rng(123)

    n_particles = 5000
    R_base = 10.0 + rng.normal(0, 2, n_particles)
    phi_base = rng.uniform(0, 2 * np.pi, n_particles)
    z_base = rng.normal(0, 3, n_particles)

    pos = np.stack([
        R_base * np.cos(phi_base),
        R_base * np.sin(phi_base),
        z_base,
    ], axis=-1)

    vT = 180.0 + rng.normal(0, 30, n_particles)
    vR = rng.normal(0, 50, n_particles)
    vz = rng.normal(0, 20, n_particles)
    vel = np.stack([
        vR * np.cos(phi_base) - vT * np.sin(phi_base),
        vR * np.sin(phi_base) + vT * np.cos(phi_base),
        vz,
    ], axis=-1)

    unique_times = np.linspace(-3.0, -0.03, 100)
    t_release = rng.choice(unique_times, n_particles)

    # Second call — JIT and force table already cached
    t0 = time.time()
    pos_jax, vel_jax = integrate_particles_jax(
        pos, vel, t_release, pot, ro=_RO, vo=_VO, dt_myr=2.0,
    )
    jax_time = time.time() - t0
    log.info("JAX (cached): %.2f s for %d particles", jax_time, n_particles)

    # Check outputs are finite
    finite = np.isfinite(pos_jax).all() and np.isfinite(vel_jax).all()
    status = "PASS" if finite else "FAIL"
    log.info("All outputs finite: %s", status)

    return finite, jax_time


def main():
    if not JAX_AVAILABLE:
        log.error("JAX not available — cannot run tests")
        sys.exit(1)

    log.info("JAX available. Running integrator validation tests...")

    results = {}

    # Test 1: circular orbit
    results["circular"] = test_single_orbit_circular()

    # Test 2: eccentric orbit
    results["eccentric"] = test_single_orbit_eccentric()

    # Test 3: batch timing
    batch_ok, jax_t, galpy_t = test_batch_timing()
    results["batch"] = batch_ok

    # Test 4: cached timing
    cached_ok, cached_t = test_jax_second_call_timing()
    results["cached"] = cached_ok

    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    for name, passed in results.items():
        log.info("  %-15s  %s", name, "PASS" if passed else "FAIL")
    log.info("  Timing: JAX first=%.2f s, JAX cached=%.2f s, galpy=%.2f s",
             jax_t + cached_t, cached_t, galpy_t)
    if galpy_t > 0 and cached_t > 0:
        log.info("  Speedup (cached): %.1fx", galpy_t / cached_t)

    n_pass = sum(1 for v in results.values() if v)
    log.info("  %d/%d tests passed", n_pass, len(results))


if __name__ == "__main__":
    main()
