"""
JAX-accelerated orbit integrator for stream particle integration.

Replaces galpy's per-batch orbit integration loop in stream_gen.py with a
single JIT-compiled call that integrates ALL particles simultaneously.

Strategy
--------
1. Pre-tabulate the MW potential forces F_R(R,z) and F_z(R,z) from galpy on a
   2D grid.  This is done once and cached.
2. Use JAX bilinear interpolation to evaluate forces at arbitrary (R,z) — no
   Python callback, pure JIT.
3. Leapfrog integrator (symplectic, 2nd order) in cylindrical coordinates
   (R, phi, z, vR, vT, vz).  Fixed timestep — same as galpy's leapfrog_c.
4. jax.vmap vectorises over particles; jax.jit compiles the whole thing.

Expected speedup: 10-50× over galpy's batched C integrator (eliminates Python
loop over ~200 release-time groups, replaces with one fused kernel).

On GPU (if jaxlib[cuda] installed): additional 10-50× from SIMD parallelism.

Coordinate system: cylindrical (R, phi, z) with physical units (kpc, km/s, Gyr).
Matches galpy's internal convention for MWPotential2014.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    # Enable 64-bit precision BEFORE importing jax.numpy.
    # Without this, JAX silently truncates float64 to float32, causing
    # ~7-15 kpc position errors after thousands of leapfrog steps.
    import jax
    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    from jax import jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    log.warning("JAX not available — falling back to galpy integrator")

# Physical constants in kpc / km/s / Gyr unit system
# 1 Gyr = 3.0857e16 km/s × kpc → conversion factor:
_KPC_PER_KMS_PER_GYR = 1.0227  # 1 kpc = 1.0227 km/s × Gyr
# So velocity in kpc/Gyr = velocity_kms * 1.0227


# ---------------------------------------------------------------------------
# Force table from galpy
# ---------------------------------------------------------------------------

class ForceTable:
    """Tabulated axisymmetric potential forces on a 2D (R, z) grid.

    Stores F_R(R, z) and F_z(R, z) as JAX arrays for bilinear interpolation.
    The potential is axisymmetric so forces depend only on (R, |z|) — but we
    store the full z range (both positive and negative) because F_z is
    antisymmetric: F_z(R, -z) = -F_z(R, z).  Using full z avoids edge cases.
    """

    def __init__(
        self,
        R_grid: np.ndarray,    # [N_R] kpc
        z_grid: np.ndarray,    # [N_z] kpc  (symmetric about 0)
        F_R: np.ndarray,       # [N_R, N_z] km/s/Gyr (acceleration in R)
        F_z: np.ndarray,       # [N_R, N_z] km/s/Gyr (acceleration in z)
    ):
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX required for ForceTable")
        self.R_grid = jnp.array(R_grid, dtype=jnp.float64)
        self.z_grid = jnp.array(z_grid, dtype=jnp.float64)
        self.F_R = jnp.array(F_R, dtype=jnp.float64)
        self.F_z = jnp.array(F_z, dtype=jnp.float64)
        self.R_min = float(R_grid[0])
        self.R_max = float(R_grid[-1])
        self.z_min = float(z_grid[0])
        self.z_max = float(z_grid[-1])
        self.dR = float(R_grid[1] - R_grid[0])
        self.dz = float(z_grid[1] - z_grid[0])
        self.N_R = len(R_grid)
        self.N_z = len(z_grid)


def build_force_table(
    potential,
    R_range_kpc: tuple[float, float] = (0.1, 60.0),
    z_range_kpc: float = 40.0,
    N_R: int = 1024,
    N_z: int = 512,
    ro: float = 8.0,
    vo: float = 220.0,
) -> ForceTable:
    """Tabulate F_R and F_z from a galpy potential on a 2D grid.

    Forces are in km/s/Gyr (acceleration units matching our leapfrog).

    The grid is linear in R and z.  For the MW potential, R ∈ [0.1, 300] kpc
    and z ∈ [-150, 150] kpc covers all bound orbits including distant halo
    streams like Orphan-Chenab (d ~ 55 kpc).

    Parameters
    ----------
    potential : galpy potential (list or single)
    R_range_kpc : (R_min, R_max) in kpc
    z_range_kpc : half-range; grid covers [-z_range, +z_range]
    N_R, N_z : grid resolution
    ro, vo : galpy natural units
    """
    R_grid = np.linspace(R_range_kpc[0], R_range_kpc[1], N_R)
    z_grid = np.linspace(-z_range_kpc, z_range_kpc, N_z)

    log.info("Building force table: R=[%.1f, %.1f] kpc (%d pts), "
             "z=[%.1f, %.1f] kpc (%d pts)",
             R_grid[0], R_grid[-1], N_R, z_grid[0], z_grid[-1], N_z)

    # Use galpy natural units throughout:
    #   evaluateRforces(pot, R/ro, z/ro) → force in vo^2/ro units
    #   Leapfrog uses (R, z) in ro units, v in vo units, t in ro/vo units
    #   Then F is directly usable as acceleration.
    #
    # Build a meshgrid and evaluate each potential component individually
    # to avoid the array-broadcasting bug in galpy's CompositePotential.
    # Sum the forces from all components manually.

    R_nu = R_grid / ro  # [N_R]
    z_nu = z_grid / ro  # [N_z]

    # Evaluate forces row-by-row: fix R, vectorize over z where possible.
    # galpy evaluateRforces handles (scalar R, array z) for some potentials,
    # but evaluatezforces on CompositePotential/MiyamotoNagai crashes with
    # array z. Use scalar calls via np.vectorize as a safe fallback.
    from galpy.potential import evaluateRforces, evaluatezforces

    F_R_grid = np.zeros((N_R, N_z))
    F_z_grid = np.zeros((N_R, N_z))

    log.info("Evaluating %d force grid points...", N_R * N_z)

    # Try vectorized row (scalar R, array z) first; fall back to scalar
    _use_vectorized = True
    try:
        _test_fr = evaluateRforces(potential, R_nu[N_R // 2], z_nu)
        _test_fz = evaluatezforces(potential, R_nu[N_R // 2], z_nu)
        if not (isinstance(_test_fr, np.ndarray) and len(_test_fr) == N_z):
            _use_vectorized = False
    except Exception:
        _use_vectorized = False

    if _use_vectorized:
        log.info("Using vectorized z evaluation (fast path)")
        for i in range(N_R):
            F_R_grid[i, :] = evaluateRforces(potential, R_nu[i], z_nu)
            F_z_grid[i, :] = evaluatezforces(potential, R_nu[i], z_nu)
    else:
        log.info("Using scalar evaluation (slow path; %d calls)", 2 * N_R * N_z)

        def _fr(R, z):
            return float(evaluateRforces(potential, R, z))

        def _fz(R, z):
            return float(evaluatezforces(potential, R, z))

        fr_vec = np.vectorize(_fr)
        fz_vec = np.vectorize(_fz)

        RR, ZZ = np.meshgrid(R_nu, z_nu, indexing="ij")
        F_R_grid = fr_vec(RR, ZZ)
        F_z_grid = fz_vec(RR, ZZ)

    log.info("Force table built: F_R range [%.4f, %.4f], F_z range [%.4f, %.4f] (natural units)",
             F_R_grid.min(), F_R_grid.max(), F_z_grid.min(), F_z_grid.max())

    # Store everything in natural units
    return ForceTable(
        R_grid=R_nu,
        z_grid=z_nu,
        F_R=F_R_grid,
        F_z=F_z_grid,
    )


# Cached force table (built once per potential)
_CACHED_FORCE_TABLE: Optional[ForceTable] = None
_CACHED_POTENTIAL_ID: Optional[int] = None

# Cached JIT-compiled integrator functions (keyed by force-table id + ro/vo).
# Creating a new @jit-decorated function on every call defeats JIT caching:
# JAX keys its XLA compilation cache by Python function *identity* (id()),
# so a fresh function object always triggers a fresh compilation (~1-2 s).
# By caching the (integrate_one, integrate_batch) pair we guarantee the same
# Python object is returned on every call and JAX reuses the compiled kernel.
_INTEGRATOR_CACHE: dict = {}  # {(ft_id, ro, vo): (integrate_one, integrate_batch)}


def get_force_table(potential, ro: float = 8.0, vo: float = 220.0) -> ForceTable:
    """Get or build cached force table for the given potential."""
    global _CACHED_FORCE_TABLE, _CACHED_POTENTIAL_ID
    pot_id = id(potential)
    if _CACHED_FORCE_TABLE is not None and _CACHED_POTENTIAL_ID == pot_id:
        return _CACHED_FORCE_TABLE
    ft = build_force_table(potential, ro=ro, vo=vo)
    _CACHED_FORCE_TABLE = ft
    _CACHED_POTENTIAL_ID = pot_id
    return ft


# ---------------------------------------------------------------------------
# JAX bilinear interpolation
# ---------------------------------------------------------------------------

if JAX_AVAILABLE:

    @jit
    def _interp2d(R_nu, z_nu, grid, R_arr, z_arr, dR, dz, N_R, N_z):
        """Bilinear interpolation on a regular 2D grid.

        All inputs are scalars (R_nu, z_nu) or the grid arrays.
        Returns interpolated value at (R_nu, z_nu).
        """
        # Clamp to grid bounds
        R_nu = jnp.clip(R_nu, R_arr[0], R_arr[-1])
        z_nu = jnp.clip(z_nu, z_arr[0], z_arr[-1])

        # Grid indices (continuous)
        ri = (R_nu - R_arr[0]) / dR
        zi = (z_nu - z_arr[0]) / dz

        # Integer indices
        i0 = jnp.clip(jnp.floor(ri).astype(jnp.int32), 0, N_R - 2)
        j0 = jnp.clip(jnp.floor(zi).astype(jnp.int32), 0, N_z - 2)

        # Fractional parts
        fr = ri - i0.astype(jnp.float64)
        fz = zi - j0.astype(jnp.float64)

        # Bilinear interpolation
        v00 = grid[i0, j0]
        v10 = grid[i0 + 1, j0]
        v01 = grid[i0, j0 + 1]
        v11 = grid[i0 + 1, j0 + 1]

        return (v00 * (1 - fr) * (1 - fz) +
                v10 * fr * (1 - fz) +
                v01 * (1 - fr) * fz +
                v11 * fr * fz)


    # ---------------------------------------------------------------------------
    # Leapfrog integrator in cylindrical coordinates (natural units)
    # ---------------------------------------------------------------------------

    def _make_leapfrog_step(ft: ForceTable):
        """Create a JIT-compiled leapfrog step function for the given force table.

        Returns a function: step(state, dt) → new_state
        where state = (R, phi, z, vR, vT, vz) all in natural units.
        """
        F_R_grid = ft.F_R
        F_z_grid = ft.F_z
        R_arr = ft.R_grid
        z_arr = ft.z_grid
        dR = ft.dR
        dz = ft.dz
        N_R = ft.N_R
        N_z = ft.N_z

        @jit
        def get_forces(R, z):
            """Get (a_R, a_z) at position (R, z) in natural units."""
            f_R = _interp2d(R, z, F_R_grid, R_arr, z_arr, dR, dz, N_R, N_z)
            f_z = _interp2d(R, z, F_z_grid, R_arr, z_arr, dR, dz, N_R, N_z)
            return f_R, f_z

        @jit
        def leapfrog_step(state, dt):
            """One leapfrog (velocity Verlet) step in cylindrical coords.

            State: (R, phi, z, vR, vT, vz) in natural units.

            Equations of motion in cylindrical (R, phi, z):
              dR/dt   = vR
              dphi/dt = vT / R
              dz/dt   = vz
              dvR/dt  = F_R + vT^2 / R   (centrifugal term)
              dvT/dt  = -vR * vT / R     (Coriolis term)
              dvz/dt  = F_z
            """
            R, phi, z, vR, vT, vz = state

            # Clamp R to avoid division by zero
            R = jnp.maximum(R, 1e-4)

            # Forces at current position
            f_R, f_z = get_forces(R, z)

            # Full radial acceleration (force + centrifugal)
            a_R = f_R + vT**2 / R
            a_T = -vR * vT / R
            a_z = f_z

            # Kick velocities half step
            vR_half  = vR  + 0.5 * dt * a_R
            vT_half  = vT  + 0.5 * dt * a_T
            vz_half  = vz  + 0.5 * dt * a_z

            # Drift positions full step
            R_new   = jnp.maximum(R + dt * vR_half, 1e-4)
            phi_new = phi + dt * vT_half / jnp.maximum(R, 1e-4)
            z_new   = z + dt * vz_half

            # Forces at new position
            f_R_new, f_z_new = get_forces(R_new, z_new)

            # Full acceleration at new position
            a_R_new = f_R_new + vT_half**2 / R_new
            a_T_new = -vR_half * vT_half / R_new
            a_z_new = f_z_new

            # Kick velocities another half step
            vR_new  = vR_half  + 0.5 * dt * a_R_new
            vT_new  = vT_half  + 0.5 * dt * a_T_new
            vz_new  = vz_half  + 0.5 * dt * a_z_new

            return (R_new, phi_new, z_new, vR_new, vT_new, vz_new)

        return leapfrog_step

    def _make_integrate_one(ft: ForceTable, ro: float, vo: float):
        """Create a JIT-compiled function to integrate one particle.

        Parameters are in physical units (kpc, km/s, Gyr) on input/output.
        Internal computation uses galpy natural units.

        Returns
        -------
        integrate_one(x, y, z, vx, vy, vz, t_start_gyr, t_end_gyr, n_steps)
            → (x_f, y_f, z_f, vx_f, vy_f, vz_f) in physical units
        """
        step_fn = _make_leapfrog_step(ft)

        # Natural time unit: ro/vo in Gyr
        # ro kpc / vo km/s = ro * 3.0857e16 km / (vo km/s) = ro/vo * 3.0857e16 s
        # In Gyr: ro/vo * 3.0857e16 / 3.15576e16
        _TU_GYR = ro / vo * 3.0857e16 / 3.15576e16  # ~0.03556 Gyr for ro=8, vo=220

        @jit
        def integrate_one(x, y, z, vx, vy, vz, t_start_gyr, n_steps, dt_gyr):
            """Integrate one particle from t_start to t_start + n_steps * dt.

            Input/output in physical units (kpc, km/s).
            """
            # Convert to cylindrical natural units
            R = jnp.sqrt(x**2 + y**2) / ro
            R = jnp.maximum(R, 1e-6)
            phi = jnp.arctan2(y, x)
            z_nu = z / ro

            vR_phys = (x * vx + y * vy) / (R * ro)  # km/s
            vT_phys = (x * vy - y * vx) / (R * ro)  # km/s

            vR_nu = vR_phys / vo
            vT_nu = vT_phys / vo
            vz_nu = vz / vo

            dt_nu = dt_gyr / _TU_GYR

            state = (R, phi, z_nu, vR_nu, vT_nu, vz_nu)

            # Use jax.lax.fori_loop for JIT-compatible fixed iteration
            def body(i, s):
                return step_fn(s, dt_nu)

            state_f = jax.lax.fori_loop(0, n_steps, body, state)

            R_f, phi_f, z_f, vR_f, vT_f, vz_f = state_f

            # Convert back to Cartesian physical units
            x_f = R_f * ro * jnp.cos(phi_f)
            y_f = R_f * ro * jnp.sin(phi_f)
            z_f_phys = z_f * ro

            vx_f = (vR_f * jnp.cos(phi_f) - vT_f * jnp.sin(phi_f)) * vo
            vy_f = (vR_f * jnp.sin(phi_f) + vT_f * jnp.cos(phi_f)) * vo
            vz_f_phys = vz_f * vo

            return x_f, y_f, z_f_phys, vx_f, vy_f, vz_f_phys

        return integrate_one


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def integrate_particles_jax(
    pos_arr: np.ndarray,         # [N, 3] kpc (x, y, z)
    vel_arr: np.ndarray,         # [N, 3] km/s (vx, vy, vz)
    t_release_arr: np.ndarray,   # [N] Gyr (negative = past)
    potential,
    ro: float = 8.0,
    vo: float = 220.0,
    dt_myr: float = 2.0,         # timestep in Myr
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate particles from their release times to t=0 using JAX.

    Drop-in replacement for _integrate_particles() in stream_gen.py.

    Parameters
    ----------
    pos_arr : [N, 3] initial positions in kpc (galactocentric Cartesian)
    vel_arr : [N, 3] initial velocities in km/s
    t_release_arr : [N] release times in Gyr (negative values = past)
    potential : galpy potential list
    ro, vo : galpy natural units
    dt_myr : integration timestep in Myr (default 2.0 for <=0.1 deg accuracy)

    Returns
    -------
    pos_final : [N, 3] kpc at t=0
    vel_final : [N, 3] km/s at t=0
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX not available")

    n = len(pos_arr)
    if n == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    # Build / retrieve cached force table
    ft = get_force_table(potential, ro=ro, vo=vo)

    # Retrieve (or build-and-cache) JIT-compiled integrators.
    # Keyed by force-table identity so that different potentials (which produce
    # different force tables) each get their own compiled kernel.
    _cache_key = (id(ft), ro, vo)
    if _cache_key not in _INTEGRATOR_CACHE:
        _integrate_one = _make_integrate_one(ft, ro, vo)
        _integrate_batch = vmap(_integrate_one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))
        _INTEGRATOR_CACHE[_cache_key] = (_integrate_one, _integrate_batch)
    _, integrate_batch = _INTEGRATOR_CACHE[_cache_key]

    dt_gyr = dt_myr / 1000.0

    results_pos = np.empty((n, 3))
    results_vel = np.empty((n, 3))

    # --- Group by EXACT release time (mirrors galpy's np.unique grouping) ----
    # This is critical for correctness: each particle must be integrated for
    # EXACTLY |t_release| Gyr.  Using a coarser grouping (e.g. 20 bins based
    # on sorted size) causes the shortest-release particles to be over-integrated
    # by up to (bin_max - actual) Gyr, placing them far off the stream track.
    # With n_steps_back=100, there are ~100 unique release times; each group has
    # ~50 particles, so the per-group vmap overhead is negligible.
    unique_t, inverse = np.unique(t_release_arr, return_inverse=True)

    for gi, t_rel in enumerate(unique_t):
        grp = np.where(inverse == gi)[0]

        # t_rel = 0 means released at present day — no integration needed
        if abs(float(t_rel)) < 1e-9:
            results_pos[grp] = pos_arr[grp]
            results_vel[grp] = vel_arr[grp]
            continue

        dur = abs(float(t_rel))
        n_steps = max(1, int(np.round(dur / dt_gyr)))

        grp_pos = pos_arr[grp]
        grp_vel = vel_arr[grp]

        x = jnp.array(grp_pos[:, 0], dtype=jnp.float64)
        y = jnp.array(grp_pos[:, 1], dtype=jnp.float64)
        z = jnp.array(grp_pos[:, 2], dtype=jnp.float64)
        vx = jnp.array(grp_vel[:, 0], dtype=jnp.float64)
        vy = jnp.array(grp_vel[:, 1], dtype=jnp.float64)
        vz = jnp.array(grp_vel[:, 2], dtype=jnp.float64)

        m = len(grp)
        # t_start is unused by the integrator (steps forward n_steps x dt from IC)
        t_start    = jnp.zeros(m, dtype=jnp.float64)
        n_steps_arr = jnp.full(m, n_steps, dtype=jnp.int32)
        dt_arr     = jnp.full(m, dt_gyr, dtype=jnp.float64)

        xf, yf, zf, vxf, vyf, vzf = integrate_batch(
            x, y, z, vx, vy, vz, t_start, n_steps_arr, dt_arr,
        )

        results_pos[grp, 0] = np.asarray(xf)
        results_pos[grp, 1] = np.asarray(yf)
        results_pos[grp, 2] = np.asarray(zf)
        results_vel[grp, 0] = np.asarray(vxf)
        results_vel[grp, 1] = np.asarray(vyf)
        results_vel[grp, 2] = np.asarray(vzf)

    return results_pos, results_vel


def integrate_progenitor_jax(
    pos0: np.ndarray,     # [3] kpc
    vel0: np.ndarray,     # [3] km/s
    t_back_gyr: float,    # backward integration time (positive value)
    n_steps: int,         # number of output timesteps
    potential,
    ro: float = 8.0,
    vo: float = 220.0,
    dt_myr: float = 1.0,  # fine timestep for progenitor orbit
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate progenitor orbit backward using JAX leapfrog.

    Returns positions and velocities at n_steps equally-spaced times
    from t=0 to t=-t_back_gyr.

    Returns
    -------
    pos_orbit : [n_steps, 3] kpc (x, y, z) at each timestep
    vel_orbit : [n_steps, 3] km/s at each timestep
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX not available")

    ft = get_force_table(potential, ro=ro, vo=vo)
    step_fn = _make_leapfrog_step(ft)

    _TU_GYR = ro / vo * 3.0857e16 / 3.15576e16

    # Convert IC to cylindrical natural units
    x0, y0, z0 = pos0
    vx0, vy0, vz0 = vel0

    R0 = np.sqrt(x0**2 + y0**2) / ro
    phi0 = np.arctan2(y0, x0)
    z0_nu = z0 / ro
    vR0 = (x0 * vx0 + y0 * vy0) / (R0 * ro) / vo
    vT0 = (x0 * vy0 - y0 * vx0) / (R0 * ro) / vo
    vz0_nu = vz0 / vo

    # Backward integration: use negative dt
    dt_gyr = dt_myr / 1000.0
    total_steps = max(1, int(np.round(t_back_gyr / dt_gyr)))
    dt_nu = -dt_gyr / _TU_GYR  # negative for backward

    # Output sampling: save state at n_steps equally-spaced intervals
    save_every = max(1, total_steps // n_steps)

    state = jnp.array([R0, phi0, z0_nu, vR0, vT0, vz0_nu])

    # Use scan for efficient sequential integration with saves
    @jit
    def integrate_full(init_state):
        def step_body(state, _):
            new_state = jnp.array(step_fn(
                (state[0], state[1], state[2], state[3], state[4], state[5]),
                dt_nu,
            ))
            return new_state, new_state

        _, trajectory = jax.lax.scan(step_body, init_state, None, length=total_steps)
        return trajectory

    trajectory = integrate_full(state)  # [total_steps, 6]

    # Subsample to n_steps output points
    indices = np.linspace(0, total_steps - 1, n_steps, dtype=int)
    traj_sub = np.asarray(trajectory[indices])  # [n_steps, 6]

    # Convert back to Cartesian physical units
    R = traj_sub[:, 0] * ro
    phi = traj_sub[:, 1]
    z = traj_sub[:, 2] * ro
    vR = traj_sub[:, 3] * vo
    vT = traj_sub[:, 4] * vo
    vz = traj_sub[:, 5] * vo

    x = R * np.cos(phi)
    y = R * np.sin(phi)
    vx = vR * np.cos(phi) - vT * np.sin(phi)
    vy = vR * np.sin(phi) + vT * np.cos(phi)

    pos_orbit = np.stack([x, y, z], axis=-1)
    vel_orbit = np.stack([vx, vy, vz], axis=-1)

    # Prepend t=0 state
    pos_t0 = pos0.reshape(1, 3)
    vel_t0 = vel0.reshape(1, 3)
    pos_orbit = np.concatenate([pos_t0, pos_orbit], axis=0)[:n_steps]
    vel_orbit = np.concatenate([vel_t0, vel_orbit], axis=0)[:n_steps]

    return pos_orbit, vel_orbit


# ---------------------------------------------------------------------------
# Reusable orbit evaluator for IC optimiser (JIT-compiles once, reuses)
# ---------------------------------------------------------------------------

def make_orbit_evaluator(
    potential,
    t_back_gyr: float,
    n_output: int = 200,
    ro: float = 8.0,
    vo: float = 220.0,
    dt_myr: float = 1.0,
):
    """Create a JIT-compiled backward orbit evaluator for the IC optimiser.

    The returned callable integrates a progenitor orbit backward for
    ``t_back_gyr`` using the JAX leapfrog and returns positions/velocities
    at ``n_output`` equally-spaced times from *t* = 0 to *t* = -T.

    JIT compilation happens on the first call (~10-30 s); subsequent calls
    with different pos/vel reuse the compiled kernel — ideal for
    ``scipy.optimize`` loops (~3000 evaluations).

    Returns
    -------
    evaluate : callable(pos_kpc[3], vel_kms[3])
        → (xs, ys, zs, vxs, vys, vzs) each shape [n_output], physical units.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX not available for orbit evaluator")

    ft = get_force_table(potential, ro=ro, vo=vo)
    step_fn = _make_leapfrog_step(ft)

    _TU_GYR = ro / vo * 3.0857e16 / 3.15576e16
    dt_gyr = dt_myr / 1000.0
    total_steps = max(1, int(np.round(t_back_gyr / dt_gyr)))
    dt_nu_val = -dt_gyr / _TU_GYR  # negative for backward

    # Pre-compute subsample indices as compile-time constant
    _indices = jnp.linspace(0, total_steps, n_output).astype(jnp.int32)
    _dt_nu = jnp.float64(dt_nu_val)

    @jit
    def _run(x0, y0, z0, vx0, vy0, vz0):
        R0 = jnp.maximum(jnp.sqrt(x0**2 + y0**2) / ro, 1e-6)
        phi0 = jnp.arctan2(y0, x0)
        z0_nu = z0 / ro
        vR0 = (x0 * vx0 + y0 * vy0) / (R0 * ro) / vo
        vT0 = (x0 * vy0 - y0 * vx0) / (R0 * ro) / vo
        vz0_nu = vz0 / vo

        init = jnp.array([R0, phi0, z0_nu, vR0, vT0, vz0_nu])

        def body(s, _):
            ns = jnp.array(step_fn(
                (s[0], s[1], s[2], s[3], s[4], s[5]), _dt_nu))
            return ns, ns

        _, traj = jax.lax.scan(body, init, None, length=total_steps)
        # Prepend initial state so index 0 = t=0, index total_steps = t=-T
        full = jnp.concatenate([init[None, :], traj], axis=0)
        out = full[_indices]  # [n_output, 6]

        R   = out[:, 0] * ro
        phi = out[:, 1]
        z   = out[:, 2] * ro
        vR  = out[:, 3] * vo
        vT  = out[:, 4] * vo
        vz  = out[:, 5] * vo

        return (R * jnp.cos(phi),
                R * jnp.sin(phi),
                z,
                vR * jnp.cos(phi) - vT * jnp.sin(phi),
                vR * jnp.sin(phi) + vT * jnp.cos(phi),
                vz)

    _compiled = [False]

    def evaluate(pos_kpc, vel_kms):
        """pos_kpc: [3] kpc, vel_kms: [3] km/s → 6 arrays of [n_output]."""
        result = _run(
            float(pos_kpc[0]), float(pos_kpc[1]), float(pos_kpc[2]),
            float(vel_kms[0]), float(vel_kms[1]), float(vel_kms[2]),
        )
        if not _compiled[0]:
            _compiled[0] = True
            log.info("JAX orbit evaluator compiled: %d leapfrog steps -> %d output pts",
                     total_steps, n_output)
        return tuple(np.asarray(a) for a in result)

    return evaluate


# ---------------------------------------------------------------------------
# Progenitor orbit for Fardal spray (drop-in replacement for galpy section)
# ---------------------------------------------------------------------------

def integrate_progenitor_spray_jax(
    pos0: np.ndarray,      # [3] kpc
    vel0: np.ndarray,      # [3] km/s
    t_back_gyr: float,     # positive backward integration time
    n_steps_back: int,     # number of output timesteps
    potential,
    ro: float = 8.0,
    vo: float = 220.0,
    dt_myr: float = 1.0,
) -> dict:
    """Integrate progenitor orbit backward for Fardal particle spray.

    Drop-in replacement for the galpy progenitor orbit section in
    ``generate_stream()``.  Returns all quantities the spray code needs.

    Returns
    -------
    dict with keys:
        'pos'   : [n_steps_back, 3] Cartesian positions (kpc)
        'vel'   : [n_steps_back, 3] Cartesian velocities (km/s)
        'r_mag' : [n_steps_back, 1] radial distance (kpc)
        'r_hat' : [n_steps_back, 3] radial unit vector
        'v_c'   : [n_steps_back]    circular velocity (km/s)
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX not available")

    ft = get_force_table(potential, ro=ro, vo=vo)
    step_fn = _make_leapfrog_step(ft)

    _TU_GYR = ro / vo * 3.0857e16 / 3.15576e16
    dt_gyr = dt_myr / 1000.0
    total_steps = max(1, int(np.round(t_back_gyr / dt_gyr)))
    dt_nu_val = -dt_gyr / _TU_GYR
    _indices = jnp.linspace(0, total_steps, n_steps_back).astype(jnp.int32)
    _dt_nu = jnp.float64(dt_nu_val)

    # Convert IC to natural units
    x0, y0, z0 = float(pos0[0]), float(pos0[1]), float(pos0[2])
    vx0, vy0, vz0 = float(vel0[0]), float(vel0[1]), float(vel0[2])
    R0 = max(np.sqrt(x0**2 + y0**2) / ro, 1e-6)
    phi0 = np.arctan2(y0, x0)
    z0_nu = z0 / ro
    vR0_nu = (x0 * vx0 + y0 * vy0) / (R0 * ro) / vo
    vT0_nu = (x0 * vy0 - y0 * vx0) / (R0 * ro) / vo
    vz0_nu = vz0 / vo

    init = jnp.array([R0, phi0, z0_nu, vR0_nu, vT0_nu, vz0_nu])

    @jit
    def _scan(init_state):
        def body(s, _):
            ns = jnp.array(step_fn(
                (s[0], s[1], s[2], s[3], s[4], s[5]), _dt_nu))
            return ns, ns
        _, traj = jax.lax.scan(body, init_state, None, length=total_steps)
        full = jnp.concatenate([init_state[None, :], traj], axis=0)
        return full[_indices]

    traj = np.asarray(_scan(init))  # [n_steps_back, 6] natural units

    # Convert to physical units
    R_phys  = traj[:, 0] * ro
    phi     = traj[:, 1]
    z_phys  = traj[:, 2] * ro
    vR_phys = traj[:, 3] * vo
    vT_phys = traj[:, 4] * vo
    vz_phys = traj[:, 5] * vo

    x  = R_phys * np.cos(phi)
    y  = R_phys * np.sin(phi)
    vx = vR_phys * np.cos(phi) - vT_phys * np.sin(phi)
    vy = vR_phys * np.sin(phi) + vT_phys * np.cos(phi)

    pos = np.stack([x, y, z_phys], axis=-1)        # [n_steps_back, 3]
    vel_arr = np.stack([vx, vy, vz_phys], axis=-1)  # [n_steps_back, 3]

    r_mag = np.linalg.norm(pos, axis=-1, keepdims=True)  # [n_steps_back, 1]
    r_hat = pos / np.clip(r_mag, 1e-6, None)

    # Circular velocity from force table: v_c = vo * sqrt(R_nu * |F_R(R_nu, 0)|)
    R_nu_arr = jnp.array(traj[:, 0])

    @jit
    def _vc(R_arr):
        def _one(r):
            fR = _interp2d(r, jnp.float64(0.0),
                           ft.F_R, ft.R_grid, ft.z_grid,
                           ft.dR, ft.dz, ft.N_R, ft.N_z)
            return jnp.sqrt(jnp.maximum(r * jnp.abs(fR), 0.0)) * vo
        return vmap(_one)(R_arr)

    v_c = np.asarray(_vc(R_nu_arr))

    log.info("JAX progenitor orbit: %d steps -> %d outputs, v_c range [%.0f, %.0f] km/s",
             total_steps, n_steps_back, v_c.min(), v_c.max())

    return {
        'pos': pos,
        'vel': vel_arr,
        'r_mag': r_mag,
        'r_hat': r_hat,
        'v_c': v_c,
    }


def vcirc_jax(
    R_kpc: np.ndarray,
    potential,
    ro: float = 8.0,
    vo: float = 220.0,
) -> np.ndarray:
    """Circular velocity at given radii using the tabulated force table.

    Parameters
    ----------
    R_kpc : array-like of cylindrical R values in kpc

    Returns
    -------
    v_c : ndarray of circular velocity in km/s
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX not available")

    ft = get_force_table(potential, ro=ro, vo=vo)
    R_nu = jnp.array(np.atleast_1d(R_kpc).astype(np.float64)) / ro

    @jit
    def _compute(R_arr):
        def _one(r):
            fR = _interp2d(r, jnp.float64(0.0),
                           ft.F_R, ft.R_grid, ft.z_grid,
                           ft.dR, ft.dz, ft.N_R, ft.N_z)
            return jnp.sqrt(jnp.maximum(r * jnp.abs(fR), 0.0)) * vo
        return vmap(_one)(R_arr)

    return np.asarray(_compute(R_nu))
