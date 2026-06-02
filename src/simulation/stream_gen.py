"""
Unperturbed stellar stream generation using galpy orbit integration
with a Fardal-like particle spray.

Replaces the gala FardalMockStreamGenerator with an equivalent implementation
using galpy.orbit.Orbit (vectorised integration).  The physical result is
identical: test-particle "spray" that releases lead and trail stars at
uniformly-spaced times along the progenitor orbit, then integrates each to
the present day.

Reference: Fardal et al. (2015), Bovy (2015).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import scipy.optimize as opt
import yaml
from galpy.orbit import Orbit
from galpy.potential import vcirc

log = logging.getLogger(__name__)

# Galactocentric frame constants (mirror of potentials.py)
# Must match MWPotential2014 defaults (Bovy 2015) for internal consistency.
_RO: float = 8.0     # kpc
_VO: float = 220.0   # km/s
_STREAM_CONFIG_CACHE: dict[tuple[str, str], dict] = {}
_PROGENITOR_IC_CACHE: dict[tuple[str, int], dict] = {}


@dataclass
class StreamParticles:
    """Container for simulated stream star positions/velocities in stream frame."""
    phi1:    np.ndarray   # stream longitude [deg]
    phi2:    np.ndarray   # stream latitude  [deg]
    dist:    np.ndarray   # heliocentric distance [kpc]
    pm1:     np.ndarray   # pm along phi1 cos(phi2) [mas/yr]
    pm2:     np.ndarray   # pm along phi2 [mas/yr]
    vrad:    np.ndarray   # line-of-sight velocity [km/s]
    # Full 6D galactocentric Cartesian (used by subhalo.py for impulse kicks)
    xyz_kpc:  np.ndarray  # [3, N]
    vxyz_kms: np.ndarray  # [3, N]


def _load_stream_config(stream_name: str, config_path: str) -> dict:
    key = (str(Path(config_path).resolve()), stream_name)
    if key not in _STREAM_CONFIG_CACHE:
        with open(key[0]) as f:
            _STREAM_CONFIG_CACHE[key] = yaml.safe_load(f)["streams"][stream_name]
    return _STREAM_CONFIG_CACHE[key]


# ---------------------------------------------------------------------------
# Progenitor initial conditions
# ---------------------------------------------------------------------------

def set_progenitor_ic_track6d(
    stream_name: str,
    config_path: str = "config/streams.yaml",
    cache_dir: str | Path = "data/processed",
    mws=None,
) -> dict:
    """Derive progenitor IC DIRECTLY from the galstreams 6D track (no optimiser).

    The galstreams tracks carry literature 6D phase-space (position + velocity),
    already fit to the real stream's kinematics. Taking the track point nearest
    the centre of the observed phi1 range and using its galactocentric 6D as the
    progenitor IC reproduces the stream's true orbit (proper motions, phi2 track)
    by construction. This replaces the 5D phi2-RMS optimiser, which matched phi2
    geometry but landed on kinematically-wrong orbits (e.g. GD-1 pm1 -8.9 instead
    of the correct -12.8). Verified 2026-05-30: track-6D IC reproduces the GD-1
    track pm1/pm2/phi2 to within measurement scatter.
    """
    _IC_CACHE_VERSION = 15  # v15: direct galstreams 6D track IC
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"{stream_name}_progenitor_ic.json"
    ic_key = (str(cache_file.resolve()), _IC_CACHE_VERSION)
    if ic_key in _PROGENITOR_IC_CACHE:
        return _PROGENITOR_IC_CACHE[ic_key]
    if cache_file.exists():
        with open(cache_file) as f:
            d = json.load(f)
        if d.get("_cache_version") == _IC_CACHE_VERSION and d.get("_ic_method") == "track6d":
            _PROGENITOR_IC_CACHE[ic_key] = d
            return d

    if mws is None:
        import galstreams  # noqa: PLC0415
        mws = galstreams.MWStreams(verbose=False)
    sc = _load_stream_config(stream_name, config_path)
    track = mws[sc["galstreams_key"]]
    tr = track.track
    frame = track.stream_frame

    gcf = coord.Galactocentric(galcen_distance=_RO * u.kpc, z_sun=0.0208 * u.kpc)
    gc = tr.transform_to(gcf)
    trsf = tr.transform_to(frame)
    tphi1 = (np.array(trsf.phi1.deg) + 180.0) % 360.0 - 180.0
    phi1_min, phi1_max = sc["phi1_range_deg"]
    centre = 0.5 * (phi1_min + phi1_max)
    # Restrict to the observed phi1 window, pick the point nearest its centre.
    in_win = (tphi1 >= phi1_min) & (tphi1 <= phi1_max)
    idx_all = np.where(in_win)[0] if in_win.any() else np.arange(len(tphi1))
    j = idx_all[int(np.argmin(np.abs(tphi1[idx_all] - centre)))]

    pos = np.array([gc.x[j].to(u.kpc).value, gc.y[j].to(u.kpc).value, gc.z[j].to(u.kpc).value])
    try:
        cd = gc.cartesian.differentials["s"]
        vel = np.array([cd.d_x[j].to(u.km / u.s).value,
                        cd.d_y[j].to(u.km / u.s).value,
                        cd.d_z[j].to(u.km / u.s).value])
        if not np.all(np.isfinite(vel)) or np.linalg.norm(vel) < 1.0:
            raise ValueError("no usable track velocity")
    except (KeyError, AttributeError, ValueError) as e:
        log.warning("track6d IC for %s: no 6D velocity (%s); falling back to optimiser", stream_name, e)
        return set_progenitor_ic(stream_name, None, config_path, cache_dir, mws=mws)

    d = {
        "pos_kpc": pos.tolist(), "vel_kms": vel.tolist(),
        "_cache_version": _IC_CACHE_VERSION, "_ic_method": "track6d",
        "_track_phi1_deg": float(tphi1[j]),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(d, f)
    _PROGENITOR_IC_CACHE[ic_key] = d
    log.info("track6d IC for %s at phi1=%.1f: |v|=%.1f km/s", stream_name, tphi1[j], np.linalg.norm(vel))
    return d


def set_progenitor_ic(
    stream_name: str,
    potential: list,
    config_path: str = "config/streams.yaml",
    cache_dir: str | Path = "data/processed",
    mws=None,                    # pre-loaded galstreams.MWStreams (avoids re-load)
    use_jax: bool = False,       # use JAX force-table orbit integration for IC optimisation
) -> dict:
    """Derive progenitor initial conditions from galstreams track.

    NOTE (2026-05-30): the 5D phi2-RMS optimiser below was found to produce
    kinematically-wrong orbits (matches phi2 geometry, wrong proper motions).
    ``set_progenitor_ic_track6d`` (direct 6D track IC) is now the default used by
    ``generate_stream``. This optimiser is retained as a fallback for tracks that
    lack 6D velocities.

    Returns a dict with keys ``pos_kpc`` [3] and ``vel_kms`` [3]
    (galactocentric Cartesian, present-day).  Result is cached to JSON.
    """
    # JAX-optimised ICs use a separate cache file and version because the
    # tabulated force field is slightly different from galpy's exact potential.
    if use_jax:
        _IC_CACHE_VERSION = 14  # v14: JAX force-table-optimised ICs
        cache_dir = Path(cache_dir)
        cache_file = cache_dir / f"{stream_name}_progenitor_ic_jax.json"
    else:
        _IC_CACHE_VERSION = 13  # v13: short orbit window (1 Gyr) for dense near-t=0 sampling
        cache_dir = Path(cache_dir)
        cache_file = cache_dir / f"{stream_name}_progenitor_ic.json"

    ic_key = (str(cache_file.resolve()), _IC_CACHE_VERSION)
    if ic_key in _PROGENITOR_IC_CACHE:
        return _PROGENITOR_IC_CACHE[ic_key]

    if cache_file.exists():
        with open(cache_file) as f:
            d = json.load(f)
        if d.get("_cache_version") == _IC_CACHE_VERSION:
            log.info("Loaded cached progenitor IC for %s (v%d)", stream_name, _IC_CACHE_VERSION)
            _PROGENITOR_IC_CACHE[ic_key] = d
            return d
        else:
            log.warning("Stale IC cache for %s (v%s != v%d) — recomputing",
                        stream_name, d.get("_cache_version", "?"), _IC_CACHE_VERSION)
            cache_file.unlink()

    if mws is None:
        import galstreams  # noqa: PLC0415
        mws = galstreams.MWStreams(verbose=False)

    sc = _load_stream_config(stream_name, config_path)
    track = mws[sc["galstreams_key"]]

    # galstreams 1.0.2: track.track is an ICRS SkyCoord with full 6D kinematics.
    tr = track.track          # ICRS SkyCoord with 6D

    gc_frame = coord.Galactocentric(
        galcen_distance=_RO * u.kpc,
        z_sun=0.0208 * u.kpc,
    )

    # ── Convert the FULL galstreams track to galactocentric Cartesian ──────
    # This allows us to pick any point along the track as a candidate
    # progenitor position (not just the midpoint).
    gcen_all = tr.transform_to(gc_frame)
    pos_track = np.array([
        gcen_all.x.to(u.kpc).value,
        gcen_all.y.to(u.kpc).value,
        gcen_all.z.to(u.kpc).value,
    ])  # [3, N_track]

    # Extract velocity along the full track
    try:
        cd_all = gcen_all.cartesian.differentials["s"]
        vel_track = np.array([
            cd_all.d_x.to(u.km / u.s).value,
            cd_all.d_y.to(u.km / u.s).value,
            cd_all.d_z.to(u.km / u.s).value,
        ])  # [3, N_track]
        has_6d = True
    except (KeyError, AttributeError):
        has_6d = False
        vel_track = None

    # Use midpoint as baseline IC (same as v2)
    mid = len(tr) // 2
    pos0 = pos_track[:, mid].copy()
    if has_6d:
        vel0 = vel_track[:, mid].copy()
        log.info("Progenitor IC for %s from track 6D: r=%.1f kpc |v|=%.1f km/s",
                 stream_name, float(np.linalg.norm(pos0)), float(np.linalg.norm(vel0)))
    else:
        r   = float(np.linalg.norm(pos0))
        v_c = float(vcirc(potential, r * u.kpc, use_physical=True, ro=_RO, vo=_VO))
        phi = np.arctan2(pos0[1], pos0[0])
        vel0 = np.array([-v_c * np.sin(phi), v_c * np.cos(phi), 0.0])
        log.warning("No 6D kinematics for %s — using circular orbit fallback", stream_name)

    # ── 5D Progenitor IC optimisation ─────────────────────────────────────
    # Parameters optimised:
    #   p[0] = track_frac: position along galstreams track [0=start, 1=end]
    #   p[1] = a1: velocity rotation angle 1 (rad, about perp axis e1)
    #   p[2] = a2: velocity rotation angle 2 (rad, about perp axis e2)
    #   p[3] = v_scale: velocity magnitude scale factor (0.9 to 1.1)
    #   p[4] = pos_offset: offset perpendicular to orbit plane (kpc, ±2)
    #
    # Uses differential_evolution (global optimiser) — avoids local minima
    # that trapped Nelder-Mead in v2.  Each function evaluation takes ~0.5s
    # (orbit integration), and DE typically needs ~200-500 evals → 2-4 min.

    sf  = track.stream_frame
    tr_sf = tr.transform_to(sf)
    tr_phi1_all = (np.array(tr_sf.phi1.deg) + 180.0) % 360.0 - 180.0
    tr_phi2_all = np.array(tr_sf.phi2.deg)
    phi1_min, phi1_max = sc["phi1_range_deg"]
    tmask = (tr_phi1_all >= phi1_min) & (tr_phi1_all <= phi1_max)

    # Ensure cache_dir exists early (needed for checkpoint files during optimisation)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if tmask.sum() > 5:
        from scipy.interpolate import interp1d  # noqa: PLC0415
        from scipy.optimize import differential_evolution, minimize  # noqa: PLC0415

        sort_idx = np.argsort(tr_phi1_all[tmask])
        ref_interp = interp1d(
            tr_phi1_all[tmask][sort_idx],
            tr_phi2_all[tmask][sort_idx],
            kind="linear", bounds_error=False, fill_value="extrapolate",
        )

        n_track = pos_track.shape[1]
        stream_age = sc.get("isochrone_age_gyr", 10.0)
        # IC optimiser orbit integration time: use a SHORT window (1 Gyr,
        # ~2 orbital periods for typical halo streams) so that all 200
        # orbit samples are concentrated near t=0.  This gives dense
        # coverage of the first orbital passage through the phi1 range.
        #
        # Why not use the full disruption age (3 Gyr)?  With 200 samples
        # over 3 Gyr (6 orbital periods), only ~11 samples fall in the
        # first passage through the phi1 range.  Since 20 phi1 bins are
        # used, many bins miss the first passage and fall back to later
        # passages at wildly different phi2.  The "closest-to-t=0" metric
        # then mixes present-day and ancient orbit positions, producing
        # a 0.8° orbit RMS that hides a 1.6° systematic stream offset.
        #
        # 1 Gyr with 200 samples → 5 Myr spacing → ~34 points per
        # passage through the 120° phi1 range → every bin gets ≥1 point
        # from the first passage.  The stream generator still uses the
        # full disruption_age_gyr for particle spray.
        _ic_orbit_time_gyr = min(1.0, sc.get("disruption_age_gyr", min(3.0, stream_age)))
        n_orb = 200
        t_samp = np.linspace(0.0, -_ic_orbit_time_gyr, n_orb) * u.Gyr

        # Build rotation basis from the midpoint velocity
        v_hat = vel0 / np.linalg.norm(vel0)
        ref_axis = np.array([0.0, 0.0, 1.0]) if abs(v_hat[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(v_hat, ref_axis); e1 /= np.linalg.norm(e1)
        e2 = np.cross(v_hat, e1);       e2 /= np.linalg.norm(e2)
        # Orbit-normal direction for position offset
        r_hat = pos0 / np.linalg.norm(pos0)
        e_norm = np.cross(v_hat, r_hat)
        e_norm_mag = np.linalg.norm(e_norm)
        if e_norm_mag > 0.01:
            e_norm /= e_norm_mag
        else:
            e_norm = e2  # fallback

        # ── Build orbit evaluator (JAX or galpy) ──────────────────────────
        if use_jax:
            from src.simulation.jax_integrator import make_orbit_evaluator  # noqa: PLC0415
            _jax_orbit_eval = make_orbit_evaluator(
                potential, t_back_gyr=_ic_orbit_time_gyr, n_output=n_orb,
                ro=_RO, vo=_VO, dt_myr=1.0,
            )
            log.info("  Using JAX orbit evaluator for IC optimisation")
        else:
            _jax_orbit_eval = None

        _eval_count = [0]  # mutable counter for progress logging

        def _orbit_phi2_rms_5d(params):
            """RMS phi2 offset of progenitor orbit vs galstreams track (5D).

            Entire body wrapped in try/except to survive ANY Python-level
            error (orbit integration, coordinate transform, astropy ERFA).
            C-level segfaults are mitigated by pre/post-checks that reject
            degenerate configurations before they reach C code.
            """
            _eval_count[0] += 1
            if _eval_count[0] % 50 == 0:
                log.info("  DE eval %d …", _eval_count[0])
            try:
                track_frac, a1, a2, v_scale, pos_off = params

                # Pick position along track
                idx_float = track_frac * (n_track - 1)
                idx_lo = int(np.floor(idx_float))
                idx_hi = min(idx_lo + 1, n_track - 1)
                w = idx_float - idx_lo
                p = (1.0 - w) * pos_track[:, idx_lo] + w * pos_track[:, idx_hi]
                # Add perpendicular offset
                p = p + pos_off * e_norm

                # Pre-check 1: reject positions too close to galactic centre.
                # At R < 1 kpc the vR/vT cylindrical decomposition blows up
                # and galpy's C integrator can segfault.
                R_cyl = float(np.sqrt(p[0]**2 + p[1]**2))
                if R_cyl < 1.0:
                    return 999.0

                # Pick velocity: interpolate along track (if 6D), then rotate + scale
                if has_6d:
                    v_base = (1.0 - w) * vel_track[:, idx_lo] + w * vel_track[:, idx_hi]
                else:
                    v_base = vel0.copy()
                v_mag_base = float(np.linalg.norm(v_base))
                if v_mag_base < 1.0:
                    return 999.0
                # Rotate velocity direction
                v_rot = v_base + v_mag_base * (np.sin(a1) * e1 + np.sin(a2) * e2)
                v_rot_mag = float(np.linalg.norm(v_rot))
                if v_rot_mag < 1.0:
                    return 999.0
                v_rot *= (v_mag_base * v_scale) / v_rot_mag

                # Pre-check 2: reject extreme velocities (escape velocity ~ 550 km/s
                # at 8 kpc; anything above 1500 is unphysical and crashes galpy C code)
                if float(np.linalg.norm(v_rot)) > 1500.0:
                    return 999.0

                if _jax_orbit_eval is not None:
                    # JAX orbit evaluator — uses tabulated force table
                    xs, ys, zs, vxs, vys, vzs = _jax_orbit_eval(p, v_rot)
                else:
                    orb = _pos_vel_to_orbit(p, v_rot)
                    orb.integrate(t_samp, potential, method="leapfrog_c", progressbar=False)
                    # Vectorised extraction — galpy Orbit stores arrays internally;
                    # calling orb.x(t_array) returns the full array in one shot.
                    xs  = np.atleast_1d(orb.x(t_samp,  use_physical=True))
                    ys  = np.atleast_1d(orb.y(t_samp,  use_physical=True))
                    zs  = np.atleast_1d(orb.z(t_samp,  use_physical=True))
                    vxs = np.atleast_1d(orb.vx(t_samp, use_physical=True))
                    vys = np.atleast_1d(orb.vy(t_samp, use_physical=True))
                    vzs = np.atleast_1d(orb.vz(t_samp, use_physical=True))

                # Post-check 1: all orbit values must be finite
                if not all(np.isfinite(v).all() for v in [xs, ys, zs, vxs, vys, vzs]):
                    return 999.0

                # Post-check 2: orbit must not pass through galactic centre.
                # The Galactocentric → ICRS coordinate transform in astropy uses
                # ERFA C code that segfaults when R → 0 (angle computation degeneracy).
                R_orbit = np.sqrt(xs**2 + ys**2)
                if float(np.min(R_orbit)) < 0.3:
                    return 999.0

                # Post-check 3: reject overly eccentric orbits.
                # GD-1 and similar thin cold streams are on mildly eccentric orbits
                # (R_apo/R_peri < 4).  Highly eccentric orbits can match the phi2
                # track at arbitrary past times but produce wildly wrong streams at
                # t=0.  The 3D radius r (not just cylindrical R) is used because
                # some streams have significant z excursions.
                r_orbit = np.sqrt(xs**2 + ys**2 + zs**2)
                ecc_ratio = float(np.max(r_orbit)) / max(float(np.min(r_orbit)), 0.1)
                if ecc_ratio > 5.0:
                    return 999.0

                phi1_o, phi2_o = _galactocentric_to_stream_phi12(
                    np.stack([xs, ys, zs]), sf
                )
                in_range = (phi1_o >= phi1_min) & (phi1_o <= phi1_max)
                if in_range.sum() < 5:
                    return 999.0

                # Time-weighted comparison: the orbit wraps through the phi1
                # range multiple times over the disruption age.  The median
                # phi2 over all passes conflates present-day and ancient passes,
                # letting the optimizer find orbits that matched the track
                # *in the past* but are 10+ deg off at t=0.  Stream particles
                # at t=0 cluster around the present-day orbital passage.
                #
                # Fix: for each phi1 bin, use only the orbit point closest to
                # t=0.  This forces the optimizer to fit the CURRENT passage.
                # |t| array aligned with orbit points (t_samp goes 0 → -T):
                t_abs = np.abs(np.linspace(0.0, _ic_orbit_time_gyr, n_orb))

                n_bins = 20
                bin_e = np.linspace(phi1_min, phi1_max, n_bins + 1)
                bc    = 0.5 * (bin_e[:-1] + bin_e[1:])
                offsets = []
                for k in range(n_bins):
                    mk = in_range & (phi1_o >= bin_e[k]) & (phi1_o < bin_e[k + 1])
                    if mk.sum() >= 1:
                        # Pick the orbit point closest to t=0 in this bin
                        bin_indices = np.where(mk)[0]
                        best_idx = bin_indices[np.argmin(t_abs[bin_indices])]
                        offsets.append(float(phi2_o[best_idx]) - float(ref_interp(bc[k])))
                if not offsets:
                    return 999.0
                return float(np.sqrt(np.mean(np.array(offsets) ** 2)))

            except Exception as exc:
                if _eval_count[0] <= 5:
                    log.debug("  IC eval %d exception: %s", _eval_count[0], exc)
                return 999.0

        # Phase 1: Global search with differential_evolution
        # TIGHT bounds: keep orbit realistic.  Wide v_scale ([0.7, 1.3])
        # lets the optimizer find high-energy orbits that pass through the
        # stream region at arbitrary past times but are wildly eccentric
        # (R_max > 100 kpc) and wrong at t=0.  Constraining v_scale to ±10%
        # and angles to ±17° keeps the orbit close to the galstreams track
        # velocity, preventing unrealistic eccentricity.
        bounds = [
            (0.0, 1.0),      # track_frac — full range along reference track
            (-0.3, 0.3),     # a1 (rad, ~17 deg) — velocity rotation angle 1
            (-0.3, 0.3),     # a2 (rad, ~17 deg) — velocity rotation angle 2
            (0.90, 1.10),    # v_scale — ±10% velocity rescaling (keep orbit realistic)
            (-2.0, 2.0),     # pos_offset (kpc) — moderate spatial shift
        ]

        rms_before = _orbit_phi2_rms_5d([0.5, 0.0, 0.0, 1.0, 0.0])
        log.info("Optimising progenitor IC for %s (5D global search, ~3-6 min) …", stream_name)
        log.info("  Initial phi2 RMS = %.3f deg", rms_before)

        # ── Crash-resilient checkpoint callback ──────────────────────────
        # galpy's C extension can segfault on certain orbit configurations,
        # killing the entire process mid-optimisation.  The callback saves
        # the best solution after each DE generation so we can recover.
        _checkpoint_file = cache_dir / f"{stream_name}_ic_checkpoint.json"
        _best_checkpoint = {"fun": 999.0, "x": None, "gen": 0, "nfev": 0}

        def _de_checkpoint_callback(xk, convergence):
            """Save best-so-far after each DE generation."""
            current_rms = _orbit_phi2_rms_5d(xk)
            _eval_count[0] -= 1  # don't count this extra eval in the progress log
            gen = _best_checkpoint["gen"] + 1
            _best_checkpoint["gen"] = gen
            _best_checkpoint["nfev"] = _eval_count[0]
            if current_rms < _best_checkpoint["fun"]:
                _best_checkpoint["fun"] = current_rms
                _best_checkpoint["x"] = xk.tolist()
                try:
                    import json as _json
                    with open(_checkpoint_file, "w") as _f:
                        _json.dump(_best_checkpoint, _f)
                except Exception:
                    pass
            if gen % 10 == 0:
                log.info("  DE gen %d: best RMS = %.3f deg (convergence=%.4f)",
                         gen, _best_checkpoint["fun"], convergence)
            return False  # don't stop

        # Check for a checkpoint from a crashed previous run
        _recovered_from_checkpoint = False
        if _checkpoint_file.exists():
            try:
                import json as _json
                with open(_checkpoint_file) as _f:
                    _ckpt = _json.load(_f)
                if _ckpt.get("fun", 999.0) < 50.0 and _ckpt.get("x") is not None:
                    log.info("  Recovered checkpoint from crashed run: gen=%d RMS=%.3f deg",
                             _ckpt["gen"], _ckpt["fun"])
                    _recovered_from_checkpoint = True
                    # Use recovered solution as seed — skip straight to polish
                    _checkpoint_best_x = np.array(_ckpt["x"])
                    _checkpoint_best_fun = _ckpt["fun"]
            except Exception:
                pass

        if _recovered_from_checkpoint:
            # Skip DE global search — use recovered checkpoint
            log.info("  Skipping DE (using recovered checkpoint) — proceeding to local polish")
            _de_best_x = _checkpoint_best_x
            _de_best_fun = _checkpoint_best_fun
            _de_nfev = _ckpt.get("nfev", 0)
        else:
            res_global = differential_evolution(
                _orbit_phi2_rms_5d,
                bounds=bounds,
                seed=42,
                maxiter=45,          # 45 generations × popsize~75 ≈ 3375 evals
                popsize=15,          # 15 × 5D = 75 individuals
                tol=0.002,           # tighter convergence tolerance
                polish=False,        # we'll do our own polish
                workers=1,           # must be 1 — galpy orbit is not thread-safe
                updating="deferred",
                callback=_de_checkpoint_callback,
            )
            log.info("  DE global search: phi2 RMS = %.3f deg (after %d evals)",
                     res_global.fun, res_global.nfev)
            _de_best_x = res_global.x
            _de_best_fun = res_global.fun
            _de_nfev = res_global.nfev

        # Phase 2: Local polish with Nelder-Mead around the DE solution
        res_local = minimize(
            _orbit_phi2_rms_5d,
            x0=_de_best_x,
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 0.001, "maxiter": 1000},
        )
        rms_after = res_local.fun
        log.info("  Local polish: phi2 RMS = %.3f deg (total %d + %d evals)",
                 rms_after, _de_nfev, res_local.nfev)

        best_params = res_local.x if res_local.fun < _de_best_fun else _de_best_x
        best_rms = min(res_local.fun, _de_best_fun)

        # Clean up checkpoint file on successful completion
        if _checkpoint_file.exists():
            try:
                _checkpoint_file.unlink()
            except Exception:
                pass

        if best_rms < rms_before:
            track_frac, a1_opt, a2_opt, v_scale_opt, pos_off_opt = best_params

            # Reconstruct the optimised pos and vel
            idx_float = track_frac * (n_track - 1)
            idx_lo = int(np.floor(idx_float))
            idx_hi = min(idx_lo + 1, n_track - 1)
            w = idx_float - idx_lo
            pos0 = (1.0 - w) * pos_track[:, idx_lo] + w * pos_track[:, idx_hi]
            pos0 = pos0 + pos_off_opt * e_norm

            if has_6d:
                v_base = (1.0 - w) * vel_track[:, idx_lo] + w * vel_track[:, idx_hi]
            else:
                v_base = vel0.copy()
            v_mag_base = float(np.linalg.norm(v_base))
            vel0 = v_base + v_mag_base * (np.sin(a1_opt) * e1 + np.sin(a2_opt) * e2)
            vel0 *= (v_mag_base * v_scale_opt) / np.linalg.norm(vel0)

            log.info("  IC optimisation PASS: phi2 RMS %.3f -> %.3f deg", rms_before, best_rms)
            log.info("    track_frac=%.3f  a1=%.4f  a2=%.4f  v_scale=%.4f  pos_off=%.3f kpc",
                     track_frac, a1_opt, a2_opt, v_scale_opt, pos_off_opt)
        else:
            log.warning("  IC optimisation did not improve phi2 RMS (%.3f deg); keeping raw IC",
                        rms_before)

        if best_rms > 1.0:
            log.warning("  WARNING: phi2 RMS = %.3f deg > 1.0 deg target for %s. "
                        "Consider adjusting MW potential parameters.", best_rms, stream_name)

    else:
        best_rms = None
        log.warning("Too few galstreams track points in phi1 range for %s — skipping IC optimisation",
                    stream_name)

    # ── Cache and return ───────────────────────────────────────────────────
    d = {
        "pos_kpc": pos0.tolist(),
        "vel_kms": vel0.tolist(),
        "_cache_version": _IC_CACHE_VERSION,
        "_rms_deg": float(best_rms) if best_rms is not None else None,
        "_use_jax": use_jax,
    }
    with open(cache_file, "w") as f:
        json.dump(d, f)
    _PROGENITOR_IC_CACHE[ic_key] = d
    return d


# ---------------------------------------------------------------------------
# Fardal-like particle spray using galpy vectorised integration
# ---------------------------------------------------------------------------

def _galactocentric_to_stream_phi12(
    pos: np.ndarray,   # [3, N] kpc
    frame,             # galstreams astropy frame
) -> tuple:
    """Fast: convert galactocentric positions → stream (phi1, phi2) only.

    Skips velocity transforms and ICRS conversion — ~3-5× faster than
    the full _galactocentric_to_stream_coords.  Used in the IC optimiser
    inner loop where only phi1/phi2 are needed.
    """
    gc = coord.Galactocentric(
        x=pos[0] * u.kpc, y=pos[1] * u.kpc, z=pos[2] * u.kpc,
        galcen_distance=_RO * u.kpc,
        z_sun=0.0208 * u.kpc,
    )
    stream_sc = gc.transform_to(frame)
    phi1 = np.asarray(stream_sc.phi1.deg)
    phi1 = (phi1 + 180.0) % 360.0 - 180.0
    phi2 = np.asarray(stream_sc.phi2.deg)
    return phi1, phi2


def _galactocentric_to_stream_coords(
    pos: np.ndarray,   # [3, N] kpc
    vel: np.ndarray,   # [3, N] km/s
    frame,             # galstreams astropy frame
) -> tuple:
    """Convert galactocentric Cartesian arrays to stream-frame observables."""
    # Use SkyCoord (not bare Galactocentric frame) so .transform_to() accepts
    # both frame instances and string names (e.g. "icrs").
    sc = coord.SkyCoord(
        x=pos[0] * u.kpc, y=pos[1] * u.kpc, z=pos[2] * u.kpc,
        v_x=vel[0] * u.km / u.s, v_y=vel[1] * u.km / u.s, v_z=vel[2] * u.km / u.s,
        frame=coord.Galactocentric(
            galcen_distance=_RO * u.kpc,
            z_sun=0.0208 * u.kpc,
        ),
    )
    stream_sc = sc.transform_to(frame)

    phi1   = stream_sc.phi1.deg
    # Astropy Longitude defaults to [0, 360]; normalise to [-180, 180] to match
    # gala/stream convention (e.g. GD-1 phi1 ∈ [-100, +20], not [260, 380])
    phi1   = (phi1 + 180.0) % 360.0 - 180.0
    phi2   = stream_sc.phi2.deg
    pm1    = stream_sc.pm_phi1_cosphi2.to(u.mas / u.yr).value
    pm2    = stream_sc.pm_phi2.to(u.mas / u.yr).value
    icrs   = sc.transform_to("icrs")
    dist   = icrs.distance.to(u.kpc).value
    vrad   = icrs.radial_velocity.to(u.km / u.s).value
    return phi1, phi2, dist, pm1, pm2, vrad


def generate_stream(
    stream_name: str,
    potential: list,
    n_stars: int = 2000,
    seed: int = 0,
    stream_age_gyr: Optional[float] = None,
    config_path: str = "config/streams.yaml",
    progenitor_ic: Optional[dict] = None,
    cache_dir: str | Path = "data/processed",
    mws=None,                    # pre-loaded galstreams.MWStreams (avoids re-load)
    n_steps_back: int = 100,     # release-time bins; fewer = faster (fewer galpy calls)
) -> StreamParticles:
    """Generate an unperturbed stream via Fardal-like particle spray (galpy).

    Algorithm
    ---------
    1. Integrate progenitor orbit backward for T_age.
    2. Release n_stars test particles at uniformly-spaced times (half lead,
       half trail), each offset by the tidal radius and a velocity kick
       σ_v = k_v × Ω × r_tidal  (Fardal et al. 2015, k_v=0.3).
    3. Integrate particles forward to t = 0, grouped by release time
       to exploit galpy's vectorised orbit integration.
    4. Convert to stream-frame observables via galstreams frame.

    Pass ``mws=galstreams.MWStreams()`` from a worker process to avoid
    reloading the 141-track catalog (~13 s) on every simulation call.

    Returns
    -------
    StreamParticles with phi1/phi2/dist/pm1/pm2/vrad and galactocentric 6D.
    """
    # Load galstreams ONCE here; pass to set_progenitor_ic to avoid double load
    if mws is None:
        import galstreams  # noqa: PLC0415
        mws = galstreams.MWStreams(verbose=False)

    rng = np.random.default_rng(seed)
    sc  = _load_stream_config(stream_name, config_path)

    if stream_age_gyr is None:
        # spray_age_gyr is a LENGTH-CALIBRATION parameter: the effective spray
        # timescale tuned so the generated stream matches the observed angular
        # extent (the simple Fardal spray under-produces length per unit time,
        # so this is larger than the literature "recent disruption rate" age).
        # It is kept separate from disruption_age_gyr (the physical timescale the
        # forward model uses to cap time-since-impact). Falls back to
        # disruption_age_gyr, then isochrone_age_gyr.
        stream_age_gyr = sc.get("spray_age_gyr",
                                sc.get("disruption_age_gyr",
                                       sc.get("isochrone_age_gyr", 10.0)))

    # ── Detect JAX mode early ─────────────────────────────────────────
    # JAX mode uses the tabulated force field for ALL orbit calculations
    # (IC optimisation, progenitor backward orbit, particle integration)
    # to ensure full consistency.
    import os as _os
    _use_jax = _os.environ.get("USE_JAX_INTEGRATOR", "0") == "1"
    if _use_jax:
        try:
            from src.simulation.jax_integrator import (  # noqa: PLC0415
                JAX_AVAILABLE,
                integrate_particles_jax,
                integrate_progenitor_spray_jax,
            )
            if not JAX_AVAILABLE:
                _use_jax = False
        except ImportError:
            _use_jax = False

    if progenitor_ic is None:
        if _use_jax:
            progenitor_ic = set_progenitor_ic(
                stream_name, potential, config_path, cache_dir, mws=mws, use_jax=True)
        else:
            # Direct 6D track IC (correct kinematics) — replaces the phi2-RMS
            # optimiser that produced wrong-orbit streams.
            progenitor_ic = set_progenitor_ic_track6d(
                stream_name, config_path, cache_dir, mws=mws)

    pos0 = np.array(progenitor_ic["pos_kpc"])   # [3]
    vel0 = np.array(progenitor_ic["vel_kms"])    # [3]

    # ---- 1. Integrate progenitor orbit backward ---------------------
    # Release-time bins: n_steps_back bins, each batch holds n_stars/n_steps_back
    # particles on average.  Fewer bins → fewer galpy orbit calls → faster.
    # 100 bins is the sweet spot: coverage is still adequate for the Fardal spray
    # while batch sizes (~20 particles) are large enough to amortise per-call
    # overhead. Use 200 for high-fidelity runs (e.g. real-data inference).
    t_back_gyr = np.linspace(0.0, -stream_age_gyr, n_steps_back)

    if _use_jax:
        # JAX progenitor orbit — uses tabulated force table (consistent with
        # JAX-optimised ICs and JAX particle integration downstream).
        spray = integrate_progenitor_spray_jax(
            pos0, vel0, t_back_gyr=stream_age_gyr,
            n_steps_back=n_steps_back, potential=potential,
            ro=_RO, vo=_VO, dt_myr=1.0,
        )
        prog_pos  = spray['pos']      # [n_steps_back, 3]
        prog_vel  = spray['vel']      # [n_steps_back, 3]
        r_mag     = spray['r_mag']    # [n_steps_back, 1]
        r_hat     = spray['r_hat']    # [n_steps_back, 3]
        v_c_orbit = spray['v_c']      # [n_steps_back]

    else:
        t_back = t_back_gyr * u.Gyr

        prog_orbit = _pos_vel_to_orbit(pos0, vel0)
        # Use leapfrog_c (NOT dop853_c): dop853_c segfaults on degenerate potential
        # configurations via STATUS_ACCESS_VIOLATION at C level — no Python fallback
        # is possible. leapfrog_c uses a fixed-size buffer and is immune.
        prog_orbit.integrate(t_back, potential, method="leapfrog_c", progressbar=False)

        # Positions/velocities of progenitor along backward orbit: [N_steps, 3]
        # t_back is already a Quantity (Gyr) — do NOT multiply by u.Gyr again
        prog_R   = prog_orbit.R(t_back, use_physical=True)    # kpc
        prog_vR  = prog_orbit.vR(t_back, use_physical=True)   # km/s
        prog_vT  = prog_orbit.vT(t_back, use_physical=True)   # km/s
        prog_z   = prog_orbit.z(t_back, use_physical=True)    # kpc
        prog_vz  = prog_orbit.vz(t_back, use_physical=True)   # km/s
        prog_phi = prog_orbit.phi(t_back, use_physical=False)  # rad

        prog_x = prog_R * np.cos(prog_phi)
        prog_y = prog_R * np.sin(prog_phi)
        prog_vx = prog_vR * np.cos(prog_phi) - prog_vT * np.sin(prog_phi)
        prog_vy = prog_vR * np.sin(prog_phi) + prog_vT * np.cos(prog_phi)

        # Normalised radial direction (for tidal offset)
        prog_pos = np.stack([prog_x, prog_y, prog_z], axis=-1)       # [N_steps, 3]
        prog_vel = np.stack([prog_vx, prog_vy, prog_vz], axis=-1)     # [N_steps, 3]
        r_mag    = np.linalg.norm(prog_pos, axis=-1, keepdims=True)   # [N_steps, 1]
        r_hat    = prog_pos / np.clip(r_mag, 1e-6, None)              # [N_steps, 3]

        # Pre-compute circular velocity at every progenitor orbit position (vectorised).
        # Avoids 5000 scalar vcirc calls inside _make_perturbed_ic.
        # galpy accepts array-valued R — one C call instead of one per particle.
        v_c_orbit = np.atleast_1d(np.asarray(
            vcirc(potential, r_mag[:, 0] * u.kpc, use_physical=True, ro=_RO, vo=_VO)
        ))   # shape: [n_steps_back], km/s

    # ---- 2. Sample release times and build perturbed ICs ------------
    n_lead  = n_stars // 2
    n_trail = n_stars - n_lead

    # Progenitor mass and tidal radius scaling
    m_prog_msun = sc.get("prog_mass_solar", 2e4)

    # Particle-spray release, configurable via a per-stream `spray` block.
    #
    # The release is written in the orbital frame (radial / in-plane-tangential /
    # out-of-plane) so individual components can be tuned. IMPORTANT empirical
    # finding (2026-05-30): a naive "proper Fardal" release with the literature
    # radial Lagrange offset kr_mean=2.0 (without the correlated velocity that
    # keeps the star bound at the Lagrange point, as in gala's FardalStreamDF)
    # is STRICTLY WORSE here than the simple isotropic kick: it over-lengthens the
    # stream, fans phi2 wider, and shifts the net proper motion off the track
    # (GD-1 pm1 -12.8 -> -10.3). The phi2 width is dominated by energy-spread /
    # orbital-precession fanning over the long spray age, not by the out-of-plane
    # release component, so the release prescription alone cannot fix it. The
    # DEFAULTS below therefore reproduce the simpler isotropic spray (kr_mean=1,
    # zero-mean isotropic velocity dispersion 0.3*v_scale), which preserves the
    # correct orbit. A correctly-correlated Fardal release (or a working gala
    # install / N-body) is the real fix and is left as future work; the knobs are
    # exposed for that.
    spray = sc.get("spray", {})
    kr_mean  = float(spray.get("kr_mean", 1.0))     # radial Lagrange offset [r_tidal]
    kr_std   = float(spray.get("kr_std", 0.0))
    kvt_mean = float(spray.get("kvt_mean", 0.0))    # in-plane tangential velocity offset [v_scale]
    kvt_std  = float(spray.get("kvt_std", 0.3))     # in-plane velocity dispersion [v_scale]
    k_perp   = float(spray.get("k_perp", 0.3))      # out-of-plane dispersion [v_scale] (0.3 = isotropic)
    kr_pos_perp = float(spray.get("kr_pos_perp", 0.0))  # out-of-plane position scatter [r_tidal]

    # Sample release times uniformly along the backward orbit
    idx_lead  = rng.integers(0, n_steps_back, n_lead)
    idx_trail = rng.integers(0, n_steps_back, n_trail)

    def _make_perturbed_ic(idx_arr: np.ndarray, sign: float) -> tuple:
        """sign=+1 for lead, sign=-1 for trail. Anisotropic orbital-frame release."""
        p_pos = prog_pos[idx_arr]          # [N, 3]
        p_vel = prog_vel[idx_arr]          # [N, 3]
        p_r   = r_mag[idx_arr, 0]          # [N]
        p_rh  = r_hat[idx_arr]             # [N, 3]
        n = len(idx_arr)

        # Tidal (Jacobi) radius via M_enc ~ v_c^2 r / G
        G_kpc = 4.3009e-6
        v_c_arr = v_c_orbit[idx_arr]       # [N], km/s
        m_enc = v_c_arr**2 * p_r / G_kpc
        r_tidal = p_r * (m_prog_msun / (3.0 * np.clip(m_enc, 1.0, None))) ** (1.0 / 3.0)
        omega = v_c_arr / p_r              # km/s / kpc
        v_scale = omega * r_tidal          # km/s  (Fardal velocity scale)

        # Orbital-frame basis per particle: radial r_hat, normal n_hat (out of
        # orbital plane), in-plane tangential t_hat.
        v_hat = p_vel / np.clip(np.linalg.norm(p_vel, axis=1, keepdims=True), 1e-6, None)
        n_hat = np.cross(p_rh, v_hat)
        n_hat = n_hat / np.clip(np.linalg.norm(n_hat, axis=1, keepdims=True), 1e-6, None)
        t_hat = np.cross(n_hat, p_rh)
        t_hat = t_hat / np.clip(np.linalg.norm(t_hat, axis=1, keepdims=True), 1e-6, None)

        # Position offset: radial Lagrange point + small out-of-plane scatter.
        kr = rng.normal(kr_mean, kr_std, n)
        kz = rng.normal(0.0, kr_pos_perp, n)
        pos_new = (p_pos
                   + sign * (kr * r_tidal)[:, None] * p_rh
                   + (kz * r_tidal)[:, None] * n_hat)

        # Velocity offset: mean tangential drift (sets along-stream length) +
        # in-plane radial scatter + SMALL out-of-plane scatter (sets thinness).
        kvt = rng.normal(kvt_mean, kvt_std, n)
        kvr = rng.normal(0.0, kvt_std, n)
        kvz = rng.normal(0.0, k_perp, n)
        dv = ((kvt * v_scale)[:, None] * t_hat
              + (kvr * v_scale)[:, None] * p_rh
              + (kvz * v_scale)[:, None] * n_hat)
        vel_new = p_vel + sign * dv

        t_release = t_back_gyr[idx_arr]    # Gyr, negative
        return pos_new, vel_new, t_release

    pos_lead,  vel_lead,  t_lead  = _make_perturbed_ic(idx_lead,  +1.0)
    pos_trail, vel_trail, t_trail = _make_perturbed_ic(idx_trail, -1.0)

    # ---- 3. Integrate each particle from release time to t=0 -----------
    # Fardal spray particles have different release times so they cannot share
    # a single time grid.  We group particles by unique release-time index and
    # use galpy's vectorised Orbit to integrate each group as a batch.
    # Benchmark: ~15x faster than a per-particle loop (1.0 ms vs 14 ms/orbit).

    def _integrate_particles(pos_arr, vel_arr, t_release_arr):
        """Return final [N, 3] pos/vel at t=0 using batched galpy orbits.

        Particles sharing the same release time are integrated together in a
        single vectorised galpy Orbit call.  For n_stars=5000 with
        n_steps_back=200 the typical batch size is ~25 → ~15× speedup over
        the per-orbit loop.

        Robustness notes:
        - Uses leapfrog_c (fixed-step symplectic) instead of dop853_c (adaptive
          RK8).  dop853_c can segfault on degenerate initial conditions because
          its adaptive step-controller allocates variable-size buffers; leapfrog_c
          uses a fixed-size buffer and is immune to this.
        - Guards against R < 0.5 kpc: near the galactic centre vR and vT diverge
          due to division by R, producing NaN/Inf that cause an access violation
          inside galpy's C extension.  Affected particles are moved to R=0.5 kpc
          and will be scatter-screened by the phi1/phi2 selection cut downstream.
        """
        # Minimum cylindrical radius to avoid division-by-R blow-up
        _R_MIN_KPC = 0.5
        # Maximum speed: escape velocity at 8 kpc from NFW halo ≈ 550 km/s;
        # anything above 2000 km/s is unphysical and will crash galpy's C code.
        _V_MAX_KMS = 2000.0

        n = len(pos_arr)
        results_pos = np.empty((n, 3))
        results_vel = np.empty((n, 3))

        unique_t, inverse = np.unique(t_release_arr, return_inverse=True)

        for gi, t_rel in enumerate(unique_t):
            grp = np.where(inverse == gi)[0]   # indices into pos_arr for this t_rel

            # t_rel = 0 means released at present day — no integration needed.
            # Also guards against np.linspace(0, 0, n) which gives dt=0 and
            # hangs galpy's C integrator in an infinite loop.
            if abs(float(t_rel)) < 1e-9:
                results_pos[grp] = pos_arr[grp]
                results_vel[grp] = vel_arr[grp]
                continue

            x  = pos_arr[grp, 0];  y  = pos_arr[grp, 1];  z  = pos_arr[grp, 2]
            vx = vel_arr[grp, 0];  vy = vel_arr[grp, 1];  vz = vel_arr[grp, 2]

            R   = np.sqrt(x**2 + y**2)

            # Guard 1: clamp tiny R to avoid vR/vT divergence
            R = np.maximum(R, _R_MIN_KPC)

            phi = np.arctan2(y, x)
            vR  = (x * vx + y * vy) / R
            vT  = (x * vy - y * vx) / R   # positive = prograde

            # Guard 2: clip extreme velocities that would NaN the C integrator
            speed = np.sqrt(vR**2 + vT**2 + vz**2)
            too_fast = speed > _V_MAX_KMS
            if too_fast.any():
                scale = np.where(too_fast, _V_MAX_KMS / np.maximum(speed, 1.0), 1.0)
                vR *= scale;  vT *= scale;  vz *= scale

            # Use ~25 Myr resolution; leapfrog_c requires ≥2 steps
            n_int = max(8, int(abs(float(t_rel)) * 40))   # 25 Myr resolution
            t_fwd = np.linspace(float(t_rel), 0.0, n_int) * u.Gyr

            # galpy vectorised Orbit: vxvv components are 1-D arrays of length |grp|
            orb = Orbit(
                vxvv=[R * u.kpc, vR * u.km / u.s, vT * u.km / u.s,
                      z * u.kpc, vz * u.km / u.s, phi * u.rad],
                ro=_RO, vo=_VO,
            )
            # Use dop853_c (fast adaptive RK8).
            # The pre-guards above (R ≥ 0.5 kpc, speed ≤ 2000 km/s) prevent the
            # degenerate initial conditions that trigger STATUS_ACCESS_VIOLATION.
            # Crash isolation is handled at the process level in generate_training_data.py
            # (multiprocessing.Pool with maxtasksperchild), so any remaining crash
            # loses only one simulation, not an entire batch.
            # If dop853_c returns NaN (non-crash error), fall back to leapfrog_c.
            orb.integrate(t_fwd, potential, method="dop853_c", progressbar=False, numcores=1)

            if not np.isfinite(orb.orbit).all():
                orb = Orbit(
                    vxvv=[R * u.kpc, vR * u.km / u.s, vT * u.km / u.s,
                          z * u.kpc, vz * u.km / u.s, phi * u.rad],
                    ro=_RO, vo=_VO,
                )
                orb.integrate(t_fwd, potential, method="leapfrog_c", progressbar=False, numcores=1)
                if not np.isfinite(orb.orbit).all():
                    results_pos[grp] = np.array([500.0, 0.0, 0.0])
                    results_vel[grp] = 0.0
                    continue

            t0 = t_fwd[-1]

            results_pos[grp, 0] = np.atleast_1d(np.asarray(orb.x( t0, use_physical=True)))
            results_pos[grp, 1] = np.atleast_1d(np.asarray(orb.y( t0, use_physical=True)))
            results_pos[grp, 2] = np.atleast_1d(np.asarray(orb.z( t0, use_physical=True)))
            results_vel[grp, 0] = np.atleast_1d(np.asarray(orb.vx(t0, use_physical=True)))
            results_vel[grp, 1] = np.atleast_1d(np.asarray(orb.vy(t0, use_physical=True)))
            results_vel[grp, 2] = np.atleast_1d(np.asarray(orb.vz(t0, use_physical=True)))

        return results_pos, results_vel

    # _use_jax was determined at the top of generate_stream() and JAX modules
    # were imported there if available.  Use the same flag for particle integration.
    if _use_jax:
        log.info("Using JAX integrator for particle integration")
        pos_lead_f, vel_lead_f = integrate_particles_jax(
            pos_lead, vel_lead, t_lead, potential, ro=_RO, vo=_VO, dt_myr=2.0,
        )
        pos_trail_f, vel_trail_f = integrate_particles_jax(
            pos_trail, vel_trail, t_trail, potential, ro=_RO, vo=_VO, dt_myr=2.0,
        )
    else:
        pos_lead_f,  vel_lead_f  = _integrate_particles(pos_lead,  vel_lead,  t_lead)
        pos_trail_f, vel_trail_f = _integrate_particles(pos_trail, vel_trail, t_trail)

    pos_all = np.concatenate([pos_lead_f, pos_trail_f], axis=0).T   # [3, N]
    vel_all = np.concatenate([vel_lead_f, vel_trail_f], axis=0).T   # [3, N]

    # ---- 4. Convert to stream-frame observables ---------------------
    # mws was already loaded at the top of this function (or passed in)
    track = mws[sc["galstreams_key"]]
    frame = track.stream_frame

    phi1, phi2, dist, pm1, pm2, vrad = _galactocentric_to_stream_coords(
        pos_all, vel_all, frame,
    )

    # Apply selection cuts
    phi1_min, phi1_max = sc["phi1_range_deg"]
    mask  = (phi1 >= phi1_min) & (phi1 <= phi1_max)
    mask &= np.abs(phi2) < sc["phi2_selection_deg"]

    return StreamParticles(
        phi1=phi1[mask],
        phi2=phi2[mask],
        dist=dist[mask],
        pm1=pm1[mask],
        pm2=pm2[mask],
        vrad=vrad[mask],
        xyz_kpc=pos_all[:, mask],
        vxyz_kms=vel_all[:, mask],
    )


def resample_to_observational_selection(
    particles: StreamParticles,
    target_n: int,
    seed: int = 0,
) -> StreamParticles:
    """Subsample stream to approximately match observed member count."""
    rng = np.random.default_rng(seed)
    n = len(particles.phi1)
    if n <= target_n:
        return particles
    idx = rng.choice(n, target_n, replace=False)
    return StreamParticles(
        phi1=particles.phi1[idx],
        phi2=particles.phi2[idx],
        dist=particles.dist[idx],
        pm1=particles.pm1[idx],
        pm2=particles.pm2[idx],
        vrad=particles.vrad[idx],
        xyz_kpc=particles.xyz_kpc[:, idx],
        vxyz_kms=particles.vxyz_kms[:, idx],
    )


# ---------------------------------------------------------------------------
# Helper: build a galpy Orbit from galactocentric Cartesian pos/vel
# ---------------------------------------------------------------------------

def _pos_vel_to_orbit(pos_kpc: np.ndarray, vel_kms: np.ndarray) -> Orbit:
    """Construct a galpy Orbit from galactocentric Cartesian pos/vel (physical)."""
    x, y, z       = float(pos_kpc[0]), float(pos_kpc[1]), float(pos_kpc[2])
    vx, vy, vz    = float(vel_kms[0]), float(vel_kms[1]), float(vel_kms[2])

    R   = np.sqrt(x**2 + y**2)
    # Guard: clamp R to avoid division-by-zero / vR,vT blow-up that produces
    # NaN → crash inside galpy's C extension.  Mirrors the _R_MIN_KPC = 0.5 kpc
    # guard in _integrate_particles.  Progenitor ICs are never this close to the
    # GC in practice; the clamp is a pure safety net.
    R   = max(R, 0.5)   # kpc
    phi = np.arctan2(y, x)
    vR  = (x * vx + y * vy) / R
    vT  = (x * vy - y * vx) / R   # positive = prograde

    return Orbit(
        vxvv=[R * u.kpc, vR * u.km / u.s, vT * u.km / u.s,
              z * u.kpc, vz * u.km / u.s, phi * u.rad],
        ro=_RO, vo=_VO,
    )


def generate_stream_spray(
    stream_name: str,
    potential: list,
    n_stars: int = 1000,
    seed: int = 0,
    tdisrupt_gyr: Optional[float] = None,
    prog_mass_msun: Optional[float] = None,
    config_path: str = "config/streams.yaml",
    mws=None,
    subhalo_orbit_pot=None,   # optional MovingObjectPotential for an impact
) -> StreamParticles:
    """Generate a stream with galpy's VALIDATED Fardal (2015) particle spray.

    Replaces the homemade isotropic spray. streamspraydf releases particles from
    the Lagrange points with the correct correlated offsets and integrates them in
    ``potential`` to the present -> smooth, thin streams by construction (no
    spray_age hack, no isotropic-kick fudge).

    If ``subhalo_orbit_pot`` (a galpy MovingObjectPotential for a subhalo on its
    flyby) is given, it is added to the integration potential so the stream forms
    WITH the impact gap -- the physically-correct, fast impact injection.

    Returns StreamParticles (same schema as generate_stream).
    """
    import numpy as _np
    import astropy.units as _u
    from galpy.df import streamspraydf

    if mws is None:
        import galstreams
        mws = galstreams.MWStreams(verbose=False)
    sc = _load_stream_config(stream_name, config_path)
    frame = mws[sc["galstreams_key"]].stream_frame

    if tdisrupt_gyr is None:
        tdisrupt_gyr = float(sc.get("spray_tdisrupt_gyr",
                                    sc.get("disruption_age_gyr", 3.0)))
    if prog_mass_msun is None:
        prog_mass_msun = float(sc.get("prog_mass_solar", 1e4))

    ic = set_progenitor_ic_track6d(stream_name, config_path, mws=mws)
    prog = _pos_vel_to_orbit(_np.array(ic["pos_kpc"]), _np.array(ic["vel_kms"]))

    # Integration potential: MW (+ the moving subhalo, if an impact is requested).
    # Adding the subhalo-on-its-flyby makes the gap form naturally during the
    # spray integration (the physically-correct, fast impact injection).
    base_pot = list(potential) if isinstance(potential, list) else [potential]
    pot_int = base_pot if subhalo_orbit_pot is None else base_pot + [subhalo_orbit_pot]

    _np.random.seed(seed)
    spdf = streamspraydf(prog_mass_msun * _u.Msun, progenitor=prog,
                         pot=pot_int, tdisrupt=tdisrupt_gyr * _u.Gyr)
    o = spdf.sample(n=n_stars, return_orbit=True, integrate=True)

    pos = _np.array([o.x(use_physical=True), o.y(use_physical=True), o.z(use_physical=True)])
    vel = _np.array([o.vx(use_physical=True), o.vy(use_physical=True), o.vz(use_physical=True)])
    phi1, phi2, dist, pm1, pm2, vrad = _galactocentric_to_stream_coords(pos, vel, frame)

    # phi1/phi2 selection to the observed window (mirror generate_stream)
    phi1_min = sc.get("phi1_min", -180.0); phi1_max = sc.get("phi1_max", 180.0)
    mask = (phi1 >= phi1_min) & (phi1 <= phi1_max) & _np.isfinite(phi1) & _np.isfinite(phi2)
    return StreamParticles(
        phi1=phi1[mask], phi2=phi2[mask], dist=dist[mask],
        pm1=pm1[mask], pm2=pm2[mask], vrad=vrad[mask],
        xyz_kpc=pos[:, mask], vxyz_kms=vel[:, mask],
    )


# ---------------------------------------------------------------------------
# streamdf / streamgapdf generator  (Bovy 2014; Sanders, Bovy & Erkal 2016)
#
# This is the IMPACT generator that replaces the homemade rewind/kick. The
# no-impact class is sampled from streamdf and the impact class from streamgapdf
# (a streamdf subclass) so both share the same smooth action-angle track -- the
# ONLY difference between the two classes is the subhalo gap, so a detector
# cannot cheat on a generator artefact. Validated 2026-06-02: strong impacts give
# gap-depth ~0.86-1.0 vs ~0.3 no-impact (G6 separability >> homemade 0.57).
#
# Hard-won setup (do not change without re-checking):
#   * b for actionAngleIsochroneApprox MUST come from estimateBIsochrone(pot,
#     R/ro, z/ro) -- a wrong b raises "time not in integration domain".
#   * impact_angle sign must match the arm (leading=True -> positive angle).
#   * nTrackChunks=5 is the speed/quality sweet spot (~15s vs ~21s at 11).
# ---------------------------------------------------------------------------

# Per-process cache: stream_name -> (streamdf, common_kwargs, frame, sc, sigv)
_STREAMDF_BASE_CACHE: dict = {}


def _streamdf_sample_to_cartesian(xv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a streamdf/streamgapdf sample [6,N] to galactocentric Cartesian.

    streamdf.sample returns (R, vR, vT, z, vz, phi). Older/newer galpy versions
    differ on whether this is in natural units (R in units of ro) or physical kpc;
    detect by magnitude (Galactic R is many kpc, so natural R is O(1)).
    """
    xv = np.asarray(xv, float)
    R, vR, vT, z, vz, phi = xv
    if np.nanmedian(np.abs(R)) < 3.0:        # natural units -> physical
        R, z = R * _RO, z * _RO
        vR, vT, vz = vR * _VO, vT * _VO, vz * _VO
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    vx = vR * np.cos(phi) - vT * np.sin(phi)
    vy = vR * np.sin(phi) + vT * np.cos(phi)
    pos = np.array([x, y, z])
    vel = np.array([vx, vy, vz])
    return pos, vel


def _get_streamdf_base(stream_name: str, potential, config_path: str, mws):
    """Build (and cache per process) the smooth streamdf base for a stream.

    Returns (sdf, common_kwargs, frame, sc, sigv). The ~12s action-angle setup is
    paid once per worker per stream; subsequent no-impact samples are ~free and
    each streamgapdf reuses the cached aA/progenitor (only the impact transform,
    ~15s, is re-done per impact).
    """
    if stream_name in _STREAMDF_BASE_CACHE:
        return _STREAMDF_BASE_CACHE[stream_name]
    from galpy.df import streamdf  # noqa: PLC0415
    from galpy.actionAngle import (  # noqa: PLC0415
        actionAngleIsochroneApprox,
        estimateBIsochrone,
    )
    if mws is None:
        import galstreams  # noqa: PLC0415
        mws = galstreams.MWStreams(verbose=False)
    sc = _load_stream_config(stream_name, config_path)
    frame = mws[sc["galstreams_key"]].stream_frame
    ic = set_progenitor_ic_track6d(stream_name, config_path, mws=mws)
    prog = _pos_vel_to_orbit(np.array(ic["pos_kpc"]), np.array(ic["vel_kms"]))
    pot = potential
    b = float(estimateBIsochrone(
        pot, prog.R(use_physical=True) / _RO, prog.z(use_physical=True) / _RO))
    aA = actionAngleIsochroneApprox(pot=pot, b=b)
    sigv = float(sc.get("sigv_kms", 0.5))
    tdis = float(sc.get("spray_tdisrupt_gyr", sc.get("disruption_age_gyr", 3.0)))
    common = dict(progenitor=prog, pot=pot, aA=aA, leading=True,
                  nTrackChunks=5, tdisrupt=tdis * u.Gyr, ro=_RO, vo=_VO)
    sdf = streamdf(sigv * u.km / u.s, **common)
    base = (sdf, common, frame, sc, sigv)
    _STREAMDF_BASE_CACHE[stream_name] = base
    return base


def sample_impact_params(rng: np.random.Generator,
                         log10_mass_range=(7.5, 8.7),
                         impactb_kpc_range=(0.0, 0.35),
                         timpact_gyr_range=(0.2, 1.5),
                         impact_angle_rad_range=(0.2, 0.7),
                         vsub_kms: float = 150.0) -> dict:
    """Draw a single-subhalo encounter for streamgapdf (leading arm, +angle).

    Default ranges target the DETECTABLE regime (mass >= 10^7.5 Msun, impact
    parameter b <= 0.35 kpc): the gap-depth scan (2026-06-02) shows these give
    gap-depth ~0.75-1.0, cleanly above the ~0.33 no-impact Poisson floor.
    Weaker encounters (high b, low mass) are physically undetectable and only add
    label noise to a binary detector; characterising them is the SBI stage's job.
    """
    return {
        "mass": float(10.0 ** rng.uniform(*log10_mass_range)),
        "impactb_kpc": float(rng.uniform(*impactb_kpc_range)),
        "timpact_gyr": float(rng.uniform(*timpact_gyr_range)),
        "impact_angle_rad": float(rng.uniform(*impact_angle_rad_range)),
        "vsub_kms": float(vsub_kms),
    }


def generate_stream_df(
    stream_name: str,
    potential,
    n_stars: int = 1000,
    seed: int = 0,
    impact: bool = False,
    impact_params: Optional[dict] = None,
    config_path: str = "config/streams.yaml",
    mws=None,
) -> StreamParticles:
    """Generate a stream from streamdf (no-impact) or streamgapdf (impact).

    The smooth action-angle base is cached per process; no-impact draws are fast,
    each impact pays ~15s for the streamgapdf impact transform. Returns
    StreamParticles in the same schema as generate_stream / generate_stream_spray.
    """
    from src.simulation.subhalo import scale_radius_from_mass  # noqa: PLC0415
    sdf, common, frame, sc, sigv = _get_streamdf_base(
        stream_name, potential, config_path, mws)
    np.random.seed(seed)

    if not impact:
        df = sdf
    else:
        from galpy.df import streamgapdf  # noqa: PLC0415
        p = impact_params or sample_impact_params(np.random.default_rng(seed))
        m = float(p["mass"])
        df = streamgapdf(
            sigv * u.km / u.s, **common,
            impactb=float(p["impactb_kpc"]) * u.kpc,
            subhalovel=np.array([0.0, float(p["vsub_kms"]), 0.0]) * u.km / u.s,
            timpact=float(p["timpact_gyr"]) * u.Gyr,
            impact_angle=float(p["impact_angle_rad"]) * u.rad,
            GM=m * u.Msun, rs=scale_radius_from_mass(m) * u.kpc,
        )

    xv = df.sample(n=n_stars)
    pos, vel = _streamdf_sample_to_cartesian(xv)
    phi1, phi2, dist, pm1, pm2, vrad = _galactocentric_to_stream_coords(pos, vel, frame)

    phi1_min = sc.get("phi1_min", -180.0); phi1_max = sc.get("phi1_max", 180.0)
    mask = (phi1 >= phi1_min) & (phi1 <= phi1_max) & np.isfinite(phi1) & np.isfinite(phi2)
    return StreamParticles(
        phi1=phi1[mask], phi2=phi2[mask], dist=dist[mask],
        pm1=pm1[mask], pm2=pm2[mask], vrad=vrad[mask],
        xyz_kpc=pos[:, mask], vxyz_kms=vel[:, mask],
    )
