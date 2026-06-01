"""
Milky Way gravitational potential setup using galpy.

Uses galpy potential components tuned to Bovy (2015) + updated solar position
from GRAVITY Collaboration (2018).  All mass parameters are read from
config/streams.yaml and can be perturbed for MW potential uncertainty
marginalisation.

galpy "composite potential" is simply a Python list of Potential objects.
Physical units are enabled throughout via (ro, vo) = (8.122 kpc, 229 km/s).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

# --- Windows DLL fix: must run before any galpy import ---
# galpy's C extension (libgalpy) depends on GSL (gsl-25.dll) which lives
# in the conda env's Library/bin.  Python 3.8+ no longer adds PATH dirs
# to the DLL search path.  Pre-loading GSL via ctypes.CDLL makes it
# available when Python later loads libgalpy.pyd via importlib.
if sys.platform == "win32":
    import ctypes as _ctypes
    try:
        _env_dir = Path(sys.executable).parent
        _lib_bin = _env_dir / "Library" / "bin"
        # Pre-load GSL shared libraries that libgalpy depends on
        for _dll_name in ("gsl-25.dll", "gslcblas-0.dll"):
            _dll_path = _lib_bin / _dll_name
            if _dll_path.exists():
                _ctypes.CDLL(str(_dll_path))
        # Also register directories for any remaining DLL lookups
        if _lib_bin.exists():
            os.add_dll_directory(str(_lib_bin))
        _site_pkg = _env_dir / "Lib" / "site-packages"
        if _site_pkg.exists():
            os.add_dll_directory(str(_site_pkg))
    except Exception:
        pass
    del _ctypes
# --- end Windows DLL fix ---

import astropy.units as u
import numpy as np
import yaml

from galpy.potential import (
    HernquistPotential,
    MiyamotoNagaiPotential,
    NFWPotential,
    evaluateRforces,
)

# --- Force galpy's parallel_map to run serially (numcores=1) ---
# When galpy falls back to its Python orbit integrator (integrateFullOrbit ->
# parallel_map), parallel_map defaults numcores to the CPU count and spawns that
# many multiprocessing workers. On Windows this intermittently crashes the
# process with STATUS_ACCESS_VIOLATION / STATUS_ILLEGAL_INSTRUCTION inside the
# spawned numerical workers (an uncatchable C-level fault). Pinning galpy's
# worker count to 1 makes the fallback run in-process (no spawn), which removes
# the crash. The C integrator (dop853_c) is unaffected. Importing this module
# (the central potential factory) applies the patch everywhere integration runs.
try:
    import galpy.util.multi as _galpy_multi
    _galpy_multi._ncpus = 1
except Exception:  # pragma: no cover - defensive
    pass

log = logging.getLogger(__name__)

# Galactocentric reference frame
# Use galpy MWPotential2014 defaults (Bovy 2015) to ensure internal
# consistency between potential forces and orbit integration.
_RO: float = 8.0     # kpc  – solar galactocentric distance (galpy default)
_VO: float = 220.0   # km/s – local circular velocity (galpy default)

# Type alias: a galpy potential is a list (or single) Potential object(s)
MWPotentialList = List  # list[galpy.potential.Potential]


# ---------------------------------------------------------------------------
# MW potential components
# ---------------------------------------------------------------------------

def get_mw_potential(config_path: str = "config/streams.yaml") -> MWPotentialList:
    """Return galpy's MWPotential2014 (Bovy 2015) — the standard MW potential.

    Components (built into galpy)
    ----------
    bulge : PowerSphericalPotentialwCutoff, M ~ 5e9 Msun
    disk  : MiyamotoNagai, M = 6.8e10 Msun, a = 3 kpc, b = 0.28 kpc
    halo  : NFW, M_vir ~ 0.8e12 Msun, c ~ 15.3

    This is the most validated potential for stellar stream work.
    Physical units enabled: (ro = 8.0 kpc, vo = 220 km/s).

    The ``config_path`` parameter is accepted for API compatibility but
    MWPotential2014 does not read stream-specific potential overrides.
    """
    from galpy.potential import MWPotential2014  # noqa: PLC0415
    return MWPotential2014


def _nfw_amp_from_m200(m200_msun: float, r_s_kpc: float, r200_kpc: float) -> float:
    """Solve for the galpy NFWPotential ``amp`` that reproduces M_200.

    galpy NFW enclosed mass: M(<r) = amp * [ln(1 + r/a) - r/(r+a)]
    """
    x = r200_kpc / r_s_kpc
    f = np.log(1.0 + x) - x / (1.0 + x)
    return m200_msun / f


def vary_potential(seed: int, config_path: str = "config/streams.yaml") -> MWPotentialList:
    """Return a MW potential with halo and disk perturbed within observational uncertainties.

    Starts from MWPotential2014 component values and applies log-normal
    perturbations to the halo and disk masses and halo scale radius.

    Perturbation model (log-normal):
        halo amplitude  : σ = 0.15
        halo scale radius: σ = 0.10
        disk amplitude  : σ = 0.10
    """
    from galpy.potential import PowerSphericalPotentialwCutoff  # noqa: PLC0415

    rng = np.random.default_rng(seed)

    # MWPotential2014 base values (Bovy 2015)
    m_halo_base = 0.80e12   # M_vir
    r_s_base    = 16.0 / 8.0 * _RO  # galpy internal: a=16/ro; convert to physical kpc
    m_disk_base = 6.80e10

    m_halo = m_halo_base * np.exp(rng.normal(0.0, 0.15))
    r_s    = r_s_base    * np.exp(rng.normal(0.0, 0.10))
    m_disk = m_disk_base * np.exp(rng.normal(0.0, 0.10))

    # Bulge: keep MWPotential2014's bulge (power-law with cutoff) unperturbed
    bulge = PowerSphericalPotentialwCutoff(
        amp=1.0, alpha=1.8, rc=1.9 / _RO, normalize=0.05,
        ro=_RO, vo=_VO,
    )
    disk = MiyamotoNagaiPotential(
        amp=m_disk * u.Msun, a=3.0 * u.kpc, b=0.28 * u.kpc, ro=_RO, vo=_VO,
    )
    nfw_amp = _nfw_amp_from_m200(m200_msun=m_halo, r_s_kpc=r_s, r200_kpc=230.0)
    halo = NFWPotential(amp=nfw_amp * u.Msun, a=r_s * u.kpc, ro=_RO, vo=_VO)

    return [bulge, disk, halo]


# ---------------------------------------------------------------------------
# Validation utility
# ---------------------------------------------------------------------------

def circular_velocity_curve(
    potential: MWPotentialList,
    r_kpc: np.ndarray,
) -> np.ndarray:
    """Return circular velocity [km/s] at galactocentric radii ``r_kpc``.

    Validation: should give 220–240 km/s at r = 8 kpc.
    """
    from galpy.potential import vcirc  # noqa: PLC0415

    r_arr = np.atleast_1d(r_kpc)
    return np.array([
        vcirc(potential, r * u.kpc, use_physical=True, ro=_RO, vo=_VO)
        for r in r_arr
    ])
