"""
Quick test of the v3 progenitor IC optimiser.

Generates one unperturbed GD-1 stream and measures the phi2 track offset
against the galstreams reference. Target: RMS < 1.0 deg.

Usage:
    python -u scripts/test_ic_optimizer.py
    python -u scripts/test_ic_optimizer.py --stream Pal5
    python -u scripts/test_ic_optimizer.py --all-streams
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import sys
import time
from pathlib import Path

# Enable faulthandler to get a traceback on segfault (galpy C extension crashes)
faulthandler.enable()

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Fix galpy C extension DLL path (must be before any galpy import)
from src.simulation.potentials import _RO, _VO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "test_ic_optimizer.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def _fix_galpy_dll_path():
    """Add galpy C extension + GSL DLL directories to DLL search path (Windows only).

    galpy's C extension (libgalpy.cp311-win_amd64.pyd) depends on GSL
    (gsl-25.dll) which lives in the conda env's Library/bin.  Python 3.8+
    no longer searches PATH for DLL dependencies, so we must explicitly
    register the directory via os.add_dll_directory().
    """
    if sys.platform != "win32":
        return
    import os
    try:
        # Add conda env's Library/bin (contains gsl-25.dll and other runtime DLLs)
        env_dir = Path(sys.executable).parent
        lib_bin = env_dir / "Library" / "bin"
        if lib_bin.exists():
            os.add_dll_directory(str(lib_bin))

        # Also add galpy's own directories
        import galpy
        galpy_dir = Path(galpy.__file__).parent
        for subdir in ["", "potential", "orbit", "actionAngle", "util"]:
            dll_dir = galpy_dir / subdir if subdir else galpy_dir
            if dll_dir.exists():
                os.add_dll_directory(str(dll_dir))

        # Add the site-packages root (where libgalpy.pyd lives)
        site_pkg = galpy_dir.parent
        os.add_dll_directory(str(site_pkg))
    except Exception:
        pass


_fix_galpy_dll_path()

from src.simulation.potentials import get_mw_potential
from src.simulation.stream_gen import generate_stream, set_progenitor_ic


def test_one_stream(stream_name: str) -> float:
    """Generate one stream, measure track offset, return RMS in degrees."""
    import galstreams
    import yaml
    from scipy.interpolate import interp1d

    log.info("=" * 60)
    log.info("Testing IC optimiser for stream: %s", stream_name)
    log.info("=" * 60)

    t0 = time.time()

    with open(ROOT / "config" / "streams.yaml") as f:
        sc = yaml.safe_load(f)["streams"][stream_name]

    potential = get_mw_potential(str(ROOT / "config" / "streams.yaml"))
    mws = galstreams.MWStreams(verbose=False)

    # This will trigger the v3 optimiser (no cache exists)
    log.info("Running v3 IC optimiser ...")
    ic = set_progenitor_ic(
        stream_name, potential,
        config_path=str(ROOT / "config" / "streams.yaml"),
        cache_dir=str(ROOT / "data" / "processed"),
        mws=mws,
    )
    ic_time = time.time() - t0
    log.info("IC optimisation took %.1f s", ic_time)
    log.info("IC result: pos=%s  vel=%s  rms=%.3f deg",
             [f"{x:.3f}" for x in ic["pos_kpc"]],
             [f"{x:.3f}" for x in ic["vel_kms"]],
             ic.get("_rms_deg", -1))

    # Generate one unperturbed stream
    log.info("Generating unperturbed stream ...")
    t1 = time.time()
    particles = generate_stream(
        stream_name, potential, n_stars=5000, seed=0,
        config_path=str(ROOT / "config" / "streams.yaml"),
        progenitor_ic=ic, mws=mws, n_steps_back=200,
    )
    gen_time = time.time() - t1
    log.info("Stream generation took %.1f s (%d particles)", gen_time, len(particles.phi1))

    # Measure track offset against galstreams reference
    track = mws[sc["galstreams_key"]]
    sf = track.stream_frame
    tr_sf = track.track.transform_to(sf)
    tr_phi1 = (np.array(tr_sf.phi1.deg) + 180.0) % 360.0 - 180.0
    tr_phi2 = np.array(tr_sf.phi2.deg)

    phi1_min, phi1_max = sc["phi1_range_deg"]
    tmask = (tr_phi1 >= phi1_min) & (tr_phi1 <= phi1_max)
    sort_idx = np.argsort(tr_phi1[tmask])
    ref_interp = interp1d(
        tr_phi1[tmask][sort_idx], tr_phi2[tmask][sort_idx],
        kind="linear", bounds_error=False, fill_value="extrapolate",
    )

    # Bin simulated stream
    sim_phi1 = particles.phi1
    sim_phi2 = particles.phi2
    in_range = (sim_phi1 >= phi1_min) & (sim_phi1 <= phi1_max)

    n_bins = 20
    bin_edges = np.linspace(phi1_min, phi1_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    offsets = []
    for k in range(n_bins):
        mk = in_range & (sim_phi1 >= bin_edges[k]) & (sim_phi1 < bin_edges[k + 1])
        if mk.sum() >= 3:
            med_phi2 = float(np.median(sim_phi2[mk]))
            ref_phi2 = float(ref_interp(bin_centers[k]))
            offsets.append(med_phi2 - ref_phi2)

    if offsets:
        rms = float(np.sqrt(np.mean(np.array(offsets) ** 2)))
        mean_off = float(np.mean(offsets))
    else:
        rms = 999.0
        mean_off = 999.0

    total_time = time.time() - t0
    status = "PASS ✓" if rms < 1.0 else "FAIL ✗"

    log.info("-" * 40)
    log.info("Stream %s: phi2 RMS = %.3f deg  mean = %.3f deg  [%s]",
             stream_name, rms, mean_off, status)
    log.info("  Bins populated: %d/%d", len(offsets), n_bins)
    log.info("  Particles in range: %d/%d", in_range.sum(), len(sim_phi1))
    log.info("  Total time: %.1f s (IC: %.1f s + gen: %.1f s)", total_time, ic_time, gen_time)
    log.info("  IC cache RMS: %.3f deg", ic.get("_rms_deg", -1))
    log.info("-" * 40)

    return rms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", default="GD1")
    parser.add_argument("--all-streams", action="store_true")
    args = parser.parse_args()

    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    if args.all_streams:
        import yaml
        with open(ROOT / "config" / "streams.yaml") as f:
            streams = list(yaml.safe_load(f)["streams"].keys())
    else:
        streams = [args.stream]

    results = {}
    for stream in streams:
        try:
            rms = test_one_stream(stream)
            results[stream] = rms
        except Exception as e:
            log.error("FAILED for %s: %s", stream, e, exc_info=True)
            results[stream] = None

    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    n_pass = 0
    for stream, rms in results.items():
        if rms is None:
            log.info("  %-10s  ERROR", stream)
        else:
            status = "PASS" if rms < 1.0 else "FAIL"
            if rms < 1.0:
                n_pass += 1
            log.info("  %-10s  phi2 RMS = %.3f deg  [%s]", stream, rms, status)
    log.info("  %d/%d streams passed (<1.0 deg)", n_pass, len(results))


if __name__ == "__main__":
    main()
