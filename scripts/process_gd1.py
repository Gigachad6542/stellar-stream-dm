"""
Download and process the GD-1 stellar stream from Gaia DR3.

Two data sources are tried in order:
  1. Price-Whelan & Bonaca 2018 (PWB18) membership catalog (Zenodo 1295543).
     This is a high-purity selection based on Gaia DR2 proper motions.
  2. Fallback: direct Gaia DR3 TAP box query + quality cuts.

Output: data/processed/streams.h5
        with group /streams/GD1/ matching the schema in src/data/stream_process.py

Usage:
    python scripts/process_gd1.py
    python scripts/process_gd1.py --source tap     # force Gaia TAP query
    python scripts/process_gd1.py --force           # re-process even if cached
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

import astropy.units as u
import astropy.coordinates as coord
import numpy as np
import yaml
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.gaia_query import query_stream_members, load_membership_catalog
from src.data.stream_process import process_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "streams.h5"
STREAMS_CFG = ROOT / "config" / "streams.yaml"

# PWB18 Zenodo download links (Gaia DR2 membership; source_id matches DR3 source_id)
PWB18_ZENODO_URL = "https://zenodo.org/record/1295543/files/PWB18_gd1_member_stars.fits"
PWB18_LOCAL = RAW_DIR / "GD1_pwb18_members.fits"


def download_pwb18(force: bool = False) -> Path | None:
    """Download the Price-Whelan & Bonaca 2018 GD-1 membership catalog.

    Returns the path to the downloaded FITS file, or None on failure.
    """
    if PWB18_LOCAL.exists() and not force:
        log.info("PWB18 catalog already cached: %s", PWB18_LOCAL)
        return PWB18_LOCAL

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading PWB18 GD-1 catalog from Zenodo...")
    try:
        urllib.request.urlretrieve(PWB18_ZENODO_URL, str(PWB18_LOCAL))
        log.info("Downloaded PWB18 catalog (%d bytes)", PWB18_LOCAL.stat().st_size)
        return PWB18_LOCAL
    except Exception as e:
        log.warning("Could not download PWB18 catalog: %s", e)
        return None


def load_pwb18(path: Path) -> Table | None:
    """Load the PWB18 catalog and standardize column names.

    The PWB18 FITS file has Gaia DR2 source_ids and proper motions.
    We rename columns to match the pipeline schema.
    """
    try:
        t = Table.read(str(path))
        log.info("Loaded PWB18 catalog: %d stars, columns: %s", len(t), t.colnames)
    except Exception as e:
        log.error("Failed to read PWB18 catalog: %s", e)
        return None

    # Standardize column names (PWB18 uses 'ra', 'dec', 'pm_ra_cosdec', 'pm_dec')
    rename_map = {
        "pm_ra_cosdec": "pmra",
        "pm_ra_cosdec_error": "pmra_error",
        "pm_dec_error": "pmdec_error",
    }
    for old, new in rename_map.items():
        if old in t.colnames and new not in t.colnames:
            t.rename_column(old, new)

    # Add membership_prob column (all stars in PWB18 have p_member >= 0.5 by construction)
    if "membership_prob" not in t.colnames:
        t["membership_prob"] = np.ones(len(t), dtype=np.float32)

    return t


def get_galstreams_track() -> Table | None:
    """Extract the GD-1 galstreams reference track as an astropy Table."""
    try:
        import galstreams  # noqa: PLC0415
        with open(STREAMS_CFG) as f:
            sc = yaml.safe_load(f)["streams"]["GD1"]
        mws = galstreams.MWStreams(verbose=False)
        track = mws[sc["galstreams_key"]]
        tr = track.track
        sf = track.stream_frame
        tr_sf = tr.transform_to(sf)

        phi1 = (np.array(tr_sf.phi1.deg) + 180.0) % 360.0 - 180.0
        phi2 = np.array(tr_sf.phi2.deg)

        # Sort by phi1 for clean interpolation
        order = np.argsort(phi1)
        t = Table()
        t["phi1"] = phi1[order]
        t["phi2"] = phi2[order]
        t["width"] = np.full(len(t), 0.5)  # approximate track half-width [deg]
        return t
    except Exception as e:
        log.warning("Could not extract galstreams track: %s", e)
        return None


def main(args) -> None:
    # Check if output already exists
    import h5py  # noqa: PLC0415
    if PROCESSED_PATH.exists() and not args.force:
        try:
            with h5py.File(str(PROCESSED_PATH), "r") as f:
                if "streams/GD1" in f:
                    n = f["streams/GD1"].attrs.get("n_members", 0)
                    log.info("GD-1 already processed: %d stars in %s. Use --force to redo.", n, PROCESSED_PATH)
                    return
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Step 1: Obtain member catalog
    # ------------------------------------------------------------------
    table = None

    if args.source in ("pwb18", "auto"):
        pwb18_path = download_pwb18(force=args.force)
        if pwb18_path is not None:
            table = load_pwb18(pwb18_path)

    if table is None and args.source in ("tap", "auto"):
        log.info("Falling back to Gaia TAP query for GD-1...")
        tap_cache = RAW_DIR / "GD1_gaia_raw.fits"
        try:
            table = query_stream_members("GD1", tap_cache, config_path=str(STREAMS_CFG), force=args.force)
        except Exception as e:
            log.error("Gaia TAP query failed: %s", e)
            log.error("Cannot obtain GD-1 data. Exiting.")
            return

    if table is None:
        log.error("No data source available for GD-1.")
        return

    log.info("Obtained %d candidate GD-1 members", len(table))

    # ------------------------------------------------------------------
    # Step 2: Get galstreams reference track
    # ------------------------------------------------------------------
    track_table = get_galstreams_track()
    if track_table is not None:
        log.info("Loaded galstreams GD-1 track: %d points", len(track_table))

    # ------------------------------------------------------------------
    # Step 3: Process into HDF5
    # ------------------------------------------------------------------
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    process_stream(
        stream_name="GD1",
        raw_table=table,
        output_path=PROCESSED_PATH,
        config_path=str(STREAMS_CFG),
        track_table=track_table,
    )

    # ------------------------------------------------------------------
    # Step 4: Quick sanity check
    # ------------------------------------------------------------------
    with h5py.File(str(PROCESSED_PATH), "r") as f:
        grp = f["streams/GD1/members"]
        n = len(grp["phi1"][:])
        phi1_min = float(grp["phi1"][:].min())
        phi1_max = float(grp["phi1"][:].max())
        pm1_med = float(np.median(grp["pm1"][:]))

    log.info("=" * 60)
    log.info("GD-1 processing complete:")
    log.info("  %d stars in HDF5", n)
    log.info("  phi1 range: [%.1f, %.1f] deg", phi1_min, phi1_max)
    log.info("  Median pm_phi1: %.2f mas/yr (expect ~-8 to -10)", pm1_med)
    log.info("  Output: %s", PROCESSED_PATH)
    log.info("=" * 60)

    if n < 100:
        log.warning("Very few stars (%d) — check the data source or selection cuts.", n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process GD-1 stream data.")
    parser.add_argument(
        "--source", choices=["pwb18", "tap", "auto"], default="auto",
        help="Data source: pwb18 (membership catalog), tap (Gaia TAP), auto (try pwb18 first).",
    )
    parser.add_argument("--force", action="store_true", help="Re-process even if cached.")
    main(parser.parse_args())
