"""Process GD-1 from the cached raw FITS file (skip Gaia TAP re-query).

This script loads the already-downloaded GD1_gaia_raw.fits and runs only
the processing pipeline: coordinate transform → dust correction → HDF5.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.stream_process import process_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_FITS = ROOT / "data" / "raw" / "GD1_gaia_raw.fits"
PROCESSED_PATH = ROOT / "data" / "processed" / "streams.h5"
STREAMS_CFG = ROOT / "config" / "streams.yaml"


def main() -> None:
    if not RAW_FITS.exists():
        log.error("Cached FITS not found: %s — run process_gd1.py --source tap first.", RAW_FITS)
        return

    log.info("Loading cached GD-1 Gaia query: %s", RAW_FITS)
    table = Table.read(str(RAW_FITS))
    log.info("Loaded %d rows", len(table))

    # Get galstreams track
    track_table = None
    try:
        import galstreams
        with open(STREAMS_CFG) as f:
            sc = yaml.safe_load(f)["streams"]["GD1"]
        mws = galstreams.MWStreams(verbose=False)
        track = mws[sc["galstreams_key"]]
        tr_sf = track.track.transform_to(track.stream_frame)
        phi1 = (np.array(tr_sf.phi1.deg) + 180.0) % 360.0 - 180.0
        phi2 = np.array(tr_sf.phi2.deg)
        order = np.argsort(phi1)
        track_table = Table()
        track_table["phi1"] = phi1[order]
        track_table["phi2"] = phi2[order]
        track_table["width"] = np.full(len(phi1), 0.5)
        log.info("Loaded galstreams GD-1 track: %d points", len(track_table))
    except Exception as e:
        log.warning("Could not load galstreams track: %s", e)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    process_stream(
        stream_name="GD1",
        raw_table=table,
        output_path=PROCESSED_PATH,
        config_path=str(STREAMS_CFG),
        track_table=track_table,
    )

    # Sanity check
    import h5py
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


if __name__ == "__main__":
    main()
