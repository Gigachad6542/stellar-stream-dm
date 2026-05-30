"""
Download Gaia DR3 and process the remaining stellar streams into the HDF5 database.

Workflow per stream:
1. Check if already present in data/processed/streams.h5 (skip if so).
2. Download Gaia DR3 via async TAP+; cache FITS to data/raw/{stream}_gaia_raw.fits.
3. Transform to stream frame, apply dust/distance pipeline, write to HDF5.

Streams are processed sequentially to respect Gaia TAP rate limits.

Usage:
    python scripts/process_streams.py
    python scripts/process_streams.py --streams Pal5 Orphan ATLAS
    python scripts/process_streams.py --force          # re-download even if cached
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import h5py
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gaia_query import query_stream_members
from src.data.stream_process import process_stream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("process_streams.log"),
    ],
)
log = logging.getLogger(__name__)

PROCESSED_HDF5 = "data/processed/streams.h5"
RAW_DIR = Path("data/raw")
CONFIG_PATH = "config/streams.yaml"

# Default stream order: GD1 is already done, process the remaining 6
DEFAULT_STREAMS = ["Pal5", "Orphan", "ATLAS", "Jhelum", "Fjorm", "Sylgr"]


def already_in_hdf5(stream_name: str, hdf5_path: str = PROCESSED_HDF5) -> bool:
    """Return True if the stream group already exists in the HDF5 file."""
    p = Path(hdf5_path)
    if not p.exists():
        return False
    try:
        with h5py.File(str(p), "r") as f:
            return f"streams/{stream_name}" in f
    except OSError:
        return False


def process_one_stream(stream_name: str, force: bool = False) -> bool:
    """Download + process one stream. Returns True on success."""
    if already_in_hdf5(stream_name) and not force:
        log.info("Stream %s already in HDF5 — skipping (use --force to reprocess).", stream_name)
        return True

    raw_path = RAW_DIR / f"{stream_name}_gaia_raw.fits"

    # ------------------------------------------------------------------ #
    # Step 1 — Gaia download
    # ------------------------------------------------------------------ #
    log.info("=" * 60)
    log.info("STREAM: %s", stream_name)
    log.info("=" * 60)

    t0 = time.time()
    try:
        raw_table = query_stream_members(
            stream_name,
            output_path=raw_path,
            config_path=CONFIG_PATH,
            force=force,
        )
        log.info("Download complete: %d rows in %.1f s", len(raw_table), time.time() - t0)
    except Exception as exc:
        log.error("Gaia download FAILED for %s: %s", stream_name, exc, exc_info=True)
        return False

    if len(raw_table) == 0:
        log.warning("Empty Gaia result for %s — skipping processing.", stream_name)
        return False

    # ------------------------------------------------------------------ #
    # Step 2 — Stream-frame transform + HDF5 write
    # ------------------------------------------------------------------ #
    t1 = time.time()
    try:
        process_stream(
            stream_name=stream_name,
            raw_table=raw_table,
            output_path=PROCESSED_HDF5,
            config_path=CONFIG_PATH,
        )
        log.info("Processed %s in %.1f s", stream_name, time.time() - t1)
    except Exception as exc:
        log.error("Processing FAILED for %s: %s", stream_name, exc, exc_info=True)
        return False

    log.info("Stream %s complete. Total: %.1f s", stream_name, time.time() - t0)
    return True


def print_hdf5_summary(hdf5_path: str = PROCESSED_HDF5) -> None:
    """Print a table of all streams and their member counts."""
    if not Path(hdf5_path).exists():
        log.info("No HDF5 file found at %s", hdf5_path)
        return
    try:
        with h5py.File(hdf5_path, "r") as f:
            streams = sorted(f.get("streams", {}).keys())
            if not streams:
                log.info("No streams in HDF5 yet.")
                return
            log.info("\n--- HDF5 Summary: %s ---", hdf5_path)
            for s in streams:
                grp = f[f"streams/{s}"]
                n = grp.attrs.get("n_members", "?")
                log.info("  %-10s  %s members", s, n)
    except OSError as e:
        log.warning("Could not read HDF5: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process stellar streams into HDF5.")
    parser.add_argument(
        "--streams", nargs="+", default=DEFAULT_STREAMS,
        help="Stream names to process (default: all 6 non-GD1 streams)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download and reprocess even if already cached/in HDF5",
    )
    args = parser.parse_args()

    # Validate stream names against config
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    valid_streams = set(cfg["streams"].keys())
    unknown = [s for s in args.streams if s not in valid_streams]
    if unknown:
        log.error("Unknown stream names: %s. Valid: %s", unknown, sorted(valid_streams))
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    log.info("Processing %d streams: %s", len(args.streams), args.streams)
    print_hdf5_summary()

    results = {}
    for stream in args.streams:
        ok = process_one_stream(stream, force=args.force)
        results[stream] = "OK" if ok else "FAILED"
        # Brief pause between Gaia queries to avoid rate-limiting
        if stream != args.streams[-1]:
            log.info("Waiting 10 s before next Gaia query...")
            time.sleep(10)

    # Final summary
    log.info("\n--- Processing Results ---")
    for stream, status in results.items():
        log.info("  %-10s  %s", stream, status)

    print_hdf5_summary()

    n_failed = sum(1 for s in results.values() if s == "FAILED")
    if n_failed:
        log.warning("%d stream(s) failed. Check process_streams.log for details.", n_failed)
        sys.exit(1)
    else:
        log.info("All streams processed successfully.")


if __name__ == "__main__":
    main()
