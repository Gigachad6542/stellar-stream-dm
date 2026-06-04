#!/usr/bin/env python
"""Ingest the published PWB18 masked GD-1 region catalog.

The source file is not a posterior membership catalog.  This script creates
dedicated HDF5 files for the published PM+CMD+track selection and, optionally,
the wider PM+CMD control selection.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pwb18 import (
    PWB18_EXPECTED_BYTES,
    PWB18_MD5,
    PWB18_URL,
    compute_pwb18_selection_profile,
    download_pwb18,
    verify_md5,
    write_pwb18_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/gd1-with-masks.fits")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--selection",
        nargs="+",
        choices=["track", "pmcmd"],
        default=["track", "pmcmd"],
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-md5", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None, help="Smoke-test truncation only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source = Path(args.input)
    if args.download:
        download_pwb18(source, force=args.force)
    if not source.exists():
        raise FileNotFoundError(
            f"{source} does not exist. Download from {PWB18_URL} or pass --download."
        )
    source_size = source.stat().st_size
    if source_size != PWB18_EXPECTED_BYTES and args.max_rows is None:
        raise RuntimeError(
            f"Unexpected PWB18 file size {source_size}; expected {PWB18_EXPECTED_BYTES}."
        )
    md5_verified = False
    if not args.skip_md5 and args.max_rows is None:
        md5_verified = verify_md5(source, PWB18_MD5)
        if not md5_verified:
            raise RuntimeError("PWB18 FITS MD5 does not match the published Zenodo checksum")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_profile = compute_pwb18_selection_profile(source)
    results = []
    for selection in args.selection:
        output = output_dir / f"GD1_pwb18_{selection}.h5"
        if output.exists() and not args.force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        if output.exists():
            output.unlink()
        results.append(
            write_pwb18_selection(
                source,
                output,
                selection=selection,
                max_rows=args.max_rows,
                selection_profile=selection_profile,
            )
        )

    manifest = output_dir / "GD1_pwb18_ingest_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": str(source),
                "source_url": PWB18_URL,
                "source_size_bytes": source_size,
                "published_md5": PWB18_MD5,
                "md5_verified": md5_verified,
                "max_rows_smoke_truncation": args.max_rows,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
