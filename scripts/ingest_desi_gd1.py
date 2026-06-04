#!/usr/bin/env python
"""Ingest the verified Jarvis et al. DESI DR2 GD-1 v3 member catalog."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.desi_gd1 import (
    DESI_GD1_ARCHIVE_BYTES,
    DESI_GD1_ARCHIVE_MD5,
    DESI_GD1_ARCHIVE_URL,
    DESI_GD1_RECORD_ID,
    DESI_GD1_VERSION,
    verify_md5,
    write_desi_gd1_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--table7", required=True)
    parser.add_argument("--table1", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--selection", nargs="+", choices=["thin", "member"], default=["thin", "member"]
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive = Path(args.archive)
    if archive.stat().st_size != DESI_GD1_ARCHIVE_BYTES:
        raise RuntimeError(
            f"Unexpected DESI v3 archive size {archive.stat().st_size}; "
            f"expected {DESI_GD1_ARCHIVE_BYTES}."
        )
    if not verify_md5(archive, DESI_GD1_ARCHIVE_MD5):
        raise RuntimeError("DESI v3 archive MD5 does not match the Zenodo checksum")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for selection in args.selection:
        output = output_dir / f"GD1_desi_dr2_v3_{selection}.h5"
        if output.exists() and not args.force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        if output.exists():
            output.unlink()
        results.append(
            write_desi_gd1_selection(
                args.table7,
                args.table1,
                output,
                selection=selection,
            )
        )

    manifest = output_dir / "GD1_desi_dr2_v3_ingest_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "result_label": "Observed Catalog Ingestion",
                "source_url": DESI_GD1_ARCHIVE_URL,
                "zenodo_record_id": DESI_GD1_RECORD_ID,
                "zenodo_version": DESI_GD1_VERSION,
                "archive": str(archive),
                "archive_size_bytes": archive.stat().st_size,
                "archive_md5": DESI_GD1_ARCHIVE_MD5,
                "md5_verified": True,
                "table7": str(args.table7),
                "table1": str(args.table1),
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
