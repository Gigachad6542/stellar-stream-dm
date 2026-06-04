#!/usr/bin/env python
"""Fuse PWB18 GD-1 density selection with DESI v3 spectroscopy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gd1_fusion import fuse_pwb18_desi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pwb18", default="data/processed/GD1_pwb18_track.h5")
    parser.add_argument("--desi", default="data/processed/GD1_desi_dr2_v3_member.h5")
    parser.add_argument("--out", default="data/processed/GD1_pwb18_desi_v3_fused.h5")
    parser.add_argument("--match-radius-arcsec", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.out)
    if output.exists():
        if not args.force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        output.unlink()
    result = fuse_pwb18_desi(
        args.pwb18,
        args.desi,
        output,
        match_radius_arcsec=args.match_radius_arcsec,
    )
    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
