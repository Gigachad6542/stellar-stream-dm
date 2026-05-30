#!/usr/bin/env python
"""
Injection-recovery experiment for the timeline forward model.

Injects a subhalo impact with KNOWN parameters into a synthetic stream, then
runs the full timeline pipeline and reports how well the known parameters are
recovered. This is the validation that must pass before trusting the pipeline on
real data.

Usage:
    # Fast (impulse) self-consistency check
    python scripts/run_injection_recovery.py --stream GD1 --fast \
        --truth-mass 8.0 --truth-time 1.5 --truth-phi1 20

    # Full orbit-integrated (slower, physically correct)
    python scripts/run_injection_recovery.py --stream GD1 \
        --truth-mass 8.0 --truth-time 1.5 --truth-phi1 20 --n-workers 4

Output:
    outputs/injection_recovery/{stream}/injection_recovery_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Injection-recovery test for the timeline forward model.")
    p.add_argument("--stream", type=str, default="GD1")
    p.add_argument("--config", type=str, default="config/streams.yaml")
    p.add_argument("--output-dir", type=str, default="outputs/injection_recovery")

    # Truth (injected) encounter
    p.add_argument("--truth-mass", type=float, default=8.0, help="Injected log10(M/Msun)")
    p.add_argument("--truth-time", type=float, default=1.5, help="Injected t_since_impact [Gyr]")
    p.add_argument("--truth-phi1", type=float, default=20.0, help="Injected impact phi1 [deg]")
    p.add_argument("--truth-seed", type=int, default=123,
                   help="Seed for the truth stream (kept distinct from --seed)")

    # Recovery grid
    p.add_argument("--mass-range", type=float, nargs=2, default=[7.0, 8.5])
    p.add_argument("--mass-step", type=float, default=0.5)
    p.add_argument("--t-range", type=float, nargs=2, default=[0.5, 2.5])
    p.add_argument("--t-step", type=float, default=0.5)
    p.add_argument("--phi1", type=float, nargs="+", default=None,
                   help="phi1 grid (deg). Default: truth +/- 15 and truth.")

    p.add_argument("--n-stars", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42, help="Candidate base seed")
    p.add_argument("--fast", action="store_true",
                   help="Impulse approximation (fast). Default: full orbit integration.")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("injection_recovery")

    from src.forward_model.injection import run_injection_recovery
    from src.forward_model.pipeline import ForwardModelConfig

    phi1_grid = args.phi1 if args.phi1 is not None else [
        args.truth_phi1 - 15.0, args.truth_phi1, args.truth_phi1 + 15.0,
    ]

    cfg = ForwardModelConfig(
        stream_name=args.stream,
        config_path=args.config,
        log10_mass_range=tuple(args.mass_range),
        log10_mass_step=args.mass_step,
        t_since_range=tuple(args.t_range),
        t_since_step=args.t_step,
        impact_phi1_values=phi1_grid,
        n_stars_sim=args.n_stars,
        base_seed=args.seed,
        use_gnn_scorer=False,           # functional signals only (density+gap+kinematic)
        use_fast_mode=args.fast,
        n_workers=args.n_workers,
        output_dir=args.output_dir,
    )

    log.info("=" * 70)
    log.info("INJECTION-RECOVERY: truth M=10^%.2f, t=%.2f Gyr, phi1=%.1f deg",
             args.truth_mass, args.truth_time, args.truth_phi1)
    log.info("=" * 70)

    result, _ = run_injection_recovery(
        cfg, args.truth_mass, args.truth_time, args.truth_phi1, truth_seed=args.truth_seed,
    )

    out_dir = Path(args.output_dir) / args.stream
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "injection_recovery_results.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    log.info("=" * 70)
    log.info("RECOVERY SUMMARY")
    log.info("  truth     : M=10^%.2f  t=%.2f Gyr  phi1=%.1f",
             result.truth_log10_mass, result.truth_t_since_gyr, result.truth_phi1)
    log.info("  recovered : M=10^%.2f  t=%.2f Gyr  phi1=%.1f  (score=%.4f)",
             result.recovered_log10_mass, result.recovered_t_since_gyr,
             result.recovered_phi1, result.recovered_score)
    log.info("  errors    : dM=%+.2f dex  dt=%+.2f Gyr  dphi1=%+.1f deg",
             result.mass_error_dex, result.t_since_error_gyr, result.phi1_error_deg)
    log.info("  best beats null: %s (null=%.4f)", result.best_beats_null, result.null_score)
    log.info("  truth grid-point rank: %d / %d", result.truth_rank, result.n_candidates)
    log.info("  Results saved to %s", out_path)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
