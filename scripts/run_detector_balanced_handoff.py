"""Post-generation handoff for the detector-balanced V2 dataset.

This script is meant to run after data/simulations_v2_detector_balanced has all
expected chunk files.  It performs the boring but important handoff steps:

  1. Verify the Python/PyTorch environment.
  2. Check the chunk count before touching HDF5 files.
  3. Compute impact_strength labels in place.
  4. Validate the HDF5 contract expected by train_v2.py.
  5. Run threshold and cheap-baseline signal audits.
  6. Write the exact detector training/evaluation commands.

Example:
    C:/Users/Dwthe/miniforge3/envs/stellar-stream-dm/python.exe scripts/run_detector_balanced_handoff.py
"""

from __future__ import annotations

import argparse
import json
import logging
import site
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep user-site packages from shadowing the project env.  This matters on this
# machine because user site previously exposed a CPU-only torch installation.
try:
    user_site = Path(site.getusersitepackages()).resolve()
    sys.path = [p for p in sys.path if not p or Path(p).resolve() != user_site]
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _threshold_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _run(cmd: list[str], dry_run: bool) -> None:
    log.info("RUN: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _chunk_count(sim_dir: Path) -> int:
    return len(list(sim_dir.glob("chunk_*.h5")))


def _environment_report(require_cuda: bool) -> dict:
    report: dict = {
        "python": sys.executable,
        "errors": [],
        "warnings": [],
    }

    try:
        import numpy as np  # noqa: PLC0415
        report["numpy"] = np.__version__
    except Exception as exc:
        report["errors"].append(f"numpy import failed: {exc}")

    try:
        import h5py  # noqa: PLC0415
        report["h5py"] = h5py.__version__
    except Exception as exc:
        report["errors"].append(f"h5py import failed: {exc}")

    try:
        import sklearn  # noqa: PLC0415
        report["sklearn"] = sklearn.__version__
    except Exception as exc:
        report["errors"].append(f"scikit-learn import failed: {exc}")

    try:
        import torch  # noqa: PLC0415
        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["cuda_device"] = torch.cuda.get_device_name(0)
        elif require_cuda:
            report["errors"].append("CUDA is required but torch.cuda.is_available() is false.")
    except Exception as exc:
        report["errors"].append(f"torch import failed: {exc}")

    try:
        import torch_geometric  # noqa: PLC0415
        report["torch_geometric"] = torch_geometric.__version__
    except Exception as exc:
        report["errors"].append(f"torch_geometric import failed: {exc}")

    try:
        import galstreams  # noqa: F401, PLC0415
        report["galstreams"] = "available"
    except Exception as exc:
        # Handoff itself does not need galstreams, but its absence means this is
        # not the same environment that generated the data.
        report["warnings"].append(f"galstreams import failed: {exc}")

    return report


def _write_next_commands(args: argparse.Namespace, out_path: Path) -> dict:
    tag = _threshold_tag(args.strength_threshold)
    ckpt_dir = Path(args.checkpoint_dir or f"checkpoints/gnn_v2_detector_balanced_s{tag}")
    train_cmd = [
        sys.executable,
        "scripts/train_v2.py",
        "--sim-dir",
        str(args.sim_dir),
        "--checkpoint-dir",
        str(ckpt_dir),
        "--epochs",
        str(args.train_epochs),
        "--binary-target",
        "impact_strong",
        "--strength-threshold",
        str(args.strength_threshold),
        "--use-profile-branch",
        "--normalizer-samples",
        str(args.normalizer_samples),
        "--batch-size",
        str(args.batch_size),
    ]
    eval_cmd = [
        sys.executable,
        "scripts/evaluate_v2_classifier.py",
        "--checkpoint",
        str(ckpt_dir / "gnn_v2_best_acc.pt"),
        "--checkpoint-dir",
        str(ckpt_dir),
        "--sim-dir",
        str(args.sim_dir),
        "--target",
        "impact_strong",
        "--strength-threshold",
        str(args.strength_threshold),
        "--batch-size",
        str(args.batch_size),
        "--out",
        f"outputs/diagnostics/gnn_v2_detector_balanced_s{tag}_eval_best_acc.json",
    ]
    plan = {
        "sim_dir": str(args.sim_dir),
        "strength_threshold": args.strength_threshold,
        "checkpoint_dir": str(ckpt_dir),
        "train_command": train_cmd,
        "eval_command": eval_cmd,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2) + "\n")
    log.info("Wrote next-step command plan to %s", out_path)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Postprocess detector-balanced V2 simulations.")
    parser.add_argument("--sim-dir", type=Path, default=Path("data/simulations_v2_detector_balanced"))
    parser.add_argument("--expected-chunks", type=int, default=200)
    parser.add_argument("--expected-sims", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--impact-strength-write-threshold", type=float, default=0.5)
    parser.add_argument("--strength-threshold", type=float, default=1.0)
    parser.add_argument("--feature-sample-size", type=int, default=12000)
    parser.add_argument("--diagnose-sample-size", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--normalizer-samples", type=int, default=2000)
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--out-prefix", default="detector_balanced")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-impact-strength", action="store_true")
    parser.add_argument("--skip-diagnose", action="store_true")
    parser.add_argument("--no-require-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env = _environment_report(require_cuda=not args.no_require_cuda)
    log.info("Environment: %s", json.dumps(env, indent=2))
    if env["errors"]:
        raise SystemExit("Environment check failed: " + "; ".join(env["errors"]))

    chunks = _chunk_count(args.sim_dir)
    log.info("Chunk count under %s: %d/%d", args.sim_dir, chunks, args.expected_chunks)
    if chunks != args.expected_chunks and not args.allow_incomplete:
        raise SystemExit(
            f"Dataset incomplete: found {chunks} chunks, expected {args.expected_chunks}. "
            "Use --allow-incomplete only for smoke tests."
        )

    validation_out = Path(f"outputs/validation/{args.out_prefix}_integrity.json")
    threshold_out = Path(f"outputs/diagnostics/{args.out_prefix}_threshold_audit.json")
    signal_out = Path(
        f"outputs/diagnostics/{args.out_prefix}_impact_strong_s{_threshold_tag(args.strength_threshold)}_signal_audit.json"
    )
    plan_out = Path(f"outputs/run_plans/{args.out_prefix}_next_steps.json")

    if not args.skip_impact_strength:
        _run([
            sys.executable,
            "scripts/compute_impact_strength.py",
            "--sim-dir",
            str(args.sim_dir),
            "--threshold",
            str(args.impact_strength_write_threshold),
            "--write-in-place",
        ], args.dry_run)

    _run([
        sys.executable,
        "scripts/validate_v2_dataset.py",
        "--sim-dir",
        str(args.sim_dir),
        "--expected-chunks",
        str(args.expected_chunks),
        "--expected-sims",
        str(args.expected_sims),
        "--chunk-size",
        str(args.chunk_size),
        "--output",
        str(validation_out),
    ], args.dry_run)

    _run([
        sys.executable,
        "scripts/audit_v2_signal_thresholds.py",
        "--sim-dir",
        str(args.sim_dir),
        "--feature-sample-size",
        str(args.feature_sample_size),
        "--out",
        str(threshold_out),
    ], args.dry_run)

    if not args.skip_diagnose:
        _run([
            sys.executable,
            "scripts/diagnose_v2_signal.py",
            "--sim-dir",
            str(args.sim_dir),
            "--target",
            "impact_strong",
            "--strength-threshold",
            str(args.strength_threshold),
            "--sample-size",
            str(args.diagnose_sample_size),
            "--out",
            str(signal_out),
        ], args.dry_run)

    plan = _write_next_commands(args, plan_out)
    print(json.dumps({
        "ok": True,
        "environment": env,
        "chunks": chunks,
        "validation_output": str(validation_out),
        "threshold_audit_output": str(threshold_out),
        "signal_audit_output": None if args.skip_diagnose else str(signal_out),
        "next_steps": plan,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
