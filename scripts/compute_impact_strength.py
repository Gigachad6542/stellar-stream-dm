"""
Compute per-simulation impact strength scores for existing HDF5 datasets.

The core problem: many "impacted" sims have encounters that are morphologically
invisible (low mass M_sub < 10^6, distant flyby, old impact). Labeling all
n_encounters > 0 as "detectable" creates noisy positive labels that cap GNN
accuracy at ~83%.

This script reads encounter parameters from existing HDF5 chunks and computes
a compact impact_strength score plus a boolean impact_strong label. By default
it runs read-only; pass --write-in-place to add those labels to the HDF5 files.
No resimulation is required.

Impact strength model (Erkal & Belokurov 2015a,b):
    The fractional gap depth scales as:
        delta ~ (M_sub / M_stream) * (t / t_cross)^(1/2) / (b / r_s)

    For a simplified detectability proxy:
        S_i = (M_i / 10^7)^(1/3) * sqrt(min(t_i, 8) / 5)
              / [(b_i/0.15 kpc) * (v_i/200 km/s)]

    The stored score is max_i(S_i), since the strongest encounter usually
    dominates detectability in a sparse stream-gap search.

Usage:
    python scripts/compute_impact_strength.py --sim-dir data/simulations_v2_plan2
    python scripts/compute_impact_strength.py --sim-dir data/simulations_v2_plan2 --write-in-place
    python scripts/compute_impact_strength.py --sim-dir data/simulations_signal_ladder/baryonic_2k --threshold 0.3 --write-in-place
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def encounter_strength(
    mass_solar: np.ndarray,
    impact_param_kpc: np.ndarray,
    flyby_vel_kms: np.ndarray,
    t_since_gyr: np.ndarray,
) -> float:
    """Compute a scalar impact strength score for a set of encounters.

    Based on the Erkal & Belokurov (2015) gap formation model. The gap depth
    scales as M^(1/3) and grows with time until phase mixing saturates the
    observable signal. Distant/fast flybys produce weaker gaps.

    We compute a per-encounter score and return the maximum because the
    strongest individual impact dominates detectability in this proxy.

    Args:
        mass_solar: Array of subhalo masses [M_sun].
        impact_param_kpc: Array of impact parameters [kpc].
        flyby_vel_kms: Array of flyby velocities [km/s].
        t_since_gyr: Array of time since impact [Gyr].

    Returns:
        max_strength: Max single-encounter strength score.
    """
    if len(mass_solar) == 0:
        return 0.0

    # Normalize to characteristic scales
    m_norm = mass_solar / 1e7  # normalize to 10^7 M_sun
    b_norm = np.clip(impact_param_kpc, 0.01, None) / 0.15  # normalize to 150 pc
    v_norm = flyby_vel_kms / 200.0  # normalize to 200 km/s
    # Time factor: gap grows with sqrt(time), but saturates for very old impacts
    # as phase mixing washes them out
    t_eff = np.clip(t_since_gyr, 0.1, 8.0)  # cap older impacts after phase mixing
    t_factor = (t_eff / 5.0) ** 0.5

    # Higher mass, closer, slower, and older up to saturation = stronger.
    per_enc = (m_norm ** (1.0 / 3.0)) * t_factor / (b_norm * v_norm)

    return float(np.max(per_enc))


def process_chunk(path: Path, threshold: float, dry_run: bool = False) -> dict:
    """Process one HDF5 chunk file, adding impact strength attributes.

    Args:
        path: Path to chunk HDF5 file.
        threshold: Strength threshold for "strong impact" boolean.
        dry_run: If True, don't write to file.

    Returns:
        Dict with counts and statistics.
    """
    mode = "r" if dry_run else "r+"
    stats = {"n_sims": 0, "n_impacted": 0, "n_strong": 0, "strengths": []}

    with h5py.File(str(path), mode) as f:
        if "simulations" not in f:
            return stats

        for run_id in f["simulations"]:
            grp = f["simulations"][run_id]
            stats["n_sims"] += 1

            n_sub = int(grp["labels"]["n_subhalos"][()])

            if n_sub > 0 and "subhalos" in grp:
                sub = grp["subhalos"]
                masses = np.asarray(sub["mass"], dtype=np.float64)
                bkpc = np.asarray(sub["impact_param"], dtype=np.float64)
                vkms = np.asarray(sub["flyby_vel"], dtype=np.float64)
                t_since = np.asarray(sub["t_since_impact_gyr"], dtype=np.float64)

                strength = encounter_strength(masses, bkpc, vkms, t_since)
                stats["n_impacted"] += 1
            else:
                strength = 0.0

            is_strong = strength > threshold
            if is_strong:
                stats["n_strong"] += 1
            stats["strengths"].append(strength)

            if not dry_run:
                grp["labels"].attrs["impact_strength"] = float(strength)
                grp["labels"].attrs["impact_strong"] = int(is_strong)
                grp["labels"].attrs["impact_strength_threshold"] = float(threshold)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute impact strength scores for HDF5 simulations")
    parser.add_argument("--sim-dir", required=True, help="Directory containing chunk_*.h5 files")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Strength threshold for 'strong impact' label (default: 0.5)")
    parser.add_argument("--write-in-place", action="store_true",
                        help="Write impact_strength attrs into the HDF5 files. Default is read-only.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Read-only mode; kept as an explicit alias for the default.")
    parser.add_argument("--sample", type=int, default=0,
                        help="Only process this many chunks (0 = all)")
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir)
    dry_run = args.dry_run or not args.write_in_place
    if dry_run:
        log.info("Read-only mode: no HDF5 files will be modified. Use --write-in-place to store attrs.")
    else:
        log.warning("Writing impact_strength attrs in place under %s", sim_dir)
    chunks = sorted(sim_dir.glob("**/chunk_*.h5"))
    if not chunks:
        log.error("No chunk files found in %s", sim_dir)
        return 1

    if args.sample > 0:
        chunks = chunks[:args.sample]

    log.info("Processing %d chunk files in %s (threshold=%.3f, dry_run=%s)",
             len(chunks), sim_dir, args.threshold, dry_run)

    total_stats = {"n_sims": 0, "n_impacted": 0, "n_strong": 0, "all_strengths": []}

    for i, path in enumerate(chunks):
        stats = process_chunk(path, args.threshold, dry_run=dry_run)
        total_stats["n_sims"] += stats["n_sims"]
        total_stats["n_impacted"] += stats["n_impacted"]
        total_stats["n_strong"] += stats["n_strong"]
        total_stats["all_strengths"].extend(stats["strengths"])

        if (i + 1) % 50 == 0 or i == len(chunks) - 1:
            log.info("Processed %d/%d chunks (%d sims, %d impacted, %d strong)",
                     i + 1, len(chunks),
                     total_stats["n_sims"], total_stats["n_impacted"], total_stats["n_strong"])

    strengths = np.array(total_stats["all_strengths"])
    impacted_strengths = strengths[strengths > 0]

    log.info("=" * 60)
    log.info("Summary:")
    log.info("  Total sims: %d", total_stats["n_sims"])
    log.info("  With any impact: %d (%.1f%%)",
             total_stats["n_impacted"],
             100 * total_stats["n_impacted"] / max(total_stats["n_sims"], 1))
    log.info("  With strong impact (S > %.3f): %d (%.1f%%)",
             args.threshold,
             total_stats["n_strong"],
             100 * total_stats["n_strong"] / max(total_stats["n_sims"], 1))

    if len(impacted_strengths) > 0:
        log.info("  Impact strength distribution (impacted sims only):")
        for q in [5, 25, 50, 75, 95]:
            log.info("    q%02d = %.4f", q, np.percentile(impacted_strengths, q))

    # Suggest threshold based on distribution
    if len(impacted_strengths) > 100:
        log.info("  Suggested thresholds:")
        for pct in [25, 33, 50]:
            thresh = np.percentile(impacted_strengths, pct)
            n_above = (strengths > thresh).sum()
            log.info("    p%d = %.4f (%d strong = %.1f%% of total)",
                     pct, thresh, n_above, 100 * n_above / len(strengths))

    return 0


if __name__ == "__main__":
    sys.exit(main())
