"""
Create synthetic test HDF5 simulations for Phase 4 GNN validation.

Does NOT use gala or stream generation — generates realistic-looking stream
data analytically so the DataLoader / GNN pipeline can be validated instantly.

Usage:
    python scripts/make_synthetic_test_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import uuid

OUTPUT = Path("data/simulations_test")
N_SIMS = 20
DM_MODELS = ["CDM", "WDM", "FDM", "SIDM"]

def make_stream(n_stars: int, dm_idx: int, seed: int) -> dict:
    """Analytic GD-1-like stream with optional subhalo gap."""
    rng = np.random.default_rng(seed)

    # Stream spine parameters
    phi1 = rng.uniform(-60.0, 0.0, n_stars)  # deg
    phi2 = -0.08 * phi1 + rng.normal(0.0, 0.15, n_stars)  # slight tilt + scatter
    dist = 9.0 + 0.05 * phi1 + rng.normal(0.0, 0.3, n_stars)  # kpc

    # Realistic GD-1 proper motions
    pm1 = -8.5 + 0.02 * phi1 + rng.normal(0.0, 0.15, n_stars)   # mas/yr
    pm2 = -2.1 + 0.005 * phi1 + rng.normal(0.0, 0.12, n_stars)  # mas/yr
    vrad = -200.0 + rng.normal(0.0, 5.0, n_stars)                # km/s

    # For suppressed WDM/FDM models or randomly: add a gap. SIDM is kept as an
    # unsuppressed mass-function reference in the current project framing.
    if dm_idx in (1, 2) or rng.random() < 0.3:
        gap_phi1 = rng.uniform(-55.0, -5.0)
        gap_hw = rng.uniform(2.0, 8.0)
        in_gap = np.abs(phi1 - gap_phi1) < gap_hw
        # Kick velocities of stars near the gap
        dv = rng.uniform(0.5, 3.0)
        sign = np.where(phi1 > gap_phi1, 1.0, -1.0)
        phi1[in_gap] += sign[in_gap] * gap_hw * 0.1
        pm2[in_gap] += dv * 0.05 * sign[in_gap]

    # Observational errors
    e_pm1 = np.abs(rng.normal(0.10, 0.02, n_stars)).clip(0.02, 0.5)
    e_pm2 = np.abs(rng.normal(0.10, 0.02, n_stars)).clip(0.02, 0.5)
    e_dist = np.abs(rng.normal(0.5, 0.1, n_stars)).clip(0.05, 2.0)

    # Labels
    n_sub = int(rng.integers(0, 4))
    log_m = float(np.log10(rng.uniform(1e6, 1e8) + 1.0)) if n_sub > 0 else 0.0

    return {
        "phi1": phi1.astype(np.float32),
        "phi2": phi2.astype(np.float32),
        "dist": dist.astype(np.float32),
        "pm1": pm1.astype(np.float32),
        "pm2": pm2.astype(np.float32),
        "vrad": vrad.astype(np.float32),
        "e_pm1": e_pm1.astype(np.float32),
        "e_pm2": e_pm2.astype(np.float32),
        "e_dist": e_dist.astype(np.float32),
        "dm_model_idx": dm_idx,
        "dm_model": DM_MODELS[dm_idx],
        "log_m_sub_mean": log_m,
        "n_subhalos": n_sub,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    out_path = OUTPUT / "chunk_00000.h5"

    with h5py.File(str(out_path), "w") as f:
        for i in range(N_SIMS):
            run_id = str(uuid.uuid4())[:8]
            dm_idx = i % 4
            seed = int(rng.integers(0, 2**31))
            n_stars = int(rng.integers(300, 1500))

            d = make_stream(n_stars, dm_idx, seed)

            grp = f.create_group(f"simulations/{run_id}")
            grp.attrs["dm_model"] = np.bytes_(d["dm_model"])
            grp.attrs["stream_name"] = np.bytes_("GD1")

            sd = grp.create_group("stream_data")
            for key in ["phi1", "phi2", "dist", "pm1", "pm2", "vrad", "e_pm1", "e_pm2", "e_dist"]:
                sd.create_dataset(key, data=d[key], compression="gzip", compression_opts=4)

            sub = grp.create_group("subhalos")
            sub.create_dataset("mass", data=np.array([], dtype=np.float32))
            sub.create_dataset("impact_param", data=np.array([], dtype=np.float32))
            sub.create_dataset("flyby_vel", data=np.array([], dtype=np.float32))

            lab = grp.create_group("labels")
            lab["dm_model_idx"] = d["dm_model_idx"]
            lab["log_m_sub_mean"] = d["log_m_sub_mean"]
            lab["n_subhalos"] = d["n_subhalos"]

    print(f"Wrote {N_SIMS} synthetic simulations to {out_path}")


if __name__ == "__main__":
    main()
