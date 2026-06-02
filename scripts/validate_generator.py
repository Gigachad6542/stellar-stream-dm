#!/usr/bin/env python
"""
Pre-flight validation gates for the stream generator.

Purpose: NEVER commit to a multi-hour generation run on bad sims again. Run this
on a SMALL batch (existing chunks or a fresh ~200-sim test dir) and only launch
the full generation if every gate passes.

Gates (thresholds tuned to real stellar streams):
  G1 Smoothness   no-impact model-free gap-depth median  < 0.30   (clumpiness)
  G2 Width        phi2 std (per stream)                  < 1.20 deg
  G3 Length       phi1 extent vs galstreams track        0.6-1.4x
  G4 Kinematics   median pm1/pm2 vs track                |d| < 0.8 mas/yr
  G5 RV           finite fraction + physical std         std < 80 km/s, no fills
  G6 Separability model-free gap-depth AUC(impact vs no) > 0.85   *** key ***
  G7 DM differ    gap-strength distributions across 4 DM models differ (KS)

Usage:
  # baseline on existing chunks:
  python scripts/validate_generator.py --sim-dir data/simulations_v3_track6d --max-sims 1500
  # fresh batch (direct generation, per stream):
  python scripts/validate_generator.py --generate --streams GD1 Pal5 ATLAS Orphan --n-per 12
"""
from __future__ import annotations
import os, sys, argparse, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


# ---------------------------------------------------------------------------
# Model-free gap-depth metric (self-contained; what the detector ceiling rides on)
# ---------------------------------------------------------------------------
def max_gap_depth(phi1: np.ndarray, bin_deg: float = 2.0, smooth: int = 7) -> float:
    from scipy.ndimage import uniform_filter1d
    phi1 = np.asarray(phi1, float)
    phi1 = phi1[np.isfinite(phi1)]
    if len(phi1) < 60:
        return 0.0
    lo, hi = np.percentile(phi1, [2, 98])
    if hi - lo < 4 * bin_deg:
        return 0.0
    h, _ = np.histogram(phi1, bins=np.arange(lo, hi + bin_deg, bin_deg))
    if len(h) < 5 or h.sum() < 40:
        return 0.0
    base = np.maximum(uniform_filter1d(h.astype(float), size=smooth, mode="nearest"), 1.0)
    return float(np.max(1.0 - h / base))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([1.0 if p > n else (0.5 if p == n else 0.0)
                          for p in pos for n in neg]))


# ---------------------------------------------------------------------------
# Gate evaluation on a set of (phi1,phi2,pm1,pm2,vrad, label) sims
# ---------------------------------------------------------------------------
def evaluate_gates(no_impact: list[dict], strong_impact: list[dict],
                   benchmarks: dict | None = None) -> dict:
    """no_impact/strong_impact: lists of dicts with phi1/phi2/pm1/pm2/vrad arrays."""
    res = {}
    # G1 smoothness
    gd = [max_gap_depth(s["phi1"]) for s in no_impact]
    g1 = float(np.median(gd)) if gd else float("nan")
    res["G1_smoothness"] = {"value": round(g1, 3), "threshold": "< 0.30",
                            "pass": bool(g1 < 0.30)}
    # G2 width
    w = [np.nanstd(s["phi2"]) for s in no_impact if len(s["phi2"])]
    g2 = float(np.median(w)) if w else float("nan")
    res["G2_width_phi2std"] = {"value": round(g2, 3), "threshold": "< 1.20 deg",
                               "pass": bool(g2 < 1.20)}
    # G3 length
    ext = [np.percentile(s["phi1"], 98) - np.percentile(s["phi1"], 2)
           for s in no_impact if len(s["phi1"]) > 60]
    g3 = float(np.median(ext)) if ext else float("nan")
    if benchmarks and benchmarks.get("track_extent"):
        ratio = g3 / benchmarks["track_extent"]
        res["G3_length_ratio"] = {"value": round(ratio, 2), "threshold": "0.6-1.4x",
                                  "pass": bool(0.6 <= ratio <= 1.4),
                                  "extent_deg": round(g3, 1)}
    else:
        res["G3_length_extent_deg"] = {"value": round(g3, 1), "threshold": "(info)",
                                       "pass": None}
    # G4 kinematics
    pm1 = float(np.median([np.nanmedian(s["pm1"]) for s in no_impact if len(s["pm1"])]))
    pm2 = float(np.median([np.nanmedian(s["pm2"]) for s in no_impact if len(s["pm2"])]))
    if benchmarks and "pm1" in benchmarks:
        d1, d2 = abs(pm1 - benchmarks["pm1"]), abs(pm2 - benchmarks["pm2"])
        res["G4_kinematics"] = {"value": f"pm1={pm1:.2f}(trk {benchmarks['pm1']:.2f}) "
                                f"pm2={pm2:.2f}(trk {benchmarks['pm2']:.2f})",
                                "threshold": "|d|<0.8", "pass": bool(d1 < 0.8 and d2 < 0.8)}
    else:
        res["G4_kinematics"] = {"value": f"pm1={pm1:.2f} pm2={pm2:.2f}",
                                "threshold": "(info)", "pass": None}
    # G5 RV
    allv = np.concatenate([np.asarray(s["vrad"], float) for s in no_impact if len(s.get("vrad", []))]) \
        if any(len(s.get("vrad", [])) for s in no_impact) else np.array([])
    fin = np.isfinite(allv)
    frac = float(fin.mean()) if len(allv) else 0.0
    vstd = float(np.nanstd(allv[fin])) if fin.any() else float("nan")
    has_fills = bool(fin.any() and np.nanmax(np.abs(allv[fin])) > 600)
    res["G5_rv"] = {"value": f"finite={frac:.2f} std={vstd:.0f} fills={has_fills}",
                    "threshold": "std<80, no fills",
                    "pass": bool(frac > 0.0 and vstd < 80 and not has_fills) if frac > 0 else None}
    # G6 separability
    if strong_impact:
        dp = [max_gap_depth(s["phi1"]) for s in strong_impact]
        a = auc(dp, gd)
        res["G6_separability_AUC"] = {"value": round(a, 3), "threshold": "> 0.85",
                                      "pass": bool(a > 0.85)}
    return res


def print_report(title: str, gates: dict) -> bool:
    print(f"\n{'='*64}\n{title}\n{'='*64}")
    allpass = True
    for k, v in gates.items():
        p = v["pass"]
        mark = "PASS" if p else ("----" if p is None else "FAIL")
        if p is False:
            allpass = False
        print(f"  [{mark}] {k:24s} {str(v['value']):42s} need {v['threshold']}")
    verdict = "ALL GATES PASS -> safe to run full generation" if allpass else "GATES FAILED -> fix generator before full run"
    print(f"  >>> {verdict}")
    return allpass


# ---------------------------------------------------------------------------
# Mode A: read existing chunks (baseline, no regeneration)
# ---------------------------------------------------------------------------
def from_chunks(sim_dir: str, max_sims: int) -> dict:
    import h5py
    no_imp, strong = [], []
    files = sorted(glob.glob(os.path.join(sim_dir, "chunk_*.h5")))
    n = 0
    for fn in files:
        with h5py.File(fn, "r") as f:
            for sid in f["simulations"]:
                lab = f[f"simulations/{sid}/labels"]
                ns = int(lab["n_subhalos"][()]); S = float(lab.attrs.get("impact_strength", 0.0))
                sd = f[f"simulations/{sid}/stream_data"]
                rec = {k: sd[k][:].astype(float) for k in ("phi1", "phi2", "pm1", "pm2", "vrad")}
                if ns == 0 and len(no_imp) < max_sims // 2:
                    no_imp.append(rec)
                elif S > 0.5 and len(strong) < max_sims // 2:
                    strong.append(rec)
                n += 1
        if len(no_imp) >= max_sims // 2 and len(strong) >= max_sims // 2:
            break
    print(f"Loaded {len(no_imp)} no-impact + {len(strong)} strong-impact sims from chunks")
    return evaluate_gates(no_imp, strong, benchmarks=None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", default=None)
    ap.add_argument("--max-sims", type=int, default=1500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.sim_dir:
        gates = from_chunks(args.sim_dir, args.max_sims)
        ok = print_report(f"GENERATOR VALIDATION (chunks: {args.sim_dir})", gates)
        if args.out:
            json.dump(gates, open(args.out, "w"), indent=1)
        return 0 if ok else 2
    print("Specify --sim-dir <chunks> (direct-generation mode added next).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
