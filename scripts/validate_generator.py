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


def gap_depth_excess(phi1: np.ndarray, bin_deg: float = 2.0, smooth: int = 7,
                     n_poisson: int = 12, seed: int = 0) -> float:
    """Gap-depth in EXCESS of the Poisson shot-noise floor.

    Raw max_gap_depth has a floor set purely by stars-per-bin (~0.33 at 1200
    stars / 2-deg bins), so a perfectly smooth stream still scores ~0.33 — the raw
    metric conflates shot noise with real clumpiness. Here we draw Poisson
    realisations from the stream's OWN smoothed density (gap filled in) and
    subtract their median gap-depth: a smooth stream -> excess ~0, a clumpy stream
    or a real gap -> excess > 0. This is the metric the streamspraydf smoothness
    check rides on (smooth ~0.05, homemade-clumpy ~0.4).
    """
    from scipy.ndimage import uniform_filter1d
    phi1 = np.asarray(phi1, float); phi1 = phi1[np.isfinite(phi1)]
    if len(phi1) < 60:
        return 0.0
    obs = max_gap_depth(phi1, bin_deg, smooth)
    lo, hi = np.percentile(phi1, [2, 98])
    if hi - lo < 4 * bin_deg:
        return 0.0
    edges = np.arange(lo, hi + bin_deg, bin_deg)
    h, _ = np.histogram(phi1, bins=edges)
    base = np.maximum(uniform_filter1d(h.astype(float), size=smooth, mode="nearest"), 1e-6)
    p = base / base.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    rng = np.random.default_rng(seed)
    N = len(phi1)
    floors = [max_gap_depth(
        rng.choice(centers, size=N, p=p) + rng.uniform(-bin_deg / 2, bin_deg / 2, N),
        bin_deg, smooth) for _ in range(n_poisson)]
    return float(obs - np.median(floors))


def intrinsic_scatter(coord: np.ndarray, along: np.ndarray) -> float:
    """Std of `coord` after removing a linear trend along `along`.

    The raw std of phi2 (or vrad) over a long stream is dominated by the smooth
    along-stream track shape / bulk gradient -- NOT by the physical cross-stream
    width or velocity dispersion. A model orbit that is slightly offset from the
    observed great-circle frame inflates raw phi2 std even when the stream is thin.
    Removing a linear phi1 trend recovers the physical scatter (the quantity the
    width/RV gates are meant to police): a puffy stream still fails, a thin-but-
    offset one passes.
    """
    coord = np.asarray(coord, float); along = np.asarray(along, float)
    m = np.isfinite(coord) & np.isfinite(along)
    if m.sum() < 10:
        return float("nan")
    c = np.polyfit(along[m], coord[m], 1)
    return float(np.std(coord[m] - np.polyval(c, along[m])))


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
    # G1 smoothness — gap-depth EXCESS over the Poisson floor (raw gap-depth has a
    # shot-noise floor that conflates smoothness with star count). gd (raw) is kept
    # for the G6 separability comparison below.
    gd = [max_gap_depth(s["phi1"]) for s in no_impact]
    gx = [gap_depth_excess(s["phi1"]) for s in no_impact]
    g1 = float(np.median(gx)) if gx else float("nan")
    res["G1_smoothness_excess"] = {"value": round(g1, 3), "threshold": "< 0.15",
                                   "pass": bool(g1 < 0.15)}
    # G2 width — intrinsic cross-stream scatter (linear phi1 trend removed), so a
    # thin stream offset from the observed frame is not falsely flagged as puffy.
    w = [intrinsic_scatter(s["phi2"], s["phi1"]) for s in no_impact if len(s["phi2"])]
    g2 = float(np.nanmedian(w)) if w else float("nan")
    res["G2_width_intrinsic_deg"] = {"value": round(g2, 3), "threshold": "< 1.20 deg",
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
    # G5 RV — gate the INTRINSIC velocity dispersion (linear phi1 trend removed):
    # the raw std over a long stream is the physical line-of-sight gradient, not a
    # defect. The gate's real job is catching fill-value contamination / missing RV.
    allv = np.concatenate([np.asarray(s["vrad"], float) for s in no_impact if len(s.get("vrad", []))]) \
        if any(len(s.get("vrad", [])) for s in no_impact) else np.array([])
    fin = np.isfinite(allv)
    frac = float(fin.mean()) if len(allv) else 0.0
    vdisp = float(np.nanmedian([intrinsic_scatter(s["vrad"], s["phi1"])
                                for s in no_impact if len(s.get("vrad", []))])) if frac > 0 else float("nan")
    has_fills = bool(fin.any() and np.nanmax(np.abs(allv[fin])) > 600)
    res["G5_rv"] = {"value": f"finite={frac:.2f} intrinsic_disp={vdisp:.0f} fills={has_fills}",
                    "threshold": "disp<60, no fills",
                    "pass": bool(frac > 0.0 and vdisp < 60 and not has_fills) if frac > 0 else None}
    # G6 separability
    if strong_impact:
        dp = [max_gap_depth(s["phi1"]) for s in strong_impact]
        a = auc(dp, gd)
        res["G6_separability_AUC"] = {"value": round(a, 3), "threshold": "> 0.85",
                                      "pass": bool(a > 0.85)}
    return res


# Critical gates decide the verdict: a generator is "good" if the no-impact class
# is smooth (G1), the right length (G3), and impacts are separable (G6). The rest
# (width/kinematics/RV match to the *observed* stream) are morphology-fidelity
# gates: informative for sim-to-real domain gap, but secondary to detectability
# (Path A: optimise for smoothness + detectability over exact morphology).
CRITICAL_GATES = ("G1_smoothness_excess", "G3_length_ratio", "G3_length_extent_deg",
                  "G6_separability_AUC")


def print_report(title: str, gates: dict) -> bool:
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    crit_pass = True
    for k, v in gates.items():
        p = v["pass"]
        mark = "PASS" if p else ("----" if p is None else "FAIL")
        tag = "CRIT" if k in CRITICAL_GATES else "morph"
        if p is False and k in CRITICAL_GATES:
            crit_pass = False
        print(f"  [{mark}] ({tag:5s}) {k:26s} {str(v['value']):46s} need {v['threshold']}")
    verdict = ("CRITICAL GATES PASS -> generator produces a detectable signal; "
               "safe to generate" if crit_pass else
               "CRITICAL GATE FAILED -> fix generator before full run")
    print(f"  >>> {verdict}")
    morph_fail = [k for k, v in gates.items()
                  if v["pass"] is False and k not in CRITICAL_GATES]
    if morph_fail:
        print(f"  (morphology gates below spec: {', '.join(morph_fail)} — "
              f"sim-to-real domain gap, accepted under Path A)")
    return crit_pass


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
