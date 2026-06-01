#!/usr/bin/env python
"""
Regenerate publication figures from v3 result JSONs (standalone; no src imports).

Currently:
  - multistream significance (per-stream look-elsewhere z + joint annotation)

Usage:
  python scripts/plot_paper_figures.py --significance \
      --multistream outputs/multistream/joint_significance_v3.json \
      --out paper/figures
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path


def plot_significance(mj_path: str, out_dir: str) -> str:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.load(open(mj_path))
    ps = d["per_stream"]
    names = [s["stream"] for s in ps]
    le_z = np.array([s["le_z"] for s in ps])
    order = np.argsort(le_z)
    names = [names[i] for i in order]
    le_z = le_z[order]
    colors = ["#c0392b" if z > 0 else "#2c3e50" for z in le_z]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.barh(names, le_z, color=colors, edgecolor="k", linewidth=0.5, alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    for thr in (-2, 2):
        ax.axvline(thr, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Look-elsewhere-corrected significance  z")
    ax.set_xlim(-3.2, 3.2)
    ax.set_title("Per-stream subhalo-impact significance (v3)")
    txt = (f"Joint (look-elsewhere):\n"
           f"Stouffer Z = {d['stouffer_z']:.2f} (p = {d['stouffer_p']:.2f})\n"
           f"Fisher $\\chi^2$ = {d['fisher_chi2']:.1f} (p = {d['fisher_p']:.2f})\n"
           f"→ no joint detection")
    ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="#f7f7f7", ec="gray", alpha=0.9))
    fig.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = os.path.join(out_dir, "significance_v3.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--significance", action="store_true")
    ap.add_argument("--multistream", default="outputs/multistream/joint_significance_v3.json")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    if args.significance:
        print("wrote", plot_significance(args.multistream, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
