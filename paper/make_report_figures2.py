#!/usr/bin/env python
"""
Methodology + real-data figures for the full report (current pipeline).
  fig11_detector_vs_N   detector p_impact vs member count (sparse-sampling caveat)
  fig12_multistream     7-stream significance (naive vs look-elsewhere) — real data
  fig13_stream_tracks   sky tracks of the target streams (galstreams)
  fig14_erkal_kick      Erkal & Belokurov (2015) Plummer impulse kick
  fig15_mass_function   subhalo mass function per DM model (counts per dex)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_training_data import _fix_galpy_dll_path
_fix_galpy_dll_path()
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("paper/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True, "figure.facecolor": "white"})
C_S, C_I, C_G, C_B = "#2c7fb8", "#d95f0e", "#31a354", "#de2d26"


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {FIG/name}")


def fig_detector_vs_N():
    # measured on real GD-1 subsampled (controlled sparse-sampling test)
    N = np.array([811, 400, 200, 100, 50]); p = np.array([0.774, 0.896, 0.923, 0.990, 0.988])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axvspan(500, 3000, color="#e8f5e9", label="reliable regime (N≳500)")
    ax.plot(N, p, "o-", color=C_I, lw=2, ms=8)
    ax.axhline(0.5, ls=":", color="#999")
    ax.set_xscale("log"); ax.set_xlabel("number of clean member stars  N")
    ax.set_ylabel("detector $p_\\mathrm{impact}$ (real GD-1)")
    ax.set_title("The detector needs enough members")
    ax.annotate("sparse sampling\nmimics gaps", xy=(70, 0.97), xytext=(120, 0.7),
                color=C_B, fontsize=10, arrowprops=dict(arrowstyle="->", color=C_B))
    ax.legend(loc="lower left"); ax.set_ylim(0.4, 1.05)
    save(fig, "fig11_detector_vs_N.png")


def fig_multistream():
    d = json.loads(Path("outputs/multistream/joint_significance_corrected.json").read_text())
    rows = d.get("per_stream", [])
    names = [r["stream"] for r in rows]
    zn = [r.get("z", np.nan) for r in rows]; zl = [r.get("le_z", np.nan) for r in rows]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.bar(x - w/2, zn, w, label="naive (single best cell)", color="#bdbdbd")
    ax.bar(x + w/2, zl, w, label="look-elsewhere corrected", color=C_S)
    ax.axhline(0, color="#444", lw=0.8); ax.axhline(3, ls="--", color=C_B, lw=1)
    ax.text(len(names)-0.5, 3.15, "3$\\sigma$", color=C_B, ha="right", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("significance z")
    sj = d.get("stouffer_z", np.nan); fp = d.get("fisher_p", np.nan)
    ax.set_title(f"No coherent detection across 7 streams\n"
                 f"naive z collapses under look-elsewhere; joint Stouffer Z={sj:.2f}, "
                 f"Fisher p={fp:.2f} (CDM-consistent)", fontsize=11)
    ax.legend(loc="upper left")
    save(fig, "fig12_multistream.png")


def fig_stream_tracks():
    import galstreams, astropy.units as u
    mws = galstreams.MWStreams(verbose=False)
    keys = {"GD1": "GD-1-I21", "ATLAS": "ATLAS-I21", "Jhelum": "Jhelum-I21",
            "Orphan": "Orphan-I21", "Pal5": "Pal5-I21", "Fjorm": "Fjorm-I21", "Sylgr": "Sylgr-I21"}
    cols = plt.cm.tab10(np.linspace(0, 1, len(keys)))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for (name, key), c in zip(keys.items(), cols):
        tr = None
        for k in (key, name):
            try: tr = mws[k]; break
            except Exception: continue
        if tr is None: continue
        icrs = tr.track.icrs
        ax.scatter(icrs.ra.deg, icrs.dec.deg, s=3, color=c, label=name)
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
    ax.set_title("Target stream sample (galstreams tracks)")
    ax.legend(ncol=4, fontsize=8, loc="lower center")
    save(fig, "fig13_stream_tracks.png")


def fig_erkal_kick():
    from src.simulation.subhalo import scale_radius_from_mass
    G = 4.30091e-6  # kpc (km/s)^2 / Msun
    w = 150.0       # km/s relative speed
    b = np.linspace(0.01, 2.0, 300)  # kpc perpendicular distance
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for logM, c in ((7.0, C_S), (8.0, C_G), (9.0, C_I)):
        M = 10**logM; rs = scale_radius_from_mass(M)
        dv = (2 * G * M / w) * b / (b**2 + rs**2)  # km/s
        ax.plot(b, dv, color=c, lw=2, label=f"$10^{{{logM:.0f}}}\\,M_\\odot$ ($r_s$={rs:.2f} kpc)")
    ax.set_xlabel("impact parameter b [kpc]"); ax.set_ylabel("velocity kick $\\Delta v$ [km/s]")
    ax.set_title("Erkal & Belokurov (2015) impulse kick")
    ax.legend(); ax.set_xlim(0, 2)
    save(fig, "fig14_erkal_kick.png")


def fig_mass_function():
    M = np.logspace(6.5, 9.0, 200); alpha = -1.9
    dN = M ** (alpha + 1)  # counts per dex (dN/dlogM ∝ M^(α+1))
    from src.simulation.mass_functions import wdm_suppression, fdm_suppression, wdm_half_mode_mass, fdm_jeans_mass
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(M, dN/dN.max(), color="#444", lw=2, label="CDM / SIDM")
    ax.plot(M, dN*wdm_suppression(M, wdm_half_mode_mass(3.0))/dN.max(), color="#41b6c4", lw=2, label="WDM 3 keV")
    ax.plot(M, dN*fdm_suppression(M, fdm_jeans_mass(1e-22))/dN.max(), color=C_I, lw=2, label="FDM 10$^{-22}$eV")
    ax.axvspan(10**7.5, 10**8.7, color="#fdd", alpha=0.5, label="our sensitive band")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("subhalo mass  $M/M_\\odot$"); ax.set_ylabel("relative counts per dex")
    ax.set_title("Subhalo mass function by DM model")
    ax.legend(fontsize=8); ax.set_ylim(1e-3, 1.5)
    save(fig, "fig15_mass_function.png")


def fig_detection_vs_strength():
    # measured completeness vs realised gap strength (detector_completeness.py)
    edges = ["<0.3", "0.3–0.5", "0.5–0.7", ">0.7"]; p = [0.048, 0.144, 0.803, 1.000]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.bar(edges, p, color=[C_S, C_S, C_G, C_G], width=0.7)
    for i, v in enumerate(p):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    ax.axhline(0.5, ls=":", color="#999")
    ax.set_xlabel("realised gap strength (depth)"); ax.set_ylabel("P(detect | impact)")
    ax.set_title("Detection is a step function in gap strength")
    ax.set_ylim(0, 1.1)
    save(fig, "fig16_detection_vs_strength.png")


def fig_dm_forecast():
    d = json.loads(Path("outputs/dm/discrimination_forecast.json").read_text())
    order = ["FDM 1e-22", "WDM 3 keV", "WDM 4 keV", "WDM 6 keV", "FDM 1e-21", "SIDM"]
    order = [m for m in order if m in d]
    rate = [d[m]["rate_ratio"] for m in order]
    def finite_or_inf(value):
        if value is None:
            return np.inf
        try:
            value = float(value)
        except (TypeError, ValueError):
            return np.inf
        return value if np.isfinite(value) else np.inf
    ndet = [finite_or_inf(d[m].get("N_det_3sig", np.inf)) for m in order]
    lbl = [m.replace(" ", "\n") for m in order]
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.2))
    cols = [C_G if (np.isfinite(n) and n < 50) else ("#fec44f" if np.isfinite(n) and n < 1e4 else C_B) for n in ndet]
    ax[0].bar(lbl, rate, color=cols, width=0.7); ax[0].axhline(1.0, ls=":", color="#444")
    ax[0].set_ylabel("detectable-impact rate / CDM"); ax[0].set_title("(a) Abundance suppression")
    ax[0].set_ylim(0, 1.15)
    nd_plot = [min(n, 1e6) for n in ndet]
    ax[1].bar(lbl, nd_plot, color=cols, width=0.7); ax[1].set_yscale("log")
    ax[1].set_ylabel("detections for 3σ vs CDM"); ax[1].set_title("(b) Detections needed (rate+mass)")
    for i, n in enumerate(ndet):
        ax[1].text(i, min(n, 1e6)*1.4, ("∞" if not np.isfinite(n) or n > 1e5 else f"{n:.0f}"),
                   ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("Baseline DM discrimination using abundance + mass (SIDM needs gap shape)",
                 y=1.02, fontsize=12)
    save(fig, "fig17_dm_forecast.png")


def fig_sidm_morphology():
    d = json.loads(Path("outputs/dm/sidm_morphology.json").read_text())
    rows = d["rows"]; logM = [r[0] for r in rows]; cdm = [r[1] for r in rows]; sidm = [r[2] for r in rows]
    x = np.arange(len(logM)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - w/2, cdm, w, color="#444", label="CDM (cuspy NFW)")
    ax.bar(x + w/2, sidm, w, color=C_I, label=f"SIDM (cored, {d['core_factor']:.0f}× r_s)")
    ax.set_xticks(x); ax.set_xticklabels([f"$10^{{{m:.1f}}}$" for m in logM])
    ax.set_xlabel("subhalo mass  $M/M_\\odot$"); ax.set_ylabel("gap depth (at fixed mass)")
    ax.set_title(f"SIDM signal: cored subhalos carve shallower gaps (AUC {d['mean_auc']:.2f})")
    ax.legend(); ax.set_ylim(0, 1.15)
    save(fig, "fig18_sidm_morphology.png")


if __name__ == "__main__":
    import traceback
    for fn in (fig_detector_vs_N, fig_multistream, fig_erkal_kick, fig_mass_function,
               fig_stream_tracks, fig_detection_vs_strength, fig_dm_forecast, fig_sidm_morphology):
        try: fn()
        except Exception: print(f"FAILED {fn.__name__}"); traceback.print_exc()
