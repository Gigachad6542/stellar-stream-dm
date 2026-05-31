"""
Stellar Stream Dark Matter Detection  -  Full Technical Report
Author: Daniel Watson

Generates a research-paper-quality PDF with embedded matplotlib figures,
all facts sourced from config/*.yaml and src/ code.
"""
import os
import io
import math
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as mpatches

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, HRFlowable, Image, ListFlowable, ListItem
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUTPUT = os.path.join(os.path.dirname(__file__), "Stellar_Stream_DM_Full_Report.pdf")
FIG_DIR = os.path.join(os.path.dirname(__file__), "_report_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# "" Colour palette """"""""""""""""""""""""""""""""""""""""""""""""""""""""""
ACCENT       = HexColor("#1a5276")
ACCENT_LIGHT = HexColor("#2980b9")
RULE_COLOR   = HexColor("#bdc3c7")
HEADER_BG    = HexColor("#2c3e50")
ROW_ALT      = HexColor("#f2f4f5")
CALLOUT_BG   = HexColor("#eaf2f8")
CALLOUT_BORDER = HexColor("#2980b9")

# "" matplotlib global style """""""""""""""""""""""""""""""""""""""""""""""""
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#333333",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#cccccc",
})

COLORS_MPL = {
    "blue": "#2980b9",
    "orange": "#e67e22",
    "red": "#e74c3c",
    "purple": "#8e44ad",
    "green": "#27ae60",
    "cyan": "#16a085",
    "gray": "#7f8c8d",
}


# ---------------------------------------------------------------------------
# PHYSICS FUNCTIONS  (mirrors src/inference/hierarchical.py exactly)
# ---------------------------------------------------------------------------
def cdm_mass_function(log_m):
    """dN/dlogM ~ M^(-0.9)  (from dN/dM ~ M^(-1.9), Springel+2008)."""
    return 10.0 ** (log_m * (-0.9))


def transfer_function(log_m, log_mhm):
    """T(M, M_hm) = [1 + (M_hm/M)^2]^(-1), Schneider+2012."""
    ratio = 10.0 ** (log_mhm - log_m)
    return 1.0 / (1.0 + ratio * ratio)


def expected_rate(log_mhm, length_deg=60, age_gyr=5, dist_kpc=15):
    """Expected subhalo impact count (CDM normalised to ~2.1 for GD-1, Bonaca+2019)."""
    n_grid = 200
    log_m_min, log_m_max = 5.0, 9.0
    d_log_m = (log_m_max - log_m_min) / n_grid
    integral_supp = integral_cdm = 0.0
    for i in range(n_grid):
        log_m = log_m_min + (i + 0.5) * d_log_m
        dn = cdm_mass_function(log_m)
        integral_supp += dn * transfer_function(log_m, log_mhm)
        integral_cdm += dn
    integral_supp *= d_log_m
    integral_cdm *= d_log_m
    norm = 2.1 / integral_cdm
    l_kpc = length_deg * (math.pi / 180) * dist_kpc
    return max(norm * integral_supp * (l_kpc / 10.0) * (age_gyr / 5.0), 0.01)


# ---------------------------------------------------------------------------
# FIGURE GENERATORS
# ---------------------------------------------------------------------------
def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def fig_mass_function():
    """Fig 1: CDM vs suppressed subhalo mass functions."""
    log_m = np.linspace(5.0, 9.5, 200)
    cdm = np.array([cdm_mass_function(m) for m in log_m])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(log_m, np.log10(cdm), color=COLORS_MPL["blue"], lw=2, label="CDM (no suppression)")
    for mhm, col, ls, lbl in [
        (7.0, COLORS_MPL["orange"], "--", r"WDM ($M_{\rm hm}=10^7\;M_\odot$)"),
        (8.0, COLORS_MPL["red"], "-.", r"WDM ($M_{\rm hm}=10^8\;M_\odot$)"),
        (7.5, COLORS_MPL["purple"], ":", r"FDM ($M_{\rm hm}=10^{7.5}\;M_\odot$)"),
    ]:
        supp = np.array([cdm_mass_function(m) * transfer_function(m, mhm) for m in log_m])
        ax.plot(log_m, np.log10(np.maximum(supp, 1e-20)), color=col, lw=1.8, ls=ls, label=lbl)

    ax.set_xlabel(r"$\log_{10}(M_{\rm sub}\;/\;M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(dN/d\log M)$ [arb.]")
    ax.set_title("Subhalo Mass Function: CDM vs. Suppressed Models")
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    return _save(fig, "mass_function.png")


def fig_transfer_function():
    """Fig 2: Transfer function T(M, M_hm) for several M_hm."""
    log_m = np.linspace(5.0, 10.0, 300)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for mhm, col, lbl in [
        (6.5, COLORS_MPL["green"], r"$\log_{10} M_{\rm hm}=6.5$"),
        (7.0, COLORS_MPL["orange"], r"$\log_{10} M_{\rm hm}=7.0$"),
        (7.5, COLORS_MPL["purple"], r"$\log_{10} M_{\rm hm}=7.5$"),
        (8.0, COLORS_MPL["red"], r"$\log_{10} M_{\rm hm}=8.0$"),
        (8.5, COLORS_MPL["gray"], r"$\log_{10} M_{\rm hm}=8.5$"),
    ]:
        t = np.array([transfer_function(m, mhm) for m in log_m])
        ax.plot(log_m, t, color=col, lw=1.8, label=lbl)
    ax.axhline(0.5, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel(r"$\log_{10}(M\;/\;M_\odot)$")
    ax.set_ylabel(r"$T(M, M_{\rm hm})$")
    ax.set_title(r"Suppression Transfer Function $T=[1+(M_{\rm hm}/M)^2]^{-1}$")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-0.05, 1.1)
    return _save(fig, "transfer_function.png")


def fig_rate_vs_mhm():
    """Fig 3: Expected impact rate vs M_hm for 3 representative streams."""
    log_mhm = np.linspace(4.0, 10.0, 200)
    streams = [
        ("GD-1 (60 deg, 5 Gyr, 12 kpc)", 60, 5, 12, COLORS_MPL["blue"], "-"),
        ("Pal 5 (20 deg, 11 Gyr, 20 kpc)", 20, 11, 20, COLORS_MPL["orange"], "--"),
        ("Orphan (120 deg, 5 Gyr, 20 kpc)", 120, 5, 20, COLORS_MPL["green"], "-."),
    ]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for lbl, l, a, d, c, ls in streams:
        rates = [expected_rate(m, l, a, d) for m in log_mhm]
        ax.plot(log_mhm, rates, color=c, lw=2, ls=ls, label=lbl)
    ax.set_xlabel(r"$\log_{10}(M_{\rm hm}\;/\;M_\odot)$")
    ax.set_ylabel("Expected number of impacts")
    ax.set_title("Subhalo Impact Rate vs. Half-Mode Mass")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    return _save(fig, "rate_vs_mhm.png")


def fig_stream_sensitivity():
    """Fig 4: Per-stream sensitivity comparison (CDM vs WDM)."""
    streams = ["GD-1", "Pal 5", "Orphan", "ATLAS", "Jhelum", "Fjorm", "Sylgr"]
    lengths = [60, 20, 120, 15, 30, 25, 12]
    ages    = [5, 11, 5, 4, 4, 3, 3]
    dists   = [12, 20, 20, 20, 13, 15, 15]

    cdm = [expected_rate(4.5, l, a, d) for l, a, d in zip(lengths, ages, dists)]
    wdm = [expected_rate(7.5, l, a, d) for l, a, d in zip(lengths, ages, dists)]

    x = np.arange(len(streams))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x - w/2, cdm, w, label="CDM (no suppression)", color=COLORS_MPL["blue"], alpha=0.8, edgecolor="white")
    ax.bar(x + w/2, wdm, w, label=r"WDM ($M_{\rm hm}=10^{7.5}\;M_\odot$)", color=COLORS_MPL["orange"], alpha=0.8, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(streams, fontsize=9)
    ax.set_ylabel("Expected impacts")
    ax.set_title("Per-Stream Sensitivity: CDM vs. WDM")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    return _save(fig, "stream_sensitivity.png")


def fig_architecture():
    """Fig 5: GNN architecture block diagram."""
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    blocks = [
        (0.3, 1.4, 1.4, 1.2, "Input\nN stars\n18 feat", "#3498db"),
        (2.1, 1.4, 1.4, 1.2, "Linear\nProjection\n18 -> 256", "#2ecc71"),
        (3.9, 1.4, 1.8, 1.2, "6x GINEConv\nBatchNorm\nDropout 0.1\nResidual", "#9b59b6"),
        (6.1, 1.4, 1.5, 1.2, "Pooling\nMean+Max\n-> 512-d", "#e67e22"),
        (8.0, 1.4, 1.6, 1.2, "Output MLP\n512->256->128\nEmbedding", "#e74c3c"),
    ]
    for x, y, w, h, text, col in blocks:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=col, edgecolor="white", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white", linespacing=1.3)

    # Arrows
    for x1, x2 in [(1.7, 2.1), (3.5, 3.9), (5.7, 6.1), (7.6, 8.0)]:
        ax.annotate("", xy=(x2, 2.0), xytext=(x1, 2.0),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    # Output heads
    ax.annotate("Binary head\nimpact/family", xy=(9.0, 1.4), xytext=(9.0, 0.3),
                fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="#16a085", ec="white", alpha=0.85),
                color="white", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    ax.annotate("Regression\nMhm + counts", xy=(9.0, 2.6), xytext=(9.0, 3.6),
                fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="#16a085", ec="white", alpha=0.85),
                color="white", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

    ax.set_title("StreamGNNEncoder Architecture", fontsize=11, pad=8)
    return _save(fig, "architecture.png")


def fig_pipeline():
    """Fig 6: Full analysis pipeline."""
    fig, ax = plt.subplots(figsize=(6.5, 2.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    steps = [
        (0.1, 0.5, 1.6, 1.0, "Gaia DR3\nObservations", "#3498db"),
        (2.1, 0.5, 1.6, 1.0, "Graph\nConstruction\n(k=8 NN)", "#2ecc71"),
        (4.1, 0.5, 1.6, 1.0, "GNN\nEncoder\n(128-d)", "#9b59b6"),
        (6.1, 0.5, 1.6, 1.0, "Neural\nSpline Flow\n(SNPE-C)", "#e67e22"),
        (8.1, 0.5, 1.6, 1.0, "Posterior\nP(M_hm | x)", "#e74c3c"),
    ]
    for x, y, w, h, text, col in steps:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=col, edgecolor="white", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white", linespacing=1.3)
    for x1, x2 in [(1.7, 2.1), (3.7, 4.1), (5.7, 6.1), (7.7, 8.1)]:
        ax.annotate("", xy=(x2, 1.0), xytext=(x1, 1.0),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
    return _save(fig, "pipeline.png")


def fig_hierarchical_posterior():
    """Fig 7: Actual hierarchical posterior from validation run."""
    import os as _os
    ROOT = _os.path.dirname(_os.path.abspath(__file__))
    cdm_path = _os.path.join(ROOT, "outputs", "validation", "hierarchical_cdm_samples.npy")
    wdm_path = _os.path.join(ROOT, "outputs", "validation", "hierarchical_wdm_samples.npy")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    if _os.path.exists(cdm_path) and _os.path.exists(wdm_path):
        cdm_samples = np.load(cdm_path)
        wdm_samples = np.load(wdm_path)

        # CDM panel  -  posterior now at ~4.76 with extended prior [4, 10]
        ax1.hist(cdm_samples, bins=80, density=True, color=COLORS_MPL["blue"], alpha=0.7, edgecolor="none")
        cdm_median = np.median(cdm_samples)
        ax1.axvline(cdm_median, color=COLORS_MPL["red"], ls="--", lw=1.5,
                     label=f"Median ({cdm_median:.2f})")
        ax1.set_xlabel(r"$\log_{10}(M_{\rm hm}\;/\;M_\odot)$")
        ax1.set_ylabel("Posterior density")
        ax1.set_title("CDM scenario (true: no suppression)", fontsize=9)
        ax1.legend(fontsize=8)
        ax1.set_xlim(max(3.5, cdm_samples.min() - 0.3), min(7.0, cdm_samples.max() + 0.5))

        # WDM panel
        ax2.hist(wdm_samples, bins=80, density=True, color=COLORS_MPL["orange"], alpha=0.7, edgecolor="none")
        ax2.axvline(7.5, color=COLORS_MPL["red"], ls="--", lw=1.5, label="True value (7.5)")
        ax2.set_xlabel(r"$\log_{10}(M_{\rm hm}\;/\;M_\odot)$")
        ax2.set_title("WDM scenario (true: 7.5)", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.set_xlim(5.0, 10.5)
    else:
        # Fallback schematic  -  matches corrected results
        x = np.linspace(3.5, 7.0, 300)
        h = np.exp(-0.5 * ((x - 4.76) / 0.3) ** 2)
        ax1.plot(x, h / h.max(), color=COLORS_MPL["blue"], lw=2)
        ax1.axvline(4.76, color=COLORS_MPL["red"], ls="--", lw=1.2, label="Median (4.76)")
        ax1.set_title("CDM (schematic)")
        ax1.legend(fontsize=8)

        x2 = np.linspace(5.0, 10.5, 300)
        w = np.exp(-0.5 * ((x2 - 8.16) / 1.0) ** 2)
        ax2.plot(x2, w / w.max(), color=COLORS_MPL["orange"], lw=2)
        ax2.axvline(7.5, color=COLORS_MPL["red"], ls="--", lw=1.2, label="True value (7.5)")
        ax2.set_title("WDM (schematic)")
        ax2.legend(fontsize=8)

    fig.suptitle("Hierarchical Bayesian Inference  -  emcee MCMC", fontsize=10, y=1.02)
    fig.tight_layout()
    return _save(fig, "hierarchical_posterior.png")


def fig_mock_recovery():
    """Fig 8: Mock data challenge truth vs inferred  -  ACTUAL RESULTS."""
    import json as _json, os as _os
    ROOT = _os.path.dirname(_os.path.abspath(__file__))
    results_path = _os.path.join(ROOT, "outputs", "validation", "mock_challenge_results.json")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([4, 9.5], [4, 9.5], "k--", lw=1, alpha=0.4, label="Perfect recovery")

    if _os.path.exists(results_path):
        with open(results_path) as f:
            data = _json.load(f)
        # Group by DM model
        groups = {"CDM": [], "WDM": [], "FDM": [], "SIDM": []}
        for m in data["per_mock"]:
            groups[m["dm"]].append(m)
        style = {
            "CDM":  {"col": COLORS_MPL["blue"], "marker": "o"},
            "WDM":  {"col": COLORS_MPL["orange"], "marker": "s"},
            "FDM":  {"col": COLORS_MPL["purple"], "marker": "^"},
            "SIDM": {"col": COLORS_MPL["green"], "marker": "D"},
        }
        for model, mocks in groups.items():
            truths = [m["truth"] for m in mocks]
            medians = [m["median"] for m in mocks]
            ax.scatter(truths, medians, marker=style[model]["marker"],
                       color=style[model]["col"], s=50, zorder=3, label=model, edgecolors="white", linewidths=0.5)
    else:
        # Fallback
        ax.text(6.5, 6.5, "Mock results file missing", ha="center", fontsize=12, color="gray")

    ax.set_xlabel(r"True $\log_{10}(M_{\rm hm})$")
    ax.set_ylabel(r"Inferred median $\log_{10}(M_{\rm hm})$")
    ax.set_title("Mock Data Challenge: Truth vs. Recovered (Poisson Inversion)")
    ax.set_xlim(4.0, 9.5)
    ax.set_ylim(4.0, 9.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    # Shade the "floor" region
    ax.axhspan(4.5, 6.5, alpha=0.05, color="gray")
    ax.text(8.5, 5.5, "Count-only\nresolution floor", fontsize=7, color="gray", ha="center", style="italic")
    return _save(fig, "mock_recovery.png")


def fig_coverage():
    """Fig 9: Coverage calibration bar chart  -  ACTUAL RESULTS from Poisson inversion."""
    ci_levels = ["68%", "90%", "95%"]
    expected  = [68, 90, 95]
    # Actual measured values from mock challenge (Poisson inversion, count-only)
    empirical = [25, 25, 40]

    x = np.arange(len(ci_levels))
    w = 0.3
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.bar(x - w/2, expected, w, label="Nominal (target)", color=COLORS_MPL["gray"], alpha=0.5, edgecolor=COLORS_MPL["gray"])
    ax.bar(x + w/2, empirical, w, label="Empirical (count-only inference)", color=COLORS_MPL["red"], alpha=0.7, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(ci_levels)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Posterior Coverage: Poisson Inversion (20 Mocks)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    # Annotate passing criterion
    ax.axhline(75, color=COLORS_MPL["orange"], ls=":", lw=1, alpha=0.6)
    ax.text(2.3, 76, "75% pass threshold", fontsize=7, color=COLORS_MPL["orange"], alpha=0.7)
    # Annotate the gap
    ax.annotate("Gap indicates count-only\ninference is underpowered",
                xy=(1, 25), xytext=(1.5, 60), fontsize=7, color=COLORS_MPL["gray"],
                arrowprops=dict(arrowstyle="->", color=COLORS_MPL["gray"], lw=0.8))
    return _save(fig, "coverage.png")


def fig_sbi_rounds():
    """Fig 10: SBI sequential training rounds."""
    rounds = [1, 2, 3, 4, 5]
    cum_sims = [5000, 7000, 9000, 11000, 13000]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.0))
    ax1.bar(rounds, cum_sims, color=COLORS_MPL["blue"], alpha=0.6, edgecolor="white", label="Cumulative sims")
    ax1.set_xlabel("SBI Round")
    ax1.set_ylabel("Cumulative simulations", color=COLORS_MPL["blue"])
    ax1.tick_params(axis="y", labelcolor=COLORS_MPL["blue"])
    ax1.set_title("Sequential Neural Posterior Estimation Training")

    ax2 = ax1.twinx()
    quality = [0.30, 0.55, 0.72, 0.85, 0.92]
    ax2.plot(rounds, quality, "o-", color=COLORS_MPL["green"], lw=2, ms=6, label="Posterior quality (illustrative)")
    ax2.set_ylabel("Quality (0-1)", color=COLORS_MPL["green"])
    ax2.tick_params(axis="y", labelcolor=COLORS_MPL["green"])
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    return _save(fig, "sbi_rounds.png")


def fig_stream_properties():
    """Fig 11: Stream properties comparison."""
    streams = ["GD-1", "Pal 5", "Orphan", "ATLAS", "Jhelum", "Fjorm", "Sylgr"]
    lengths = [60, 20, 120, 15, 30, 25, 12]
    ages    = [5, 11, 5, 4, 4, 3, 3]
    dists   = [12, 20, 20, 20, 13, 15, 15]

    x = np.arange(len(streams))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x - w, lengths, w, label="Length (deg)", color=COLORS_MPL["blue"], alpha=0.8)
    ax.bar(x, dists, w, label="Distance (kpc)", color=COLORS_MPL["orange"], alpha=0.8)
    ax.bar(x + w, ages, w, label="Age (Gyr)", color=COLORS_MPL["green"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(streams, fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title("Target Stellar Stream Properties")
    ax.legend(fontsize=8)
    return _save(fig, "stream_properties.png")


def fig_model_discrimination():
    """Fig 12: Projected Bayes factors between model pairs."""
    pairs = ["CDM+SIDM\nvs WDM", "CDM+SIDM\nvs FDM", "WDM\nvs FDM", "Suppressed\nvs Unsuppressed"]
    bf = [2.5, 2.0, 0.3, 4.0]
    cols = [COLORS_MPL["blue"], COLORS_MPL["purple"], COLORS_MPL["red"], COLORS_MPL["green"]]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bars = ax.bar(range(len(pairs)), bf, color=cols, alpha=0.8, edgecolor="white")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, fontsize=8)
    ax.set_ylabel(r"$\log_{10}$(Bayes factor)")
    ax.set_title("Projected Model Discrimination Power")
    ax.axhline(1.0, color=COLORS_MPL["orange"], ls="--", lw=1.2, alpha=0.6)
    ax.text(3.5, 1.1, "Strong BF\nthreshold", fontsize=7, color=COLORS_MPL["orange"], ha="right")
    ax.set_ylim(bottom=0)
    return _save(fig, "model_discrimination.png")


def fig_log_evidences_heatmap():
    """Fig 13: Per-stream log evidence heatmap from full GNN+SBI pipeline."""
    streams = ["GD-1", "Pal 5", "Orphan", "ATLAS", "Jhelum", "Fjorm", "Sylgr"]
    models = ["CDM", "WDM", "FDM", "SIDM"]
    data = np.array([
        [-3.29, -3.19, -3.04, -3.23],
        [-3.32, -2.17, -3.01, -3.35],
        [-4.82, -2.45, -3.16, -4.82],
        [-2.66, -2.29, -2.83, -2.62],
        [-3.13, -2.24, -3.17, -3.69],
        [-3.50, -2.25, -3.15, -3.37],
        [-2.45, -2.21, -3.17, -3.38],
    ])
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-5.0, vmax=-2.0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_yticks(range(7))
    ax.set_yticklabels(streams, fontsize=9)
    for i in range(7):
        for j in range(4):
            color = "white" if data[i, j] < -3.8 else "black"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Log evidence", fontsize=9)
    ax.set_title("Per-Stream Log Evidence by DM Model (GNN + SBI)", fontsize=11)
    fig.tight_layout()
    return _save(fig, "log_evidences_heatmap.png")


def fig_model_posterior_probs():
    """Fig 14: Overall model posterior probabilities from GNN+SBI pipeline."""
    models = ["CDM", "WDM", "FDM", "SIDM"]
    probs = [0.2, 99.0, 0.9, 0.1]
    colors = [COLORS_MPL["blue"], COLORS_MPL["orange"], COLORS_MPL["purple"], COLORS_MPL["green"]]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    bars = ax.bar(models, probs, color=colors, alpha=0.85, edgecolor="white", width=0.6)
    ax.set_ylabel("Posterior Probability (%)")
    ax.set_title("Model Preference Within Simulated GNN + SBI Framework")
    ax.set_ylim(0, 115)
    for bar, prob in zip(bars, probs):
        ypos = bar.get_height() + 1.5
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{prob:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(50, color=COLORS_MPL["gray"], ls=":", lw=0.8, alpha=0.4)
    return _save(fig, "model_posterior_probs.png")


# ---------------------------------------------------------------------------
# FIGURES FOR THE TIMELINE FORWARD MODEL + SIM-TO-REAL ROBUSTNESS (2026-05-30)
# ---------------------------------------------------------------------------
def fig_timeline_pipeline():
    """Timeline forward-model workflow: detect -> date -> simulate -> evolve -> compare."""
    fig, ax = plt.subplots(figsize=(6.7, 2.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 2); ax.axis("off")
    steps = [
        (0.05, "Detect impact\n(GNN p, gaps)", "#3498db"),
        (2.45, "Estimate time\nt_since (<= age)", "#16a085"),
        (4.85, "Simulate impacts\non past stream", "#9b59b6"),
        (7.25, "Evolve to today\n(orbit integ.)", "#e67e22"),
        (9.65, "Compare to real\n+ significance", "#e74c3c"),
    ]
    w, h = 2.2, 1.0
    for x, text, col in steps:
        rect = mpatches.FancyBboxPatch((x, 0.5), w, h, boxstyle="round,pad=0.08",
                                       facecolor=col, edgecolor="white", alpha=0.88)
        ax.add_patch(rect)
        ax.text(x + w / 2, 1.0, text, ha="center", va="center",
                fontsize=7.6, fontweight="bold", color="white", linespacing=1.3)
    for x1 in [2.25, 4.65, 7.05, 9.45]:
        ax.annotate("", xy=(x1 + 0.2, 1.0), xytext=(x1, 1.0),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
    ax.set_title("Timeline Forward Model: Rewind, Re-impact, Re-evolve, Compare", fontsize=10.5, pad=6)
    return _save(fig, "timeline_pipeline.png")


def fig_ood_fix():
    """Headline sim-to-real result: error-DR retrain brings real GD-1 in-distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0))

    # Panel 1: e_vrad normalizer std (clamped vs error-DR) — the root cause.
    ax1.bar([0, 1], [0.0100, 1.9516], color=[COLORS_MPL["red"], COLORS_MPL["green"]],
            alpha=0.85, edgecolor="white", width=0.6)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["baseline\n(clamped)", "error-DR"], fontsize=8)
    ax1.set_ylabel(r"$e_{\rm vrad}$ feature std (norm.)")
    ax1.set_title("Root cause: near-constant\nerror feature was std-clamped", fontsize=8.5)
    ax1.text(0, 0.12, "0.01", ha="center", fontsize=8, fontweight="bold")
    ax1.text(1, 1.80, "1.95", ha="center", fontsize=8, fontweight="bold")

    # Panel 2: real GD-1 OOD (log) + p_impact, old vs error-DR.
    x = np.arange(2)
    ax2.bar(x - 0.2, [391.3, 13.4], 0.4, color=COLORS_MPL["red"], alpha=0.8,
            edgecolor="white", label=r"OOD max $\sigma$")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"OOD max $\sigma$ (log)", color=COLORS_MPL["red"])
    ax2.set_xticks(x); ax2.set_xticklabels(["old\ndetector", "error-DR\ndetector"], fontsize=8)
    ax2.tick_params(axis="y", labelcolor=COLORS_MPL["red"])
    ax2.axhline(5.0, color=COLORS_MPL["gray"], ls=":", lw=1)
    ax2.text(1.4, 5.6, r"$\pm5\sigma$ clip", fontsize=6.5, color=COLORS_MPL["gray"])
    axb = ax2.twinx()
    axb.plot(x, [0.98, 0.16], "o-", color=COLORS_MPL["blue"], lw=2, ms=7, label="p_impact")
    axb.set_ylabel("p_impact (real GD-1)", color=COLORS_MPL["blue"])
    axb.tick_params(axis="y", labelcolor=COLORS_MPL["blue"]); axb.set_ylim(0, 1.05)
    ax2.set_title("Real GD-1: 391$\\sigma$ artifact -> 13$\\sigma$,\np 0.98 -> 0.16", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "ood_fix.png")


def fig_erkal_kick():
    """Erkal+2015 bounded Plummer impulse vs the old capped point-mass heuristic."""
    G = 4.3009e-6; M = 1e8; w = 200.0; rs = 0.4
    d = np.linspace(0.01, 3.0, 400)
    plummer = 2 * G * M / w * d / (d**2 + rs**2)
    point = np.minimum(2 * G * M / (d * w), 50.0)   # old: 2GM/(dw), capped at 50
    fig, ax = plt.subplots(figsize=(6, 3.3))
    ax.plot(d, plummer, color=COLORS_MPL["green"], lw=2.2,
            label=r"Erkal+2015 Plummer: $\frac{2GM}{w}\frac{d}{d^2+r_s^2}$ (bounded)")
    ax.plot(d, point, color=COLORS_MPL["red"], lw=1.8, ls="--",
            label=r"Old: $2GM/(dw)$, capped at 50 km/s")
    ax.axvline(rs, color=COLORS_MPL["gray"], ls=":", lw=1)
    ax.text(rs + 0.03, ax.get_ylim()[1] * 0.5, r"$d=r_s$ (peak)", fontsize=8, color=COLORS_MPL["gray"])
    ax.set_xlabel(r"perpendicular distance to subhalo path $d$ [kpc]")
    ax.set_ylabel(r"$|\Delta v|$ [km/s]")
    ax.set_title(r"Velocity Kick: Bounded Erkal+2015 vs. Capped Heuristic ($M=10^8\,M_\odot$)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 12)
    return _save(fig, "erkal_kick.png")


def fig_significance():
    """Null distribution discriminating injected impact (z=3.3) from no impact (z=0.6)."""
    rng = np.random.default_rng(7)
    null = rng.normal(0.483, 0.046, 4000)   # measured null (injected-impact run)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(null, bins=50, density=True, color=COLORS_MPL["gray"], alpha=0.55,
            edgecolor="none", label="No-impact null distribution")
    ax.axvline(0.332, color=COLORS_MPL["green"], lw=2.2,
               label=r"Injected impact (best): $z=3.3\sigma$")
    ax.axvline(0.389, color=COLORS_MPL["orange"], lw=2.0, ls="--",
               label=r"No-impact stream (best): $z=0.6\sigma$")
    ax.set_xlabel("Combined score (lower = better fit)")
    ax.set_ylabel("Null density")
    ax.set_title("Statistical Significance vs. a No-Impact Null Distribution")
    ax.legend(fontsize=8, loc="upper right")
    return _save(fig, "significance.png")


def fig_multiepoch_rv():
    """Real radial-velocity coverage from multi-epoch / multi-survey fusion."""
    streams = ["ATLAS", "Jhelum", "Orphan", "GD-1", "Pal 5"]
    s5 = [296, 257, 0, 0, 0]
    gaia = [15, 8, 12, 1, 2]   # approximate Gaia DR3 RVS matches (bright members)
    x = np.arange(len(streams)); wbar = 0.38
    fig, ax = plt.subplots(figsize=(6, 3.0))
    ax.bar(x - wbar/2, s5, wbar, label="S5 survey", color=COLORS_MPL["purple"], alpha=0.85, edgecolor="white")
    ax.bar(x + wbar/2, gaia, wbar, label="Gaia DR3 RVS", color=COLORS_MPL["blue"], alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(streams, fontsize=9)
    ax.set_ylabel("Members with real RV")
    ax.set_title("Real Radial Velocities Fused In (the missing 6th dimension)")
    ax.legend(fontsize=8)
    ax.text(2.0, 30, "S5 footprint is southern;\nGD-1 needs APOGEE/DESI", fontsize=7,
            color=COLORS_MPL["gray"], style="italic", ha="center")
    return _save(fig, "multiepoch_rv.png")


def fig_mc_time():
    """Monte-Carlo impact-time posterior width vs measurement-error scale."""
    scale = np.array([0.0, 0.5, 1.0, 2.0])
    std = np.array([0.000, 0.05, 0.109, 0.354])   # measured (injected 1.5 Gyr truth)
    fig, ax = plt.subplots(figsize=(6, 3.1))
    ax.plot(scale, std, "o-", color=COLORS_MPL["purple"], lw=2.2, ms=7)
    ax.fill_between(scale, 0, std, color=COLORS_MPL["purple"], alpha=0.12)
    ax.set_xlabel("Measurement-error scale (1.0 = current; <1 = future / multi-epoch)")
    ax.set_ylabel(r"Impact-time posterior std [Gyr]")
    ax.set_title("Dated-Impact Precision Scales with Measurement Precision")
    ax.annotate("zero error ->\ndeterministic fit", xy=(0.0, 0.0), xytext=(0.4, 0.18),
                fontsize=8, color=COLORS_MPL["gray"],
                arrowprops=dict(arrowstyle="->", color=COLORS_MPL["gray"], lw=0.8))
    ax.set_ylim(bottom=-0.01)
    return _save(fig, "mc_time.png")


# ---------------------------------------------------------------------------
# REPORTLAB STYLES
# ---------------------------------------------------------------------------
def build_styles():
    s = getSampleStyleSheet()

    s.add(ParagraphStyle("PaperTitle", parent=s["Title"],
        fontSize=22, leading=28, textColor=HexColor("#1a1a2e"),
        spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold"))
    s.add(ParagraphStyle("AuthorLine", parent=s["Normal"],
        fontSize=12, leading=16, textColor=HexColor("#34495e"),
        spaceAfter=2, alignment=TA_CENTER, fontName="Helvetica"))
    s.add(ParagraphStyle("DateLine", parent=s["Normal"],
        fontSize=10, leading=14, textColor=HexColor("#7f8c8d"),
        spaceAfter=16, alignment=TA_CENTER, fontName="Helvetica-Oblique"))
    s.add(ParagraphStyle("Abstract", parent=s["Normal"],
        fontSize=10, leading=14, textColor=HexColor("#2c3e50"),
        spaceAfter=10, alignment=TA_LEFT, fontName="Helvetica",
        leftIndent=36, rightIndent=36))
    s.add(ParagraphStyle("AbstractHead", parent=s["Normal"],
        fontSize=10, leading=14, textColor=HexColor("#2c3e50"),
        spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold"))
    s.add(ParagraphStyle("SectionHead", parent=s["Heading1"],
        fontSize=14, leading=18, textColor=HexColor("#1a1a2e"),
        spaceBefore=18, spaceAfter=6, fontName="Helvetica-Bold"))
    s.add(ParagraphStyle("SubHead", parent=s["Heading2"],
        fontSize=12, leading=15, textColor=HexColor("#2c3e50"),
        spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold"))
    s.add(ParagraphStyle("SubHead3", parent=s["Heading3"],
        fontSize=10.5, leading=13, textColor=HexColor("#34495e"),
        spaceBefore=6, spaceAfter=3, fontName="Helvetica-Bold"))
    s.add(ParagraphStyle("Body", parent=s["Normal"],
        fontSize=10, leading=13.5, textColor=HexColor("#2c3e50"),
        spaceAfter=6, alignment=TA_LEFT, fontName="Helvetica"))
    s.add(ParagraphStyle("BodySmall", parent=s["Normal"],
        fontSize=9, leading=12, textColor=HexColor("#2c3e50"),
        spaceAfter=4, alignment=TA_LEFT, fontName="Helvetica"))
    s.add(ParagraphStyle("FigCaption", parent=s["Normal"],
        fontSize=9, leading=12, textColor=HexColor("#555555"),
        spaceAfter=10, alignment=TA_CENTER, fontName="Helvetica-Oblique"))
    s.add(ParagraphStyle("Equation", parent=s["Normal"],
        fontSize=10, leading=14, textColor=HexColor("#2c3e50"),
        spaceBefore=6, spaceAfter=6, alignment=TA_CENTER, fontName="Courier"))
    s.add(ParagraphStyle("BulletItem", parent=s["Normal"],
        fontSize=10, leading=13, textColor=HexColor("#2c3e50"),
        spaceAfter=2, leftIndent=24, bulletIndent=12, fontName="Helvetica"))
    s.add(ParagraphStyle("Footer", parent=s["Normal"],
        fontSize=8, leading=10, textColor=HexColor("#95a5a6"),
        alignment=TA_CENTER, fontName="Helvetica"))
    return s


class ThinRule(Flowable):
    """Horizontal rule."""
    def __init__(self, width=468, thickness=0.75, color=RULE_COLOR):
        Flowable.__init__(self)
        self.width = width
        self.thickness = thickness
        self.color = color
    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


def make_table(headers, rows, col_widths=None, font_size=9, leading=None):
    table_leading = leading or (font_size + 1.5)
    header_style = ParagraphStyle(
        "LocalTableHeader",
        fontName="Helvetica-Bold",
        fontSize=font_size,
        leading=table_leading,
        textColor=white,
    )
    body_style = ParagraphStyle(
        "LocalTableBody",
        fontName="Helvetica",
        fontSize=font_size,
        leading=table_leading,
        textColor=HexColor("#2c3e50"),
    )

    def cell(value, style):
        text = "" if value is None else str(value)
        text = (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("\n", "<br/>"))
        return Paragraph(text, style)

    data = [[cell(h, header_style) for h in headers]]
    data.extend([[cell(v, body_style) for v in row] for row in rows])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), font_size),
        ("FONTSIZE", (0, 1), (-1, -1), font_size),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 1), (-1, -1), HexColor("#2c3e50")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, ROW_ALT]),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#d5d8dc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    t = Table(data, colWidths=col_widths, repeatRows=1, splitByRow=1)
    t.setStyle(TableStyle(style_cmds))
    return t


def add_figure(story, img_path, caption, styles, width=5.8*inch):
    """Insert a matplotlib figure with caption."""
    img = Image(img_path, width=width, height=width * 0.55)
    story.append(Spacer(1, 6))
    story.append(img)
    story.append(Paragraph(caption, styles["FigCaption"]))


# ---------------------------------------------------------------------------
# BUILD PDF
# ---------------------------------------------------------------------------
def add_page_number(canvas, doc):
    """Footer with page numbers."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#95a5a6"))
    canvas.drawCentredString(letter[0] / 2, 0.5 * inch,
                              f"Watson (2026)   -   Stellar Stream DM   -   Page {doc.page}")
    canvas.restoreState()


def build_pdf():
    styles = build_styles()
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=letter,
        leftMargin=0.85*inch, rightMargin=0.85*inch,
        topMargin=0.75*inch, bottomMargin=0.85*inch,
        title="Stellar Stream DM  -  Full Technical Report",
        author="Daniel Watson"
    )
    story = []
    W = doc.width

    # Generate all figures first
    print("Generating figures...")
    fig_paths = {
        "mass_fn": fig_mass_function(),
        "transfer": fig_transfer_function(),
        "rate": fig_rate_vs_mhm(),
        "sensitivity": fig_stream_sensitivity(),
        "arch": fig_architecture(),
        "pipeline": fig_pipeline(),
        "hier": fig_hierarchical_posterior(),
        "mock": fig_mock_recovery(),
        "coverage": fig_coverage(),
        "sbi_rounds": fig_sbi_rounds(),
        "streams": fig_stream_properties(),
        "discrimination": fig_model_discrimination(),
        "log_ev_heatmap": fig_log_evidences_heatmap(),
        "model_probs": fig_model_posterior_probs(),
        "timeline": fig_timeline_pipeline(),
        "ood_fix": fig_ood_fix(),
        "erkal": fig_erkal_kick(),
        "significance": fig_significance(),
        "multiepoch_rv": fig_multiepoch_rv(),
        "mc_time": fig_mc_time(),
    }
    print("Figures generated.")

    # -----------------------------------------------------------------------
    # TITLE PAGE
    # -----------------------------------------------------------------------
    story.append(Spacer(1, 1.2*inch))
    story.append(Paragraph(
        "A Framework for Constraining Dark Matter Substructure<br/>"
        "via Graph Neural Networks and Simulation-Based Inference<br/>"
        "Applied to Gaia DR3 Stellar Streams",
        styles["PaperTitle"]))
    story.append(Spacer(1, 14))
    story.append(ThinRule(W, 1.0, ACCENT_LIGHT))
    story.append(Spacer(1, 14))
    story.append(Paragraph("Daniel Watson", styles["AuthorLine"]))
    story.append(Paragraph("May 2026", styles["DateLine"]))
    story.append(Spacer(1, 20))

    # Abstract
    story.append(Paragraph("Abstract", styles["AbstractHead"]))
    story.append(ThinRule(W * 0.3, 0.5, RULE_COLOR))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "We present a computational framework for constraining dark matter substructure using stellar "
        "streams observed by Gaia DR3. The publication-ready result in the current implementation is "
        "a hierarchical count/rate analysis of seven Milky Way streams; the graph neural network "
        "(GNN) and simulation-based inference (SBI) components are reported as a methods pipeline "
        "under active validation. A 6-layer GINEConv encoder maps stream phase-space graphs into "
        "128-dimensional embeddings for a future calibrated SNPE-C posterior engine. "
        "First, a hierarchical Bayesian rate model applied to literature-reported gap counts yields "
        "log<sub>10</sub>(M<sub>hm</sub>) = 4.56 (95% upper limit: 5.14), with the total predicted "
        "CDM impact count (17.2) matching observations (17) to within 1%. An extended model with free "
        "mass function slope finds alpha = -1.89 +/- 0.20, consistent with the CDM prediction of "
        "-1.9 (Springel et al. 2008). "
        "Second, the current V2 GNN run reframes the learning task as impact-strong binary "
        "classification plus regression. On 15,000 held-out simulations it reaches ROC AUC 0.668 "
        "and balanced accuracy 61.4%, so it is not yet suitable for discovery claims. First-pass "
        "SBI calibration also undercovers key parameters. We therefore label every result as "
        "Observed, Synthetic Validation, Forecast, or Simulated-Framework Diagnostic, and do not "
        "claim evidence for WDM. Injection tests establish a detection threshold near "
        "10<super>7</super> M<sub>sun</sub> for count-based methods. Automated tests validate "
        "implementation behavior; they do not validate astrophysical discovery claims. "
        "Third (Section 13), we add a timeline forward model that rewinds a stream, injects "
        "candidate subhalo impacts using the closed-form Erkal &amp; Belokurov (2015) Plummer "
        "impulse, integrates them forward to the present, and scores each against the data with a "
        "calibrated significance versus a no-impact null. In the course of this we diagnose and fix "
        "a sim-to-real failure that had pinned the detector at p_impact = 1.0 on real data: error "
        "domain-randomised retraining reduces the real-GD-1 input out-of-distribution level from "
        "391 sigma to 13 sigma and turns a saturated probability into a meaningful one, while real "
        "radial velocities from the S5 survey and Gaia RVS are fused in to constrain the rewind.",
        styles["Abstract"]))
    story.append(Spacer(1, 10))
    story.append(ThinRule(W, 0.5, RULE_COLOR))

    # Keywords
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Keywords:</b> dark matter substructure, stellar streams, graph neural networks, "
        "simulation-based inference, half-mode mass, Gaia DR3, hierarchical Bayesian inference",
        styles["BodySmall"]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # TABLE OF CONTENTS
    # -----------------------------------------------------------------------
    story.append(Paragraph("Contents", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 8))
    toc = [
        "1. Introduction",
        "2. Scientific Background and Model Context",
        "3. Observational Data and Stream Sample",
        "4. Methods: GNN, SBI, and Hierarchical Counts",
        "   4.1  Graph Construction",
        "   4.2  GNN Architecture",
        "   4.3  Multi-Task Training",
        "   4.4  Simulation-Based Inference",
        "   4.5  Hierarchical Bayesian Model",
        "5. Implementation Features",
        "6. Synthetic Validation and Mock Challenges",
        "7. Dark Matter Model Definitions",
        "8. Target Stream Sample",
        "9. Software Validation and Reproducibility",
        "10. Reproducibility and Configuration Snapshot",
        "11. Observed Application to Gaia DR3",
        "   11.1  Hierarchical Inference on Literature Gap Counts",
        "   11.2  Extended Model: Free Mass Function Slope",
        "   11.3  Observed + Synthetic Injection Tests",
        "   11.4  Equivariant GNN Architecture",
        "   11.5  Forecast: Radial Velocity Extension",
        "   11.6  Synthetic/Diagnostic GNN + SBI Results",
        "   11.7  Discussion: Reconciling the Two Analyses",
        "12. Discussion, Limitations, and Outlook",
        "13. Timeline Forward Model and Sim-to-Real Robustness",
        "   13.1  Detection to Timeline Handoff",
        "   13.2  Erkal & Belokurov (2015) Velocity Kick",
        "   13.3  Diagnosing and Fixing Detector Over-Confidence",
        "   13.4  GD-1 Gap Localisation and the Frame Transform",
        "   13.5  Multi-Epoch / Multi-Survey Data Fusion",
        "   13.6  Validation: Injection-Recovery and Significance",
        "   13.7  Uncertainty-Aware Impact Time",
    ]
    for item in toc:
        indent = 24 if item.startswith("   ") else 0
        story.append(Paragraph(item.strip(),
            ParagraphStyle("TOCItem", parent=styles["Body"],
                           leftIndent=indent, spaceAfter=3)))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 1. INTRODUCTION
    # -----------------------------------------------------------------------
    story.append(Paragraph("1. Introduction", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "Approximately 85% of all matter in the universe is dark matter  -  a substance that does not "
        "emit or absorb light. Its existence is firmly established through gravitational effects: "
        "galaxy rotation curves, gravitational lensing, and the cosmic microwave background all require "
        "dark matter to explain observations. The central open question in particle physics and cosmology "
        "is: <i>what are the fundamental properties of dark matter</i>",
        styles["Body"]))
    story.append(Paragraph(
        "Different theoretical models predict dark matter with different properties. Cold Dark Matter "
        "(CDM) predicts particles that move slowly and clump into structures at every scale  -  from "
        "galaxy clusters down to Earth-mass objects. Alternative models  -  Warm Dark Matter (WDM), "
        "Fuzzy Dark Matter (FDM), and Self-Interacting Dark Matter (SIDM)  -  each predict fewer "
        "small-scale structures, but for different physical reasons. The number of small dark matter "
        "clumps (subhalos) orbiting our galaxy is a direct observable that distinguishes these theories.",
        styles["Body"]))
    story.append(Paragraph(
        "When a globular cluster orbits too close to the galactic centre, tidal forces tear it apart, "
        "stretching its stars into a thin ribbon called a <i>stellar stream</i>. These streams act as "
        "gravitational antennae for dark matter substructure: when an invisible subhalo passes near a "
        "stream, it creates a density gap and velocity perturbation. By counting gaps and characterising "
        "their morphology across multiple streams, we can constrain the subhalo mass function.",
        styles["Body"]))
    story.append(Paragraph(
        "This work describes a graph neural network (GNN) framework that operates directly on the "
        "6-dimensional phase-space data of every star in a stream, preserving spatial and kinematic "
        "correlations that traditional 1D methods (Bovy et al. 2017; Banik et al. 2021) discard. The "
        "GNN embedding is combined with Simulation-Based Inference (SBI) to produce full posterior "
        "distributions over the half-mode mass M<sub>hm</sub>  -  a model-independent measure of the "
        "smallest dark matter structures that can exist. We apply both a count-based hierarchical "
        "rate model and the full GNN + SBI pipeline to seven Gaia DR3 stellar streams, and discuss "
        "the implications of their contrasting findings.",
        styles["Body"]))

    # Pipeline figure
    add_figure(story, fig_paths["pipeline"],
        "<b>Figure 1.</b> End-to-end analysis pipeline: Gaia DR3 observations are transformed into "
        "k-nearest-neighbour graphs, encoded by a 6-layer GINEConv GNN into 128-dimensional "
        "embeddings, and processed by SNPE-C with Neural Spline Flows to obtain posterior "
        "distributions over dark matter parameters.",
        styles, width=5.5*inch)
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 2. SCIENTIFIC BACKGROUND
    # -----------------------------------------------------------------------
    story.append(Paragraph("2. Scientific Background and Model Context", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))

    story.append(Paragraph("2.1  The Subhalo Mass Function", styles["SubHead"]))
    story.append(Paragraph(
        "CDM N-body simulations (Springel et al. 2008, Aquarius) predict a subhalo mass function "
        "following dN/dM ~ M<super>-1.9</super>, meaning far more low-mass subhalos than high-mass "
        "ones. For every subhalo at 10<super>9</super> M<sub>sun</sub>, CDM predicts roughly 80 at "
        "10<super>8</super> M<sub>sun</sub> and ~6,000 at 10<super>7</super> M<sub>sun</sub>. "
        "Alternative dark matter models suppress this abundance below a characteristic half-mode mass.",
        styles["Body"]))

    add_figure(story, fig_paths["mass_fn"],
        "<b>Figure 2.</b> Subhalo mass functions for CDM and suppressed models. CDM follows an "
        "unbroken power law (blue). WDM and FDM models suppress abundances below their respective "
        "half-mode masses, producing fewer low-mass subhalos. All curves use dN/dM ~ M<super>-1.9</super> "
        "(Springel et al. 2008) as the base CDM mass function.",
        styles)

    story.append(Paragraph("2.2  The Transfer Function", styles["SubHead"]))
    story.append(Paragraph(
        "The suppression is modelled by the transfer function T(M, M<sub>hm</sub>) = "
        "[1 + (M<sub>hm</sub>/M)<super>2</super>]<super>-1</super>, following the WDM suppression "
        "form from Schneider et al. (2012). When M<sub>hm</sub> is well below the minimum subhalo "
        "mass we consider (10<super>5</super> M<sub>sun</sub>), there is no suppression (CDM regime). "
        "When M<sub>hm</sub> is large (10<super>8</super>-10<super>9</super> M<sub>sun</sub>), "
        "most subhalos are suppressed.",
        styles["Body"]))

    add_figure(story, fig_paths["transfer"],
        "<b>Figure 3.</b> The suppression transfer function T(M, M_hm) for several half-mode mass "
        "values. The function transitions from 0 (full suppression) to 1 (no suppression) around "
        "M = M_hm. The dotted line marks T = 0.5, the defining level of the half-mode mass.",
        styles)

    story.append(Paragraph("2.3  Model-Independent Constraints via M<sub>hm</sub>", styles["SubHead"]))
    story.append(Paragraph(
        "Rather than attempting to discriminate between specific particle models  -  which is degenerate "
        "at Gaia DR3 precision (Banik et al. 2021; Dalal et al. 2022)  -  this project infers "
        "M<sub>hm</sub> directly as a model-independent constraint. A single M<sub>hm</sub> upper "
        "limit maps to particle mass limits for any model: for WDM, "
        "M<sub>hm</sub> = 1.7 x 10<super>10</super> (m<sub>WDM</sub>/keV)<super>-3.33</super> "
        "M<sub>sun</sub> (Lovell et al. 2014). For FDM, the mapping follows "
        "M<sub>Jeans</sub> = 1.5 x 10<super>8</super> (m<sub>axion</sub>/10<super>-22</super> eV)<super>-1.5</super> "
        "M<sub>sun</sub> (Hui et al. 2017).",
        styles["Body"]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 3. DATA AND OBSERVATIONS
    # -----------------------------------------------------------------------
    story.append(Paragraph("3. Observational Data and Stream Sample", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "Observational data are drawn from the Gaia DR3 catalogue (Gaia Collaboration 2023), which "
        "provides positions, parallaxes, proper motions, and (where available) radial velocities for "
        "approximately 1.8 billion stars. Distances are supplemented with photogeometric estimates "
        "from Bailer-Jones et al. (2021).",
        styles["Body"]))

    story.append(Paragraph(
        "The current Plan 2 training set comprises 100,000 simulated streams (25,000 per dark "
        "matter model) generated with galpy (Bovy 2015) using a realistic Milky Way potential. "
        "The V2 fast profile run uses up to 1,200 stars per graph with domain randomisation over "
        "foreground contamination, noise scale, stream membership, baryonic perturbations, and "
        "potential variations. Baryonic perturbations are applied across all dark-matter families "
        "at the configured 80% rate so the model cannot use baryons as a CDM-only shortcut.",
        styles["Body"]))

    story.append(Paragraph("<b>Node Features (18 per star)</b>", styles["SubHead3"]))
    node_tbl = [
        ["Index", "Feature", "Description", "Source"],
        ["0", "phi1", "Stream longitude (along stream)", "Gaia astrometry"],
        ["1", "phi2", "Stream latitude (across stream)", "Gaia astrometry"],
        ["2", "dist", "Heliocentric distance", "Photogeometric"],
        ["3", "pm1", "Proper motion along stream", "Gaia astrometry"],
        ["4", "pm2", "Proper motion across stream", "Gaia astrometry"],
        ["5", "vrad", "Radial velocity", "Gaia RVS"],
        ["6-9", "e_*", "Measurement uncertainties (4)", "Gaia error model"],
        ["10", "membership_prob", "Stream membership probability", "Mixture model"],
        ["11-17", "stream_*", "One-hot stream identity (7)", "Catalogue"],
    ]
    story.append(make_table(node_tbl[0], node_tbl[1:],
                            col_widths=[0.6*inch, 1.2*inch, 2.5*inch, 1.6*inch]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Edge Features (5 base, 7 with orbital extension)</b>", styles["SubHead3"]))
    edge_tbl = [
        ["Feature", "Description"],
        ["delta_phi1", "Difference in stream longitude between connected stars"],
        ["delta_phi2", "Difference in stream latitude"],
        ["delta_pm1", "Difference in proper motion along stream"],
        ["delta_pm2", "Difference in proper motion across stream"],
        ["dist_4d", "Euclidean distance in 4D phase space"],
        ["delta_R_cyl (optional)", "Galactocentric cylindrical radius difference"],
        ["delta_z_cyl (optional)", "Height above galactic disk difference"],
    ]
    story.append(make_table(edge_tbl[0], edge_tbl[1:],
                            col_widths=[1.8*inch, 4.1*inch]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 4. METHODS
    # -----------------------------------------------------------------------
    story.append(Paragraph("4. Methods: GNN, SBI, and Hierarchical Counts", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))

    # 4.1 Graph Construction
    story.append(Paragraph("4.1  Graph Construction", styles["SubHead"]))
    story.append(Paragraph(
        "Each stellar stream is represented as a k-nearest-neighbour graph with k = 8 in the current "
        "V2 fast profile runs. Stars are "
        "connected based on proximity in the 4-dimensional space of normalised stream coordinates and "
        "proper motions (phi1, phi2, pm1, pm2). Streams with more than 1,200 stars are randomly "
        "downsampled. Data augmentation includes phi1 offset (U[-5, 5] deg), phi2 noise "
        "(N[0, 0.05] deg), distance scatter (log-normal, sigma = 0.02), proper motion noise "
        "(N[0, 0.02] mas/yr). Mirror reflection is disabled for stream-identity-aware V2 runs.",
        styles["Body"]))

    # 4.2 GNN Architecture
    story.append(Paragraph("4.2  GNN Architecture", styles["SubHead"]))
    story.append(Paragraph(
        "The core encoder is a Graph Isomorphism Network with Edge features (GINEConv; Xu et al. 2019). "
        "GINEConv is provably maximally expressive among message-passing GNNs, meaning it can "
        "distinguish between any two non-isomorphic graphs. The GINE variant incorporates edge features "
        "during message aggregation  -  critical because relative velocity differences between "
        "neighbouring stars are the primary signal of subhalo perturbation.",
        styles["Body"]))

    add_figure(story, fig_paths["arch"],
        "<b>Figure 4.</b> StreamGNNEncoder architecture. Input node features (18-d) and edge features "
        "(5-d base; 7-d with orbital extension) are projected into a 256-d hidden space, processed "
        "by six GINEConv layers with batch normalisation, residual connections, and 10% dropout, "
        "then globally pooled and compressed to a 128-d embedding. The V2 heads are a configurable "
        "binary classifier (model family, any impact, or strong impact) and regression heads for "
        "half-mode mass and impact counts.",
        styles, width=5.5*inch)

    arch_tbl = [
        ["Component", "Specification"],
        ["Convolution type", "GINEConv (Xu et al. 2019)"],
        ["Layers", "6"],
        ["Hidden dimension", "256"],
        ["Embedding dimension", "128"],
        ["Dropout", "0.1"],
        ["Pooling", "Mean + Max concatenation (512-d)"],
        ["Residual connections", "From layer 2 onward"],
        ["Batch normalisation", "After each GINEConv layer"],
    ]
    story.append(make_table(arch_tbl[0], arch_tbl[1:],
                            col_widths=[2.2*inch, 3.7*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Table 1.</b> GNN encoder architecture parameters (from config/training.yaml).",
        styles["FigCaption"]))

    # 4.3 Multi-Task Training
    story.append(Paragraph("4.3  Multi-Task Training", styles["SubHead"]))
    story.append(Paragraph(
        "The V2 embedding feeds a binary head and regression heads. The binary target is configurable: "
        "suppressed versus unsuppressed model family, any injected impact, or the current impact-strong "
        "target based on a morphology-strength proxy. Regression targets include n_impacts and "
        "log<sub>10</sub>(M<sub>hm</sub>) for WDM/FDM simulations only; CDM and SIDM are treated as "
        "unsuppressed reference families, not high-M<sub>hm</sub> physical targets. Training uses "
        "AdamW (lr = 10<super>-3</super>, weight decay = 10<super>-4</super>), batch size 128 in the "
        "fast V2 run, mixed precision where supported, and a 70/15/15 split. SBI training now uses "
        "train+val only, leaving the test split for calibration.",
        styles["Body"]))
    story.append(PageBreak())

    # 4.4 SBI
    story.append(Paragraph("4.4  Simulation-Based Inference", styles["SubHead"]))
    story.append(Paragraph(
        "Traditional likelihood-based inference is intractable for stellar stream perturbation models "
        "because evaluating the likelihood requires a full N-body simulation for each parameter set. "
        "We use Sequential Neural Posterior Estimation, variant C (SNPE-C / APT), which trains a "
        "conditional density estimator to approximate the posterior P(parameters | data). The density "
        "estimator is a Neural Spline Flow (NSF) with 8 rational-quadratic spline transforms, 8 bins "
        "per spline, and hidden layers [256, 256].",
        styles["Body"]))

    add_figure(story, fig_paths["sbi_rounds"],
        "<b>Figure 5.</b> Sequential training schedule. Round 1 uses 5,000 simulations drawn from "
        "the prior; subsequent rounds draw 2,000 simulations each from the previous round's posterior, "
        "concentrating the simulation budget near high-probability regions. Green line shows "
        "illustrative posterior quality improvement across rounds.",
        styles, width=4.8*inch)

    sbi_tbl = [
        ["Parameter", "Value"],
        ["Algorithm", "SNPE-C (APT)"],
        ["Density estimator", "Neural Spline Flow"],
        ["NSF transforms", "8"],
        ["NSF bins", "8"],
        ["NSF hidden dims", "[256, 256]"],
        ["Rounds", "5"],
        ["Sims (round 1)", "5,000"],
        ["Sims (rounds 2-5)", "2,000 each"],
        ["Posterior samples", "10,000"],
        ["Credible interval", "90%"],
    ]
    story.append(make_table(sbi_tbl[0], sbi_tbl[1:],
                            col_widths=[2.2*inch, 3.7*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Table 2.</b> SBI configuration (from config/training.yaml).",
        styles["FigCaption"]))

    # 4.5 Hierarchical
    story.append(Paragraph("4.5  Hierarchical Bayesian Multi-Stream Model", styles["SubHead"]))
    story.append(Paragraph(
        "A hierarchical Bayesian model combines evidence from all seven streams into a single "
        "M<sub>hm</sub> constraint, replacing the statistically incorrect product-of-posteriors "
        "approach used in prior work. The model assumes:",
        styles["Body"]))
    story.append(Paragraph(
        "M<sub>hm</sub> ~ Uniform(10<super>4</super>, 10<super>10</super> M<sub>sun</sub>)",
        styles["Equation"]))
    story.append(Paragraph(
        "n<sub>s</sub> ~ Poisson(lambda(M<sub>hm</sub>, L<sub>s</sub>, T<sub>s</sub>, D<sub>s</sub>))  for each stream s",
        styles["Equation"]))
    story.append(Paragraph(
        "The expected rate lambda integrates the suppressed subhalo mass function over "
        "[10<super>5</super>, 10<super>9</super>] M<sub>sun</sub>, weighted by stream geometric "
        "cross-section and age. The normalisation constant C is calibrated so that CDM produces "
        "approximately 2.1 detectable impacts for a GD-1-like stream (60 deg, 5 Gyr, 12 kpc), "
        "consistent with Bonaca et al. (2019) who identified 2 gaps and 1 spur in GD-1. "
        "NumPyro NUTS is the primary MCMC backend, with emcee as fallback.",
        styles["Body"]))

    add_figure(story, fig_paths["rate"],
        "<b>Figure 6.</b> Expected subhalo impact rate as a function of half-mode mass for three "
        "representative streams. At low M_hm (CDM regime), rates are set by stream geometry and age. "
        "As M_hm increases, low-mass subhalos are suppressed and the rate drops. Orphan-Chenab has "
        "the highest rate due to its 120-degree length.",
        styles)

    add_figure(story, fig_paths["hier"],
        "<b>Figure 7.</b> Hierarchical posterior distributions from validation run (emcee MCMC, "
        "32 walkers, 8,000 steps). Left: CDM scenario  -  posterior peaks at ~4.8, correctly "
        "indicating no suppression above the minimum subhalo mass considered. Right: WDM scenario "
        " -  broad posterior with truth (7.5) near the lower 68% HDI bound.",
        styles, width=5.5*inch)
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 5. NOVEL FEATURES
    # -----------------------------------------------------------------------
    story.append(Paragraph("5. Implementation Features", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The framework integrates seven machine learning and statistical components, each addressing "
        "a specific requirement of the stellar stream dark matter inference problem. The executable "
        "software tests verify implementation behavior, while astrophysical validation requires the "
        "mock, injection, and calibration tests discussed below.",
        styles["Body"]))

    features = [
        ("F1: Attention-Based Interpretability",
         "A learnable query vector attends to final hidden states via per-graph softmax, producing "
         "per-star importance weights summing to 1. This reveals which stars the GNN considers most "
         "informative  -  typically stars at gap edges, spur features, and velocity outliers. When "
         "enabled, pooling becomes mean + max + attention (768-d total). This interpretability is "
         "unique to graph architectures; power-spectrum methods cannot produce per-star importance maps."),
        ("F2: Multi-Scale Graph Construction",
         "The stream is divided into 20 equal-width bins along phi1. Each bin becomes a segment node "
         "with 8 features (mean phi1, mean phi2, std phi2, mean pm1, mean pm2, std pm1, "
         "log(n_stars), density contrast). A separate 3-layer GINEConv (64 hidden, 64 output) "
         "processes the segment graph. CDM creates many clustered gaps while WDM creates few "
         "large ones  -  segment-level structure captures this distinction."),
        ("F3: MC Dropout Uncertainty",
         "Monte Carlo dropout runs T = 20 stochastic forward passes at inference time, producing "
         "per-dimension mean and standard deviation of the embedding. High std flags "
         "out-of-distribution inputs. No retraining required  -  it activates existing dropout layers."),
        ("F4: Orbit-Phase Edge Features",
         "Galactocentric cylindrical coordinates R<sub>cyl</sub> and z<sub>cyl</sub> are added as "
         "node features, with delta_R_cyl and delta_z_cyl as edge features (5 -> 7 dimensions). Stars "
         "at different orbital phases respond differently to a perturbation. Backward-compatible: "
         "defaults to 5 base features if orbital data is absent."),
        ("F5: Perturbation Age Target",
         "A third regression target  -  log<sub>10</sub>(t<sub>since_last_impact</sub>)  -  is added "
         "alongside M<sub>sub</sub> and n_impacts. Gap width encodes time-since-impact due to "
         "phase-mixing. Controlled by config parameter n_reg_targets (default 2, set to 3 for age)."),
        ("F6: Hierarchical Bayesian Model",
         "Replaces the product-of-posteriors with a proper hierarchical model sharing M<sub>hm</sub> "
         "across streams. Per-stream counts follow Poisson(rate(M_hm, L, T, D)). NumPyro NUTS "
         "primary backend; emcee fallback."),
        ("F7: Mock Data Challenge",
         "Twenty mock streams with known ground truth (5 CDM at log<sub>10</sub> M<sub>hm</sub> = 4.5, "
         "5 WDM, 5 FDM, and 5 SIDM as unsuppressed references) are run through the count-rate "
         "validation path. Coverage is evaluated at 68%, 90%, and 95% CI levels. The full blind "
         "GNN+SBI challenge remains a required next validation step."),
    ]
    for title, desc in features:
        story.append(Paragraph(title, styles["SubHead3"]))
        story.append(Paragraph(desc, styles["Body"]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 6. MOCK DATA CHALLENGE
    # -----------------------------------------------------------------------
    story.append(Paragraph("6. Synthetic Validation and Mock Challenges", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "We validated the statistical inference engine on synthetic data with known ground truth. "
        "This exercises the rate model, Poisson likelihood inversion, hierarchical Bayesian inference "
        "via emcee MCMC, and coverage metrics. It does not use the GNN (which requires a trained "
        "checkpoint)  -  it tests the mathematical machinery downstream of the embedding step.",
        styles["Body"]))

    story.append(Paragraph("6.1  Poisson Inversion on 20 Mock Streams", styles["SubHead"]))
    story.append(Paragraph(
        "Twenty mock streams with known ground truth were processed through Poisson likelihood "
        "inversion on the rate model grid. Each mock uses only its observed impact count n_impacts "
        "to infer M<sub>hm</sub>  -  no morphological or embedding information.",
        styles["Body"]))

    mock_tbl = [
        ["DM Model", "# Mocks", "log10(M_hm / Msun)", "Physical Meaning"],
        ["CDM", "5", "4.5 (no suppression)", "Full subhalo abundance"],
        ["WDM", "5", "6.5, 7.0, 7.5, 8.0, 8.5", "Free-streaming suppression"],
        ["FDM", "5", "6.5, 7.0, 7.5, 8.0, 8.5", "Quantum pressure suppression"],
        ["SIDM", "5", "4.5 (no suppression)", "Same mass function as CDM, cored profiles"],
    ]
    story.append(make_table(mock_tbl[0], mock_tbl[1:],
                            col_widths=[0.9*inch, 0.7*inch, 2.1*inch, 2.2*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Table 3.</b> <b>Synthetic Validation.</b> Mock data challenge parameters. CDM uses log<sub>10</sub> M<sub>hm</sub> = 4.5, "
        "placing the half-mode mass below the minimum subhalo mass considered.",
        styles["FigCaption"]))

    # Results table
    story.append(Spacer(1, 6))
    results_tbl = [
        ["Metric", "Value", "Interpretation"],
        ["Coverage at 68% CI", "25%", "Strongly under-covering"],
        ["Coverage at 90% CI", "25%", "Fails 75% threshold"],
        ["Coverage at 95% CI", "40%", "Fails 75% threshold"],
        ["Mean bias", "-1.87 dex", "Systematic underestimation of M_hm"],
        ["RMSE", "2.21 dex", "Large scatter"],
    ]
    story.append(make_table(results_tbl[0], results_tbl[1:],
                            col_widths=[1.5*inch, 1.0*inch, 3.4*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Table 3b.</b> <b>Synthetic Validation.</b> Mock challenge results from Poisson count-only inference.",
        styles["FigCaption"]))

    story.append(Paragraph(
        "<b>Interpretation:</b> The poor coverage is expected and informative. With only an integer "
        "impact count per stream and default stream parameters, the Poisson inversion cannot "
        "distinguish between \"low M<sub>hm</sub> with unlucky low count\" and \"high M<sub>hm</sub> "
        "with expected low count.\" All non-CDM mocks are systematically pulled toward a floor near "
        "log<sub>10</sub>(M<sub>hm</sub>) ~ 5-6, regardless of their true value. This directly "
        "demonstrates why morphological information  -  gap depth, width, velocity profile  -  captured "
        "by the GNN embedding is essential for constraining M<sub>hm</sub> beyond what raw counts provide.",
        styles["Body"]))

    add_figure(story, fig_paths["mock"],
        "<b>Figure 8.</b> <b>Synthetic Validation.</b> Mock data challenge results: true vs. recovered log<sub>10</sub>(M_hm) "
        "from Poisson count-only inference. All non-CDM points cluster between 4.6 and 5.7 regardless of "
        "true value, showing the resolution floor of count-based inference. CDM mocks (blue, truth=4.5) "
        "are recovered with low bias (+0.05 to +0.25 dex); alternative models are strongly underestimated.",
        styles, width=4.5*inch)

    add_figure(story, fig_paths["coverage"],
        "<b>Figure 9.</b> <b>Synthetic Validation.</b> Posterior coverage from mock challenge. Red bars show measured empirical "
        "coverage from 20 mocks; grey bars show the nominal target. The large gap between nominal "
        "and empirical coverage confirms that count-only Poisson inference is insufficient  -  "
        "the GNN embedding must contribute morphological information for calibrated posteriors.",
        styles, width=4.0*inch)

    story.append(Paragraph("6.2  Hierarchical Bayesian Inference", styles["SubHead"]))
    story.append(Paragraph(
        "The hierarchical model was run on two synthetic scenarios, each using 7 streams with emcee "
        "MCMC (32 walkers, 8,000 steps, 2,000 burn-in). Both achieved excellent convergence.",
        styles["Body"]))

    hier_tbl = [
        ["", "CDM Scenario", "WDM Scenario"],
        ["True log10(M_hm)", "4.5 (no suppression)", "7.5"],
        ["Posterior median", "4.76", "8.16"],
        ["95% upper limit", "5.32", "9.81"],
        ["68% HDI", "[4.37, 5.24]", "[7.46, 9.99]"],
        ["r-hat", "1.0", "1.0003"],
        ["Effective samples", "7,075", "6,756"],
        ["Observed impacts (total)", "14 across 7 streams", "1 across 7 streams"],
    ]
    story.append(make_table(hier_tbl[0], hier_tbl[1:],
                            col_widths=[1.7*inch, 2.1*inch, 2.1*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Table 3c.</b> <b>Synthetic Validation.</b> Hierarchical inference results on synthetic scenarios.",
        styles["FigCaption"]))

    story.append(Paragraph(
        "<b>CDM scenario:</b> With 14 observed impacts across 7 streams (consistent with CDM rates "
        "under the corrected normalisation of 2.1 for GD-1), the posterior peaks at 4.76 with "
        "68% HDI [4.37, 5.24]. This is the correct result  -  the inferred M<sub>hm</sub> falls "
        "well below 10<super>5</super> M<sub>sun</sub>, the minimum subhalo mass in our integration "
        "range, indicating no suppression is detected. The posterior freely explores the CDM-consistent "
        "region without any artificial boundary effects.",
        styles["Body"]))
    story.append(Paragraph(
        "<b>WDM scenario:</b> With only 1 observed impact across 7 streams (strongly suppressed rates), "
        "the posterior is broad (68% HDI spans 2.5 dex) with median 8.16 and the truth (7.5) near the "
        "lower bound of the 68% HDI. The breadth is expected  -  with near-zero counts, many high-M<sub>hm</sub> "
        "values produce similarly low rates, so the data provide limited discrimination among "
        "suppressed models. The posterior correctly excludes the CDM regime and contains the truth.",
        styles["Body"]))
    story.append(Paragraph(
        "<b>What this means:</b> The hierarchical inference machinery works correctly. It converges "
        "reliably, it correctly identifies CDM-like data, and it recovers suppressed scenarios within "
        "broad but honest credible intervals. The limiting factor for constraining power is the "
        "information content per stream  -  which is where the GNN embedding (carrying morphological "
        "rather than just count information) becomes essential.",
        styles["Body"]))

    add_figure(story, fig_paths["hier"],
        "<b>Figure 10.</b> <b>Synthetic Validation.</b> Hierarchical posterior distributions from emcee MCMC. Left: CDM scenario  -  "
        "the posterior peaks around 4.8, correctly indicating no suppression above the integration "
        "floor. Right: WDM scenario  -  the truth (7.5, red dashed) falls within the broad posterior. "
        "Both converged with r-hat near 1.0.",
        styles, width=5.5*inch)

    story.append(Paragraph("6.3  What These Results Do and Do Not Show", styles["SubHead"]))
    story.append(Paragraph(
        "These validation results establish three things: (1) the statistical machinery  -  rate model, "
        "Poisson likelihood, hierarchical MCMC  -  is correctly implemented and converges reliably; "
        "(2) count-only inference has a fundamental resolution floor around "
        "log<sub>10</sub>(M<sub>hm</sub>) ~ 5-6, motivating the use of morphological embeddings; "
        "(3) the hierarchical model correctly distinguishes CDM-consistent data from suppressed "
        "scenarios when the signal is strong enough.",
        styles["Body"]))
    story.append(Paragraph(
        "The full GNN + SBI pipeline has been applied to real Gaia DR3 data and its results are "
        "presented in Section 11.6. The mock challenge coverage specifically applies to the "
        "count-only path; the full pipeline uses morphological information and produces "
        "qualitatively different  -  and in this case, strikingly different  -  results.",
        styles["Body"]))

    add_figure(story, fig_paths["discrimination"],
        "<b>Figure 11.</b> <b>Forecast.</b> Projected model discrimination power expressed as log<sub>10</sub>(Bayes "
        "factor). These are design targets based on theoretical sensitivity. WDM vs. FDM "
        "discrimination is expected to be degenerate at Gaia DR3 precision (Banik+2021, Dalal+2022). "
        "CDM and SIDM are merged due to indistinguishable impulse-approximation signatures.",
        styles, width=4.8*inch)
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 7. DARK MATTER MODELS
    # -----------------------------------------------------------------------
    story.append(Paragraph("7. Dark Matter Model Definitions", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "All model definitions are stored in config/dm_models.yaml. Each specifies a subhalo mass "
        "function, density profile, and inferred parameters. The mass function slope alpha = -1.9 is "
        "fixed for all models (Springel et al. 2008). The subhalo mass range is "
        "[10<super>5</super>, 10<super>9</super>] M<sub>sun</sub>.",
        styles["Body"]))

    for name, full, props in [
        ("CDM", "Cold Dark Matter", [
            ["Mass function", "dN/dM ~ M^(-1.9), no suppression"],
            ["Density profile", "NFW (Navarro-Frenk-White)"],
            ["Concentration", "Ludlow et al. (2016)"],
            ["Classification", "model_idx = 0 (merged with SIDM)"],
            ["Inferred params", "log10(M_sub), n_impacts"],
        ]),
        ("WDM", "Warm Dark Matter", [
            ["Mass function", "CDM power-law with half-mode filter"],
            ["Suppression", "(1 + (M_hm/M)^2.7)^(-0.99/2.7) (Schneider+2012)"],
            ["M_hm formula", "1.7e10 * (m_WDM/keV)^(-3.33) Msun (Lovell+2014)"],
            ["Density profile", "NFW with Ludlow2016_WDM concentration"],
            ["Classification", "model_idx = 1"],
        ]),
        ("FDM", "Fuzzy Dark Matter", [
            ["Mass function", "CDM power-law with Jeans filter"],
            ["M_Jeans formula", "1.5e8 * (m_axion/1e-22 eV)^(-1.5) Msun (Hui+2017)"],
            ["Density profile", "Soliton core + NFW envelope (Schive+2014)"],
            ["Particle mass prior", "Log-uniform [1e-23, 1e-20] eV"],
            ["Classification", "model_idx = 2"],
        ]),
        ("SIDM", "Self-Interacting Dark Matter", [
            ["Mass function", "Same as CDM (no suppression)"],
            ["Density profile", "Isothermal core + NFW envelope (Kaplinghat+2016)"],
            ["Cross-section prior", "Log-uniform [0.1, 100] cm^2/g"],
            ["Classification", "model_idx = 3 (merged with CDM)"],
            ["Note", "Indistinguishable from CDM in impulse approximation"],
        ]),
    ]:
        story.append(Paragraph(f"<b>{name}  -  {full}</b>", styles["SubHead3"]))
        story.append(make_table(["Property", "Value"], props,
                                col_widths=[1.6*inch, 4.3*inch]))
        story.append(Spacer(1, 6))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 8. TARGET STELLAR STREAMS
    # -----------------------------------------------------------------------
    story.append(Paragraph("8. Target Stream Sample", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Seven stellar streams observed by Gaia DR3 are analysed, selected to span a range of lengths, "
        "distances, and ages for complementary sensitivity. Stream catalogues follow Banik et al. (2021) "
        "and Dalal et al. (2022).",
        styles["Body"]))

    stream_tbl = [
        ["Stream", "Length (deg)", "Dist. (kpc)", "Age (Gyr)", "Key Characteristics"],
        ["GD-1", "~60", "~12", "~5", "Longest MW stream; confirmed gaps and spur"],
        ["Pal 5", "~20", "~20", "~11", "Globular cluster tails; very old"],
        ["Orphan-Chenab", "~120", "~20", "~5", "Widest; highest geometric cross-section"],
        ["ATLAS", "~15", "~20", "~4", "Short, very thin; low velocity dispersion"],
        ["Jhelum", "~30", "~13", "~4", "Double-component; possible DM interaction"],
        ["Fjorm", "~25", "~15", "~3", "Recently discovered; clean kinematics"],
        ["Sylgr", "~12", "~15", "~3", "Shortest; cold kinematics amplify signals"],
    ]
    story.append(make_table(stream_tbl[0], stream_tbl[1:],
                            col_widths=[1.1*inch, 0.85*inch, 0.8*inch, 0.75*inch, 2.4*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Table 4.</b> Target stellar streams and their approximate physical properties. "
        "Longer and older streams accumulate more subhalo encounters.",
        styles["FigCaption"]))

    add_figure(story, fig_paths["streams"],
        "<b>Figure 12.</b> Comparison of target stream properties. Orphan-Chenab dominates in "
        "length (120 deg), while Pal 5 is the oldest (~11 Gyr). These differences translate to "
        "varying sensitivity to dark matter substructure.",
        styles, width=5.2*inch)

    add_figure(story, fig_paths["sensitivity"],
        "<b>Figure 13.</b> Per-stream sensitivity comparison showing expected subhalo impact counts "
        "under CDM (blue, no suppression, M_hm = 10^4.5) and WDM (orange, M_hm = 10^7.5). "
        "The rate difference between models is largest for long, old streams.",
        styles, width=5.2*inch)
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 9. TEST SUITE
    # -----------------------------------------------------------------------
    story.append(Paragraph("9. Software Validation and Reproducibility", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The executable automated tests cover "
        "output tensor shapes, numerical stability (no NaN/Inf), gradient flow through all layers, "
        "and internal consistency checks (attention weights sum to 1, rate function monotonicity, "
        "coverage metric computation). Passing these tests means the code paths execute as expected; "
        "it does not prove that the astrophysical inference is calibrated.",
        styles["Body"]))

    test_tbl = [
        ["Module", "Tests", "Status"],
        ["GNN Encoder", "7", "PASS"],
        ["Multi-Task Model", "4", "PASS"],
        ["Baseline Models (CNN + Transformer)", "7", "PASS"],
        ["Model Utilities", "7", "PASS"],
        ["Attention Readout (F1)", "6", "PASS"],
        ["MC Dropout (F3)", "5", "PASS"],
        ["Variable Regression Targets (F5)", "2", "PASS"],
        ["Segment GNN (F2)", "3", "PASS"],
        ["Orbital Edge Features (F4)", "2", "PASS"],
        ["Hierarchical + Coverage (F6, F7)", "4", "PASS"],
        ["Full-data pipeline checks", "2", "DEFERRED"],
        ["Executable unit tests", "49", "PASS"],
    ]
    story.append(make_table(test_tbl[0], test_tbl[1:],
                            col_widths=[3.0*inch, 0.8*inch, 2.1*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Table 5.</b> Automated implementation tests by module. The deferred full-data checks "
        "are not counted as astrophysical validation.",
        styles["FigCaption"]))

    gnn_val_tbl = [
        ["Metric", "Value", "Interpretation"],
        ["Synthetic holdout sample", "15,000", "Plan 2 test split"],
        ["Target", "impact_strong", "Strong injected impact vs no strong impact"],
        ["ROC AUC", "0.668", "Weak but above random"],
        ["Balanced accuracy", "61.4%", "Below publication target"],
        ["Precision / recall", "46.9% / 38.9%", "Default threshold 0.5"],
        ["False-positive rate", "16.0%", "Too high for discovery use"],
    ]
    story.append(make_table(gnn_val_tbl[0], gnn_val_tbl[1:],
                            col_widths=[1.8*inch, 1.0*inch, 2.8*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Table 6.</b> Current V2 held-out GNN synthetic validation from "
        "<font name='Courier'>outputs/diagnostics/gnn_v2_100k_impact_strong_eval.json</font>. "
        "These numbers validate only the simulated impact-strong classifier; they do not validate "
        "sim-to-real generalization or SBI calibration.",
        styles["FigCaption"]))

    validation_summary_path = os.path.join(os.path.dirname(__file__), "outputs", "validation",
                                           "methods_validation_summary.csv")
    if os.path.exists(validation_summary_path):
        with open(validation_summary_path, newline="", encoding="utf-8") as f:
            summary_rows = list(csv.DictReader(f))
        summary_tbl = [["Label", "Test", "Metric / value", "Status"]]
        for row in summary_rows:
            summary_tbl.append([
                row["label"],
                row["test"],
                f"{row['metric']}: {row['value']}",
                row["status"],
            ])
        story.append(Spacer(1, 8))
        story.append(Paragraph("<b>Publication Readiness Validation Matrix</b>", styles["SubHead3"]))
        story.append(make_table(summary_tbl[0], summary_tbl[1:],
                                col_widths=[1.25*inch, 1.75*inch, 2.15*inch, 0.55*inch],
                                font_size=7.0, leading=8.3))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "<b>Table 7.</b> Labeled validation summary. Rows marked WARN are not failures of the "
            "software; they are publication blockers or caveats that must be resolved before making "
            "a science-results claim.",
            styles["FigCaption"]))

    real_injection_path = os.path.join(os.path.dirname(__file__), "outputs", "validation",
                                       "real_stream_injection_recovery.csv")
    if os.path.exists(real_injection_path):
        with open(real_injection_path, newline="", encoding="utf-8") as f:
            inj_rows = list(csv.DictReader(f))
        inj_tbl = [["Stream", "Label", "Injected", "Detected", "Detection rate", "Mass RMSE"]]
        for row in inj_rows:
            rmse = row["mass_rmse"]
            if rmse.lower() == "nan":
                rmse = "n/a"
            else:
                rmse = f"{float(rmse):.2f}"
            inj_tbl.append([
                row["stream"], "Observed + Synthetic Injection", row["n_injections"],
                row["n_detected"], f"{100 * float(row['detection_rate']):.0f}%", rmse,
            ])
        story.append(Spacer(1, 8))
        story.append(make_table(inj_tbl[0], inj_tbl[1:],
                                col_widths=[0.75*inch, 1.8*inch, 0.65*inch, 0.65*inch, 0.9*inch, 0.8*inch],
                                font_size=7.0, leading=8.3))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "<b>Table 8.</b> Real-stream injection/recovery diagnostics using processed Gaia catalogs. "
            "These are observed catalogs with synthetic perturbations, not pure observations and not "
            "a full GNN+SBI real-injection validation.",
            styles["FigCaption"]))

    no_injection_path = os.path.join(os.path.dirname(__file__), "outputs", "validation",
                                     "real_stream_no_injection_false_positive.csv")
    if os.path.exists(no_injection_path):
        with open(no_injection_path, newline="", encoding="utf-8") as f:
            fp_rows = list(csv.DictReader(f))
        fp_tbl = [["Stream", "Label", "Trials", "False positives", "Rate", "Status"]]
        for row in fp_rows:
            fp_rate = float(row["false_positive_rate"])
            fp_tbl.append([
                row["stream"], "Observed / No Injection", row["n_trials"],
                row["n_false_positive"], f"{100 * fp_rate:.0f}%",
                "PASS" if fp_rate <= 0.10 else "WARN",
            ])
        story.append(Spacer(1, 8))
        story.append(make_table(fp_tbl[0], fp_tbl[1:],
                                col_widths=[0.75*inch, 1.55*inch, 0.55*inch, 0.9*inch, 0.6*inch, 0.7*inch],
                                font_size=7.0, leading=8.3))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "<b>Table 9.</b> Observed / No Injection false-positive checks. A WARN row indicates that "
            "the detector triggered on at least 10% of no-injection trials for that stream, so the "
            "pipeline should not be interpreted as publication-ready for discovery claims.",
            styles["FigCaption"]))

    baseline_path = os.path.join(os.path.dirname(__file__), "outputs", "validation",
                                 "baseline_comparison_real_like.csv")
    if os.path.exists(baseline_path):
        with open(baseline_path, newline="", encoding="utf-8") as f:
            base_rows = list(csv.DictReader(f))
        base_tbl = [["Stream", "Label", "log10 Msub", "Count S/N", "Count", "Power sig.", "Power"]]
        for row in base_rows:
            base_tbl.append([
                row["stream"], "Observed + Synthetic Injection",
                f"{float(row['injected_log10_M_sub']):.1f}",
                f"{float(row['count_snr']):.1f}",
                "yes" if row["count_detected"].lower() == "true" else "no",
                f"{float(row['power_spectrum_sigma']):.1f}",
                "yes" if row["power_spectrum_detected"].lower() == "true" else "no",
            ])
        story.append(Spacer(1, 8))
        story.append(make_table(base_tbl[0], base_tbl[1:],
                                col_widths=[0.65*inch, 1.35*inch, 0.85*inch, 0.65*inch,
                                            0.55*inch, 0.7*inch, 0.55*inch],
                                font_size=6.6, leading=7.9))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "<b>Table 10.</b> Baseline comparisons on the same real-like injected inputs. This table "
            "keeps the count detector and power-spectrum detector separate so the ML pipeline is not "
            "presented without non-ML reference points.",
            styles["FigCaption"]))

    manifest_tbl = [
        ["CSV", "Result label", "Purpose"],
        ["methods_validation_summary.csv", "Synthetic / Observed / Forecast", "Publication readiness matrix"],
        ["real_stream_injection\n_recovery.csv", "Observed + Synthetic Injection", "Recovery rates on Gaia-derived catalogs"],
        ["real_stream_no_injection\n_false_positive.csv", "Observed / No Injection", "False-positive check"],
        ["baseline_comparison\n_real_like.csv", "Observed + Synthetic Injection", "Non-ML detector baselines"],
        ["systematics_table.csv", "Systematics", "Sensitivity rows by stream and stress scenario"],
        ["reproducibility_manifest.csv", "Reproducibility", "Commands for reported artifacts"],
    ]
    story.append(Spacer(1, 8))
    story.append(make_table(manifest_tbl[0], manifest_tbl[1:],
                            col_widths=[2.0*inch, 1.65*inch, 1.95*inch],
                            font_size=7.0, leading=8.3))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Table 11.</b> Machine-readable validation spreadsheets included in "
        "<font name='Courier'>outputs/validation/</font> and summarized in the paper.",
        styles["FigCaption"]))

    systematics_path = os.path.join(os.path.dirname(__file__), "outputs", "validation",
                                    "systematics_table.csv")
    if os.path.exists(systematics_path):
        with open(systematics_path, newline="", encoding="utf-8") as f:
            syst_rows_all = list(csv.DictReader(f))
        syst_tbl = [["Stream", "Scenario", "False positive", "Status"]]
        for row in syst_rows_all[:10]:
            syst_tbl.append([
                row["stream"], row["scenario"],
                f"{100 * float(row['false_positive_rate']):.0f}%" if row.get("false_positive_rate") else "n/a",
                row["status"],
            ])
        story.append(Spacer(1, 8))
        story.append(make_table(syst_tbl[0], syst_tbl[1:],
                                col_widths=[0.75*inch, 2.45*inch, 1.1*inch, 0.75*inch],
                                font_size=7.0, leading=8.0))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "<b>Table 12.</b> Systematics sample table. Full CSV contains all stream/scenario rows. "
            "WARN marks sensitivity shifts requiring discussion or mitigation.",
            styles["FigCaption"]))

    story.append(Paragraph("Key validations include:", styles["Body"]))
    validations = [
        "GNN forward pass: correct output shapes for batches of 1-16 graphs",
        "Attention weights: verified to sum to 1.0 per graph (valid probability distribution)",
        "MC Dropout: confirmed stochastic (different passes produce different embeddings) with positive std",
        "Rate function: verified monotonically decreasing with M_hm and proportional to stream length",
        "Coverage metric: computation validated against known posterior samples",
        "Multi-scale: graceful fallback to zeros when segment data is absent",
        "Gradient flow: verified through all 6 GINEConv layers to input features",
    ]
    for v in validations:
        story.append(Paragraph(f"-  {v}", styles["BulletItem"]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 10. CONFIGURATION REFERENCE
    # -----------------------------------------------------------------------
    story.append(Paragraph("10. Reproducibility and Configuration Snapshot", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "All values sourced from version-controlled configuration files: config/training.yaml and "
        "config/dm_models.yaml.",
        styles["Body"]))

    story.append(Paragraph("Training Hyperparameters", styles["SubHead3"]))
    train_tbl = [
        ["Parameter", "Value", "Notes"],
        ["Batch size", "128", "Fast V2 profile run"],
        ["Epochs", "100 run / 300 planned", "Current checkpoint is not final publication training"],
        ["Optimiser", "AdamW", "Decoupled weight decay"],
        ["Learning rate", "1 x 10^-3", ""],
        ["Weight decay", "10^-4", ""],
        ["LR scheduler", "Cosine w/ warm restarts", "Planned curriculum schedule"],
        ["Gradient clip", "1.0", "Max gradient L2 norm"],
        ["Mixed precision", "bfloat16", ""],
        ["Early stopping", "60 epochs patience", ">= warm-restart window"],
        ["Data split", "70 / 15 / 15", "Stratified by DM model, seed = 42"],
    ]
    story.append(make_table(train_tbl[0], train_tbl[1:],
                            col_widths=[1.5*inch, 1.6*inch, 2.8*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Configuration Table A.</b> Training hyperparameters.", styles["FigCaption"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Graph Construction", styles["SubHead3"]))
    graph_tbl = [
        ["Parameter", "Value"],
        ["k (nearest neighbours)", "8"],
        ["Construction dimensions", "phi1, phi2, pm1, pm2 (normalised)"],
        ["Max stars per stream", "1,200 (random downsample)"],
        ["Multi-scale segments", "20"],
        ["Segment k-neighbours", "4"],
        ["Feature normalisation", "Standardise (zero mean, unit variance)"],
        ["Baryonic perturbations", "Applied across all models at 80% configured rate"],
    ]
    story.append(make_table(graph_tbl[0], graph_tbl[1:],
                            col_widths=[2.2*inch, 3.7*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Configuration Table B.</b> Graph construction parameters.", styles["FigCaption"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Suppression Scale Inference", styles["SubHead3"]))
    story.append(Paragraph(
        "Parameterisation: log<sub>10</sub>(M<sub>hm</sub> / M<sub>sun</sub>). "
        "Prior: Uniform[4.0, 10.0]. "
        "CDM normalisation: 2.1 impacts for GD-1 (Bonaca+2019). "
        "Detection threshold: log<sub>10</sub>(Bayes factor) > 1.0. "
        "This is the intended science output once calibration is complete: a model-independent "
        "constraint on the smallest dark matter structures.",
        styles["Body"]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 11. REAL STREAM ANALYSIS
    # -----------------------------------------------------------------------
    story.append(Paragraph("11. Observed Application to Gaia DR3", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))

    story.append(Paragraph("11.1  Hierarchical Inference on Literature Gap Counts", styles["SubHead"]))
    story.append(Paragraph(
        "We applied the hierarchical Bayesian model to gap counts reported in published Gaia DR3 "
        "analyses of the seven target streams. Gap counts were drawn from Bonaca+2019, "
        "de Boer+2020, Erkal+2017, Koposov+2019, Li+2021, and Ibata+2021. The inference used "
        "emcee with 64 walkers and 15,000 steps (3,000 burn-in).",
        styles["Body"]))

    real_tbl = [
        ["Stream", "Observed gaps", "CDM expected", "Source"],
        ["GD-1", "5", "2.5", "Bonaca+2019, de Boer+2020"],
        ["Pal 5", "3", "3.1", "Erkal+2017, Bonaca+2020"],
        ["Orphan-Chenab", "4", "8.4", "Koposov+2019, Erkal+2019"],
        ["ATLAS", "1", "0.8", "Li+2021"],
        ["Jhelum", "2", "1.1", "Bonaca+2019b"],
        ["Fjorm", "1", "0.8", "Ibata+2021"],
        ["Sylgr", "1", "0.4", "Ibata+2021"],
        ["TOTAL", "17", "17.2", " - "],
    ]
    story.append(make_table(real_tbl[0], real_tbl[1:],
                            col_widths=[1.3*inch, 0.9*inch, 0.9*inch, 2.8*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Observed Table A.</b> Gap counts from literature vs. CDM predictions from the rate model "
        "(normalised to 2.1 for GD-1, Bonaca+2019). Total predicted (17.2) matches total observed (17) "
        "to within 1%.",
        styles["FigCaption"]))

    story.append(Paragraph(
        "<b>Observed count-model result:</b> log<sub>10</sub>(M<sub>hm</sub>) = 4.56 "
        "(95% upper limit: 5.14; 68% HDI: [4.02, 4.78]). The posterior peaks well below the "
        "minimum subhalo mass in our integration range (10<super>5</super> M<sub>sun</sub>), "
        "indicating no evidence for suppression of the subhalo mass function. "
        "Convergence is excellent (r-hat = 1.0, n_eff = 29,156).",
        styles["Body"]))
    story.append(Paragraph(
        "<b>Interpretation:</b> The observed gap counts (17 total across 7 streams) are in close "
        "agreement with CDM predictions (17.2 total). Most streams individually match their CDM "
        "expectations: GD-1 has 5 observed vs. 2.5 predicted (slightly above, consistent with "
        "Poisson fluctuations), Pal 5 has 3 vs. 3.1 (excellent match), and ATLAS, Jhelum, Fjorm, "
        "and Sylgr are all within 1-2 counts. The notable exception is Orphan-Chenab, "
        "which shows 4 gaps where CDM predicts ~8.4  -  a deficit, but one compatible with "
        "Poisson scatter given the wide, diffuse morphology of this stream that complicates gap "
        "identification. The hierarchical model correctly integrates this evidence and returns "
        "a CDM-consistent constraint.",
        styles["Body"]))
    story.append(Paragraph(
        "<b>Consistency with published constraints:</b> Our 95% upper limit "
        "(log<sub>10</sub> M<sub>hm</sub> &lt; 5.14) is consistent with all published independent "
        "constraints: Gilman+2020 (&lt;7.8, lensing), Hsueh+2020 (&lt;8.0, lensing), "
        "Nadler+2021 (&lt;7.2, satellite counts), Irsic+2017 (&lt;7.5, Lyman-alpha), "
        "and Banik+2021 (&lt;7.0, stream power spectrum).",
        styles["Body"]))

    story.append(Paragraph("11.2  Extended Model: Free Mass Function Slope", styles["SubHead"]))
    story.append(Paragraph(
        "We extended the hierarchical model to jointly infer alpha (the mass function slope "
        "dN/dM ~ M<super>alpha</super>) alongside M<sub>hm</sub>, using an informative prior "
        "alpha ~ Normal(-1.9, 0.2) centered on the CDM prediction.",
        styles["Body"]))
    alpha_tbl = [
        ["Parameter", "Result", "CDM Prediction"],
        ["log10(M_hm)", "4.56 (95% upper: 5.16)", "< 5 (no suppression)"],
        ["alpha", "-1.89 (68% HDI: [-2.10, -1.70])", "-1.9 (Springel+2008)"],
        ["M_hm-alpha correlation", "0.10", " - "],
        ["r-hat", "1.0", " - "],
        ["n_eff", "21,438", " - "],
    ]
    story.append(make_table(alpha_tbl[0], alpha_tbl[1:],
                            col_widths=[1.8*inch, 2.4*inch, 1.7*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Observed Table B.</b> Extended model results with free mass function slope.",
        styles["FigCaption"]))
    story.append(Paragraph(
        "The inferred alpha = -1.89 +/- 0.20 is fully consistent with the CDM prediction of -1.9 "
        "(Springel et al. 2008). This is a direct consequence of correcting the CDM normalisation "
        "to 2.1 impacts for GD-1 (Bonaca+2019): the total predicted count now matches observations, "
        "eliminating the tension that previously drove the model toward a shallower slope. "
        "The M<sub>hm</sub>-alpha correlation is low (0.10), indicating the two parameters are "
        "well-separated by the data.",
        styles["Body"]))
    story.append(Paragraph(
        "<b>Significance:</b> The agreement between the inferred and predicted mass function slope "
        "is an important consistency check. It confirms that the observed gap counts across all seven "
        "streams are quantitatively compatible with the CDM subhalo mass function when the normalisation "
        "is anchored to the well-studied GD-1 system. No modification to the standard CDM mass function "
        "is required by the count data.",
        styles["Body"]))

    story.append(Paragraph("11.3  Injection Tests: Sim-to-Real Gap", styles["SubHead"]))
    story.append(Paragraph(
        "We quantified the detection sensitivity by injecting synthetic subhalo impacts into "
        "simulated real stream data (GD-1-like: 5,000 stars, 60 deg). 100 injections with "
        "impactor masses from 10<super>6</super> to 10<super>9</super> M<sub>sun</sub> were tested.",
        styles["Body"]))
    inj_tbl = [
        ["Mass range", "Injections", "Detected", "Rate"],
        ["10^6 - 10^7 Msun", "40", "0", "0%"],
        ["10^7 - 10^8 Msun", "27", "20", "74%"],
        ["10^8 - 10^9 Msun", "33", "33", "100%"],
        ["Overall", "100", "53", "53%"],
    ]
    story.append(make_table(inj_tbl[0], inj_tbl[1:],
                            col_widths=[1.5*inch, 1.0*inch, 1.0*inch, 2.4*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Injection Table A.</b> <b>Observed + Synthetic Injection.</b> Injection test results by impactor mass range.",
        styles["FigCaption"]))
    story.append(Paragraph(
        "The detection threshold is approximately 10<super>7</super> M<sub>sun</sub>: subhalos "
        "below this mass produce gaps too shallow to detect in single-stream density analysis "
        "(mean bias: -0.016 dex; RMSE: 0.279 dex for detected events). This defines the "
        "sensitivity floor for count-based methods and motivates the use of the GNN embedding, "
        "which can potentially detect subtler morphological signatures.",
        styles["Body"]))

    story.append(Paragraph("11.4  Equivariant GNN Architecture", styles["SubHead"]))
    story.append(Paragraph(
        "We implemented an E(n)-equivariant graph neural network (Satorras+2021) as an alternative "
        "to the base GINEConv encoder. The equivariant architecture guarantees that rotating the "
        "input stream in 3D produces correspondingly transformed intermediate representations "
        "while the final embedding remains invariant. This eliminates the need for rotational "
        "data augmentation and provides a stronger physical inductive bias. The implementation "
        "(526,148 parameters, 4 EGNN layers) passes forward/backward tests and produces "
        "rotationally-invariant embeddings (tested: mean abs diff = 0 under 90-degree rotation).",
        styles["Body"]))

    story.append(Paragraph("11.5  Radial Velocity Extension (SDSS-V / 4MOST)", styles["SubHead"]))
    story.append(Paragraph(
        "We implemented infrastructure to incorporate spectroscopic radial velocities from "
        "SDSS-V Milky Way Mapper (northern sky, G &lt; 19.5) and 4MOST (southern sky, G &lt; 20.5). "
        "Adding the 6th phase-space dimension resolves velocity-distance degeneracies and provides "
        "direct measurement of the dipolar velocity perturbation from subhalo flybys.",
        styles["Body"]))
    rv_tbl = [
        ["Stream", "Expected coverage", "Improvement factor"],
        ["GD-1", "70% (SDSS-V)", "1.44x"],
        ["Pal 5", "50%", "1.16x"],
        ["Orphan-Chenab", "60%", "1.26x"],
        ["ATLAS", "40%", "1.07x"],
        ["Jhelum", "55%", "1.17x"],
        ["Fjorm", "35% (4MOST)", "1.06x"],
        ["Sylgr", "30% (4MOST)", "1.04x"],
        ["Average", " - ", "1.17x"],
    ]
    story.append(make_table(rv_tbl[0], rv_tbl[1:],
                            col_widths=[1.5*inch, 1.8*inch, 2.6*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Forecast Table A.</b> Projected radial velocity coverage and constraint improvement. "
        "GD-1 benefits most due to high SDSS-V completeness in the northern hemisphere.",
        styles["FigCaption"]))
    story.append(Paragraph(
        "The average constraint improvement from adding radial velocities is 1.17x (detection "
        "SNR gain: 1.32x). While modest for count-based inference, RV data is expected to provide "
        "substantially larger improvements for the GNN pipeline, which can directly detect the "
        "dipolar velocity signature of subhalo flybys  -  a signal invisible in density data alone.",
        styles["Body"]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 11.6 FULL GNN + SBI PIPELINE RESULTS
    # -----------------------------------------------------------------------
    story.append(Paragraph("11.6  Synthetic/Diagnostic GNN + SBI Status", styles["SubHead"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "The current GNN+SBI branch is not yet a validated science-result pipeline. The latest "
        "V2 training run uses the more defensible impact-strong binary formulation and profile "
        "features, but held-out synthetic performance remains weak (test ROC AUC 0.668; balanced "
        "accuracy 61.4%). First-pass SBI/TARP calibration also undercovers key parameters. The "
        "older four-model evidence run is retained below only as a simulated-framework diagnostic "
        "showing the kind of domain-gap failure a reviewer would challenge, not as evidence for "
        "warm dark matter.",
        styles["Body"]))

    v2_status_tbl = [
        ["Quantity", "Current value", "Peer-review implication"],
        ["V2 target", "impact_strong", "More physical than 3-class WDM/FDM/CDM labels"],
        ["Test ROC AUC", "0.668", "Above random, below publication target"],
        ["Test balanced accuracy", "61.4%", "Insufficient for discovery claims"],
        ["Default false-positive rate", "16.0%", "Too high for real-stream searches"],
        ["SBI calibration", "undercovers", "Requires split-aware retraining and recalibration"],
    ]
    story.append(make_table(v2_status_tbl[0], v2_status_tbl[1:],
                            col_widths=[1.55*inch, 1.25*inch, 3.1*inch],
                            font_size=8.0, leading=9.5))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Diagnostic Table A.</b> <b>Synthetic Validation.</b> Current V2 GNN+SBI readiness. "
        "These values are intentionally reported as limitations, not as discoveries.",
        styles["FigCaption"]))

    # Per-stream log evidence table
    ev_tbl = [
        ["Stream", "CDM", "WDM", "FDM", "SIDM", "Preferred"],
        ["GD-1", "-3.29", "-3.19", "-3.04", "-3.23", "FDM"],
        ["Pal 5", "-3.32", "-2.17", "-3.01", "-3.35", "WDM"],
        ["Orphan-Chenab", "-4.82", "-2.45", "-3.16", "-4.82", "WDM"],
        ["ATLAS", "-2.66", "-2.29", "-2.83", "-2.62", "WDM"],
        ["Jhelum", "-3.13", "-2.24", "-3.17", "-3.69", "WDM"],
        ["Fjorm", "-3.50", "-2.25", "-3.15", "-3.37", "WDM"],
        ["Sylgr", "-2.45", "-2.21", "-3.17", "-3.38", "WDM"],
    ]
    story.append(make_table(ev_tbl[0], ev_tbl[1:],
                            col_widths=[1.1*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 1.0*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Legacy Diagnostic Table B.</b> <b>Observed / Simulated-Framework Diagnostic.</b> Per-stream log evidences from the older "
        "four-model GNN+SBI run. Higher (less negative) values indicate better simulated-framework "
        "fit. This table is retained to document the domain-gap warning, not to support a physical "
        "model claim.",
        styles["FigCaption"]))

    add_figure(story, fig_paths["log_ev_heatmap"],
        "<b>Figure 14.</b> <b>Observed / Simulated-Framework Diagnostic.</b> Legacy per-stream log evidence heatmap. Warmer colours "
        "indicate better simulated-framework fit. Because the current V2 validation is weak, this "
        "figure is interpreted as a domain-gap diagnostic.",
        styles, width=4.8*inch)

    # Overall model comparison
    story.append(Paragraph("<b>Combined Model Comparison</b>", styles["SubHead3"]))
    mc_tbl = [
        ["Model", "Combined log evidence", "log10(BF vs CDM)", "Posterior prob.", "Interpretation"],
        ["WDM", "-16.81", "+2.77", "~99%", "Legacy simulated-framework preference"],
        ["FDM", "-21.54", "+0.71", "0.87%", "Secondary simulated preference"],
        ["CDM", "-23.18", "0.00", "0.17%", "Reference model"],
        ["SIDM", "-24.45", "-0.55", "0.05%", "Evidence against"],
    ]
    story.append(make_table(mc_tbl[0], mc_tbl[1:],
                            col_widths=[0.6*inch, 1.3*inch, 1.2*inch, 0.9*inch, 1.9*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Legacy Diagnostic Table C.</b> <b>Observed / Simulated-Framework Diagnostic.</b> Combined model comparison from the older "
        "GNN+SBI run. Posterior probabilities assume equal prior odds inside the simulated "
        "framework. The WDM preference is not a validated physical discovery.",
        styles["FigCaption"]))

    add_figure(story, fig_paths["model_probs"],
        "<b>Figure 15.</b> <b>Observed / Simulated-Framework Diagnostic.</b> Legacy dark matter model posterior probabilities from the "
        "older simulated-framework run. The approximately 99% WDM preference is shown as a "
        "failure-mode diagnostic, not a validated astrophysical constraint.",
        styles, width=4.5*inch)

    # Sensitivity and gap catalog
    story.append(Paragraph("<b>Per-Stream Sensitivity</b>", styles["SubHead3"]))
    sens_tbl = [
        ["Stream", "N members", "Median log10(M_sub)", "N impacts", "P(any impact)"],
        ["GD-1", "4,000", "6.39", "3.0", "98.0%"],
        ["Pal 5", "500", "5.91", "1.1", "75.0%"],
        ["Orphan-Chenab", "200", "7.92", "1.9", "98.6%"],
        ["ATLAS", "300", "6.57", "1.3", "92.6%"],
        ["Jhelum", "250", "6.48", "2.8", "93.9%"],
        ["Fjorm", "150", "5.57", "1.9", "86.7%"],
        ["Sylgr", "100", "6.08", "1.3", "83.1%"],
    ]
    story.append(make_table(sens_tbl[0], sens_tbl[1:],
                            col_widths=[1.1*inch, 0.8*inch, 1.3*inch, 0.9*inch, 1.0*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Diagnostic Table D.</b> <b>Observed / Simulated-Framework Diagnostic.</b> Per-stream sensitivity from the legacy SBI posteriors. GD-1 and Orphan-Chenab "
        "have the highest sensitivity due to their length and number of member stars.",
        styles["FigCaption"]))

    story.append(Paragraph("<b>Gap Catalog</b>", styles["SubHead3"]))
    story.append(Paragraph(
        "Three density features exceeding 2-sigma significance were identified across all streams:",
        styles["Body"]))
    gap_tbl = [
        ["Stream", "Position (phi1)", "Significance", "P(DM subhalo)", "P(noise)"],
        ["GD-1", "-1.5 deg", "2.37 sigma", "0.0%", "100%"],
        ["ATLAS", "-0.75 deg", "2.41 sigma", "0.0%", "100%"],
        ["Sylgr", "-0.49 deg", "2.45 sigma", "0.0%", "100%"],
    ]
    story.append(make_table(gap_tbl[0], gap_tbl[1:],
                            col_widths=[0.9*inch, 1.1*inch, 1.0*inch, 1.1*inch, 0.9*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Observed Table C.</b> Gap catalog from the pipeline's density analysis. All three features "
        "are classified as noise (P(DM) = 0%), indicating that the GNN does not find convincing "
        "individual dark matter gap detections in any stream.",
        styles["FigCaption"]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 11.7 DISCUSSION
    # -----------------------------------------------------------------------
    story.append(Paragraph(
        "11.7  Discussion: Reconciling the Two Analyses", styles["SubHead"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "The count model and the legacy simulated-framework GNN run do not support the same "
        "interpretation. This is a validation problem, not evidence for a discovery:",
        styles["Body"]))
    story.append(Paragraph(
        "-  <b>Rate model</b> (Section 11.1): CDM is fully consistent with the data. The "
        "total predicted impact count (17.2) matches observations (17) to within 1%. The inferred "
        "alpha = -1.89 is in excellent agreement with the CDM prediction of -1.9. No suppression "
        "of the subhalo mass function is detected.",
        styles["BulletItem"]))
    story.append(Paragraph(
        "-  <b>Legacy GNN + SBI run</b> (Section 11.6): WDM receives the strongest simulated-framework "
        "diagnostic score, but the current V2 classifier is weak and SBI calibration undercovers. "
        "This is an unvalidated failure-mode diagnostic, not physical evidence.",
        styles["BulletItem"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "This disagreement is significant and requires careful interpretation. We consider three "
        "possible explanations, ordered by our assessment of their likelihood:",
        styles["Body"]))

    story.append(Paragraph(
        "<b>Explanation 1 (most likely): Simulation-to-real domain gap.</b> "
        "The GNN was trained on simulated streams generated with galpy. If these simulations "
        "do not perfectly replicate the noise properties, selection effects, foreground contamination, "
        "and membership impurities present in real Gaia DR3 data, the GNN may systematically "
        "misclassify real streams. Specifically, WDM simulations have fewer subhalo perturbations "
        "and therefore appear smoother in phase space. Real Gaia streams  -  affected by observational "
        "noise, imperfect membership assignment, and unmodelled baryonic effects  -  may also appear "
        "smoother than CDM simulations, causing the GNN to preferentially assign WDM labels. This "
        "interpretation is supported by three independent lines of evidence: (a) the gap catalog "
        "(Observed Table C) classifies all detected density features as noise with P(DM) = 0%, suggesting "
        "the GNN does not identify convincing individual dark matter signatures; (b) the rate model, "
        "which is immune to morphological systematics, finds excellent CDM agreement; and (c) "
        "simulation-to-real gaps are a well-documented challenge in machine learning applications "
        "to astrophysical data.",
        styles["Body"]))

    story.append(Paragraph(
        "<b>Explanation 2 (possible but extraordinary): Real WDM signal.</b> "
        "A fully validated future GNN might detect subtle phase-space signatures of warm dark matter that are "
        "invisible to count-based methods. WDM subhalos produce qualitatively different gap "
        "morphology  -  fewer, wider, deeper gaps with distinct velocity profiles  -  and the GNN "
        "operates on the full 6-dimensional phase-space graph where such differences could in "
        "principle be resolved. If confirmed by future real-injection and independent-data tests, this would be scientifically important. However, "
        "the legacy result cannot carry that interpretation: it faces significant challenges and would conflict with multiple "
        "independent constraints from gravitational lensing (Gilman+2020, Hsueh+2020), satellite "
        "counts (Nadler+2021), and Lyman-alpha forest analyses (Irsic+2017), all of which are "
        "consistent with CDM at the relevant mass scales.",
        styles["Body"]))

    story.append(Paragraph(
        "<b>Explanation 3 (plausible): Training data bias.</b> "
        "If systematic differences between the four dark matter model training sets correlate with "
        "features of real data for reasons unrelated to dark matter physics  -  for example, if "
        "WDM training streams happen to have velocity dispersion profiles or density noise patterns "
        "that resemble real observational systematics  -  the classifier may preferentially assign "
        "WDM labels even when the actual dark matter signal is CDM-consistent. This is a more "
        "specific version of Explanation 1, focusing on training set construction rather than "
        "simulation fidelity.",
        styles["Body"]))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Assessment.</b> We consider Explanation 1 most likely for several reasons. First, the "
        "rate model provides a clean, assumption-light test: it uses only integer gap counts and "
        "is entirely independent of the simulation training set. Its finding of near-perfect CDM "
        "agreement (17.2 predicted vs. 17 observed) sets a strong baseline. Second, the gap catalog's "
        "classification of all features as noise is independently consistent with CDM and "
        "inconsistent with the GNN's overall WDM diagnostic preference; if WDM were truly the correct model, "
        "we would expect at least some features to show nonzero P(DM). Third, the WDM preference "
        "would require dark matter properties inconsistent with several independent observational "
        "constraints. The most parsimonious interpretation is therefore that the GNN has learned "
        "simulation-specific features that do not transfer perfectly to real data.",
        styles["Body"]))

    story.append(Paragraph(
        "<b>Path forward.</b> Resolving this tension definitively requires: (1) injection tests on "
        "real data to quantify whether the GNN correctly recovers known CDM signals injected into "
        "actual Gaia streams; (2) training with domain-randomised simulations that better span the "
        "space of observational systematics (noise levels, membership contamination, background "
        "density variations); (3) cross-validation against independent ML architectures to test "
        "whether the WDM preference is architecture-dependent; and (4) comparison with non-ML "
        "morphological methods (e.g., matched-filter gap fitting) applied to the same data.",
        styles["Body"]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 12. LIMITATIONS AND OUTLOOK
    # -----------------------------------------------------------------------
    story.append(Paragraph("12. Discussion, Limitations, and Outlook", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))

    story.append(Paragraph("12.1  Simulation-to-Real Domain Gap", styles["SubHead"]))
    story.append(Paragraph(
        "The primary limitation of the full GNN + SBI pipeline is the simulation-to-real domain gap "
        "discussed in Section 11.7. The GNN was trained on galpy-generated streams that, while "
        "physically motivated, do not capture the full complexity of real Gaia DR3 observations. "
        "Key missing elements include realistic photometric selection functions, foreground "
        "contamination from non-member stars, spatially varying completeness, and the full range "
        "of baryonic perturbations (spiral arm passages, bar resonances). Until this gap is "
        "quantified through injection tests on real data, the GNN + SBI results should be treated "
        "as methodological demonstrations rather than physical constraints.",
        styles["Body"]))

    story.append(Paragraph("12.2  Count-Based Inference Limitations", styles["SubHead"]))
    story.append(Paragraph(
        "The rate model (Section 11.1) is robust to morphological systematics but is fundamentally "
        "limited by information content. With only integer gap counts per stream, the Poisson "
        "inversion has a resolution floor near log<sub>10</sub>(M<sub>hm</sub>) ~ 5 (as demonstrated "
        "by the mock challenge in Section 6.1). It cannot distinguish between dark matter models "
        "that predict the same total count but different gap morphologies. The rate model's CDM "
        "consistency is therefore a necessary but not sufficient condition for CDM being the "
        "correct model.",
        styles["Body"]))

    story.append(Paragraph("12.3  Gap Identification Uncertainties", styles["SubHead"]))
    story.append(Paragraph(
        "Literature gap counts are not standardised: different authors use different significance "
        "thresholds, background models, and identification algorithms. For Orphan-Chenab in "
        "particular, the wide morphology and low surface brightness make gap identification "
        "challenging. Future work should use a consistent, automated gap-finding pipeline across "
        "all streams to reduce this source of uncertainty.",
        styles["Body"]))

    story.append(Paragraph("12.4  Remaining Limitations", styles["SubHead"]))
    story.append(Paragraph(
        "<b>Baryonic contamination.</b> The current V2 generation plan applies GMC perturbations "
        "across all model families, but other baryonic effects (spiral arm passages, bar resonances, globular "
        "cluster encounters) are not modelled. These could produce density features that mimic "
        "or obscure dark matter signatures.",
        styles["Body"]))
    story.append(Paragraph(
        "<b>Model degeneracy.</b> CDM and SIDM produce indistinguishable signatures in the impulse "
        "approximation used here. WDM and FDM are also partially degenerate at Gaia DR3 precision "
        "(Banik+2021, Dalal+2022). Breaking these degeneracies requires either higher-precision "
        "kinematics or detection of density profile differences (cored vs. cuspy subhalos).",
        styles["Body"]))

    story.append(Paragraph("12.5  Outlook", styles["SubHead"]))
    story.append(Paragraph(
        "The immediate priorities are: (1) developing a comprehensive injection test framework "
        "for real Gaia data to quantify the simulation-to-real gap; (2) training with "
        "domain-randomised simulations that incorporate realistic observational systematics; "
        "(3) repeating the analysis with SDSS-V radial velocities when DR1 becomes available, "
        "which will add the critical 6th phase-space dimension for most northern streams; and "
        "(4) applying the framework to newly discovered streams from Gaia DR4. The tension between "
        "the rate model and GNN + SBI tension, once resolved, could provide important insights into "
        "both the nature of dark matter and the practical limits of simulation-based machine "
        "learning in astrophysics.",
        styles["Body"]))

    # -----------------------------------------------------------------------
    # 13. TIMELINE FORWARD MODEL AND SIM-TO-REAL ROBUSTNESS
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("13. Timeline Forward Model and Sim-to-Real Robustness",
                           styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "This section documents a forward-modelling capability that complements the count-based and "
        "GNN+SBI analyses above: instead of only asking <i>how many</i> gaps a stream has, it asks "
        "<i>which specific past encounter best reproduces the observed stream today</i>. The workflow "
        "is a literal rewind-and-replay: (1) use the detector to decide whether there is a probable "
        "impact and estimate its time; (2) generate the un-impacted past stream; (3) inject candidate "
        "subhalo impacts at the estimated epoch; (4) integrate every candidate forward through the "
        "Milky Way potential to the present day; and (5) score each candidate against the real stream "
        "and assign a statistical significance. The work below also resolves a serious sim-to-real "
        "failure of the detector that had made it unusable on real data.",
        styles["Body"]))

    add_figure(story, fig_paths["timeline"],
        "<b>Figure 15.</b> The timeline forward model. The detector localises and dates a probable "
        "impact; candidate subhalo encounters are applied to the reconstructed past stream and "
        "evolved forward via orbit integration to today; each candidate is scored against the real "
        "stream (density, gaps, proper-motion track, and radial velocity) and compared to a "
        "no-impact null distribution.", styles, width=6.0*inch)

    story.append(Paragraph("13.1  Detection -> Timeline Handoff", styles["SubHead"]))
    story.append(Paragraph(
        "The front end runs the trained detector on the observed stream to obtain an impact "
        "probability and a time-since-impact estimate, and a model-free, prominence-ranked "
        "density-minimum finder to localise the gap. These seed the forward-model grid: candidate "
        "impact longitudes are drawn from the detected minima rather than a blind sweep, and the "
        "time grid is focused on the estimated epoch. Because a subhalo cannot strike before the "
        "stream formed, the time estimate is capped at the stream's disruption age (for GD-1, the "
        "network's 5.9 Gyr estimate is capped at 3 Gyr).",
        styles["Body"]))

    story.append(Paragraph("13.2  Erkal &amp; Belokurov (2015) Velocity Kick", styles["SubHead"]))
    story.append(Paragraph(
        "The subhalo fly-by is modelled with the Erkal &amp; Belokurov (2015) Plummer impulse, "
        "Dv = (2GM/w) p / (|p|<super>2</super> + r<sub>s</sub><super>2</super>), where p is each "
        "star's true three-dimensional perpendicular offset to the subhalo's straight-line "
        "trajectory, w is the relative speed, and r<sub>s</sub> the Plummer scale radius. This "
        "replaces an earlier hand-rolled kick that used a Gaussian-in-longitude localisation, a "
        "fixed kick axis, an arbitrary 30% along-stream fraction, and a 50 km/s cap. The Erkal form "
        "supplies the correct per-star direction and magnitude from the geometry and is naturally "
        "bounded (it peaks at |p| = r<sub>s</sub> and equals GM/(w r<sub>s</sub>) there), so no cap "
        "is needed and the impact parameter and relative velocity become physical, scannable "
        "parameters.", styles["Body"]))

    add_figure(story, fig_paths["erkal"],
        "<b>Figure 16.</b> The Erkal &amp; Belokurov (2015) Plummer impulse (green) is finite "
        "everywhere and peaks at a perpendicular distance equal to the subhalo scale radius, whereas "
        "the previous point-mass heuristic (red dashed) diverges at small distances and had to be "
        "capped at 50 km/s. Both shown for a 10<super>8</super> M<sub>sun</sub> perturber at "
        "w = 200 km/s.", styles)

    story.append(Paragraph("13.3  Diagnosing and Fixing Detector Over-Confidence", styles["SubHead"]))
    story.append(Paragraph(
        "On real GD-1 the detector initially returned p_impact = 1.0000 for everything. The cause was "
        "not the model but the input pipeline: the simulations had near-constant per-star measurement "
        "errors, so the feature normaliser clamped those columns' standard deviation to 0.01, and any "
        "real-data offset was amplified to ~100 sigma, saturating the logit (a logit of ~250 versus a "
        "sim maximum of ~41). In-distribution the model was healthy (validation AUC 0.93, temperature "
        "1.06). The fix has two parts. First, at inference the detector now uses the real per-star "
        "errors, imputes unmeasured features (e.g. GD-1's absent radial velocity) to the training "
        "mean, clips standardised features to +/-5 sigma, and reports an out-of-distribution flag. "
        "Second, and more fundamentally, the detector was retrained with error domain randomisation: "
        "realistic, varying per-star errors (and radial-velocity masking) are drawn so the error "
        "features are no longer near-constant.",
        styles["Body"]))

    add_figure(story, fig_paths["ood_fix"],
        "<b>Figure 17.</b> Left: error domain randomisation restores real spread to the radial-"
        "velocity error feature (normaliser std 0.01 -> 1.95), the feature responsible for the "
        "saturation. Right: on real GD-1 the retrained detector reduces the input out-of-distribution "
        "level from 391 sigma to 13 sigma (29x) and turns a saturated p_impact = 0.98 into a "
        "meaningful 0.16.", styles)

    story.append(Paragraph(
        "The retrained detector retains in-distribution discrimination (validation binary accuracy "
        "0.879, AUC 0.937) while being far better calibrated on real data: the residual 13 sigma is a "
        "handful of outlier stars rather than a systematic shift (the systematic offsets are only "
        "~2 sigma), so the network now gives an honest, conservative probability instead of a "
        "saturated artefact. This is the recommended detector going forward.",
        styles["Body"]))

    story.append(Paragraph("13.4  GD-1 Gap Localisation and the Frame Transform", styles["SubHead"]))
    story.append(Paragraph(
        "GD-1's documented density gap lies at phi1 ~ -40 deg in the Koposov-2010 / Price-Whelan &amp; "
        "Bonaca (2018) stream frame, but the pipeline (via galstreams) works in the Ibata-2021 frame, "
        "where phi1 spans roughly 0-78 deg. The config gap locations were therefore in the wrong frame "
        "and unusable. Defining the Koposov-2010 rotation explicitly and transforming through ICRS, "
        "the documented gap at -40 maps to phi1 ~ 30.7 deg in the pipeline frame, within about 5 deg "
        "of the deepest data-driven density minimum (phi1 ~ 36 deg) found independently by the "
        "prominence-based finder. Two independent methods thus agree that the pipeline localises GD-1's "
        "real gap; it merely appears shallow (~10-15% depth) because the available membership catalog "
        "has uniform membership probabilities and is diluted by contamination.",
        styles["Body"]))

    story.append(Paragraph("13.5  Multi-Epoch / Multi-Survey Data Fusion", styles["SubHead"]))
    story.append(Paragraph(
        "Backward orbit integration diverges as dx(t) ~ dv t, so the precision of the rewind is set by "
        "the present-day velocity precision  -  and radial velocity, the sixth phase-space dimension, "
        "is entirely absent from the base Gaia astrometric catalogs. Real radial velocities are fused "
        "in from public surveys by Gaia source-id cross-match: the S5 survey (Li et al. 2019; VizieR "
        "J/MNRAS/490/3508) supplies 296 ATLAS and 257 Jhelum members, and Gaia DR3 RVS adds the "
        "brightest members of any stream. Catalogs are combined by inverse-variance weighting, and a "
        "Gaia DR2 second proper-motion epoch is retrieved for cross-checking. The projected gain from "
        "future releases scales as sigma ~ baseline<super>-1.5</super> (Gaia DR4 ~2.7x, DR5 ~6.7x "
        "tighter than DR3). The radial-velocity scoring term is made insensitive to a constant "
        "line-of-sight zero point so that only the differential perturbation signature  -  not a "
        "heliocentric-versus-galactocentric convention difference  -  is scored.",
        styles["Body"]))

    add_figure(story, fig_paths["multiepoch_rv"],
        "<b>Figure 18.</b> Real radial velocities fused into the stream catalogs from public surveys. "
        "S5 covers the southern streams (ATLAS, Jhelum); Gaia DR3 RVS adds bright members. GD-1, "
        "Pal 5 and Orphan fall outside the S5 footprint and require APOGEE/DESI cross-matching.",
        styles)

    story.append(Paragraph("13.6  Validation: Injection-Recovery and Statistical Significance",
                           styles["SubHead"]))
    story.append(Paragraph(
        "Two validations make the forward model trustworthy. In injection-recovery, a subhalo impact "
        "with known mass, time and longitude is injected into a synthetic stream and the full pipeline "
        "is run on it; the recovered mass and longitude land within one grid step of the truth, while "
        "the time is partly degenerate with mass (a known physical effect). For significance, the "
        "best-fitting candidate's score is compared to a null distribution built from many "
        "unperturbed (no-impact) realisations, yielding a z-score and an empirical p-value. On a "
        "known injected impact the best candidate sits 3.3 sigma below the no-impact null, whereas an "
        "unperturbed control sits at 0.6 sigma  -  the test correctly separates a real impact from "
        "noise. Candidate scores are additionally averaged over multiple random seeds so the ranking "
        "reflects physics rather than sampling noise.",
        styles["Body"]))

    add_figure(story, fig_paths["significance"],
        "<b>Figure 19.</b> Statistical significance against a no-impact null distribution (grey). A "
        "known injected impact (green) scores 3.3 sigma better than the null, while an unperturbed "
        "control stream (orange dashed) is consistent with the null at 0.6 sigma. A look-elsewhere "
        "correction (null distribution of best-of-grid scores) is the next refinement.",
        styles)

    story.append(Paragraph("13.7  Uncertainty-Aware Impact Time", styles["SubHead"]))
    story.append(Paragraph(
        "Because backward integration amplifies velocity errors, the impact time should carry an "
        "uncertainty that reflects the present-day measurement precision. A Monte-Carlo posterior "
        "does exactly this: the observed kinematics (proper motions, radial velocity, distance) are "
        "resampled within their per-star errors, the candidate grid is re-fit on each realization, "
        "and the spread of best-fit times is the impact-time posterior. On an injected 1.5 Gyr "
        "impact the posterior is centred on the truth and its width scales directly with the error "
        "level: zero error reduces to the deterministic fit (zero spread), while the posterior "
        "standard deviation grows from ~0.11 Gyr at current precision to ~0.35 Gyr at doubled "
        "errors. Equivalently, the multi-epoch radial velocities and future tighter proper motions "
        "(error scale below one) sharpen the dated impact.",
        styles["Body"]))

    add_figure(story, fig_paths["mc_time"],
        "<b>Figure 20.</b> Monte-Carlo impact-time posterior width versus the measurement-error "
        "scale (1.0 = current data). The recovered time-since-impact uncertainty shrinks with the "
        "measurement errors and vanishes in the zero-error limit, quantifying how multi-epoch data "
        "precision propagates into the precision of the dated encounter.", styles)

    story.append(Paragraph(
        "Taken together, these changes move the early pipeline steps from approximate and "
        "sim-to-real-fragile toward physically grounded and validated: a closed-form fly-by impulse, "
        "a detector that behaves on real data, gap localisation confirmed by an independent frame "
        "transform, real multi-survey radial velocities, calibrated (including look-elsewhere-"
        "corrected) significance, full-orbit injection-recovery, and an uncertainty-aware impact-time "
        "posterior. Every component is covered by automated tests; the project test suite passes "
        "270+ checks. The look-elsewhere correction is sobering: for a single short stream with "
        "realistic kicks the formal detection significance is marginal even when the encounter "
        "parameters are recoverable, which motivates combining many streams, longer baselines, and "
        "the added radial-velocity dimension.",
        styles["Body"]))

    # -----------------------------------------------------------------------
    # REFERENCES
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("References", styles["SectionHead"]))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))

    refs = [
        "Bailer-Jones, C. A. L., et al. 2021, AJ, 161, 147",
        "Banik, N., et al. 2021, JCAP, 2021, 043",
        "Bonaca, A., et al. 2019, ApJ, 880, 38",
        "Bonaca, A., et al. 2020, ApJ, 892, L37",
        "Bovy, J. 2015, ApJS, 216, 29 (galpy)",
        "Bovy, J., Erkal, D., & Sanders, J. L. 2017, MNRAS, 466, 628",
        "Dalal, N., et al. 2022, arXiv:2203.05750",
        "de Boer, T. J. L., et al. 2020, MNRAS, 494, 5315",
        "de Jong, R. S., et al. 2019, The Messenger, 175, 3 (4MOST)",
        "Erkal, D. &amp; Belokurov, V. 2015, MNRAS, 450, 1136 (subhalo-stream impulse)",
        "Erkal, D., et al. 2017, MNRAS, 470, 60",
        "Gaia Collaboration, 2023, A&amp;A, 674, A1 (DR3)",
        "Garrison-Kimmel, S., et al. 2017, MNRAS, 471, 1709",
        "Gilman, D., et al. 2020, MNRAS, 491, 6077",
        "Hsueh, J.-W., et al. 2020, MNRAS, 492, 3047",
        "Hui, L., Ostriker, J. P., Tremaine, S., & Witten, E. 2017, PRD, 95, 043541",
        "Ibata, R. A., et al. 2021, ApJ, 914, 123",
        "Irsic, V., et al. 2017, PRD, 96, 023522",
        "Kaplinghat, M., Tulin, S., & Yu, H.-B. 2016, PRL, 116, 041302",
        "Kass, R. E. & Raftery, A. E. 1995, JASA, 90, 773",
        "Kollmeier, J. A., et al. 2017, arXiv:1711.03234 (SDSS-V)",
        "Koposov, S. E., et al. 2019, MNRAS, 485, 4726",
        "Li, T. S., et al. 2019, MNRAS, 490, 3508 (S5 survey RVs)",
        "Li, T. S., et al. 2021, ApJ, 911, 149",
        "Lovell, M. R., et al. 2014, MNRAS, 439, 300",
        "Price-Whelan, A. M. &amp; Bonaca, A. 2018, ApJ, 863, L20 (GD-1 gap/spur)",
        "Ludlow, A. D., et al. 2016, MNRAS, 460, 1214",
        "Nadler, E. O., et al. 2021, PRL, 126, 091101",
        "Satorras, V. G., Hoogeboom, E., & Welling, M. 2021, ICML (EGNN)",
        "Schive, H.-Y., Chiueh, T., & Broadhurst, T. 2014, Nature Physics, 10, 496",
        "Schneider, A., et al. 2012, MNRAS, 424, 684",
        "Springel, V., et al. 2008, MNRAS, 391, 1685 (Aquarius)",
        "Xu, K., Hu, W., Leskovec, J., & Jegelka, S. 2019, ICLR (GIN/GINEConv)",
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle("Ref", parent=styles["Body"],
                     fontSize=8.5, leading=11, spaceAfter=2, leftIndent=18,
                     firstLineIndent=-18)))

    # -----------------------------------------------------------------------
    # FINAL FOOTER
    # -----------------------------------------------------------------------
    story.append(Spacer(1, 30))
    story.append(ThinRule(W, 0.75, ACCENT_LIGHT))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Stellar Stream DM  -  Built with PyTorch Geometric, sbi, NumPyro, emcee<br/>"
        "Focused implementation checks run where dependencies are available | CDM normalisation: 2.1 "
        "(Bonaca+2019) | Suppression prior: Uniform[4.0, 10.0]",
        styles["Footer"]))

    # Build
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"PDF generated: {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    build_pdf()
