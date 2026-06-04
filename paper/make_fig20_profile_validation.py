#!/usr/bin/env python
"""
Profile-identifiability validation figure for the full report.

  fig20_profile_validation  matched streamgapdf injection/recovery screen:
    (a) predeclared gate metrics (family accuracy, detection recall) for the
        real-selection screen vs the idealised no-selection screen, against the
        frozen gate thresholds;
    (b) recovered vs true perturber scale-radius factor (clean, no-selection
        screen) showing the gap->profile degeneracy and the bias toward the
        intermediate ("cored_3x") family.

Reads the two screen outputs produced by
scripts/run_gd1_streamgapdf_injection_recovery.py.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("paper/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True, "figure.facecolor": "white"})
C_S, C_I, C_G, C_B = "#2c7fb8", "#d95f0e", "#31a354", "#de2d26"

WITHSEL = Path("outputs/profile_validation/quickcheck_fixed.json")
NOSEL = Path("outputs/profile_validation/quickcheck_noselection.json")


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {FIG/name}")


def fig_profile_validation():
    ws = json.loads(WITHSEL.read_text())
    ns = json.loads(NOSEL.read_text())
    gate_fam = ws["validation_contract"]["profile_family_accuracy_gate"]
    gate_rec = ws["validation_contract"]["detection_recall_gate"]

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.3))

    # (a) gate metrics vs thresholds
    labels = ["family\naccuracy", "detection\nrecall"]
    x = np.arange(len(labels)); w = 0.36
    ws_vals = [ws["summary"]["profile_family_accuracy"], ws["summary"]["detection_recall"]]
    ns_vals = [ns["summary"]["profile_family_accuracy"], ns["summary"]["detection_recall"]]
    ax[0].bar(x - w/2, ws_vals, w, color=C_S, label="real PWB18/DESI selection")
    ax[0].bar(x + w/2, ns_vals, w, color="#9ecae1", label="idealised (no selection)")
    for i, (a, b) in enumerate(zip(ws_vals, ns_vals)):
        ax[0].text(i - w/2, a + 0.015, f"{a:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax[0].text(i + w/2, b + 0.015, f"{b:.2f}", ha="center", fontsize=9, fontweight="bold")
    # gate threshold markers
    ax[0].plot([x[0]-0.5, x[0]+0.5], [gate_fam, gate_fam], "--", color=C_B, lw=1.6)
    ax[0].plot([x[1]-0.5, x[1]+0.5], [gate_rec, gate_rec], "--", color=C_B, lw=1.6)
    ax[0].text(x[0], gate_fam + 0.02, f"gate {gate_fam:.2f}", color=C_B, ha="center", fontsize=8.5)
    ax[0].text(x[1], gate_rec + 0.02, f"gate {gate_rec:.2f}", color=C_B, ha="center", fontsize=8.5)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("value"); ax[0].set_ylim(0, 1.0)
    ax[0].set_title("(a) Predeclared gates both fail")
    ax[0].legend(loc="upper right", fontsize=8.5)

    # (b) recovered vs true scale-radius factor (clean screen)
    truth_f, rec_f, detected = [], [], []
    for t in ns["all_tasks"]:
        truth_f.append(t["truth"]["scale_radius_factor"])
        rec_f.append(t["best_candidate"]["scale_radius_factor"])
        detected.append(bool(t.get("detected", False)))
    truth_f = np.array(truth_f, float); rec_f = np.array(rec_f, float)
    detected = np.array(detected, bool)
    rng = np.random.default_rng(0)
    jx = truth_f * (1 + rng.uniform(-0.06, 0.06, truth_f.size))
    jy = rec_f * (1 + rng.uniform(-0.06, 0.06, rec_f.size))
    lim = [0.35, 14]
    ax[1].plot(lim, lim, ":", color="#777", lw=1.3, label="perfect recovery (1:1)")
    ax[1].scatter(jx[~detected], jy[~detected], s=55, facecolors="none",
                  edgecolors="#999", linewidths=1.4, label="undetected (gap too shallow)")
    ax[1].scatter(jx[detected], jy[detected], s=60, color=C_I, label="detected")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ticks = [0.5, 1, 3, 10]
    ax[1].set_xticks(ticks); ax[1].set_yticks(ticks)
    ax[1].set_xticklabels([str(t) for t in ticks]); ax[1].set_yticklabels([str(t) for t in ticks])
    ax[1].set_xlabel("true scale-radius factor")
    ax[1].set_ylabel("recovered scale-radius factor")
    rmse = ns["summary"]["profile_scale_log10_rmse"]
    ax[1].set_title(f"(b) Scale radius is under-determined (clean RMSE {rmse:.2f} dex)")
    ax[1].legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Matched streamgapdf injection/recovery screen: perturber density profile is not identifiable",
                 y=1.02, fontsize=12)
    save(fig, "fig20_profile_validation.png")


if __name__ == "__main__":
    import traceback
    try:
        fig_profile_validation()
    except Exception:
        print("FAILED fig_profile_validation"); traceback.print_exc()
