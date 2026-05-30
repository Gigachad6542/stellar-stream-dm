"""
Publication-quality figures for the stellar stream DM analysis.

Generates:
1. Stream density profiles with detected gaps overlaid
2. Corner plots of per-stream posteriors
3. Multi-stream posterior combination plot
4. Subhalo mass function constraints with model predictions
5. Gap classification pie charts / probability bars
6. GNN embedding PCA coloured by DM model
7. Suppression scale constraint (M_hm posterior + particle mass exclusion)
8. Power spectrum comparison (observed vs CDM)
9. Posterior predictive check results (PPC)
10. Gap classifier calibration and confusion matrix
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

log = logging.getLogger(__name__)

COLORS = {"CDM": "#1f77b4", "WDM": "#ff7f0e", "FDM": "#2ca02c", "SIDM": "#d62728"}


# ---------------------------------------------------------------------------
# Stream density profile
# ---------------------------------------------------------------------------

def plot_density_profile(
    phi1: np.ndarray,
    gaps: list | None = None,
    stream_name: str = "",
    phi1_range: tuple[float, float] = (-100.0, 20.0),
    n_bins: int = 200,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the stellar density along phi1 with gaps highlighted."""
    from .gap_catalog import compute_density_profile  # noqa: PLC0415

    bins = np.linspace(phi1_range[0], phi1_range[1], n_bins + 1)
    centers, density = compute_density_profile(phi1, n_bins, phi1_range)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.step(centers, density, where="mid", color="black", lw=0.8, alpha=0.8)
    ax.fill_between(centers, 0, density, step="mid", alpha=0.15, color="steelblue")

    if gaps:
        for gap in gaps:
            c = gap.phi1_center
            w = gap.phi1_width / 2.0
            color = "red" if gap.p_dm_subhalo > 0.5 else "orange"
            ax.axvspan(c - w, c + w, alpha=0.25, color=color,
                       label=f"{gap.significance:.1f}σ gap")

    ax.set_xlabel("φ₁ [deg]")
    ax.set_ylabel("Stars / bin")
    ax.set_title(f"{stream_name} density profile")
    ax.set_xlim(phi1_range)

    if output_path:
        fig.savefig(str(output_path), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Corner plot for posterior
# ---------------------------------------------------------------------------

def plot_corner(
    samples: pd.DataFrame,
    truths: list[float] | None = None,
    output_path: str | Path | None = None,
    title: str = "",
) -> "plt.Figure":
    """Corner plot of posterior samples using the `corner` package."""
    import corner  # noqa: PLC0415

    fig = corner.corner(
        samples.values,
        labels=list(samples.columns),
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".2f",
    )
    if title:
        fig.suptitle(title, y=1.02)
    if output_path:
        fig.savefig(str(output_path), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Multi-stream combined posterior
# ---------------------------------------------------------------------------

def plot_combined_posteriors(
    per_stream_samples: dict[str, pd.DataFrame],
    combined_result: dict,
    param_name: str = "log10_M_sub_mean",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Overlay per-stream posteriors and the combined posterior."""
    fig, ax = plt.subplots(figsize=(8, 4))

    from scipy.stats import gaussian_kde  # noqa: PLC0415

    colors = plt.cm.tab10(np.linspace(0, 1, len(per_stream_samples)))
    for (stream, samples), color in zip(per_stream_samples.items(), colors):
        if param_name not in samples.columns:
            continue
        vals = samples[param_name].values
        kde = gaussian_kde(vals)
        x = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 300)
        ax.plot(x, kde(x), label=stream, color=color, lw=1.5, alpha=0.7)

    # Plot combined
    if "combined" in combined_result and param_name in combined_result["combined"]:
        comb = combined_result["combined"][param_name]
        grid, log_post = comb["grid"], comb["log_posterior"]
        post = np.exp(log_post - log_post.max())
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        post /= _trapz(post, grid)
        ax.plot(grid, post, "k-", lw=2.5, label="Combined")

    ax.set_xlabel(param_name)
    ax.set_ylabel("Posterior density")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title("Per-stream and combined posterior")
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Subhalo mass function constraints
# ---------------------------------------------------------------------------

def plot_mass_function_constraints(
    samples_dict: dict[str, pd.DataFrame],
    param_name: str = "log10_M_sub_mean",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot posterior mass constraints alongside theoretical predictions."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for model, samples in samples_dict.items():
        if param_name not in samples.columns:
            continue
        vals = samples[param_name].values
        lo, med, hi = np.percentile(vals, [16, 50, 84])
        color = COLORS.get(model, "gray")
        ax.errorbar(
            x=[{"CDM": 0, "WDM": 1, "FDM": 2, "SIDM": 3}.get(model, 0)],
            y=[med],
            yerr=[[med - lo], [hi - med]],
            fmt="o", color=color, label=model, capsize=4, markersize=8,
        )

    ax.set_ylabel(f"Inferred {param_name}")
    ax.set_title("Subhalo mass constraints per DM model")
    ax.legend()
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["CDM", "WDM", "FDM", "SIDM"])
    ax.grid(axis="y", alpha=0.3)

    if output_path:
        fig.savefig(str(output_path), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    model_posterior_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of posterior model probabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: posterior probabilities
    ax = axes[0]
    models = model_posterior_df["model"].tolist()
    probs = model_posterior_df["posterior_probability"].tolist()
    colors = [COLORS.get(m, "gray") for m in models]
    ax.bar(models, probs, color=colors)
    ax.set_ylabel("Posterior probability")
    ax.set_title("DM model posterior")
    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5, label="Uniform prior")

    # Right: log Bayes factors
    ax2 = axes[1]
    bf_col = [c for c in model_posterior_df.columns if "log10_BF" in c]
    if bf_col:
        bfs = model_posterior_df[bf_col[0]].tolist()
        ax2.barh(models, bfs, color=colors)
        ax2.axvline(0, color="black", lw=0.8)
        ax2.axvline(1, color="orange", lw=0.8, linestyle="--", label="Strong (log10 BF=1)")
        ax2.axvline(2, color="red", lw=0.8, linestyle="--", label="Decisive (log10 BF=2)")
        ax2.set_xlabel(f"log₁₀ Bayes Factor vs CDM")
        ax2.set_title("Bayes factors vs CDM")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    if output_path:
        fig.savefig(str(output_path), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Embedding PCA
# ---------------------------------------------------------------------------

def plot_embedding_pca(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """PCA of GNN embeddings coloured by DM model label."""
    from sklearn.decomposition import PCA  # noqa: PLC0415

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, name in enumerate(label_names):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5,
                   color=list(COLORS.values())[i % 4], label=name)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("GNN embedding PCA")
    ax.legend(markerscale=3)

    if output_path:
        fig.savefig(str(output_path), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Suppression scale constraint
# ---------------------------------------------------------------------------

def plot_suppression_constraint(
    log10_M_hm_posterior: np.ndarray,
    upper_limit_95: float | None = None,
    particle_mass_limits: dict | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the half-mode mass posterior and derived particle mass exclusion.

    This is the paper's primary result figure: the model-independent M_hm
    constraint from combined multi-stream analysis.

    Args:
        log10_M_hm_posterior: [S] posterior samples of log10(M_hm / Msun).
        upper_limit_95: Optional 95% upper limit on log10(M_hm).
        particle_mass_limits: Optional dict with keys "m_wdm_keV_lower",
            "m_axion_eV_lower", "sigma_m_upper" for the secondary axis.
        output_path: If provided, save the figure.

    Returns:
        Matplotlib Figure.
    """
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: M_hm posterior
    ax = axes[0]
    kde = gaussian_kde(log10_M_hm_posterior)
    x = np.linspace(
        log10_M_hm_posterior.min() - 0.5,
        log10_M_hm_posterior.max() + 0.5,
        400,
    )
    density = kde(x)
    ax.fill_between(x, 0, density, alpha=0.3, color="steelblue")
    ax.plot(x, density, color="steelblue", lw=2)

    # Credible intervals
    lo, med, hi = np.percentile(log10_M_hm_posterior, [5, 50, 95])
    ax.axvline(med, color="navy", lw=1.5, ls="-", label=f"Median: {med:.2f}")
    ax.axvspan(lo, hi, alpha=0.1, color="navy", label=f"90% CI: [{lo:.2f}, {hi:.2f}]")

    if upper_limit_95 is not None:
        ax.axvline(upper_limit_95, color="red", lw=1.5, ls="--",
                   label=f"95% UL: {upper_limit_95:.2f}")

    ax.set_xlabel(r"$\log_{10}(M_{\rm hm} / M_\odot)$")
    ax.set_ylabel("Posterior density")
    ax.set_title("Half-mode mass constraint (model-independent)")
    ax.legend(fontsize=9)

    # Right panel: particle mass exclusion limits
    ax2 = axes[1]
    if particle_mass_limits:
        labels = []
        values = []
        colors_bar = []
        if "m_wdm_keV_lower" in particle_mass_limits:
            labels.append("WDM\n$m_{\\rm WDM}$")
            values.append(particle_mass_limits["m_wdm_keV_lower"])
            colors_bar.append(COLORS["WDM"])
        if "m_axion_eV_lower" in particle_mass_limits:
            labels.append("FDM\n$m_a$")
            values.append(np.log10(particle_mass_limits["m_axion_eV_lower"]))
            colors_bar.append(COLORS["FDM"])
        if "sigma_m_upper" in particle_mass_limits:
            labels.append("SIDM\n$\\sigma/m$")
            values.append(particle_mass_limits["sigma_m_upper"])
            colors_bar.append(COLORS["SIDM"])

        ax2.barh(labels, values, color=colors_bar, alpha=0.7, height=0.5)
        ax2.set_xlabel("95% exclusion limit")
        ax2.set_title("Particle mass limits")
        ax2.grid(axis="x", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No particle\nmass limits\nprovided",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11, color="gray")
        ax2.set_axis_off()

    plt.tight_layout()
    if output_path:
        fig.savefig(str(output_path), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Power spectrum comparison
# ---------------------------------------------------------------------------

def plot_power_spectrum(
    freqs: np.ndarray,
    power_observed: np.ndarray,
    power_cdm_mean: np.ndarray | None = None,
    power_cdm_std: np.ndarray | None = None,
    excess_mask: np.ndarray | None = None,
    stream_name: str = "",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the 1D angular power spectrum with CDM expectation band.

    Args:
        freqs: Frequency array [1/deg].
        power_observed: Observed power spectrum.
        power_cdm_mean: Mean CDM power spectrum from simulations.
        power_cdm_std: Standard deviation of CDM power spectra.
        excess_mask: Boolean mask of frequencies with significant excess.
        stream_name: For the title.
        output_path: If provided, save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.semilogy(freqs, power_observed, "k-", lw=1.5, label="Observed", zorder=3)

    if power_cdm_mean is not None:
        ax.semilogy(freqs, power_cdm_mean, color=COLORS["CDM"], lw=1, ls="--", label="CDM mean")
        if power_cdm_std is not None:
            ax.fill_between(
                freqs,
                np.maximum(power_cdm_mean - 2 * power_cdm_std, 1e-10),
                power_cdm_mean + 2 * power_cdm_std,
                alpha=0.2, color=COLORS["CDM"], label="CDM 2σ band",
            )

    if excess_mask is not None and np.any(excess_mask):
        ax.scatter(freqs[excess_mask], power_observed[excess_mask],
                   color="red", s=30, zorder=4, label="Significant excess")

    ax.set_xlabel("Spatial frequency [1/deg]")
    ax.set_ylabel("Power")
    ax.set_title(f"{stream_name} 1D angular power spectrum" if stream_name else "1D angular power spectrum")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    if output_path:
        fig.savefig(str(output_path), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Posterior predictive check (PPC)
# ---------------------------------------------------------------------------

def plot_ppc_results(
    summary_names: list[str],
    observed_stats: np.ndarray,
    simulated_stats: np.ndarray,
    p_values: np.ndarray | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot posterior predictive check results as rank histograms.

    For each summary statistic, shows the distribution of simulated values
    under the posterior and the position of the observed value.

    Args:
        summary_names: [K] names of summary statistics.
        observed_stats: [K] observed values.
        simulated_stats: [S, K] simulated values under the posterior.
        p_values: [K] two-sided p-values (optional, shown in subplot titles).
        output_path: If provided, save the figure.

    Returns:
        Matplotlib Figure.
    """
    k = len(summary_names)
    n_cols = min(k, 4)
    n_rows = (k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if k == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).ravel()

    for i, (name, obs_val) in enumerate(zip(summary_names, observed_stats)):
        ax = axes[i]
        sim_vals = simulated_stats[:, i]
        ax.hist(sim_vals, bins=30, density=True, alpha=0.5, color="steelblue",
                edgecolor="white", lw=0.5)
        ax.axvline(obs_val, color="red", lw=2, label="Observed")

        title = name
        if p_values is not None:
            title += f"\n(p = {p_values[i]:.3f})"
            if p_values[i] < 0.05:
                ax.set_facecolor("#fff3f3")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Posterior predictive checks", fontsize=13, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Gap classifier diagnostics
# ---------------------------------------------------------------------------

def plot_gap_classifier_diagnostics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: list[str] = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot confusion matrix and calibration curve for the gap classifier.

    Args:
        y_true: [N] true integer labels (0=DM, 1=baryonic, 2=noise).
        y_pred_proba: [N, 3] predicted probabilities.
        class_names: Names for each class.
        output_path: If provided, save the figure.

    Returns:
        Matplotlib Figure.
    """
    from sklearn.metrics import confusion_matrix  # noqa: PLC0415

    if class_names is None:
        class_names = ["DM subhalo", "Baryonic", "Noise"]

    y_pred = y_pred_proba.argmax(axis=1)
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: confusion matrix
    ax = axes[0]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (normalized)")
    for i_row in range(n_classes):
        for j_col in range(n_classes):
            color = "white" if cm_norm[i_row, j_col] > 0.5 else "black"
            ax.text(j_col, i_row, f"{cm_norm[i_row, j_col]:.2f}\n({cm[i_row, j_col]})",
                    ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Right: reliability / calibration diagram (one-vs-rest for each class)
    ax2 = axes[1]
    for cls_i, name in enumerate(class_names):
        probs = y_pred_proba[:, cls_i]
        truth = (y_true == cls_i).astype(int)

        # Bin predictions into calibration bins
        n_cal_bins = 10
        bin_edges = np.linspace(0, 1, n_cal_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_accs = np.zeros(n_cal_bins)
        bin_counts = np.zeros(n_cal_bins)

        for b in range(n_cal_bins):
            mask = (probs >= bin_edges[b]) & (probs < bin_edges[b + 1])
            if mask.sum() > 0:
                bin_accs[b] = truth[mask].mean()
                bin_counts[b] = mask.sum()

        valid = bin_counts > 0
        ax2.plot(bin_centers[valid], bin_accs[valid], "o-", label=name, markersize=4, lw=1.5)

    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Observed frequency")
    ax2.set_title("Calibration diagram")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(str(output_path), bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 11. Attention-based interpretability: per-star importance map
# ---------------------------------------------------------------------------

def plot_stream_attention(
    phi1: np.ndarray,
    phi2: np.ndarray,
    alpha: np.ndarray,
    stream_name: str = "",
    output_path: str | Path | None = None,
    cmap: str = "hot",
    figsize: tuple[float, float] = (12, 4),
) -> plt.Figure:
    """Plot a stellar stream coloured by GNN attention weights (node importance).

    Each star is plotted at its (phi1, phi2) coordinate with colour proportional
    to its attention weight. High-attention stars are the ones driving the GNN's
    DM model classification — typically stars near gap edges or velocity outliers.

    This visualization is unique to graph-based architectures; no power-spectrum
    or CNN method can produce per-star importance maps.

    Args:
        phi1: [N] stream coordinate (degrees along stream).
        phi2: [N] stream coordinate (degrees perpendicular to stream).
        alpha: [N] attention weights from GNN (sum to 1).
        stream_name: Stream label for the title.
        output_path: If provided, saves the figure to this path.
        cmap: Matplotlib colormap name.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by alpha so high-attention stars are plotted on top
    order = np.argsort(alpha)
    sc = ax.scatter(
        phi1[order], phi2[order],
        c=alpha[order], cmap=cmap,
        s=3, alpha=0.8, linewidths=0,
        vmin=0, vmax=np.percentile(alpha, 99),
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Attention weight (importance)")

    ax.set_xlabel(r"$\phi_1$ (deg)")
    ax.set_ylabel(r"$\phi_2$ (deg)")
    title = "GNN Node Importance"
    if stream_name:
        title += f" — {stream_name}"
    ax.set_title(title)
    ax.set_aspect("auto")

    # Annotate top-5 most important stars
    top_k = min(5, len(alpha))
    top_idx = np.argsort(alpha)[-top_k:]
    for idx in top_idx:
        ax.annotate(
            f"{alpha[idx]:.3f}",
            (phi1[idx], phi2[idx]),
            fontsize=7, color="white",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.6),
        )

    plt.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    return fig
