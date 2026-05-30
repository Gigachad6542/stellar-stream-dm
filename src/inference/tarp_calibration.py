"""
TARP-inspired post-hoc recalibration for SBI posteriors.

The neural posteriors from SNPE-C are amortized approximations that typically
have coverage errors (overconfident or underconfident). This module implements:

1. **Empirical coverage diagnosis**: compute the actual coverage at many nominal
   levels using a held-out calibration set.
2. **Isotonic recalibration**: learn a monotonic mapping from nominal to empirical
   coverage, then apply it to adjust credible intervals.
3. **Per-parameter correction**: each parameter dimension gets its own recalibration
   curve since miscalibration patterns differ.

References:
    Lemos+2023: "Sampling-Based Accuracy Testing of Posterior Estimators for
                 General Inference" (arxiv:2302.03026)
    Hermans+2022: "A Trust Crisis In Simulation-Based Inference"
    Zhao+2021: "Diagnostics for Conditional Density Estimators and Bayesian
                Inference Algorithms"

Usage:
    from src.inference.tarp_calibration import TARPRecalibrator

    # Build recalibrator from SBC ranks
    recal = TARPRecalibrator.from_sbc_ranks(ranks, n_posterior_samples=500)

    # Or build from (theta, embedding) calibration set
    recal = TARPRecalibrator.from_calibration_set(
        posterior, theta_cal, emb_cal, n_posterior_samples=1000
    )

    # Apply to get corrected credible intervals
    corrected_ci = recal.corrected_credible_interval(
        posterior_samples, nominal_level=0.90
    )
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy import interpolate, stats

log = logging.getLogger(__name__)


class TARPRecalibrator:
    """Post-hoc recalibration using empirical coverage curves.

    For each parameter dimension, stores a monotonic mapping from
    nominal coverage level → corrected coverage level.

    If the posterior is overconfident (common): a nominal 90% CI actually
    covers only 75% of the time. The recalibrator learns this and tells you
    to use a wider (e.g., 96%) CI to get true 90% coverage.
    """

    def __init__(
        self,
        nominal_levels: np.ndarray,
        empirical_coverages: np.ndarray,
        param_names: list[str] | None = None,
    ):
        """
        Args:
            nominal_levels: [L] array of nominal CI levels (e.g., 0.01, 0.02, ..., 0.99)
            empirical_coverages: [L, n_params] actual coverage at each nominal level
            param_names: optional names for each parameter dimension
        """
        self.nominal_levels = nominal_levels
        self.empirical_coverages = empirical_coverages
        self.n_params = empirical_coverages.shape[1]
        self.param_names = param_names or [f"param_{i}" for i in range(self.n_params)]

        # Build interpolation functions: empirical → nominal (for correction)
        # Given desired empirical coverage α, what nominal level to request?
        self._correction_fns = []
        for j in range(self.n_params):
            emp = empirical_coverages[:, j]
            # Ensure monotonicity via isotonic regression
            emp_monotone = np.maximum.accumulate(emp)
            # Interpolate: given desired true coverage, return nominal level to use
            # (inverse of the empirical coverage function)
            fn = interpolate.interp1d(
                emp_monotone, nominal_levels,
                kind="linear", bounds_error=False,
                fill_value=(nominal_levels[0], nominal_levels[-1]),
            )
            self._correction_fns.append(fn)

        # Also store forward map: nominal → empirical (for diagnostics)
        self._forward_fns = []
        for j in range(self.n_params):
            emp = empirical_coverages[:, j]
            fn = interpolate.interp1d(
                nominal_levels, emp,
                kind="linear", bounds_error=False,
                fill_value=(emp[0], emp[-1]),
            )
            self._forward_fns.append(fn)

    @classmethod
    def from_sbc_ranks(
        cls,
        ranks: np.ndarray,
        n_posterior_samples: int,
        param_names: list[str] | None = None,
        n_levels: int = 99,
    ) -> "TARPRecalibrator":
        """Build recalibrator from SBC rank statistics.

        The SBC ranks are the rank of θ_true among n_posterior_samples draws.
        For a calibrated posterior, ranks ~ Uniform(0, n_posterior_samples).
        The empirical CDF of the normalized ranks gives the coverage curve.

        Args:
            ranks: [N, n_params] integer ranks from SBC.
            n_posterior_samples: number of posterior samples used in SBC.
            param_names: optional parameter names.
            n_levels: number of nominal levels to evaluate (default 99 → 0.01..0.99).
        """
        # Normalize ranks to [0, 1]
        normalized_ranks = ranks / n_posterior_samples  # [N, n_params]

        nominal_levels = np.linspace(0.01, 0.99, n_levels)
        n_params = ranks.shape[1]
        empirical_coverages = np.zeros((n_levels, n_params))

        for j in range(n_params):
            for i, alpha in enumerate(nominal_levels):
                # For HPD-style coverage: fraction of ranks in [alpha/2, 1-alpha/2]
                # But simpler: fraction of normalized ranks ≤ alpha
                # Actually, for credible interval coverage:
                # A nominal α CI covers if the rank is in the "central" α fraction
                # i.e., rank/(N+1) is in [(1-α)/2, (1+α)/2]
                lo = (1.0 - alpha) / 2.0
                hi = (1.0 + alpha) / 2.0
                in_ci = (normalized_ranks[:, j] >= lo) & (normalized_ranks[:, j] <= hi)
                empirical_coverages[i, j] = in_ci.mean()

        log.info("Built TARP recalibrator from %d SBC trials", len(ranks))
        for j in range(n_params):
            name = param_names[j] if param_names else f"param_{j}"
            # Report calibration at key levels
            for target in [0.68, 0.90, 0.95]:
                idx = np.argmin(np.abs(nominal_levels - target))
                emp = empirical_coverages[idx, j]
                log.info("  %s: nominal %.0f%% CI -> actual %.1f%% coverage",
                         name, target * 100, emp * 100)

        return cls(nominal_levels, empirical_coverages, param_names)

    @classmethod
    def from_calibration_set(
        cls,
        posterior,
        theta_cal: torch.Tensor,
        embeddings_cal: torch.Tensor,
        n_posterior_samples: int = 1000,
        param_names: list[str] | None = None,
        n_levels: int = 99,
    ) -> "TARPRecalibrator":
        """Build recalibrator by running coverage test on calibration set.

        This is more accurate than from_sbc_ranks because it uses more
        posterior samples per trial.

        Args:
            posterior: trained sbi posterior object.
            theta_cal: [N, n_params] ground-truth parameters.
            embeddings_cal: [N, D] pre-computed GNN embeddings.
            n_posterior_samples: samples per trial (more = better calibration estimate).
            param_names: optional parameter names.
            n_levels: number of nominal levels.
        """
        n_cal = len(theta_cal)
        n_params = theta_cal.shape[1]
        nominal_levels = np.linspace(0.01, 0.99, n_levels)
        empirical_coverages = np.zeros((n_levels, n_params))

        # Detect device
        try:
            _net = posterior.posterior_estimator if hasattr(posterior, "posterior_estimator") \
                   else posterior.net
            _device = next(_net.parameters()).device
        except (StopIteration, AttributeError):
            _device = torch.device("cpu")

        log.info("Computing coverage on %d calibration samples (%d posterior draws each)...",
                 n_cal, n_posterior_samples)

        # For each calibration sample, compute quantile of true theta in posterior
        quantiles = np.zeros((n_cal, n_params))
        for i in range(n_cal):
            x_obs = embeddings_cal[i].to(_device)
            with torch.no_grad():
                samples = posterior.sample(
                    (n_posterior_samples,),
                    x=x_obs,
                    show_progress_bars=False,
                )
            samples_np = samples.cpu().numpy()
            theta_np = theta_cal[i].cpu().numpy()

            # Compute quantile of theta_true in marginal posterior for each param
            for j in range(n_params):
                quantiles[i, j] = np.mean(samples_np[:, j] < theta_np[j])

            if (i + 1) % 100 == 0:
                log.info("  Calibration sample %d/%d", i + 1, n_cal)

        # Compute empirical coverage at each nominal level
        for j in range(n_params):
            for idx, alpha in enumerate(nominal_levels):
                lo = (1.0 - alpha) / 2.0
                hi = (1.0 + alpha) / 2.0
                in_ci = (quantiles[:, j] >= lo) & (quantiles[:, j] <= hi)
                empirical_coverages[idx, j] = in_ci.mean()

        log.info("Coverage computation complete.")
        for j in range(n_params):
            name = param_names[j] if param_names else f"param_{j}"
            for target in [0.68, 0.90, 0.95]:
                idx_t = np.argmin(np.abs(nominal_levels - target))
                emp = empirical_coverages[idx_t, j]
                log.info("  %s: nominal %.0f%% CI -> actual %.1f%% coverage",
                         name, target * 100, emp * 100)

        return cls(nominal_levels, empirical_coverages, param_names)

    def correct_level(self, desired_coverage: float, param_idx: int = 0) -> float:
        """Given a desired true coverage level, return the nominal level to request.

        Example: if posteriors are overconfident and you want true 90% coverage,
        this might return 0.96 (meaning you need to use the 96% nominal CI
        to actually cover 90% of the time).

        Args:
            desired_coverage: the true coverage you want (e.g., 0.90)
            param_idx: which parameter dimension (default 0)

        Returns:
            The nominal level to request from the posterior.
        """
        return float(self._correction_fns[param_idx](desired_coverage))

    def actual_coverage(self, nominal_level: float, param_idx: int = 0) -> float:
        """Given a nominal CI level, return the actual empirical coverage.

        Example: if posteriors are overconfident, nominal 90% might actually
        cover only 75%.

        Args:
            nominal_level: the nominal CI level (e.g., 0.90)
            param_idx: which parameter dimension (default 0)

        Returns:
            The actual coverage at that nominal level.
        """
        return float(self._forward_fns[param_idx](nominal_level))

    def corrected_credible_interval(
        self,
        posterior_samples: np.ndarray,
        desired_coverage: float = 0.90,
    ) -> np.ndarray:
        """Compute corrected credible intervals from posterior samples.

        Args:
            posterior_samples: [N, n_params] samples from the posterior.
            desired_coverage: target true coverage level (e.g., 0.90)

        Returns:
            [n_params, 2] array of (lo, hi) bounds for each parameter.
        """
        n_params = posterior_samples.shape[1]
        intervals = np.zeros((n_params, 2))

        for j in range(n_params):
            # Get corrected nominal level for this parameter
            corrected_nominal = self.correct_level(desired_coverage, param_idx=j)
            alpha = 1.0 - corrected_nominal
            lo_q = alpha / 2.0
            hi_q = 1.0 - alpha / 2.0
            intervals[j, 0] = np.quantile(posterior_samples[:, j], lo_q)
            intervals[j, 1] = np.quantile(posterior_samples[:, j], hi_q)

        return intervals

    def calibration_summary(self) -> dict:
        """Return a summary of calibration diagnostics."""
        summary = {}
        for j, name in enumerate(self.param_names):
            param_summary = {}
            for target in [0.50, 0.68, 0.90, 0.95]:
                idx = np.argmin(np.abs(self.nominal_levels - target))
                emp = self.empirical_coverages[idx, j]
                corrected = self.correct_level(target, param_idx=j)
                param_summary[f"nominal_{int(target*100)}"] = {
                    "actual_coverage": float(emp),
                    "correction_needed": float(corrected),
                    "bias": float(emp - target),
                }
            summary[name] = param_summary
        return summary

    def save(self, path: str | Path) -> None:
        """Save recalibrator to disk."""
        path = Path(path)
        data = {
            "nominal_levels": self.nominal_levels,
            "empirical_coverages": self.empirical_coverages,
            "param_names": self.param_names,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        log.info("Saved TARP recalibrator to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "TARPRecalibrator":
        """Load recalibrator from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            nominal_levels=data["nominal_levels"],
            empirical_coverages=data["empirical_coverages"],
            param_names=data["param_names"],
        )

    def plot_coverage(self, output_path: str | None = None) -> None:
        """Plot coverage diagnostic: nominal vs empirical coverage."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, self.n_params, figsize=(5 * self.n_params, 4))
        if self.n_params == 1:
            axes = [axes]

        for j, (ax, name) in enumerate(zip(axes, self.param_names)):
            ax.plot(
                self.nominal_levels, self.empirical_coverages[:, j],
                "b-", linewidth=2, label="Empirical",
            )
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
            ax.fill_between(
                self.nominal_levels,
                self.nominal_levels - 0.03,
                self.nominal_levels + 0.03,
                alpha=0.15, color="gray", label="±3% tolerance",
            )
            ax.set_xlabel("Nominal coverage")
            ax.set_ylabel("Empirical coverage")
            ax.set_title(f"{name}")
            ax.legend(loc="lower right")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        plt.suptitle("TARP Coverage Diagnostic", fontsize=14, y=1.02)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            log.info("Coverage plot saved to %s", output_path)
        plt.close(fig)
