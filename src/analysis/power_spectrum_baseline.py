"""
1D angular power spectrum baseline for stream gap detection.

This provides a traditional (non-ML) summary statistic approach for comparing
against the GNN+SBI pipeline. The angular power spectrum of the stream density
profile is sensitive to subhalo-induced perturbations at characteristic spatial
scales (Bovy+2017, Banik+2019).

Scientific motivation:
    - Power at high spatial frequencies (small angular scales, ~ 1-5 deg) is
      enhanced by subhalo impacts relative to smooth streams.
    - The power spectrum ratio P_observed / P_smooth isolates the perturbation
      signal from the intrinsic density gradient.
    - This is the standard "non-ML" comparison baseline in the literature
      (e.g. Bovy+2017 used this for GD-1 constraints).

Usage:
    Compute per-stream power spectra for observed and simulated streams, then
    compare using a chi-squared statistic or as a SBI summary statistic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PowerSpectrumResult:
    """Result of a 1D power spectrum computation."""
    frequencies: np.ndarray          # spatial frequencies [1/deg]
    power: np.ndarray                # power spectral density
    power_ratio: np.ndarray | None = None  # P_obs / P_smooth (if smooth provided)
    stream_name: str = ""
    n_stars: int = 0
    phi1_range: tuple[float, float] = (0.0, 0.0)
    n_bins: int = 0


# ---------------------------------------------------------------------------
# Core power spectrum computation
# ---------------------------------------------------------------------------

def compute_density_1d(
    phi1: np.ndarray,
    n_bins: int = 256,
    phi1_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute binned 1D stellar density along phi1.

    Args:
        phi1: Star positions in stream longitude [deg].
        n_bins: Number of evenly spaced bins. Power of 2 recommended for FFT.
        phi1_range: (min, max) extent. If None, uses data range + 5% padding.

    Returns:
        (bin_centers, density) where density is star counts per bin.
    """
    if phi1_range is None:
        margin = 0.05 * (phi1.max() - phi1.min())
        phi1_range = (phi1.min() - margin, phi1.max() + margin)

    bins = np.linspace(phi1_range[0], phi1_range[1], n_bins + 1)
    counts, _ = np.histogram(phi1, bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, counts.astype(np.float64)


def detrend_density(
    density: np.ndarray,
    method: str = "polynomial",
    poly_order: int = 3,
    smooth_sigma_bins: float = 30.0,
) -> np.ndarray:
    """Remove large-scale density gradient from the stream profile.

    The intrinsic density profile of a stream has a large-scale gradient
    (higher density near the progenitor). This must be removed before
    computing the power spectrum so that perturbation power dominates.

    Args:
        density: 1D density array (counts per bin).
        method: "polynomial" for polynomial fit, "smooth" for broad Gaussian.
        poly_order: Polynomial order for the background model.
        smooth_sigma_bins: Sigma of Gaussian for "smooth" method.

    Returns:
        Residual density (density - background).
    """
    if method == "polynomial":
        x = np.arange(len(density), dtype=float)
        # Mask zeros to avoid fitting empty regions
        mask = density > 0
        if mask.sum() < poly_order + 1:
            return density - np.mean(density)
        coeffs = np.polyfit(x[mask], density[mask], poly_order)
        background = np.polyval(coeffs, x)
    elif method == "smooth":
        background = gaussian_filter1d(density, sigma=smooth_sigma_bins)
    else:
        raise ValueError(f"Unknown detrend method: {method}")

    residual = density - background
    return residual


def compute_power_spectrum(
    phi1: np.ndarray,
    n_bins: int = 256,
    phi1_range: tuple[float, float] | None = None,
    detrend: str = "polynomial",
    window: str = "hann",
    normalize: bool = True,
) -> PowerSpectrumResult:
    """Compute the 1D angular power spectrum of a stream's density profile.

    Pipeline:
        1. Bin stellar positions into 1D density profile.
        2. Detrend to remove large-scale gradient.
        3. Apply window function to reduce spectral leakage.
        4. Compute FFT and power spectral density.

    Args:
        phi1: Star phi1 positions [deg].
        n_bins: Number of bins (power of 2 recommended).
        phi1_range: (min, max) phi1 range. None = auto.
        detrend: Detrending method ("polynomial", "smooth", or "none").
        window: Window function ("hann", "blackman", or "none").
        normalize: If True, normalize power by total power (relative PSD).

    Returns:
        PowerSpectrumResult with frequencies and power arrays.
    """
    if len(phi1) < 10:
        log.warning("Too few stars (%d) for meaningful power spectrum", len(phi1))
        return PowerSpectrumResult(
            frequencies=np.array([]),
            power=np.array([]),
            stream_name="",
            n_stars=len(phi1),
        )

    centers, density = compute_density_1d(phi1, n_bins, phi1_range)

    if phi1_range is None:
        margin = 0.05 * (phi1.max() - phi1.min())
        phi1_range = (phi1.min() - margin, phi1.max() + margin)

    bin_width_deg = (phi1_range[1] - phi1_range[0]) / n_bins

    # Detrend
    if detrend != "none":
        residual = detrend_density(density, method=detrend)
    else:
        residual = density - np.mean(density)

    # Window function
    if window == "hann":
        w = np.hanning(n_bins)
    elif window == "blackman":
        w = np.blackman(n_bins)
    elif window == "none":
        w = np.ones(n_bins)
    else:
        raise ValueError(f"Unknown window: {window}")

    windowed = residual * w

    # FFT (real-valued input -> rfft)
    fft_vals = np.fft.rfft(windowed)
    power = np.abs(fft_vals) ** 2

    # Frequency axis: cycles per degree
    freqs = np.fft.rfftfreq(n_bins, d=bin_width_deg)

    # Drop DC component
    freqs = freqs[1:]
    power = power[1:]

    # Normalize
    if normalize and power.sum() > 0:
        power = power / power.sum()

    return PowerSpectrumResult(
        frequencies=freqs,
        power=power,
        stream_name="",
        n_stars=len(phi1),
        phi1_range=phi1_range,
        n_bins=n_bins,
    )


# ---------------------------------------------------------------------------
# Power spectrum ratio (observed / smooth model)
# ---------------------------------------------------------------------------

def compute_power_ratio(
    observed_phi1: np.ndarray,
    smooth_phi1: np.ndarray,
    n_bins: int = 256,
    phi1_range: tuple[float, float] | None = None,
    floor: float = 1e-10,
) -> PowerSpectrumResult:
    """Compute the ratio of observed to smooth-stream power spectra.

    The ratio P_obs(k) / P_smooth(k) isolates the signal from subhalo
    impacts. Values > 1 at high spatial frequencies indicate perturbation
    power above what the smooth model predicts.

    Args:
        observed_phi1: Observed star positions.
        smooth_phi1: Smooth (unperturbed) model star positions.
        n_bins: Number of bins.
        phi1_range: Shared phi1 range for both spectra.
        floor: Minimum denominator value to avoid division by zero.

    Returns:
        PowerSpectrumResult with power_ratio filled.
    """
    ps_obs = compute_power_spectrum(
        observed_phi1, n_bins=n_bins, phi1_range=phi1_range, normalize=False,
    )
    ps_smooth = compute_power_spectrum(
        smooth_phi1, n_bins=n_bins, phi1_range=phi1_range, normalize=False,
    )

    if len(ps_obs.power) == 0 or len(ps_smooth.power) == 0:
        return ps_obs

    ratio = ps_obs.power / np.maximum(ps_smooth.power, floor)

    return PowerSpectrumResult(
        frequencies=ps_obs.frequencies,
        power=ps_obs.power,
        power_ratio=ratio,
        stream_name=ps_obs.stream_name,
        n_stars=ps_obs.n_stars,
        phi1_range=ps_obs.phi1_range,
        n_bins=n_bins,
    )


# ---------------------------------------------------------------------------
# Summary statistics from the power spectrum
# ---------------------------------------------------------------------------

def power_spectrum_summary_vector(
    phi1: np.ndarray,
    n_bins: int = 256,
    phi1_range: tuple[float, float] | None = None,
    n_bandpowers: int = 10,
) -> np.ndarray:
    """Compress the power spectrum into a fixed-length summary vector.

    Divides the frequency range into n_bandpowers log-spaced bins and
    averages the power in each. This gives a compact summary suitable for
    use as a SBI observation vector (alternative to GNN embeddings).

    This is the approach used by Bovy+2017 and Banik+2019 to obtain
    constraints on the subhalo mass function from stream density data.

    Args:
        phi1: Star positions [deg].
        n_bins: FFT bins.
        phi1_range: Range for binning.
        n_bandpowers: Number of output bandpower bins.

    Returns:
        [n_bandpowers] array of log10(bandpower) values.
        Returns zeros if computation fails.
    """
    ps = compute_power_spectrum(phi1, n_bins=n_bins, phi1_range=phi1_range, normalize=False)

    if len(ps.power) < n_bandpowers:
        return np.zeros(n_bandpowers)

    # Log-spaced frequency bins
    freq_min = ps.frequencies[0]
    freq_max = ps.frequencies[-1]
    band_edges = np.logspace(np.log10(freq_min), np.log10(freq_max), n_bandpowers + 1)

    bandpowers = np.zeros(n_bandpowers)
    for i in range(n_bandpowers):
        mask = (ps.frequencies >= band_edges[i]) & (ps.frequencies < band_edges[i + 1])
        if mask.any():
            bandpowers[i] = np.mean(ps.power[mask])
        else:
            # Interpolate from neighbors
            bandpowers[i] = np.interp(
                0.5 * (band_edges[i] + band_edges[i + 1]),
                ps.frequencies,
                ps.power,
            )

    # Log transform (add floor to avoid log(0))
    bandpowers = np.log10(bandpowers + 1e-30)
    return bandpowers


# ---------------------------------------------------------------------------
# Chi-squared test: observed vs simulated power spectra
# ---------------------------------------------------------------------------

def chi_squared_power_spectrum(
    observed_phi1: np.ndarray,
    simulated_phi1_list: list[np.ndarray],
    n_bins: int = 256,
    phi1_range: tuple[float, float] | None = None,
    n_bandpowers: int = 10,
) -> dict:
    """Compare observed power spectrum to an ensemble of simulations.

    Computes a chi-squared statistic measuring how well the simulated
    power spectra (under a given DM model) reproduce the observed one.
    Lower chi-squared = better fit.

    This is the classical frequentist approach to stream gap constraints:
    simulate many realisations under CDM/WDM, compare power spectra.

    Args:
        observed_phi1: Observed stream positions.
        simulated_phi1_list: List of simulated position arrays.
        n_bins: FFT bins.
        phi1_range: Shared phi1 range.
        n_bandpowers: Number of bandpower summary bins.

    Returns:
        Dict with keys: chi2, p_value, observed_bandpowers, sim_mean, sim_std.
    """
    from scipy.stats import chi2 as chi2_dist  # noqa: PLC0415

    obs_bp = power_spectrum_summary_vector(observed_phi1, n_bins, phi1_range, n_bandpowers)

    sim_bps = np.array([
        power_spectrum_summary_vector(s, n_bins, phi1_range, n_bandpowers)
        for s in simulated_phi1_list
    ])

    if len(sim_bps) < 2:
        return {"chi2": np.inf, "p_value": 0.0,
                "observed_bandpowers": obs_bp, "sim_mean": obs_bp, "sim_std": np.ones_like(obs_bp)}

    sim_mean = sim_bps.mean(axis=0)
    sim_std = sim_bps.std(axis=0)
    sim_std = np.maximum(sim_std, 1e-10)  # avoid division by zero

    # Chi-squared
    chi2_val = float(np.sum(((obs_bp - sim_mean) / sim_std) ** 2))
    dof = n_bandpowers
    p_value = float(1.0 - chi2_dist.cdf(chi2_val, dof))

    return {
        "chi2": chi2_val,
        "p_value": p_value,
        "dof": dof,
        "observed_bandpowers": obs_bp,
        "sim_mean": sim_mean,
        "sim_std": sim_std,
    }


# ---------------------------------------------------------------------------
# Perturbation detection via power excess
# ---------------------------------------------------------------------------

def detect_power_excess(
    observed_phi1: np.ndarray,
    smooth_phi1: np.ndarray,
    n_bins: int = 256,
    phi1_range: tuple[float, float] | None = None,
    threshold_sigma: float = 2.0,
    freq_range: tuple[float, float] = (0.1, 1.0),
) -> dict:
    """Detect excess power at perturbation-sensitive frequencies.

    Subhalo impacts produce gaps of characteristic width 1-10 deg,
    corresponding to spatial frequencies 0.1-1.0 cycles/deg.

    Compares the mean power in this band between observed and smooth
    streams, testing whether the excess exceeds the expected Poisson noise.

    Args:
        observed_phi1: Observed star positions.
        smooth_phi1: Smooth model positions.
        n_bins: FFT bins.
        phi1_range: Phi1 range for both.
        threshold_sigma: Detection threshold in sigma.
        freq_range: (min, max) frequency band for perturbations [cycles/deg].

    Returns:
        Dict with: detected (bool), excess_sigma, mean_ratio, freq_band.
    """
    ps_result = compute_power_ratio(
        observed_phi1, smooth_phi1, n_bins=n_bins, phi1_range=phi1_range,
    )

    if ps_result.power_ratio is None or len(ps_result.power_ratio) == 0:
        return {"detected": False, "excess_sigma": 0.0, "mean_ratio": 1.0,
                "freq_band": freq_range}

    # Select perturbation frequency band
    mask = (ps_result.frequencies >= freq_range[0]) & (ps_result.frequencies <= freq_range[1])

    if not mask.any():
        return {"detected": False, "excess_sigma": 0.0, "mean_ratio": 1.0,
                "freq_band": freq_range}

    ratio_in_band = ps_result.power_ratio[mask]
    mean_ratio = float(np.mean(ratio_in_band))

    # Under null (no perturbations), ratio ~ 1 with variance from Poisson noise
    # Expected variance of power ratio: ~2/N_stars (chi-squared with 2 DOF per freq bin)
    n_obs = max(ps_result.n_stars, 1)
    expected_std = np.sqrt(2.0 / n_obs) * np.sqrt(mask.sum())  # rough estimate
    excess_sigma = (mean_ratio - 1.0) / max(expected_std, 1e-10)

    detected = excess_sigma > threshold_sigma

    return {
        "detected": detected,
        "excess_sigma": float(excess_sigma),
        "mean_ratio": mean_ratio,
        "freq_band": freq_range,
        "n_freq_bins_in_band": int(mask.sum()),
    }
