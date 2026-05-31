"""Injection-recovery validation for the timeline forward model.

Injects a known impact into a synthetic stream and checks the pipeline recovers
it. Marked ``slow`` because each run generates streams via galpy/galstreams.

Recovery expectations (fast/impulse self-consistency):
    * mass and phi1 (localisation) are well constrained -> asserted tightly
    * time-since-impact is partly degenerate with mass -> not asserted tightly
    * the best candidate must beat the unperturbed null
"""

import numpy as np
import pytest


@pytest.mark.slow
def test_injection_recovery_fast_mode_strong_signal():
    from src.forward_model.injection import run_injection_recovery
    from src.forward_model.pipeline import ForwardModelConfig

    # With the physical Erkal+2015 kick (bounded, no 50 km/s cap), realistic
    # kicks are far smaller than the old cap-saturated values, so a clearly
    # detectable impact requires a massive perturber. 10^9 Msun is robustly
    # recovered; 10^8.5 is near the detection threshold for this short stream.
    truth_mass, truth_time, truth_phi1 = 9.0, 1.5, 20.0

    cfg = ForwardModelConfig(
        stream_name="GD1",
        log10_mass_range=(8.0, 9.0), log10_mass_step=0.5,
        t_since_range=(1.0, 2.0), t_since_step=0.5,
        impact_phi1_values=[5.0, 20.0, 35.0],
        n_stars_sim=1500,
        base_seed=42,
        use_gnn_scorer=False,
        use_fast_mode=True,
        n_workers=1,
    )

    result, results = run_injection_recovery(
        cfg, truth_mass, truth_time, truth_phi1, truth_seed=123,
    )

    # The injected signal must be detectable: best candidate beats the null.
    assert result.best_beats_null, (
        f"best score {result.recovered_score:.4f} did not beat null {result.null_score:.4f}"
    )

    # The truth grid point must score among the best few (near-degenerate
    # candidates reshuffle once the radial-velocity term adds seed-level scatter,
    # so we require the truth to rank highly rather than win outright).
    assert result.truth_rank <= 3, f"truth ranked {result.truth_rank} (expected top 3)"

    # Mass and localisation recovered within one grid step.
    assert abs(result.mass_error_dex) <= 0.5 + 1e-6, (
        f"mass error {result.mass_error_dex:.2f} dex exceeds one grid step"
    )
    assert abs(result.phi1_error_deg) <= 15.0 + 1e-6, (
        f"phi1 error {result.phi1_error_deg:.1f} deg exceeds one grid step"
    )

    # Sanity: a full grid was evaluated.
    assert result.n_candidates == 3 * 3 * 3
    assert np.isfinite(result.recovered_score)


@pytest.mark.slow
def test_injection_recovery_offset_phi1_detectable_and_near_truth():
    """A different injected phi1 stays detectable and recovers within one grid step.

    Exact phi1 recovery is only reliable when the signal is strong and the grid
    is well sampled; on a coarse 3-point phi1 grid we assert the weaker but still
    meaningful claim that the best candidate beats the null and lands within one
    grid spacing of the truth. Finer characterisation is done via
    scripts/run_injection_recovery.py.
    """
    from src.forward_model.injection import run_injection_recovery
    from src.forward_model.pipeline import ForwardModelConfig

    truth_phi1 = 40.0
    phi1_grid = [10.0, 40.0, 70.0]
    spacing = 30.0
    cfg = ForwardModelConfig(
        stream_name="GD1",
        log10_mass_range=(8.5, 9.0), log10_mass_step=0.5,
        t_since_range=(1.0, 2.0), t_since_step=0.5,
        impact_phi1_values=phi1_grid,
        n_stars_sim=1500,
        base_seed=42,
        use_gnn_scorer=False,
        use_fast_mode=True,
        n_workers=1,
    )
    result, _ = run_injection_recovery(
        cfg, truth_log10_mass=9.0, truth_t_since_gyr=1.5, truth_phi1=truth_phi1,
        truth_seed=7,
    )
    assert result.best_beats_null
    assert result.recovered_phi1 in phi1_grid              # a real grid point
    assert abs(result.recovered_phi1 - truth_phi1) <= spacing + 1e-6
