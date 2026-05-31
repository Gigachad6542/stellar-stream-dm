"""Tests for forward-model statistical significance (null distribution stats)."""

import numpy as np
import pytest

from src.forward_model.significance import compute_significance


class TestComputeSignificance:
    def test_strong_candidate_is_significant(self):
        # Candidate far below the null distribution -> high z, low p.
        null = list(np.random.default_rng(0).normal(5.0, 0.3, 200))
        res = compute_significance(3.0, null)
        assert res.z_score > 3.0
        assert res.p_value < 0.05
        assert res.improvement > 0  # better than null mean

    def test_candidate_within_null_not_significant(self):
        null = list(np.random.default_rng(1).normal(5.0, 0.3, 200))
        res = compute_significance(5.0, null)            # right at the null mean
        assert abs(res.z_score) < 1.0
        assert res.p_value > 0.05

    def test_p_value_is_one_sided_fraction(self):
        null = [1.0, 2.0, 3.0, 4.0, 5.0]
        # candidate=3.0: three null (1,2,3) are <= 3 -> (3+1)/(5+1)=0.667
        res = compute_significance(3.0, null)
        assert res.p_value == pytest.approx(4 / 6)

    def test_zero_std_handled(self):
        res = compute_significance(1.0, [5.0, 5.0, 5.0])
        assert np.isfinite(res.p_value)
        assert res.z_score == float("inf")  # better than a degenerate null

    def test_to_dict_serializable(self):
        import json
        res = compute_significance(3.0, list(np.random.default_rng(2).normal(5, 0.3, 50)))
        json.dumps(res.to_dict())

    def test_empty_null_raises(self):
        with pytest.raises(ValueError):
            compute_significance(1.0, [])


@pytest.mark.slow
def test_look_elsewhere_is_more_conservative():
    """The look-elsewhere (best-of-grid) null must give a p-value no smaller than
    the naive unperturbed null for the same candidate, because fitting the best of
    many candidates to no-impact noise already achieves a low score."""
    import os
    import galstreams
    from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel
    from src.forward_model.injection import generate_injected_stream

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mws = galstreams.MWStreams(verbose=False)
    cfg = ForwardModelConfig(
        stream_name="GD1", config_path=os.path.join(root, "config", "streams.yaml"),
        log10_mass_range=(8.0, 9.0), log10_mass_step=0.5,
        t_since_range=(1.0, 2.0), t_since_step=0.5,
        impact_phi1_values=[5.0, 20.0, 35.0], n_stars_sim=1500, base_seed=42,
        use_gnn_scorer=False, use_fast_mode=True, n_workers=1)
    cfg0 = ForwardModelConfig(stream_name="GD1", config_path=cfg.config_path,
                              n_stars_sim=1500, base_seed=999, use_fast_mode=True)
    obs = generate_injected_stream(cfg0, 9.0, 1.5, 20.0, seed=123, mws=mws)
    model = TimelineForwardModel(cfg); model._mws = mws
    model.prepare(observed_override=obs)
    results = model.run_grid()
    best = results[0].score.combined

    naive = compute_significance(best, model.build_null_distribution(12, seed0=2000))
    le = compute_significance(best, model.build_lookelsewhere_null(8, seed0=3000))

    assert len(le.null_scores) == 8
    # Look-elsewhere correction never makes a detection look MORE significant.
    assert le.p_value >= naive.p_value - 1e-9
    assert le.null_mean <= naive.null_mean + 1e-6  # best-of-grid <= single-candidate null


@pytest.mark.slow
def test_monte_carlo_impact_time_tightens_with_precision():
    """The impact-time posterior must reduce to the deterministic best fit with
    zero measurement error and broaden as the errors grow -- i.e. better data
    (smaller errors, as from multi-epoch fusion) sharpens the dated impact."""
    import os
    import galstreams
    from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel
    from src.forward_model.injection import generate_injected_stream

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mws = galstreams.MWStreams(verbose=False)
    cfg = ForwardModelConfig(
        stream_name="GD1", config_path=os.path.join(root, "config", "streams.yaml"),
        log10_mass_range=(8.5, 9.0), log10_mass_step=0.5,
        t_since_range=(0.5, 2.5), t_since_step=0.5,
        impact_phi1_values=[10.0, 20.0, 30.0], n_stars_sim=1200, base_seed=42,
        use_gnn_scorer=False, use_fast_mode=True, n_workers=1)
    cfg0 = ForwardModelConfig(stream_name="GD1", config_path=cfg.config_path,
                              n_stars_sim=1200, base_seed=999, use_fast_mode=True)
    obs = generate_injected_stream(cfg0, 9.0, 1.5, 20.0, seed=123, mws=mws)
    n = len(obs["phi1"])
    obs["e_pm1"] = np.full(n, 0.3); obs["e_pm2"] = np.full(n, 0.3)
    obs["e_vrad"] = np.full(n, 3.0); obs["e_dist"] = np.full(n, 0.5)
    model = TimelineForwardModel(cfg); model._mws = mws
    model.prepare(observed_override=obs)

    p0 = model.monte_carlo_impact_time(n_realizations=8, error_scale=0.0, seed0=5000)
    p2 = model.monte_carlo_impact_time(n_realizations=8, error_scale=2.0, seed0=5000)

    assert p0.t_since_std == 0.0                       # no noise -> deterministic
    assert p2.t_since_std >= p0.t_since_std            # more error -> >= spread
    assert 0.5 <= p0.t_since_median <= 2.5             # within the grid
    assert p0.t_since_p16 <= p0.t_since_median <= p0.t_since_p84
