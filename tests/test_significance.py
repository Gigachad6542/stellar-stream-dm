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
