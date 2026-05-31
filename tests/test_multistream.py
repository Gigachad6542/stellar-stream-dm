"""Tests for multi-stream joint significance combination."""

import math

import numpy as np
import pytest

from src.forward_model.multistream import (
    combine_p_fisher,
    combine_streams,
    combine_z_stouffer,
)


class TestStouffer:
    def test_known_value(self):
        # Four streams each at z=1 -> Z = 4/sqrt(4) = 2.
        Z, p = combine_z_stouffer([1.0, 1.0, 1.0, 1.0])
        assert Z == pytest.approx(2.0)
        assert p == pytest.approx(0.5 * math.erfc(2.0 / math.sqrt(2)), rel=1e-6)  # ~0.0228

    def test_null_streams_stay_null(self):
        Z, p = combine_z_stouffer([0.1, -0.2, 0.05, 0.0, -0.1])
        assert abs(Z) < 1.0 and p > 0.05

    def test_combination_boosts_coherent_signal(self):
        # Seven weak-but-coherent streams combine above any single one.
        Z, p = combine_z_stouffer([0.8] * 7)
        assert Z > 2.0
        assert p < 0.05

    def test_weighted(self):
        # Down-weighting a noisy stream changes Z.
        Zu, _ = combine_z_stouffer([2.0, 0.0])
        Zw, _ = combine_z_stouffer([2.0, 0.0], weights=[1.0, 0.1])
        assert Zw > Zu  # the informative stream dominates more


class TestFisher:
    def test_chi2_dof(self):
        x, p = combine_p_fisher([0.5, 0.5, 0.5])
        # x = -2 sum ln(0.5) = -2*3*ln0.5 = 6*ln2
        assert x == pytest.approx(6 * math.log(2), rel=1e-6)
        assert 0 < p < 1

    def test_small_pvalues_significant(self):
        _, p = combine_p_fisher([0.01, 0.02, 0.03, 0.01])
        assert p < 0.01


class TestCombineStreams:
    def _streams(self, z, p, le_z=None, le_p=None):
        return [{"stream": f"S{i}", "z": z, "p": p,
                 **({"le_z": le_z, "le_p": le_p} if le_z is not None else {})}
                for i in range(7)]

    def test_uses_look_elsewhere_when_present(self):
        # Naive z is inflated (3.0) but look-elsewhere z is ~0 -> joint not significant.
        res = combine_streams(self._streams(3.0, 0.001, le_z=0.1, le_p=0.46), use="look_elsewhere")
        assert res.n_streams == 7
        assert res.stouffer_p > 0.05
        assert "upper limit" in res.interpretation.lower() or "no joint" in res.interpretation.lower()

    def test_naive_would_be_significant(self):
        # Same streams, but combining the (inflated) naive z's flags significance,
        # demonstrating why the look-elsewhere correction matters for the joint test.
        res = combine_streams(self._streams(3.0, 0.001, le_z=0.1, le_p=0.46), use="naive")
        assert res.stouffer_p < 0.05

    def test_incoherent_excess_is_not_a_detection(self):
        # A few streams pinned at the empirical-p floor with mixed-sign z (the
        # real multi-stream situation) must NOT be reported as a detection even
        # if Fisher's p is small -- it's a null-construction artifact.
        ps = [{"stream": "a", "le_z": 29.0, "le_p": 0.016},
              {"stream": "b", "le_z": -1.9, "le_p": 1.0},
              {"stream": "c", "le_z": -4.8, "le_p": 0.97},
              {"stream": "d", "le_z": 2.9, "le_p": 0.016},
              {"stream": "e", "le_z": 0.4, "le_p": 0.18},
              {"stream": "f", "le_z": 1.5, "le_p": 0.066},
              {"stream": "g", "le_z": 2.4, "le_p": 0.033}]
        res = combine_streams(ps, use="look_elsewhere")
        assert not res.coherent
        assert res.frac_positive < 0.85
        assert "not a detection" in res.interpretation.lower()

    def test_coherent_positive_is_flagged_detection(self):
        ps = [{"stream": f"S{i}", "le_z": 4.0, "le_p": 0.001} for i in range(7)]
        res = combine_streams(ps, use="look_elsewhere")
        assert res.coherent
        assert "detection" in res.interpretation.lower()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            combine_streams([{"stream": "x", "z": float("nan"), "p": float("nan")}])
