# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for joint calibration module."""

import numpy as np
import pytest

from eris_econ.dimensions import Dim, N_DIMS
from eris_econ.joint_calibration import (
    _public_goods_state_norm,
    _ultimatum_state_norm,
    encode_public_goods_norm,
    encode_ultimatum_norm,
    predict_game_norm,
    run_joint_calibration,
)


class TestNormalizedEncodings:
    """Tests for normalized state encodings."""

    def test_ultimatum_consequences_in_01(self):
        """Normalized consequences should be in [0, 1]."""
        for pct in range(0, 51, 5):
            state = _ultimatum_state_norm(10.0, pct)
            assert 0 <= state[Dim.CONSEQUENCES] <= 1.1, (
                f"Ult {pct}%: consequences={state[Dim.CONSEQUENCES]}"
            )

    def test_public_goods_consequences_in_01(self):
        """Normalized consequences should be in [0, 1]."""
        for pct in range(0, 101, 10):
            state = _public_goods_state_norm(20.0, pct)
            assert 0 <= state[Dim.CONSEQUENCES] <= 1.1, (
                f"PG {pct}%: consequences={state[Dim.CONSEQUENCES]}"
            )

    def test_ultimatum_50pct_high_consequences(self):
        """Fair offer with high acceptance should have decent expected money."""
        state = _ultimatum_state_norm(10.0, 50.0)
        # At 50%, p_acc ≈ 0.98, expected ≈ 0.98 * 5 / 10 ≈ 0.49
        assert state[Dim.CONSEQUENCES] > 0.3

    def test_ultimatum_0pct_low_expected(self):
        """0% offer has ~95% rejection, so expected money is very low."""
        state = _ultimatum_state_norm(10.0, 0.0)
        # At 0%, p_acc ≈ 0.05, expected ≈ 0.05 * 10 / 10 = 0.05
        assert state[Dim.CONSEQUENCES] < 0.2

    def test_public_goods_0pct_max_money(self):
        """0% contribution gives maximum remaining money."""
        state = _public_goods_state_norm(20.0, 0.0)
        assert state[Dim.CONSEQUENCES] > 0.9

    def test_public_goods_100pct_lower_money(self):
        """100% contribution gives less remaining money."""
        state_0 = _public_goods_state_norm(20.0, 0.0)
        state_100 = _public_goods_state_norm(20.0, 100.0)
        assert state_0[Dim.CONSEQUENCES] > state_100[Dim.CONSEQUENCES]


class TestEncoding:
    """Tests for the observation encoding functions."""

    def test_encode_ultimatum_returns_observations(self):
        obs = encode_ultimatum_norm()
        assert len(obs) > 0

    def test_encode_public_goods_returns_observations(self):
        obs = encode_public_goods_norm()
        assert len(obs) > 0

    def test_ultimatum_obs_structure(self):
        obs = encode_ultimatum_norm()
        o = obs[0]
        assert o.start.shape == (N_DIMS,)
        assert o.chosen.shape == (N_DIMS,)
        assert len(o.rejected) > 0

    def test_ultimatum_start_normalized(self):
        obs = encode_ultimatum_norm()
        assert obs[0].start[Dim.CONSEQUENCES] == 1.0


class TestPredictGameNorm:
    """Tests for normalized game predictions."""

    def test_ultimatum_prediction_range(self):
        sigma = np.diag([200.0] + [50.0] * 7 + [0.01])
        pct, costs = predict_game_norm(sigma, "ultimatum")
        assert 0 <= pct <= 50

    def test_public_goods_prediction_range(self):
        sigma = np.diag([200.0] + [50.0] * 7 + [0.01])
        pct, costs = predict_game_norm(sigma, "public_goods")
        assert 0 <= pct <= 100

    def test_dictator_prediction_range(self):
        sigma = np.diag([200.0] + [50.0] * 7 + [0.01])
        pct, costs = predict_game_norm(sigma, "dictator")
        assert 0 <= pct <= 50

    def test_rejection_increases_offer(self):
        """Rejection probability should increase optimal offer."""
        sigma = np.diag([200.0] + [50.0] * 7 + [0.01])
        ult_pct, _ = predict_game_norm(sigma, "ultimatum")
        ult_nr_pct, _ = predict_game_norm(sigma, "ultimatum_no_rejection")
        assert ult_pct >= ult_nr_pct

    def test_unknown_game_raises(self):
        sigma = np.diag([1.0] * N_DIMS)
        with pytest.raises(ValueError):
            predict_game_norm(sigma, "unknown_game")


class TestJointCalibration:
    """Integration tests for the full calibration pipeline."""

    def test_pipeline_runs(self):
        """Full pipeline should complete without error."""
        report = run_joint_calibration(do_cv=False, do_bootstrap=False)
        assert report is not None
        assert len(report.predictions) >= 3

    def test_ultimatum_prediction_reasonable(self):
        """Calibrated model should predict positive ultimatum offer."""
        report = run_joint_calibration(do_cv=False, do_bootstrap=False)
        ult = next(p for p in report.predictions if "rejection)" in p.game)
        assert ult.predicted_pct > 10, (
            f"Ultimatum prediction {ult.predicted_pct}% too close to Nash (0%)"
        )

    def test_public_goods_prediction_reasonable(self):
        """Calibrated model should predict positive PG contribution."""
        report = run_joint_calibration(do_cv=False, do_bootstrap=False)
        pg = next(p for p in report.predictions if "Public" in p.game)
        assert pg.predicted_pct > 10

    def test_summary_output(self):
        report = run_joint_calibration(do_cv=False, do_bootstrap=False)
        summary = report.summary()
        assert "JOINT CALIBRATION REPORT" in summary
        assert "PREDICTIONS" in summary
