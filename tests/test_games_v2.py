# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for games_v2 module: improved game encodings with rejection probability."""

import numpy as np
import pytest

from eris_econ.dimensions import Dim, N_DIMS
from eris_econ.games_v2 import (
    predict_game,
    public_goods_state,
    rejection_probability,
    ultimatum_state,
)


class TestRejectionProbability:
    """Tests for the Camerer (2003) rejection probability function."""

    def test_very_low_offer_high_rejection(self):
        """0% offer should be rejected almost certainly."""
        p = rejection_probability(0)
        assert p > 0.90, f"0% offer: rejection prob {p:.3f} should be >90%"

    def test_fair_offer_low_rejection(self):
        """50% offer should rarely be rejected."""
        p = rejection_probability(50)
        assert p < 0.05, f"50% offer: rejection prob {p:.3f} should be <5%"

    def test_monotonically_decreasing(self):
        """Higher offers should have lower rejection probability."""
        prev = 1.0
        for pct in range(0, 51, 5):
            p = rejection_probability(pct)
            assert p <= prev + 1e-10, (
                f"Rejection prob not monotonic: {pct}%={p:.3f} > prev={prev:.3f}"
            )
            prev = p

    def test_threshold_region(self):
        """Around 18% offer, rejection should be ~50%."""
        p = rejection_probability(18)
        assert 0.3 < p < 0.7, f"At threshold (18%): p={p:.3f}, expected ~0.5"

    def test_returns_float(self):
        p = rejection_probability(25.0)
        assert isinstance(float(p), float)


class TestUltimatumState:
    """Tests for the improved ultimatum state encoding."""

    def test_output_shape(self):
        state = ultimatum_state(10.0, 25.0)
        assert state.shape == (N_DIMS,)

    def test_fair_offer_high_fairness(self):
        state = ultimatum_state(10.0, 50.0)
        assert state[Dim.FAIRNESS] > 0.8

    def test_zero_offer_low_fairness(self):
        state = ultimatum_state(10.0, 0.0)
        assert state[Dim.FAIRNESS] < 0.2

    def test_rejection_lowers_expected_money(self):
        """With rejection, low offers should have lower expected money."""
        state_rej = ultimatum_state(10.0, 5.0, include_rejection=True)
        state_no_rej = ultimatum_state(10.0, 5.0, include_rejection=False)
        assert state_rej[Dim.CONSEQUENCES] < state_no_rej[Dim.CONSEQUENCES], (
            f"Expected money with rejection ({state_rej[Dim.CONSEQUENCES]:.2f}) "
            f"should be < without ({state_no_rej[Dim.CONSEQUENCES]:.2f})"
        )

    def test_fair_offer_similar_with_and_without_rejection(self):
        """50% offer has ~0% rejection, so states should be nearly identical."""
        state_rej = ultimatum_state(10.0, 50.0, include_rejection=True)
        state_no_rej = ultimatum_state(10.0, 50.0, include_rejection=False)
        assert np.allclose(state_rej, state_no_rej, atol=0.1)

    def test_expected_money_peaks_around_30_40(self):
        """Expected money should peak around 30-40% offer, not at 0%."""
        expected = []
        for pct in range(0, 51, 5):
            state = ultimatum_state(10.0, pct, include_rejection=True)
            expected.append(state[Dim.CONSEQUENCES])
        max_idx = np.argmax(expected)
        max_pct = max_idx * 5
        assert 15 <= max_pct <= 45, (
            f"Expected money peaks at {max_pct}%, expected 15-45%"
        )

    def test_epistemic_higher_for_safe_offers(self):
        """High offers should have higher epistemic certainty."""
        state_low = ultimatum_state(10.0, 5.0)
        state_high = ultimatum_state(10.0, 45.0)
        assert state_high[Dim.EPISTEMIC] > state_low[Dim.EPISTEMIC]


class TestPublicGoodsState:
    """Tests for the public goods state encoding."""

    def test_output_shape(self):
        state = public_goods_state(20.0, 50.0)
        assert state.shape == (N_DIMS,)

    def test_zero_contribution_max_money(self):
        """0% contribution should give the most remaining money."""
        state_0 = public_goods_state(20.0, 0.0)
        state_100 = public_goods_state(20.0, 100.0)
        assert state_0[Dim.CONSEQUENCES] > state_100[Dim.CONSEQUENCES]

    def test_full_contribution_max_fairness(self):
        state = public_goods_state(20.0, 100.0)
        assert state[Dim.FAIRNESS] > 0.8

    def test_zero_contribution_low_fairness(self):
        state = public_goods_state(20.0, 0.0)
        assert state[Dim.FAIRNESS] < 0.2


class TestPredictGame:
    """Tests for the predict_game function."""

    def test_ultimatum_returns_valid_pct(self):
        sigma = np.diag([25.0] + [0.25] * 8)
        pct, costs = predict_game(sigma, "ultimatum")
        assert 0 <= pct <= 50

    def test_dictator_returns_valid_pct(self):
        sigma = np.diag([25.0] + [0.25] * 8)
        pct, costs = predict_game(sigma, "dictator")
        assert 0 <= pct <= 50

    def test_public_goods_returns_valid_pct(self):
        sigma = np.diag([25.0] + [0.25] * 8)
        pct, costs = predict_game(sigma, "public_goods")
        assert 0 <= pct <= 100

    def test_costs_list_not_empty(self):
        sigma = np.diag([25.0] + [0.25] * 8)
        _, costs = predict_game(sigma, "ultimatum", resolution=5)
        assert len(costs) > 0

    def test_unknown_game_raises(self):
        sigma = np.diag([25.0] + [0.25] * 8)
        with pytest.raises(ValueError, match="Unknown game"):
            predict_game(sigma, "prisoner_dilemma")

    def test_rejection_increases_ultimatum_offer(self):
        """Rejection probability should push optimal offer higher."""
        sigma = np.diag([25.0] + [0.25] * 8)
        pct_rej, _ = predict_game(sigma, "ultimatum")
        pct_no_rej, _ = predict_game(sigma, "ultimatum_no_rejection")
        assert pct_rej >= pct_no_rej, (
            f"With rejection ({pct_rej}%) should be >= without ({pct_no_rej}%)"
        )

    def test_dictator_less_than_or_equal_ultimatum(self):
        """Without rejection threat, dictator giving should be <= ultimatum."""
        sigma = np.diag([25.0] + [0.25] * 8)
        ult_pct, _ = predict_game(sigma, "ultimatum_no_rejection")
        dic_pct, _ = predict_game(sigma, "dictator")
        assert dic_pct <= ult_pct + 5, (
            f"Dictator ({dic_pct}%) >> ultimatum no-rej ({ult_pct}%)"
        )
