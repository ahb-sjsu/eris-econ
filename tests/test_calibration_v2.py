# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for calibration_v2 module: bounded diagonal estimation."""

import numpy as np
import pytest

from eris_econ.calibration import ObservedChoice
from eris_econ.calibration_v2 import (
    CalibratedSigma,
    _softmax_nll,
    bootstrap_confidence,
    cross_validate,
    estimate_diagonal_sigma,
)
from eris_econ.dimensions import N_DIMS


def _make_synthetic_obs(n: int = 100, rng=None) -> list:
    """Generate synthetic observations where option 0 is always chosen."""
    if rng is None:
        rng = np.random.default_rng(42)

    obs = []
    for _ in range(n):
        start = rng.normal(0, 1, N_DIMS)
        # Chosen is close to start (low cost)
        chosen = start + rng.normal(0, 0.1, N_DIMS)
        # Rejected is far from start (high cost)
        rejected = [start + rng.normal(0, 1.0, N_DIMS)]
        obs.append(ObservedChoice(start=start, chosen=chosen, rejected=rejected))
    return obs


class TestSoftmaxNLL:
    """Tests for the softmax negative log-likelihood."""

    def test_returns_float(self):
        obs = _make_synthetic_obs(10)
        log_diag = np.zeros(N_DIMS)
        nll = _softmax_nll(log_diag, obs)
        assert isinstance(nll, float)

    def test_nll_is_positive(self):
        obs = _make_synthetic_obs(10)
        log_diag = np.zeros(N_DIMS)
        nll = _softmax_nll(log_diag, obs)
        assert nll > 0

    def test_lower_nll_at_correct_weights(self):
        """The NLL should be lower when weights match the data structure."""
        rng = np.random.default_rng(42)
        obs = []
        # Data: only dimension 0 differs between chosen and rejected
        for _ in range(50):
            start = np.zeros(N_DIMS)
            chosen = np.zeros(N_DIMS)
            chosen[0] = 0.1  # small change on dim 0
            rejected = [np.zeros(N_DIMS)]
            rejected[0][0] = 2.0  # large change on dim 0
            obs.append(ObservedChoice(start=start, chosen=chosen, rejected=rejected))

        # High weight on dim 0 should give lower NLL
        log_diag_high = np.zeros(N_DIMS)  # sigma=1 → weight=1
        log_diag_low = np.ones(N_DIMS) * 3  # sigma=exp(3)≈20 → weight=0.05

        nll_high = _softmax_nll(log_diag_high, obs, regularization=0)
        nll_low = _softmax_nll(log_diag_low, obs, regularization=0)
        assert nll_high < nll_low


class TestEstimateDiagonalSigma:
    """Tests for bounded diagonal sigma estimation."""

    def test_returns_calibrated_sigma(self):
        obs = _make_synthetic_obs(50)
        result = estimate_diagonal_sigma(obs)
        assert isinstance(result, CalibratedSigma)

    def test_sigma_is_diagonal(self):
        obs = _make_synthetic_obs(50)
        result = estimate_diagonal_sigma(obs)
        # Off-diagonal should be zero
        off_diag = result.sigma - np.diag(np.diag(result.sigma))
        assert np.allclose(off_diag, 0)

    def test_sigma_shape(self):
        obs = _make_synthetic_obs(50)
        result = estimate_diagonal_sigma(obs)
        assert result.sigma.shape == (N_DIMS, N_DIMS)

    def test_sigma_positive_diagonal(self):
        obs = _make_synthetic_obs(50)
        result = estimate_diagonal_sigma(obs)
        assert np.all(np.diag(result.sigma) > 0)

    def test_respects_money_bounds(self):
        obs = _make_synthetic_obs(50)
        result = estimate_diagonal_sigma(
            obs, money_bounds=(5.0, 50.0),
        )
        money_var = result.sigma[0, 0]
        assert 5.0 - 0.01 <= money_var <= 50.0 + 0.01

    def test_respects_moral_bounds(self):
        obs = _make_synthetic_obs(50)
        result = estimate_diagonal_sigma(
            obs, moral_bounds=(0.1, 10.0),
        )
        for d in range(1, N_DIMS):
            var = result.sigma[d, d]
            assert 0.1 - 0.01 <= var <= 10.0 + 0.01

    def test_n_parameters(self):
        obs = _make_synthetic_obs(50)
        result = estimate_diagonal_sigma(obs)
        assert result.n_parameters == N_DIMS

    def test_aic_bic_computed(self):
        obs = _make_synthetic_obs(50)
        result = estimate_diagonal_sigma(obs)
        assert isinstance(result.aic, float)
        assert isinstance(result.bic, float)

    def test_weights_property(self):
        obs = _make_synthetic_obs(50)
        result = estimate_diagonal_sigma(obs)
        weights = result.weights()
        assert weights.shape == (N_DIMS,)
        assert np.all(weights > 0)


class TestCrossValidate:
    """Tests for K-fold cross-validation."""

    def test_returns_best_reg_and_scores(self):
        obs = _make_synthetic_obs(50)
        best_reg, scores = cross_validate(obs, n_folds=3, regularization_values=[0.01, 0.1])
        assert best_reg in [0.01, 0.1]
        assert len(scores) == 2

    def test_scores_are_finite(self):
        obs = _make_synthetic_obs(50)
        _, scores = cross_validate(obs, n_folds=3, regularization_values=[0.01, 0.1])
        assert all(np.isfinite(s) for s in scores)


class TestBootstrapConfidence:
    """Tests for bootstrap confidence intervals."""

    def test_returns_three_arrays(self):
        obs = _make_synthetic_obs(30)
        med, lo, hi = bootstrap_confidence(obs, n_bootstrap=5)
        assert med.shape == (N_DIMS,)
        assert lo.shape == (N_DIMS,)
        assert hi.shape == (N_DIMS,)

    def test_ci_ordering(self):
        """Low should be <= median <= high."""
        obs = _make_synthetic_obs(30)
        med, lo, hi = bootstrap_confidence(obs, n_bootstrap=10)
        assert np.all(lo <= med + 1e-10)
        assert np.all(med <= hi + 1e-10)
