# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for structural fuzzing module."""

import numpy as np
import pytest

from eris_econ.dimensions import Dim, N_DIMS
from eris_econ.structural_fuzz import (
    OBSERVED,
    SubsetResult,
    _optimize_subset,
    compositional_test,
    compute_mri,
    enumerate_subsets,
    find_adversarial_threshold,
    prediction_error,
    run_structural_fuzz,
    sensitivity_profile,
)


class TestPredictionError:
    """Tests for the prediction error metric."""

    def test_returns_mae_and_errors(self):
        sigma = np.diag([200.0] + [50.0] * 7 + [0.01])
        mae, errors = prediction_error(sigma)
        assert isinstance(mae, float)
        assert mae >= 0
        assert "ultimatum" in errors
        assert "dictator" in errors
        assert "public_goods" in errors

    def test_zero_sigma_doesnt_crash(self):
        """Edge case: very small sigma values."""
        sigma = np.diag([0.01] * N_DIMS)
        mae, errors = prediction_error(sigma)
        assert np.isfinite(mae)


class TestOptimizeSubset:
    """Tests for single-subset optimization."""

    def test_empty_subset(self):
        result = _optimize_subset((), n_grid=5)
        assert result.n_dims == 0
        assert result.mae >= 0

    def test_single_dim(self):
        result = _optimize_subset((Dim.EPISTEMIC,), n_grid=10)
        assert result.n_dims == 1
        assert result.mae >= 0

    def test_two_dims(self):
        result = _optimize_subset((Dim.FAIRNESS, Dim.EPISTEMIC), n_grid=10)
        assert result.n_dims == 2
        assert result.mae >= 0

    def test_more_dims_can_improve(self):
        """Adding relevant dimensions should not systematically worsen MAE."""
        r1 = _optimize_subset((Dim.EPISTEMIC,), n_grid=15)
        r2 = _optimize_subset((Dim.FAIRNESS, Dim.EPISTEMIC), n_grid=15)
        # Not guaranteed to improve (grid resolution), but shouldn't be
        # drastically worse
        assert r2.mae < r1.mae + 5.0


class TestEnumerateSubsets:
    """Tests for exhaustive subset enumeration."""

    def test_returns_sorted(self):
        results = enumerate_subsets(max_dims=2)
        maes = [r.mae for r in results]
        assert maes == sorted(maes)

    def test_pareto_marked(self):
        results = enumerate_subsets(max_dims=2)
        pareto = [r for r in results if r.pareto_optimal]
        assert len(pareto) >= 1

    def test_best_model_under_5pct(self):
        """Best subset should achieve <5% MAE."""
        results = enumerate_subsets(max_dims=3)
        assert results[0].mae < 5.0, (
            f"Best model MAE={results[0].mae}%, expected <5%"
        )

    def test_epistemic_is_best_single_dim(self):
        """Epistemic should be the best single dimension."""
        results = enumerate_subsets(max_dims=1)
        single_dim_results = [r for r in results if r.n_dims == 1]
        best_single = min(single_dim_results, key=lambda r: r.mae)
        assert Dim.EPISTEMIC in best_single.dims


class TestSensitivityProfile:
    """Tests for dimension importance via ablation."""

    def test_returns_all_dims(self):
        sigma_diag = np.array([200, 1, 50, 1, 1, 50, 50, 1, 0.01])
        results = sensitivity_profile(sigma_diag)
        assert len(results) == N_DIMS

    def test_ranked(self):
        sigma_diag = np.array([200, 1, 50, 1, 1, 50, 50, 1, 0.01])
        results = sensitivity_profile(sigma_diag)
        ranks = [r.importance_rank for r in results]
        assert sorted(ranks) == list(range(1, N_DIMS + 1))

    def test_epistemic_most_important(self):
        """With low epistemic variance (high weight), it should rank #1."""
        sigma_diag = np.array([200, 1, 50, 1, 1, 50, 50, 1, 0.01])
        results = sensitivity_profile(sigma_diag)
        assert results[0].dim == Dim.EPISTEMIC


class TestMRI:
    """Tests for Model Robustness Index."""

    def test_returns_mri(self):
        sigma_diag = np.array([200, 1, 50, 1, 1, 50, 50, 1, 0.01])
        mri = compute_mri(sigma_diag, n_perturbations=50)
        assert mri.mri >= 0
        assert mri.n_perturbations == 50

    def test_mri_components(self):
        sigma_diag = np.array([200, 1, 50, 1, 1, 50, 50, 1, 0.01])
        mri = compute_mri(sigma_diag, n_perturbations=50)
        assert mri.mean_omega <= mri.p75_omega
        assert mri.p75_omega <= mri.p95_omega

    def test_perturbation_count(self):
        sigma_diag = np.array([200, 1, 50, 1, 1, 50, 50, 1, 0.01])
        mri = compute_mri(sigma_diag, n_perturbations=30)
        assert len(mri.perturbation_errors) == 30


class TestCompositionTest:
    """Tests for greedy compositional dimension addition."""

    def test_returns_sequence(self):
        result = compositional_test()
        assert len(result.order) >= 2
        assert len(result.mae_sequence) == len(result.order)

    def test_money_alone_is_bad(self):
        """Starting with money alone should give high MAE."""
        result = compositional_test(starting_dim=Dim.CONSEQUENCES)
        assert result.mae_sequence[0] > 15.0


class TestAdversarialThreshold:
    """Tests for adversarial perturbation search."""

    def test_finds_thresholds(self):
        sigma_diag = np.array([200, 1, 50, 1, 1, 50, 50, 1, 0.01])
        results = find_adversarial_threshold(sigma_diag, Dim.EPISTEMIC)
        assert len(results) >= 1

    def test_threshold_direction(self):
        sigma_diag = np.array([200, 1, 50, 1, 1, 50, 50, 1, 0.01])
        results = find_adversarial_threshold(sigma_diag, Dim.EPISTEMIC)
        for r in results:
            assert r.direction in ("increase", "decrease")
            assert r.threshold_ratio > 0


class TestFullCampaign:
    """Integration test for the full structural fuzz campaign."""

    def test_runs(self):
        report = run_structural_fuzz(max_subset_dims=2, n_mri_perturbations=50)
        assert report is not None
        assert report.best_model is not None
        assert report.mri is not None

    def test_summary(self):
        report = run_structural_fuzz(max_subset_dims=2, n_mri_perturbations=50)
        summary = report.summary()
        assert "STRUCTURAL FUZZ REPORT" in summary
        assert "PARETO FRONTIER" in summary
        assert "MODEL ROBUSTNESS INDEX" in summary
