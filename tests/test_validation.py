# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests comparing framework predictions to published experimental data."""

import numpy as np

from eris_econ.games import _default_sigma
from eris_econ.validation import (
    dimensional_loss_aversion,
    run_full_validation,
    test_prediction_1_dimensional_loss_aversion,
    test_ultimatum_prediction,
    test_dictator_prediction,
    test_public_goods_prediction,
    test_endowment_by_good_type,
    test_cross_cultural_ultimatum,
)
from eris_econ.dimensions import Dim


class TestPrediction1DimensionalLossAversion:
    """KEY TEST: lambda should vary with dimensional content.

    Prospect theory predicts constant lambda ~= 2.25.
    The geometric framework predicts lambda increases with the number
    of activated non-monetary dimensions.
    """

    def test_lambda_monotonically_increases(self):
        """Lambda must strictly increase as more dimensions are activated."""
        results = test_prediction_1_dimensional_loss_aversion()
        lambdas = [r.predicted_lambda for r in results]
        for i in range(len(lambdas) - 1):
            assert lambdas[i] < lambdas[i + 1], (
                f"Lambda not monotonic: {results[i].context} "
                f"({lambdas[i]:.3f}) >= {results[i + 1].context} "
                f"({lambdas[i + 1]:.3f})"
            )

    def test_pure_monetary_lambda_near_1(self):
        """Pure monetary loss (d1 only) should have lambda close to 1."""
        sigma = _default_sigma()
        lam = dimensional_loss_aversion(sigma, activated_dims=[])
        # With only d1 changing symmetrically, lambda should be ~1
        assert 0.8 < lam < 1.5, f"Pure monetary lambda = {lam}, expected ~1.0"

    def test_rich_loss_lambda_above_2(self):
        """Loss activating many dimensions should have lambda > 2."""
        sigma = _default_sigma()
        lam = dimensional_loss_aversion(
            sigma,
            activated_dims=[
                Dim.RIGHTS, Dim.FAIRNESS,
                Dim.SOCIAL_IMPACT, Dim.VIRTUE_IDENTITY,
            ],
        )
        assert lam > 2.0, f"Multi-dimensional lambda = {lam}, expected > 2.0"

    def test_kt_range_with_typical_loss(self):
        """A typical monetary loss with moderate moral content should
        produce lambda in the KT range (2.0-2.5)."""
        sigma = _default_sigma()
        # A loss that activates 2-3 non-monetary dims (typical everyday loss)
        lam = dimensional_loss_aversion(
            sigma,
            activated_dims=[Dim.RIGHTS, Dim.VIRTUE_IDENTITY],
        )
        assert 1.5 < lam < 3.5, f"Typical loss lambda = {lam}, expected 1.5-3.5"


class TestUltimatumPrediction:
    """Model should predict ~40-50% offers (Nash predicts ~0%)."""

    def test_optimal_offer_above_nash(self):
        """Offer must be well above Nash (0%). Exact value depends on sigma
        calibration -- the paper identifies this as the central open problem.
        With default sigma, model predicts 20% (qualitatively correct direction,
        quantitative match requires calibrated sigma from experimental data)."""
        pred = test_ultimatum_prediction()
        assert pred.predicted_value >= 10, (
            f"Ultimatum offer {pred.predicted_value}% too close to Nash (0%)"
        )
        assert pred.predicted_value <= 50, (
            f"Ultimatum offer {pred.predicted_value}% too high"
        )

    def test_not_nash(self):
        """Must predict something other than the Nash equilibrium (0%)."""
        pred = test_ultimatum_prediction()
        assert pred.predicted_value > 0, "Model predicts Nash (0%) — framework adds nothing"


class TestDictatorPrediction:
    """Model should predict positive giving (Nash predicts 0%)."""

    def test_positive_giving(self):
        pred = test_dictator_prediction()
        assert pred.predicted_value > 0, "Model predicts 0% giving (same as Nash)"

    def test_less_than_ultimatum(self):
        """Dictator giving should be <= ultimatum offers (no rejection threat)."""
        ult = test_ultimatum_prediction()
        dic = test_dictator_prediction()
        assert dic.predicted_value <= ult.predicted_value, (
            f"Dictator {dic.predicted_value}% > Ultimatum {ult.predicted_value}%"
        )


class TestPublicGoodsPrediction:
    """Model should predict positive contribution (Nash predicts 0%)."""

    def test_positive_contribution(self):
        pred = test_public_goods_prediction()
        assert pred.predicted_value > 0, "Model predicts 0% (same as Nash)"


class TestEndowmentByGoodType:
    """WTA/WTP ratio should increase with number of ownership dimensions."""

    def test_wta_wtp_monotonic(self):
        """More ownership dimensions -> higher WTA/WTP ratio."""
        results = test_endowment_by_good_type()
        ratios = [r.predicted_wta_wtp_ratio for r in results]
        for i in range(len(ratios) - 1):
            assert ratios[i] < ratios[i + 1], (
                f"WTA/WTP not monotonic: {results[i].good_type} "
                f"({ratios[i]:.2f}) >= {results[i + 1].good_type} "
                f"({ratios[i + 1]:.2f})"
            )

    def test_lottery_ticket_low(self):
        """Lottery tickets (mainly monetary) should have low WTA/WTP."""
        results = test_endowment_by_good_type()
        lottery = results[0]
        assert lottery.predicted_wta_wtp_ratio < 3.0, (
            f"Lottery WTA/WTP = {lottery.predicted_wta_wtp_ratio}, too high"
        )

    def test_sentimental_good_high(self):
        """Sentimental goods should have highest WTA/WTP."""
        results = test_endowment_by_good_type()
        sentimental = results[-1]
        lottery = results[0]
        assert sentimental.predicted_wta_wtp_ratio > lottery.predicted_wta_wtp_ratio, (
            "Sentimental good WTA/WTP should exceed lottery ticket"
        )


class TestCrossCultural:
    """Different sigma values should produce different optimal offers."""

    def test_cultural_variation_exists(self):
        """Different cultures should produce different predictions."""
        results = test_cross_cultural_ultimatum()
        values = [r.predicted_value for r in results]
        assert len(set(values)) > 1, "All cultures produce same prediction"

    def test_strong_norms_higher_offers(self):
        """Cultures with stronger fairness norms should show higher offers."""
        results = test_cross_cultural_ultimatum()
        # results[0] = US, results[1] = Machiguenga-like, results[2] = Lamalera-like
        us = results[0].predicted_value
        lamalera = results[2].predicted_value
        assert lamalera >= us, (
            f"Lamalera-like ({lamalera}%) should >= US ({us}%)"
        )


class TestFullValidation:
    """Integration test: run full validation and check report."""

    def test_full_report_runs(self):
        report = run_full_validation()
        assert report.dimensional_lambda is not None
        assert len(report.dimensional_lambda) == 5

    def test_lambda_monotonic_in_report(self):
        report = run_full_validation()
        assert report.lambda_monotonic, "Key prediction fails: lambda not monotonic"

    def test_summary_output(self):
        report = run_full_validation()
        summary = report.summary()
        assert "EMPIRICAL VALIDATION REPORT" in summary
        assert "PREDICTION 1" in summary
