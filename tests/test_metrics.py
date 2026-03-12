# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

import numpy as np

from eris_econ.dimensions import Dim, N_DIMS
from eris_econ.metrics import (
    mahalanobis_distance,
    boundary_penalty,
    edge_weight,
    loss_aversion_ratio,
)


class TestMahalanobisDistance:
    def test_zero_distance(self):
        a = np.zeros(N_DIMS)
        sigma_inv = np.eye(N_DIMS)
        assert mahalanobis_distance(a, a, sigma_inv) == 0.0

    def test_identity_is_euclidean(self):
        a = np.zeros(N_DIMS)
        b = np.zeros(N_DIMS)
        b[0] = 3.0
        b[1] = 4.0
        sigma_inv = np.eye(N_DIMS)
        d = mahalanobis_distance(a, b, sigma_inv)
        assert abs(d - 5.0) < 1e-6

    def test_symmetric(self):
        rng = np.random.RandomState(42)
        a = rng.randn(N_DIMS)
        b = rng.randn(N_DIMS)
        sigma_inv = np.eye(N_DIMS)
        assert (
            abs(mahalanobis_distance(a, b, sigma_inv) - mahalanobis_distance(b, a, sigma_inv))
            < 1e-10
        )

    def test_weighted(self):
        """Dimensions with higher weight in Σ^{-1} contribute more."""
        a = np.zeros(N_DIMS)
        b = np.zeros(N_DIMS)
        b[0] = 1.0

        sigma_inv_1 = np.eye(N_DIMS)
        sigma_inv_10 = np.eye(N_DIMS)
        sigma_inv_10[0, 0] = 10.0

        d1 = mahalanobis_distance(a, b, sigma_inv_1)
        d10 = mahalanobis_distance(a, b, sigma_inv_10)
        assert d10 > d1


class TestBoundaryPenalty:
    def test_no_boundaries(self):
        a = np.zeros(N_DIMS)
        b = np.ones(N_DIMS)
        assert boundary_penalty(a, b, {}) == 0.0

    def test_theft_boundary(self):
        a = np.zeros(N_DIMS)
        a[Dim.RIGHTS] = 1.0
        b = np.zeros(N_DIMS)
        b[Dim.RIGHTS] = -0.5

        pen = boundary_penalty(a, b, {"theft": 100.0})
        assert pen == 100.0

    def test_sacred_value_infinite(self):
        a = np.zeros(N_DIMS)
        a[Dim.RIGHTS] = 1.0
        b = np.zeros(N_DIMS)
        b[Dim.RIGHTS] = -0.5

        pen = boundary_penalty(a, b, {"theft": np.inf})
        assert pen == float("inf")

    def test_no_crossing_no_penalty(self):
        a = np.zeros(N_DIMS)
        a[Dim.RIGHTS] = 1.0
        b = np.zeros(N_DIMS)
        b[Dim.RIGHTS] = 0.5  # Still positive

        pen = boundary_penalty(a, b, {"theft": 100.0})
        assert pen == 0.0


class TestEdgeWeight:
    def test_combines_distance_and_penalty(self):
        a = np.zeros(N_DIMS)
        a[Dim.RIGHTS] = 1.0
        b = np.zeros(N_DIMS)
        b[Dim.CONSEQUENCES] = 1.0
        b[Dim.RIGHTS] = -0.5

        sigma_inv = np.eye(N_DIMS)
        w = edge_weight(a, b, sigma_inv, {"theft": 50.0})
        assert w > 50.0  # penalty + distance


class TestLossAversionRatio:
    def test_loss_aversion_gt_1(self):
        """Losses should be more costly than equivalent gains → λ > 1."""
        ref = np.zeros(N_DIMS)
        ref[Dim.CONSEQUENCES] = 10.0
        ref[Dim.RIGHTS] = 1.0

        gain = ref.copy()
        gain[Dim.CONSEQUENCES] += 1.0

        loss = ref.copy()
        loss[Dim.CONSEQUENCES] -= 1.0
        loss[Dim.RIGHTS] -= 0.15  # Losses activate rights dimension
        loss[Dim.VIRTUE_IDENTITY] -= 0.1

        sigma_inv = np.eye(N_DIMS)
        ratio = loss_aversion_ratio(gain, loss, ref, sigma_inv)
        assert ratio > 1.0
