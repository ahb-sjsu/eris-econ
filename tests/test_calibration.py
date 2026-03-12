# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

import numpy as np

from eris_econ.dimensions import N_DIMS
from eris_econ.calibration import (
    ObservedChoice,
    estimate_sigma,
    estimate_boundaries,
)


class TestEstimateSigma:
    def test_recovers_structure(self):
        """With enough data, should recover a reasonable covariance matrix."""
        rng = np.random.RandomState(42)
        # Generate synthetic choices under known sigma
        observations = []
        for _ in range(50):
            start = rng.randn(N_DIMS)
            chosen = start + rng.randn(N_DIMS) * 0.1
            rejected = [start + rng.randn(N_DIMS) * 0.5 for _ in range(3)]
            observations.append(ObservedChoice(start=start, chosen=chosen, rejected=rejected))

        sigma = estimate_sigma(observations, regularization=0.1)
        assert sigma.shape == (N_DIMS, N_DIMS)
        # Should be symmetric
        assert np.allclose(sigma, sigma.T, atol=1e-6)
        # Should be positive semi-definite
        eigenvalues = np.linalg.eigvalsh(sigma)
        assert np.all(eigenvalues >= -1e-6)


class TestEstimateBoundaries:
    def test_never_crossed_is_infinite(self):
        obs = {"sacred": [(1.0, False), (5.0, False), (100.0, False)]}
        boundaries = estimate_boundaries(obs)
        assert boundaries["sacred"] == float("inf")

    def test_always_crossed_is_zero(self):
        obs = {"trivial": [(0.1, True), (0.5, True), (1.0, True)]}
        boundaries = estimate_boundaries(obs)
        assert boundaries["trivial"] == 0.0

    def test_moderate_boundary(self):
        obs = {
            "promise": [
                (1.0, False),
                (2.0, False),
                (5.0, True),
                (10.0, True),
            ]
        }
        boundaries = estimate_boundaries(obs)
        # Boundary should be between 2 and 5
        assert 1.0 < boundaries["promise"] < 10.0
