# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Parameter estimation for the Economic Decision Complex.

The framework requires two empirical inputs:
1. Σ (9×9 covariance matrix) — estimated from behavioral data
2. β_k (boundary penalties) — estimated from willingness-to-violate data

This module provides estimation methods for both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from eris_econ.dimensions import N_DIMS


@dataclass
class ObservedChoice:
    """An observed economic choice for calibration.

    Each observation records the starting state, chosen end state,
    and (optionally) rejected alternatives.
    """

    start: np.ndarray  # [9] starting attribute vector
    chosen: np.ndarray  # [9] chosen end state
    rejected: List[np.ndarray]  # list of [9] rejected alternatives


def estimate_sigma(
    observations: List[ObservedChoice],
    regularization: float = 0.01,
) -> np.ndarray:
    """Estimate Σ from observed choices via maximum likelihood.

    The model assumes agents choose the minimum-cost option under
    Mahalanobis distance.  We find Σ that maximizes the likelihood
    of observed choices.

    Uses L-BFGS-B on the Cholesky factor of Σ^{-1} to ensure
    positive definiteness.

    Args:
        observations: List of observed choice data.
        regularization: L2 penalty on off-diagonal elements.

    Returns:
        Estimated [9, 9] covariance matrix.
    """
    n_params = N_DIMS * (N_DIMS + 1) // 2  # Cholesky lower triangle

    def _unpack_cholesky(params: np.ndarray) -> np.ndarray:
        """Reconstruct Σ^{-1} from Cholesky factor parameters."""
        L = np.zeros((N_DIMS, N_DIMS))
        idx = 0
        for i in range(N_DIMS):
            for j in range(i + 1):
                L[i, j] = params[idx]
                idx += 1
        # Ensure positive diagonal
        for i in range(N_DIMS):
            L[i, i] = np.exp(L[i, i])
        return L @ L.T  # Σ^{-1} = L L^T

    def neg_log_likelihood(params: np.ndarray) -> float:
        """Negative log-likelihood of observed choices."""
        sigma_inv = _unpack_cholesky(params)
        nll = 0.0

        for obs in observations:
            delta_chosen = obs.chosen - obs.start
            cost_chosen = float(delta_chosen @ sigma_inv @ delta_chosen)

            # Softmax likelihood: P(chosen) = exp(-cost_chosen) / Σ exp(-cost_j)
            costs = [cost_chosen]
            for alt in obs.rejected:
                delta_alt = alt - obs.start
                costs.append(float(delta_alt @ sigma_inv @ delta_alt))

            min_cost = min(costs)
            log_denom = min_cost + np.log(sum(np.exp(-(c - min_cost)) for c in costs))
            nll += (cost_chosen - min_cost) + log_denom

        # Regularization
        nll += regularization * np.sum(params**2)
        return nll

    # Initialize at identity
    init = np.zeros(n_params)
    idx = 0
    for i in range(N_DIMS):
        for j in range(i + 1):
            if i == j:
                init[idx] = 0.0  # exp(0) = 1 → identity diagonal
            idx += 1

    result = minimize(neg_log_likelihood, init, method="L-BFGS-B")
    sigma_inv = _unpack_cholesky(result.x)

    # Recover Σ from Σ^{-1}
    sigma = np.linalg.inv(sigma_inv + 1e-10 * np.eye(N_DIMS))
    return sigma


def estimate_boundaries(
    boundary_observations: Dict[str, List[Tuple[float, bool]]],
) -> Dict[str, float]:
    """Estimate boundary penalties β_k from willingness-to-violate data.

    For each boundary, we observe (offered_incentive, violated?) pairs.
    The penalty β_k is the incentive level at which ~50% of subjects
    choose to cross the boundary (the indifference point).

    Args:
        boundary_observations: {boundary_name: [(incentive, crossed), ...]}

    Returns:
        {boundary_name: estimated β_k}
    """
    boundaries: Dict[str, float] = {}

    for name, obs in boundary_observations.items():
        if not obs:
            boundaries[name] = 0.0
            continue

        # Simple logistic regression: P(cross) = 1 / (1 + exp(-(incentive - β)))
        # β is the 50% crossing point
        incentives = np.array([o[0] for o in obs])
        crossed = np.array([float(o[1]) for o in obs])

        if crossed.mean() == 0:
            # Never crossed → β = inf (sacred value)
            boundaries[name] = float("inf")
            continue
        if crossed.mean() == 1:
            # Always crossed → β ≈ 0
            boundaries[name] = 0.0
            continue

        def neg_ll(beta):
            p = 1.0 / (1.0 + np.exp(-(incentives - beta[0])))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.sum(crossed * np.log(p) + (1 - crossed) * np.log(1 - p))

        result = minimize(neg_ll, x0=[np.median(incentives)], method="Nelder-Mead")
        boundaries[name] = float(result.x[0])

    return boundaries
