# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Improved calibration: bounded diagonal estimation with cross-validation.

Addresses the limitations of the original MLE calibration:
1. Constrained sigma[0,0] to prevent money weight collapse
2. Diagonal-only estimation (9 parameters, not 45)
3. Cross-validation for regularization tuning
4. Bootstrap confidence intervals on sigma
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

from eris_econ.calibration import ObservedChoice
from eris_econ.dimensions import N_DIMS, Dim


@dataclass
class CalibratedSigma:
    """Result of constrained sigma calibration."""
    sigma: np.ndarray  # [9, 9] diagonal covariance
    log_likelihood: float
    n_observations: int
    n_parameters: int  # always 9 for diagonal
    converged: bool

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        return 2 * self.n_parameters - 2 * (-self.log_likelihood)

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        return (self.n_parameters * np.log(self.n_observations)
                - 2 * (-self.log_likelihood))

    def weights(self) -> np.ndarray:
        """Diagonal weights (1/sigma_ii)."""
        return 1.0 / np.diag(self.sigma)


def _softmax_nll(
    log_diag: np.ndarray,
    observations: List[ObservedChoice],
    regularization: float = 0.01,
) -> float:
    """Softmax negative log-likelihood for diagonal sigma."""
    sigma_diag = np.exp(log_diag)
    sigma_inv_diag = 1.0 / sigma_diag
    nll = 0.0

    for obs in observations:
        delta_chosen = obs.chosen - obs.start
        cost_chosen = float(np.sum(sigma_inv_diag * delta_chosen ** 2))

        costs = [cost_chosen]
        for alt in obs.rejected:
            delta_alt = alt - obs.start
            costs.append(float(np.sum(sigma_inv_diag * delta_alt ** 2)))

        min_cost = min(costs)
        log_denom = min_cost + np.log(
            sum(np.exp(-(c - min_cost)) for c in costs)
        )
        nll += (cost_chosen - min_cost) + log_denom

    # Regularization: penalize extreme values
    nll += regularization * np.sum(log_diag ** 2)
    return nll


def estimate_diagonal_sigma(
    observations: List[ObservedChoice],
    regularization: float = 0.01,
    money_bounds: Tuple[float, float] = (1.0, 1000.0),
    moral_bounds: Tuple[float, float] = (0.01, 100.0),
    init_money_var: float = 25.0,
    init_moral_var: float = 0.25,
) -> CalibratedSigma:
    """Estimate diagonal sigma with bounded parameters.

    Uses L-BFGS-B with bounds to prevent:
    - Money weight collapsing to zero (sigma[0,0] → ∞)
    - Moral dimensions having unreasonable weights

    Args:
        observations: Choice data.
        regularization: L2 penalty.
        money_bounds: (min, max) for sigma[0,0].
        moral_bounds: (min, max) for sigma[k,k], k>0.
        init_money_var: Initial money variance.
        init_moral_var: Initial moral dimension variance.

    Returns:
        CalibratedSigma with optimal diagonal covariance.
    """
    # Parameterize in log-space for positivity
    init = np.zeros(N_DIMS)
    init[Dim.CONSEQUENCES] = np.log(init_money_var)
    for d in range(1, N_DIMS):
        init[d] = np.log(init_moral_var)

    # Bounds in log-space
    bounds = []
    bounds.append((np.log(money_bounds[0]), np.log(money_bounds[1])))
    for _ in range(1, N_DIMS):
        bounds.append((np.log(moral_bounds[0]), np.log(moral_bounds[1])))

    result = minimize(
        _softmax_nll,
        init,
        args=(observations, regularization),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000},
    )

    sigma_diag = np.exp(result.x)
    sigma = np.diag(sigma_diag)

    return CalibratedSigma(
        sigma=sigma,
        log_likelihood=-result.fun,
        n_observations=len(observations),
        n_parameters=N_DIMS,
        converged=result.success,
    )


def cross_validate(
    observations: List[ObservedChoice],
    n_folds: int = 5,
    regularization_values: List[float] | None = None,
    rng: np.random.Generator | None = None,
) -> Tuple[float, List[float]]:
    """K-fold cross-validation to select regularization strength.

    Returns (best_regularization, list_of_cv_scores).
    """
    if regularization_values is None:
        regularization_values = [0.001, 0.01, 0.05, 0.1, 0.5]

    if rng is None:
        rng = np.random.default_rng(42)

    indices = np.arange(len(observations))
    rng.shuffle(indices)
    fold_size = len(indices) // n_folds

    cv_scores = []
    for reg in regularization_values:
        fold_nlls = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            train_obs = [observations[i] for i in train_idx]
            val_obs = [observations[i] for i in val_idx]

            # Train
            cal = estimate_diagonal_sigma(train_obs, regularization=reg)
            # Evaluate on validation fold
            val_nll = _softmax_nll(
                np.log(np.diag(cal.sigma)), val_obs, regularization=0.0
            )
            fold_nlls.append(val_nll / len(val_obs))

        mean_nll = np.mean(fold_nlls)
        cv_scores.append(mean_nll)

    best_idx = np.argmin(cv_scores)
    return regularization_values[best_idx], cv_scores


def bootstrap_confidence(
    observations: List[ObservedChoice],
    n_bootstrap: int = 50,
    regularization: float = 0.01,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap confidence intervals on sigma diagonal.

    Returns (median_sigma_diag, ci_low, ci_high) — 95% CI.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(observations)
    sigmas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_obs = [observations[i] for i in idx]
        cal = estimate_diagonal_sigma(boot_obs, regularization=regularization)
        sigmas.append(np.diag(cal.sigma))

    sigmas = np.array(sigmas)
    median = np.median(sigmas, axis=0)
    ci_low = np.percentile(sigmas, 2.5, axis=0)
    ci_high = np.percentile(sigmas, 97.5, axis=0)
    return median, ci_low, ci_high
