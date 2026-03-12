# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Distance metrics for the Economic Decision Complex.

The core metric is Mahalanobis distance weighted by boundary penalties:

    w(v_i, v_j) = sqrt(Δa^T Σ^{-1} Δa) + Σ_k β_k · 1[boundary k crossed]

where Δa is the 9-dimensional attribute-vector change, Σ is the covariance
matrix encoding dimensional interactions, and β_k are moral-economic
boundary penalties.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from eris_econ.dimensions import N_DIMS, Dim


def mahalanobis_distance(
    a: np.ndarray,
    b: np.ndarray,
    sigma_inv: np.ndarray,
) -> float:
    """Mahalanobis distance between two attribute vectors.

    Args:
        a, b: length-9 attribute vectors
        sigma_inv: [9, 9] inverse covariance matrix

    Returns:
        sqrt(Δ^T Σ^{-1} Δ) — the multi-dimensional distance
    """
    delta = b - a
    return float(np.sqrt(max(0.0, delta @ sigma_inv @ delta)))


def boundary_penalty(
    a: np.ndarray,
    b: np.ndarray,
    boundaries: Dict[str, float],
) -> float:
    """Compute total boundary penalty for a state transition.

    Checks for moral-economic boundary crossings and sums penalties.
    Standard boundaries:
    - "theft": rights dimension goes negative (β = inf for sacred value)
    - "coercion": autonomy dimension goes below threshold
    - "deception": epistemic dimension drops significantly
    - "exploitation": fairness dimension drops while consequences improve
    """
    penalty = 0.0
    delta = b - a

    for name, beta in boundaries.items():
        crossed = False

        if name == "theft" and b[Dim.RIGHTS] < 0 < a[Dim.RIGHTS]:
            crossed = True
        elif name == "coercion" and delta[Dim.AUTONOMY] < -0.5:
            crossed = True
        elif name == "deception" and delta[Dim.EPISTEMIC] < -0.3:
            crossed = True
        elif name == "exploitation":
            # Gaining monetary value while decreasing fairness for counterparty
            if delta[Dim.CONSEQUENCES] > 0 and delta[Dim.FAIRNESS] < -0.3:
                crossed = True
        elif name == "sacred_value":
            # Any dimension crossing from positive to zero on sacred goods
            if any(a[i] > 0 and b[i] <= 0 for i in range(N_DIMS)):
                crossed = True
        elif name == "promise_breaking" and delta[Dim.LEGITIMACY] < -0.5:
            crossed = True

        if crossed:
            if np.isinf(beta):
                return float("inf")
            penalty += beta

    return penalty


def edge_weight(
    a: np.ndarray,
    b: np.ndarray,
    sigma_inv: np.ndarray,
    boundaries: Dict[str, float],
) -> float:
    """Total edge weight: Mahalanobis distance + boundary penalties.

    This is the full cost of transitioning from economic state a to state b.
    """
    dist = mahalanobis_distance(a, b, sigma_inv)
    pen = boundary_penalty(a, b, boundaries)
    return dist + pen


def loss_aversion_ratio(
    gain_state: np.ndarray,
    loss_state: np.ndarray,
    reference: np.ndarray,
    sigma_inv: np.ndarray,
) -> float:
    """Compute the loss aversion ratio λ = w_loss / w_gain.

    In the geometric framework, λ ≈ 2.25 emerges because losses
    traverse more dimensions (rights, identity, fairness, social impact)
    than gains (primarily consequences only).

    Args:
        gain_state: state after a gain of magnitude M
        loss_state: state after a loss of magnitude M
        reference: reference point (current state)
        sigma_inv: inverse covariance matrix

    Returns:
        λ = d(reference, loss_state) / d(reference, gain_state)
    """
    d_gain = mahalanobis_distance(reference, gain_state, sigma_inv)
    d_loss = mahalanobis_distance(reference, loss_state, sigma_inv)
    if d_gain < 1e-10:
        return float("inf")
    return d_loss / d_gain
