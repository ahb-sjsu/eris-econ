# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Behavioral economics phenomena as geometric properties.

These are not ad-hoc biases — they emerge structurally from the
multi-dimensional Mahalanobis metric:

- Loss aversion: losses traverse more dimensions → λ ≈ 2.25
- Reference dependence: distance is measured FROM current state
- Framing effects: same decision, different gauge → different path weights
- Endowment effect: ownership activates rights + identity dimensions
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from eris_econ.dimensions import Dim, N_DIMS
from eris_econ.metrics import loss_aversion_ratio, mahalanobis_distance


def compute_loss_aversion(
    sigma: np.ndarray,
    magnitude: float = 1.0,
) -> float:
    """Compute the emergent loss aversion ratio λ from the metric tensor.

    A gain of magnitude M changes primarily d_1 (consequences).
    A loss of magnitude M changes d_1 AND activates d_2 (rights violation
    risk), d_3 (fairness), d_6 (social), d_7 (identity).

    λ = d(ref, loss_state) / d(ref, gain_state)

    Empirical target: λ ≈ 2.0-2.5 (Kahneman & Tversky).
    """
    sigma_inv = np.linalg.inv(sigma + 1e-10 * np.eye(N_DIMS))

    reference = np.zeros(N_DIMS)
    reference[Dim.CONSEQUENCES] = 10.0  # baseline wealth
    reference[Dim.RIGHTS] = 1.0
    reference[Dim.FAIRNESS] = 0.5
    reference[Dim.AUTONOMY] = 1.0
    reference[Dim.VIRTUE_IDENTITY] = 0.5

    # Gain: primarily monetary, small positive social
    gain = reference.copy()
    gain[Dim.CONSEQUENCES] += magnitude
    gain[Dim.SOCIAL_IMPACT] += 0.05 * magnitude

    # Loss: monetary decline + rights threat + fairness injury + identity hit
    loss = reference.copy()
    loss[Dim.CONSEQUENCES] -= magnitude
    loss[Dim.RIGHTS] -= 0.15 * magnitude  # ownership threat
    loss[Dim.FAIRNESS] -= 0.1 * magnitude  # perceived unfairness
    loss[Dim.SOCIAL_IMPACT] -= 0.1 * magnitude  # social cost
    loss[Dim.VIRTUE_IDENTITY] -= 0.1 * magnitude  # identity blow

    return loss_aversion_ratio(gain, loss, reference, sigma_inv)


def reference_dependence(
    current: np.ndarray,
    option_a: np.ndarray,
    option_b: np.ndarray,
    sigma_inv: np.ndarray,
) -> Tuple[float, float]:
    """Demonstrate reference dependence: preference depends on starting point.

    Returns (cost_a, cost_b) — costs from current state to each option.
    The SAME pair (A, B) can have different relative costs depending
    on the reference point `current`.
    """
    cost_a = mahalanobis_distance(current, option_a, sigma_inv)
    cost_b = mahalanobis_distance(current, option_b, sigma_inv)
    return cost_a, cost_b


def endowment_effect(
    sigma: np.ndarray,
    item_value: float = 5.0,
) -> Tuple[float, float]:
    """Compute willingness-to-accept vs willingness-to-pay gap.

    WTA > WTP because selling activates d_2 (loss of rights), d_7 (identity),
    and d_6 (social) on top of d_1 (monetary).

    Returns (wta_distance, wtp_distance) — the manifold distance of
    selling vs buying.
    """
    sigma_inv = np.linalg.inv(sigma + 1e-10 * np.eye(N_DIMS))

    # Owner's state: has item (strong attachment across multiple dimensions)
    owner = np.zeros(N_DIMS)
    owner[Dim.CONSEQUENCES] = item_value
    owner[Dim.RIGHTS] = 1.0
    owner[Dim.AUTONOMY] = 0.8
    owner[Dim.VIRTUE_IDENTITY] = 0.7  # identified with possession
    owner[Dim.SOCIAL_IMPACT] = 0.3

    # After selling: gained money but lost on 4 non-monetary dimensions
    sold = np.zeros(N_DIMS)
    sold[Dim.CONSEQUENCES] = item_value * 1.2  # modest cash gain
    sold[Dim.RIGHTS] = 0.2  # lost ownership rights
    sold[Dim.AUTONOMY] = 0.4  # less autonomy over the item
    sold[Dim.VIRTUE_IDENTITY] = 0.2  # identity loss (gave up "my thing")
    sold[Dim.SOCIAL_IMPACT] = -0.1  # social cost of selling

    # Buyer's state: has money, neutral attachment
    buyer = np.zeros(N_DIMS)
    buyer[Dim.CONSEQUENCES] = item_value * 2
    buyer[Dim.RIGHTS] = 0.5
    buyer[Dim.AUTONOMY] = 0.6
    buyer[Dim.VIRTUE_IDENTITY] = 0.4

    # After buying: spent money but gains are primarily on d_1 and d_2
    bought = np.zeros(N_DIMS)
    bought[Dim.CONSEQUENCES] = item_value  # spent money
    bought[Dim.RIGHTS] = 0.9  # gained ownership
    bought[Dim.AUTONOMY] = 0.7  # modest autonomy gain
    bought[Dim.VIRTUE_IDENTITY] = 0.5  # slight identity gain

    wta_distance = mahalanobis_distance(owner, sold, sigma_inv)
    wtp_distance = mahalanobis_distance(buyer, bought, sigma_inv)

    return wta_distance, wtp_distance


def framing_as_gauge(
    state: np.ndarray,
    frame_rotation: np.ndarray,
    sigma_inv: np.ndarray,
) -> np.ndarray:
    """Model framing effects as gauge transformations.

    A "frame" is a rotation of the description basis that changes
    how the same objective state is perceived.  The metric tensor
    is NOT invariant under frame rotation for boundedly-rational agents
    → framing effects emerge.

    For a gauge-invariant (rational) agent, the Mahalanobis distance
    should be identical regardless of frame.

    Args:
        state: the objective attribute vector
        frame_rotation: [9, 9] orthogonal matrix (the frame change)
        sigma_inv: inverse covariance matrix

    Returns:
        Framed state = R @ state
    """
    return frame_rotation @ state
