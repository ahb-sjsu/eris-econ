# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Improved game encodings with rejection probability and finer resolution.

Key improvement: the ultimatum game now incorporates the responder's
rejection probability, based on Camerer (2003) meta-analysis data.
This changes the proposer's expected outcome: low offers have a high
probability of rejection (yielding $0), so the expected monetary value
of low offers is much lower than the nominal keep amount.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from eris_econ.dimensions import Dim, N_DIMS
from eris_econ.metrics import mahalanobis_distance


def rejection_probability(offer_pct: float) -> float:
    """Empirical rejection probability from Camerer (2003) meta-analysis.

    Smoothed logistic fit to published rejection rates.
    """
    # Logistic: p(reject) = 1 / (1 + exp(k * (offer - threshold)))
    # Fitted to: 0%→~95%, 10%→~60%, 20%→~30%, 30%→~10%, 40%→~2%, 50%→~0%
    k = 0.15  # steepness
    threshold = 18.0  # 50% rejection at ~18% offer
    return 1.0 / (1.0 + np.exp(k * (offer_pct - threshold)))


def ultimatum_state(
    stake: float,
    offer_pct: float,
    include_rejection: bool = True,
) -> np.ndarray:
    """Map an ultimatum offer to a 9D attribute vector.

    When include_rejection=True, the monetary outcome is the EXPECTED
    value accounting for rejection probability.  Identity and social
    dimensions also account for the cost of rejection.
    """
    keep = stake * (1 - offer_pct / 100)
    give = stake * offer_pct / 100

    if include_rejection:
        p_rej = rejection_probability(offer_pct)
        p_acc = 1 - p_rej
    else:
        p_rej = 0.0
        p_acc = 1.0

    # Expected monetary outcome
    expected_money = p_acc * keep  # rejection → $0

    # Fairness: 50/50 is maximum fairness
    fairness = 0.1 + 0.8 * min(offer_pct / 50, 1.0)

    # Identity: generous offers enhance self-image
    # BUT: rejection hurts identity ("my offer was rejected")
    base_identity = 0.3 + 0.5 * min(offer_pct / 50, 1.0)
    rejection_identity_cost = 0.2  # identity hit from rejection
    identity = p_acc * base_identity + p_rej * (base_identity - rejection_identity_cost)

    # Social: offering fairly avoids social sanction
    # Rejection is a social failure
    base_social = -0.2 + 0.6 * min(offer_pct / 50, 1.0)
    rejection_social_cost = 0.3
    social = p_acc * base_social + p_rej * (base_social - rejection_social_cost)

    # Epistemic: uncertainty about rejection
    epistemic = 0.5 + 0.3 * p_acc  # more certain at high offers

    state = np.zeros(N_DIMS)
    state[Dim.CONSEQUENCES] = expected_money
    state[Dim.RIGHTS] = 1.0
    state[Dim.FAIRNESS] = fairness
    state[Dim.AUTONOMY] = 1.0
    state[Dim.PRIVACY_TRUST] = 0.5
    state[Dim.SOCIAL_IMPACT] = social
    state[Dim.VIRTUE_IDENTITY] = identity
    state[Dim.LEGITIMACY] = 0.5
    state[Dim.EPISTEMIC] = epistemic
    return state


def public_goods_state(
    endowment: float,
    contrib_pct: float,
    n_players: int = 4,
    multiplier: float = 2.0,
    others_contrib_pct: float = 50.0,
) -> np.ndarray:
    """Map a public goods contribution to a 9D attribute vector."""
    contrib = endowment * contrib_pct / 100
    others_contrib = endowment * others_contrib_pct / 100 * (n_players - 1)
    total_pool = (contrib + others_contrib) * multiplier / n_players
    remaining = endowment - contrib + total_pool

    fairness = 0.1 + 0.8 * (contrib_pct / 100)
    identity = 0.2 + 0.6 * (contrib_pct / 100)
    social = -0.4 + 0.8 * (contrib_pct / 100)

    state = np.zeros(N_DIMS)
    state[Dim.CONSEQUENCES] = remaining
    state[Dim.RIGHTS] = 1.0
    state[Dim.FAIRNESS] = fairness
    state[Dim.AUTONOMY] = 1.0
    state[Dim.PRIVACY_TRUST] = 0.5
    state[Dim.SOCIAL_IMPACT] = social
    state[Dim.VIRTUE_IDENTITY] = identity
    state[Dim.LEGITIMACY] = 0.5
    state[Dim.EPISTEMIC] = 0.5
    return state


def predict_game(
    sigma: np.ndarray,
    game: str,
    stake: float = 10.0,
    endowment: float = 20.0,
    resolution: int = 5,
) -> Tuple[float, List[Tuple[float, float]]]:
    """Predict optimal choice for a game.

    Args:
        sigma: calibrated covariance matrix
        game: "ultimatum", "ultimatum_no_rejection", "dictator", "public_goods"
        stake: for ultimatum/dictator
        endowment: for public goods
        resolution: percentage step size

    Returns:
        (optimal_pct, [(pct, cost), ...])
    """
    sigma_inv = np.linalg.inv(sigma + 1e-10 * np.eye(N_DIMS))

    if game.startswith("ultimatum"):
        include_rej = "no_rejection" not in game
        max_pct = 50
        start = np.zeros(N_DIMS)
        start[Dim.CONSEQUENCES] = stake
        start[Dim.RIGHTS] = 1.0
        start[Dim.FAIRNESS] = 0.5
        start[Dim.AUTONOMY] = 1.0
        start[Dim.PRIVACY_TRUST] = 0.5
        start[Dim.VIRTUE_IDENTITY] = 0.5
        start[Dim.LEGITIMACY] = 0.5
        start[Dim.EPISTEMIC] = 0.5 + 0.3  # certain before offering

        costs = []
        for pct in range(0, max_pct + 1, resolution):
            state = ultimatum_state(stake, pct, include_rejection=include_rej)
            cost = mahalanobis_distance(start, state, sigma_inv)
            costs.append((float(pct), cost))

    elif game == "dictator":
        max_pct = 50
        start = np.zeros(N_DIMS)
        start[Dim.CONSEQUENCES] = stake
        start[Dim.RIGHTS] = 1.0
        start[Dim.FAIRNESS] = 0.5
        start[Dim.AUTONOMY] = 1.0
        start[Dim.PRIVACY_TRUST] = 0.5
        start[Dim.VIRTUE_IDENTITY] = 0.5
        start[Dim.LEGITIMACY] = 0.5
        start[Dim.EPISTEMIC] = 0.5

        costs = []
        for pct in range(0, max_pct + 1, resolution):
            # Dictator: no rejection, lower identity/social stakes
            state = ultimatum_state(stake, pct, include_rejection=False)
            # Reduce identity/social activation (no rejection threat)
            state[Dim.VIRTUE_IDENTITY] *= 0.8
            state[Dim.SOCIAL_IMPACT] *= 0.7
            cost = mahalanobis_distance(start, state, sigma_inv)
            costs.append((float(pct), cost))

    elif game == "public_goods":
        start = np.zeros(N_DIMS)
        start[Dim.CONSEQUENCES] = endowment
        start[Dim.RIGHTS] = 1.0
        start[Dim.FAIRNESS] = 0.5
        start[Dim.AUTONOMY] = 1.0
        start[Dim.PRIVACY_TRUST] = 0.5
        start[Dim.SOCIAL_IMPACT] = 0.0
        start[Dim.VIRTUE_IDENTITY] = 0.5
        start[Dim.LEGITIMACY] = 0.5
        start[Dim.EPISTEMIC] = 0.5

        costs = []
        for pct in range(0, 101, resolution):
            state = public_goods_state(endowment, pct)
            cost = mahalanobis_distance(start, state, sigma_inv)
            costs.append((float(pct), cost))

    else:
        raise ValueError(f"Unknown game: {game}")

    optimal = min(costs, key=lambda x: x[1])
    return optimal[0], costs
