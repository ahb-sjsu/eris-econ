# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Standard economic games modeled on the 9D decision manifold.

Each game is constructed as an Economic Decision Complex where the
geometric framework explains observed deviations from Nash equilibrium:
- Ultimatum: equal splits because fairness + identity penalties > monetary gain
- Dictator: positive giving because identity + social cost of $0 offer
- Public goods: initial cooperation decays but never reaches zero
- Trust: positive returns because legitimacy + reputation penalties
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from eris_econ.dimensions import Dim, EconomicState, N_DIMS
from eris_econ.manifold import EconomicDecisionComplex


def _default_sigma() -> np.ndarray:
    """Default covariance matrix with empirically-motivated off-diagonals.

    Key structure: consequences weakly correlated with fairness;
    rights strongly correlated with legitimacy; identity correlated
    with social impact.
    """
    sigma = np.eye(N_DIMS)
    # Monetary dimension has high variance (less weight per unit)
    # because money varies on a much larger scale than moral dims
    sigma[Dim.CONSEQUENCES, Dim.CONSEQUENCES] = 25.0
    # Moral dimensions have low variance (high weight per unit)
    for d in range(1, N_DIMS):
        sigma[d, d] = 0.25
    # Consequences-Fairness interaction
    sigma[Dim.CONSEQUENCES, Dim.FAIRNESS] = 0.5
    sigma[Dim.FAIRNESS, Dim.CONSEQUENCES] = 0.5
    # Rights-Legitimacy coupling
    sigma[Dim.RIGHTS, Dim.LEGITIMACY] = 0.15
    sigma[Dim.LEGITIMACY, Dim.RIGHTS] = 0.15
    # Identity-Social coupling
    sigma[Dim.VIRTUE_IDENTITY, Dim.SOCIAL_IMPACT] = 0.1
    sigma[Dim.SOCIAL_IMPACT, Dim.VIRTUE_IDENTITY] = 0.1
    # Trust-Epistemic coupling
    sigma[Dim.PRIVACY_TRUST, Dim.EPISTEMIC] = 0.1
    sigma[Dim.EPISTEMIC, Dim.PRIVACY_TRUST] = 0.1
    return sigma


def _state(
    money: float = 0.0,
    rights: float = 1.0,
    fairness: float = 0.5,
    autonomy: float = 1.0,
    trust: float = 0.5,
    social: float = 0.0,
    identity: float = 0.5,
    legitimacy: float = 0.5,
    epistemic: float = 0.5,
) -> EconomicState:
    """Helper to create an EconomicState with named dimensions."""
    return EconomicState(
        (
            money,
            rights,
            fairness,
            autonomy,
            trust,
            social,
            identity,
            legitimacy,
            epistemic,
        )
    )


def ultimatum_game(
    stake: float = 10.0,
    sigma: np.ndarray | None = None,
) -> EconomicDecisionComplex:
    """Construct the proposer's decision complex for the ultimatum game.

    Explains why proposers offer ~40-50% (not $0.01):
    the fairness (d_3) and identity (d_7) penalty for low offers
    outweighs the monetary gain on d_1.
    """
    if sigma is None:
        sigma = _default_sigma()

    boundaries = {
        "exploitation": 5.0,  # large penalty for clearly unfair splits
    }

    E = EconomicDecisionComplex(sigma=sigma, boundaries=boundaries)

    # Starting state: has the stake, neutral on other dimensions
    E.add_vertex("start", _state(money=stake, fairness=0.5, identity=0.5))

    # Possible offers (keep, give)
    for give_pct in [0, 10, 20, 30, 40, 50]:
        give = stake * give_pct / 100
        keep = stake - give

        # Higher offers → better fairness and identity scores
        fairness = 0.1 + 0.8 * (give_pct / 50)  # 50/50 = max fairness
        identity = 0.3 + 0.5 * (give_pct / 50)
        social = -0.2 + 0.6 * (give_pct / 50)

        vid = f"offer_{give_pct}"
        E.add_vertex(
            vid,
            _state(
                money=keep,
                fairness=fairness,
                identity=identity,
                social=social,
            ),
        )
        E.add_edge("start", vid, label=f"offer {give_pct}%")

    E.compute_weights()
    return E


def dictator_game(
    stake: float = 10.0,
    sigma: np.ndarray | None = None,
) -> EconomicDecisionComplex:
    """Dictator game — no rejection threat, yet people still give ~20-30%.

    The identity (d_7) and social impact (d_6) costs of keeping everything
    are non-zero even without strategic incentive.
    """
    if sigma is None:
        sigma = _default_sigma()

    E = EconomicDecisionComplex(sigma=sigma)

    E.add_vertex("start", _state(money=stake, fairness=0.5, identity=0.5))

    for give_pct in [0, 10, 20, 30, 40, 50]:
        give = stake * give_pct / 100
        keep = stake - give
        fairness = 0.1 + 0.8 * (give_pct / 50)
        identity = 0.2 + 0.6 * (give_pct / 50)
        social = -0.3 + 0.7 * (give_pct / 50)

        vid = f"give_{give_pct}"
        E.add_vertex(
            vid,
            _state(
                money=keep,
                fairness=fairness,
                identity=identity,
                social=social,
            ),
        )
        E.add_edge("start", vid, label=f"give {give_pct}%")

    E.compute_weights()
    return E


def prisoners_dilemma(
    sigma: np.ndarray | None = None,
) -> Tuple[EconomicDecisionComplex, EconomicDecisionComplex]:
    """Prisoner's dilemma as two coupled decision complexes.

    Cooperation is the lower-cost path on the full manifold when
    the identity/social/legitimacy penalties for defection exceed
    the monetary gain.

    Returns (player_A_complex, player_B_complex).
    """
    if sigma is None:
        sigma = _default_sigma()

    boundaries = {"promise_breaking": 3.0}

    def make_player_complex(cooperate_money, defect_money):
        E = EconomicDecisionComplex(sigma=sigma, boundaries=boundaries)
        E.add_vertex("start", _state(money=0, fairness=0.5, identity=0.5, legitimacy=0.5))
        E.add_vertex(
            "cooperate",
            _state(
                money=cooperate_money,
                fairness=0.8,
                identity=0.8,
                social=0.6,
                legitimacy=0.7,
            ),
        )
        E.add_vertex(
            "defect",
            _state(
                money=defect_money,
                fairness=0.1,
                identity=0.2,
                social=-0.3,
                legitimacy=0.2,
            ),
        )
        E.add_edge("start", "cooperate", label="cooperate")
        E.add_edge("start", "defect", label="defect")
        E.compute_weights()
        return E

    # Payoffs: (C,C)=3, (C,D)=0, (D,C)=5, (D,D)=1
    # For now, build each player assuming other cooperates
    A = make_player_complex(cooperate_money=3, defect_money=5)
    B = make_player_complex(cooperate_money=3, defect_money=5)
    return A, B


def public_goods_game(
    n_players: int = 4,
    endowment: float = 10.0,
    multiplier: float = 2.0,
    sigma: np.ndarray | None = None,
) -> EconomicDecisionComplex:
    """Public goods game — a single player's decision complex.

    Explains initial high contributions (social pressure) that decay
    (learning others free-ride) but never reach zero (identity cost).
    """
    if sigma is None:
        sigma = _default_sigma()

    E = EconomicDecisionComplex(sigma=sigma)
    E.add_vertex(
        "start",
        _state(
            money=endowment,
            fairness=0.5,
            identity=0.5,
            social=0.0,
        ),
    )

    for contrib_pct in [0, 25, 50, 75, 100]:
        contrib = endowment * contrib_pct / 100
        # Assume others contribute 50% on average
        others_contrib = endowment * 0.5 * (n_players - 1)
        total_pool = (contrib + others_contrib) * multiplier / n_players
        remaining = endowment - contrib + total_pool

        fairness = 0.1 + 0.8 * (contrib_pct / 100)
        identity = 0.2 + 0.6 * (contrib_pct / 100)
        social = -0.4 + 0.8 * (contrib_pct / 100)

        vid = f"contrib_{contrib_pct}"
        E.add_vertex(
            vid,
            _state(
                money=remaining,
                fairness=fairness,
                identity=identity,
                social=social,
            ),
        )
        E.add_edge("start", vid, label=f"contribute {contrib_pct}%")

    E.compute_weights()
    return E
