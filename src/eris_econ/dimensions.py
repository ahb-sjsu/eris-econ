# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""The nine dimensions of the Economic Decision Complex.

Every economic state is a point in R^9.  These nine dimensions capture
the full space of human economic decision-making — projecting to d_1
alone (monetary value) recovers classical utility theory, but loses
information that is mathematically irrecoverable (Scalar Irrecoverability
Theorem).

Dimensions d_1 through d_4 are *transferable* in bilateral exchange
(conservation: Δd_k(A) + Δd_k(B) = 0).  Dimensions d_5 through d_9
are *evaluative* — not conserved, allowing mutual gains from trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Dim(IntEnum):
    """The nine economic decision dimensions."""

    CONSEQUENCES = 0  # d_1: monetary cost, material outcome, expected value
    RIGHTS = 1  # d_2: property rights, contractual obligations
    FAIRNESS = 2  # d_3: distributional justice, procedural justice, reciprocity
    AUTONOMY = 3  # d_4: freedom of choice, coercion aversion, voluntariness
    PRIVACY_TRUST = 4  # d_5: information asymmetry, disclosure norms, fiduciary duty
    SOCIAL_IMPACT = 5  # d_6: externalities, reputation, community effects
    VIRTUE_IDENTITY = 6  # d_7: self-image, moral identity, character consistency
    LEGITIMACY = 7  # d_8: institutional trust, rule compliance, procedural regularity
    EPISTEMIC = 8  # d_9: information quality, confidence, ambiguity


N_DIMS = 9

# Which dimensions are conserved in bilateral exchange
TRANSFERABLE_DIMS = frozenset(
    {
        Dim.CONSEQUENCES,
        Dim.RIGHTS,
        Dim.AUTONOMY,
    }
)

# Which dimensions can generate mutual gains (not conserved)
EVALUATIVE_DIMS = frozenset(
    {
        Dim.PRIVACY_TRUST,
        Dim.SOCIAL_IMPACT,
        Dim.VIRTUE_IDENTITY,
        Dim.LEGITIMACY,
        Dim.EPISTEMIC,
    }
)

# Fairness (d_3) is partially transferable — context-dependent
PARTIALLY_TRANSFERABLE = frozenset({Dim.FAIRNESS})

DIM_NAMES = {
    Dim.CONSEQUENCES: "Consequences",
    Dim.RIGHTS: "Rights/Entitlements",
    Dim.FAIRNESS: "Fairness",
    Dim.AUTONOMY: "Autonomy",
    Dim.PRIVACY_TRUST: "Privacy/Trust",
    Dim.SOCIAL_IMPACT: "Social Impact",
    Dim.VIRTUE_IDENTITY: "Virtue/Identity",
    Dim.LEGITIMACY: "Legitimacy",
    Dim.EPISTEMIC: "Epistemic Status",
}


@dataclass(frozen=True)
class EconomicState:
    """A point in the 9-dimensional economic decision space.

    Represents an agent's current configuration of goods, obligations,
    relationships, information, and moral standing.
    """

    values: tuple  # Length-9 tuple of floats

    def __post_init__(self):
        if len(self.values) != N_DIMS:
            raise ValueError(f"Expected {N_DIMS} dimensions, got {len(self.values)}")

    def __getitem__(self, dim: Dim | int) -> float:
        return self.values[int(dim)]

    def monetary(self) -> float:
        """Shorthand for the consequences/monetary dimension."""
        return self.values[Dim.CONSEQUENCES]
