# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Prospect theory problems encoded as 9D choice observations.

Maps the 17 Kahneman & Tversky (1979) problems — as replicated by
Ruggeri et al. (2020, Nature Human Behaviour) across 19 countries —
into the 9-dimensional economic decision space.

The geometric framework predicts that behavioral anomalies (certainty
effect, reflection effect, isolation effect) emerge from the metric
structure: sure outcomes have smaller Mahalanobis distance on d_5
(trust/certainty) and d_9 (epistemic), while losses activate d_2
(rights), d_3 (fairness), d_7 (identity) in addition to d_1 (money).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from eris_econ.calibration import ObservedChoice
from eris_econ.dimensions import Dim, N_DIMS


DATA_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class Prospect:
    """A lottery: list of (outcome, probability) pairs."""
    payoffs: List[Tuple[float, float]]  # [(amount, prob), ...]

    @property
    def ev(self) -> float:
        return sum(a * p for a, p in self.payoffs)

    @property
    def is_certain(self) -> bool:
        """A prospect is certain iff it has a single outcome.

        Previously used `any(p >= 0.99)` which misclassified lottery
        tickets like (5000, 0.001; 0, 0.999) as "certain".
        """
        return len(self.payoffs) == 1

    @property
    def max_loss(self) -> float:
        return min(0.0, min(a for a, _ in self.payoffs))

    @property
    def variance(self) -> float:
        ev = self.ev
        return sum(p * (a - ev) ** 2 for a, p in self.payoffs)


@dataclass
class KTProblem:
    """A Kahneman-Tversky choice problem."""
    ruggeri_id: int  # 1-17
    kt_item: str  # e.g., "Item 3" or "Item 3'"
    option_a: Prospect
    option_b: Prospect
    domain: str  # "gain", "loss", or "mixed"
    phenomenon: str  # "certainty", "reflection", "isolation", "loss_aversion"
    endowment: float = 0.0  # initial endowment for items 11-12


# The 17 problems from KT (1979) as replicated by Ruggeri et al. (2020)
KT_PROBLEMS = [
    KTProblem(
        ruggeri_id=1, kt_item="Item 1",
        option_a=Prospect([(2500, 0.33), (2400, 0.66), (0, 0.01)]),
        option_b=Prospect([(2400, 1.0)]),
        domain="gain", phenomenon="certainty",
    ),
    KTProblem(
        ruggeri_id=2, kt_item="Item 2",
        option_a=Prospect([(2500, 0.33), (0, 0.67)]),
        option_b=Prospect([(2400, 0.34), (0, 0.66)]),
        domain="gain", phenomenon="certainty",
    ),
    KTProblem(
        ruggeri_id=3, kt_item="Item 3",
        option_a=Prospect([(4000, 0.80), (0, 0.20)]),
        option_b=Prospect([(3000, 1.0)]),
        domain="gain", phenomenon="certainty",
    ),
    KTProblem(
        ruggeri_id=4, kt_item="Item 4",
        option_a=Prospect([(4000, 0.20), (0, 0.80)]),
        option_b=Prospect([(3000, 0.25), (0, 0.75)]),
        domain="gain", phenomenon="certainty",
    ),
    KTProblem(
        ruggeri_id=5, kt_item="Item 7",
        option_a=Prospect([(6000, 0.45), (0, 0.55)]),
        option_b=Prospect([(3000, 0.90), (0, 0.10)]),
        domain="gain", phenomenon="certainty",
    ),
    KTProblem(
        ruggeri_id=6, kt_item="Item 8",
        option_a=Prospect([(6000, 0.001), (0, 0.999)]),
        option_b=Prospect([(3000, 0.002), (0, 0.998)]),
        domain="gain", phenomenon="certainty",
    ),
    KTProblem(
        ruggeri_id=7, kt_item="Item 3'",
        option_a=Prospect([(-4000, 0.80), (0, 0.20)]),
        option_b=Prospect([(-3000, 1.0)]),
        domain="loss", phenomenon="reflection",
    ),
    KTProblem(
        ruggeri_id=8, kt_item="Item 4'",
        option_a=Prospect([(-4000, 0.20), (0, 0.80)]),
        option_b=Prospect([(-3000, 0.25), (0, 0.75)]),
        domain="loss", phenomenon="reflection",
    ),
    KTProblem(
        ruggeri_id=9, kt_item="Item 7'",
        option_a=Prospect([(-6000, 0.45), (0, 0.55)]),
        option_b=Prospect([(-3000, 0.90), (0, 0.10)]),
        domain="loss", phenomenon="reflection",
    ),
    KTProblem(
        ruggeri_id=10, kt_item="Item 8'",
        option_a=Prospect([(-6000, 0.001), (0, 0.999)]),
        option_b=Prospect([(-3000, 0.002), (0, 0.998)]),
        domain="loss", phenomenon="reflection",
    ),
    KTProblem(
        ruggeri_id=11, kt_item="Item 10",
        option_a=Prospect([(4000, 0.80), (0, 0.20)]),
        option_b=Prospect([(3000, 1.0)]),
        domain="gain", phenomenon="isolation",
        # Two-stage: 75% end, 25% proceed. Effective: A=(4000,0.20), B=(3000,0.25)
    ),
    KTProblem(
        ruggeri_id=12, kt_item="Item 11",
        option_a=Prospect([(1000, 0.50), (0, 0.50)]),
        option_b=Prospect([(500, 1.0)]),
        domain="gain", phenomenon="isolation",
        endowment=2000,
    ),
    KTProblem(
        ruggeri_id=13, kt_item="Item 12",
        option_a=Prospect([(-2000, 0.50), (0, 0.50)]),
        option_b=Prospect([(-1000, 1.0)]),
        domain="loss", phenomenon="isolation",
        endowment=4000,
    ),
    KTProblem(
        ruggeri_id=14, kt_item="Item 13",
        option_a=Prospect([(6000, 0.25), (0, 0.75)]),
        option_b=Prospect([(4000, 0.25), (2000, 0.25), (0, 0.50)]),
        domain="gain", phenomenon="loss_aversion",
    ),
    KTProblem(
        ruggeri_id=15, kt_item="Item 13'",
        option_a=Prospect([(-6000, 0.25), (0, 0.75)]),
        option_b=Prospect([(-4000, 0.25), (-2000, 0.25), (0, 0.50)]),
        domain="loss", phenomenon="loss_aversion",
    ),
    KTProblem(
        ruggeri_id=16, kt_item="Item 14",
        option_a=Prospect([(5000, 0.001), (0, 0.999)]),
        option_b=Prospect([(5, 1.0)]),
        domain="gain", phenomenon="loss_aversion",
    ),
    KTProblem(
        ruggeri_id=17, kt_item="Item 14'",
        option_a=Prospect([(-5000, 0.001), (0, 0.999)]),
        option_b=Prospect([(-5, 1.0)]),
        domain="loss", phenomenon="loss_aversion",
    ),
]


def prospect_to_state(
    prospect: Prospect,
    endowment: float = 0.0,
    scale: float = 1000.0,
) -> np.ndarray:
    """Map a prospect (lottery) to a 9D attribute vector.

    Key mappings:
    - d_1 (consequences): expected value / scale
    - d_2 (rights): preserved unless loss threatens property
    - d_3 (fairness): higher for fair/balanced prospects
    - d_5 (trust): certainty of outcome — FLIPS in loss domain
    - d_7 (identity): risk profile maps to self-image
    - d_9 (epistemic): information quality — FLIPS in loss domain

    Domain-dependent encoding (reflection effect):
    In gains, certainty is desirable (d5, d9 increase with certainty).
    In losses, certainty is distressing — certain losses lock you in,
    while risky losses provide hope. d5 and d9 flip so that risky
    options in the loss domain map to higher epistemic/trust values.
    """
    ev = prospect.ev + endowment
    is_loss = ev < endowment or prospect.max_loss < 0

    state = np.zeros(N_DIMS)

    # d1: expected monetary value, normalized
    state[Dim.CONSEQUENCES] = ev / scale

    # d2: rights — losses threaten property rights
    state[Dim.RIGHTS] = 1.0
    if is_loss:
        loss_severity = abs(prospect.max_loss) / scale
        state[Dim.RIGHTS] = max(0.0, 1.0 - 0.3 * loss_severity)

    # d3: fairness — self-fairness, higher for balanced prospects
    state[Dim.FAIRNESS] = 0.5

    # d4: autonomy — preserved in choice problems
    state[Dim.AUTONOMY] = 1.0

    # Certainty measure: probability concentration
    # Uses max_prob^2 to create the discontinuity at p=1.0 that
    # drives the certainty effect (Allais paradox)
    if prospect.is_certain:
        certainty = 1.0
    else:
        max_prob = max(p for _, p in prospect.payoffs)
        certainty = max_prob ** 2

    # d5: trust/certainty — domain-dependent
    if is_loss:
        # In losses: certainty is distressing (locked into loss),
        # uncertainty provides hope of avoiding loss
        state[Dim.PRIVACY_TRUST] = 0.3 + 0.7 * (1.0 - certainty)
    else:
        # In gains: certainty is desirable
        state[Dim.PRIVACY_TRUST] = 0.3 + 0.7 * certainty

    # d6: social impact — gambling has mild social connotation
    state[Dim.SOCIAL_IMPACT] = 0.0

    # d7: identity — domain-dependent
    if is_loss:
        if prospect.is_certain:
            state[Dim.VIRTUE_IDENTITY] = 0.3  # resigned acceptance
        else:
            state[Dim.VIRTUE_IDENTITY] = 0.5  # fighting against loss
    else:
        if prospect.is_certain:
            state[Dim.VIRTUE_IDENTITY] = 0.7  # "I am prudent"
        else:
            state[Dim.VIRTUE_IDENTITY] = 0.4  # "I am taking a risk"

    # d8: legitimacy — preserved in fair experiments
    state[Dim.LEGITIMACY] = 0.5

    # d9: epistemic — domain-dependent (same logic as d5)
    if is_loss:
        # Risky losses have higher epistemic value (hope)
        state[Dim.EPISTEMIC] = 0.3 + 0.7 * (1.0 - certainty)
    else:
        # Certain gains have higher epistemic value (clarity)
        state[Dim.EPISTEMIC] = 0.3 + 0.7 * certainty

    return state


def encode_ruggeri_data(
    filepath: Path | None = None,
) -> List[ObservedChoice]:
    """Encode Ruggeri et al. (2020) data as ObservedChoice objects.

    Each subject's choice on each problem becomes one observation.
    Option A = 1.0, Option B = 0.0 in the CSV.
    """
    if filepath is None:
        filepath = DATA_DIR / "ruggeri_prospect_theory.csv"

    observations = []
    problems_by_id = {p.ruggeri_id: p for p in KT_PROBLEMS}

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for pid in range(1, 18):
                val = row.get(str(pid), "")
                if not val:
                    continue

                problem = problems_by_id[pid]
                choice = float(val)  # 1.0 = chose A, 0.0 = chose B

                state_a = prospect_to_state(
                    problem.option_a, problem.endowment
                )
                state_b = prospect_to_state(
                    problem.option_b, problem.endowment
                )
                # Reference state: no gamble, just the endowment
                start = np.zeros(N_DIMS)
                start[Dim.CONSEQUENCES] = problem.endowment / 1000.0
                start[Dim.RIGHTS] = 1.0
                start[Dim.FAIRNESS] = 0.5
                start[Dim.AUTONOMY] = 1.0
                start[Dim.PRIVACY_TRUST] = 1.0
                start[Dim.VIRTUE_IDENTITY] = 0.5
                start[Dim.LEGITIMACY] = 0.5
                start[Dim.EPISTEMIC] = 1.0

                if choice >= 0.5:  # chose A
                    observations.append(ObservedChoice(
                        start=start, chosen=state_a, rejected=[state_b]
                    ))
                else:  # chose B
                    observations.append(ObservedChoice(
                        start=start, chosen=state_b, rejected=[state_a]
                    ))

    return observations


def encode_ruggeri_by_country(
    filepath: Path | None = None,
) -> Dict[str, List[ObservedChoice]]:
    """Encode Ruggeri data grouped by country."""
    if filepath is None:
        filepath = DATA_DIR / "ruggeri_prospect_theory.csv"

    country_obs: Dict[str, List[ObservedChoice]] = {}
    problems_by_id = {p.ruggeri_id: p for p in KT_PROBLEMS}

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            country = row["Country"]
            if country not in country_obs:
                country_obs[country] = []

            for pid in range(1, 18):
                val = row.get(str(pid), "")
                if not val:
                    continue

                problem = problems_by_id[pid]
                choice = float(val)

                state_a = prospect_to_state(
                    problem.option_a, problem.endowment
                )
                state_b = prospect_to_state(
                    problem.option_b, problem.endowment
                )
                start = np.zeros(N_DIMS)
                start[Dim.CONSEQUENCES] = problem.endowment / 1000.0
                start[Dim.RIGHTS] = 1.0
                start[Dim.FAIRNESS] = 0.5
                start[Dim.AUTONOMY] = 1.0
                start[Dim.PRIVACY_TRUST] = 1.0
                start[Dim.VIRTUE_IDENTITY] = 0.5
                start[Dim.LEGITIMACY] = 0.5
                start[Dim.EPISTEMIC] = 1.0

                if choice >= 0.5:
                    country_obs[country].append(ObservedChoice(
                        start=start, chosen=state_a, rejected=[state_b]
                    ))
                else:
                    country_obs[country].append(ObservedChoice(
                        start=start, chosen=state_b, rejected=[state_a]
                    ))

    return country_obs
