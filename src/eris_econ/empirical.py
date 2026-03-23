# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Calibration against real experimental economics data.

Uses published datasets to estimate Σ from observed choices, then
generates out-of-sample predictions to test the framework.

Datasets:
- Fraser & Nettle (2020): Ultimatum + public goods game (Zenodo 3764693)
- Ruggeri et al. (2020): Prospect theory replication, 19 countries (OSF esxc4)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from eris_econ.calibration import ObservedChoice, estimate_sigma
from eris_econ.dimensions import Dim, N_DIMS
from eris_econ.games import _default_sigma, _state
from eris_econ.manifold import EconomicDecisionComplex
from eris_econ.metrics import mahalanobis_distance


DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# Ultimatum game: encode offers as attribute-vector choices
# ---------------------------------------------------------------------------

def _ultimatum_state(stake: float, offer_pct: float) -> np.ndarray:
    """Map an ultimatum offer to a 9D attribute vector.

    Higher offers -> better fairness and identity, worse monetary outcome.
    """
    give = stake * offer_pct / 100
    keep = stake - give

    # Fairness: 50/50 is maximum fairness
    fairness = 0.1 + 0.8 * min(offer_pct / 50, 1.0)
    # Identity: generous offers enhance self-image
    identity = 0.3 + 0.5 * min(offer_pct / 50, 1.0)
    # Social: offering fairly avoids social sanction
    social = -0.2 + 0.6 * min(offer_pct / 50, 1.0)

    state = np.zeros(N_DIMS)
    state[Dim.CONSEQUENCES] = keep
    state[Dim.RIGHTS] = 1.0  # constant
    state[Dim.FAIRNESS] = fairness
    state[Dim.AUTONOMY] = 1.0  # constant
    state[Dim.SOCIAL_IMPACT] = social
    state[Dim.VIRTUE_IDENTITY] = identity
    state[Dim.LEGITIMACY] = 0.5  # constant
    state[Dim.EPISTEMIC] = 0.5  # constant
    return state


def load_ultimatum_data(filepath: Path | None = None) -> List[ObservedChoice]:
    """Load Fraser & Nettle ultimatum data as ObservedChoice objects.

    Each subject's proposed amount is their 'chosen' option.
    The other possible offers are 'rejected' alternatives.
    """
    if filepath is None:
        filepath = DATA_DIR / "fraser_nettle_exp1.csv"

    stake = 10.0
    possible_pcts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    start = np.zeros(N_DIMS)
    start[Dim.CONSEQUENCES] = stake
    start[Dim.RIGHTS] = 1.0
    start[Dim.FAIRNESS] = 0.5
    start[Dim.AUTONOMY] = 1.0
    start[Dim.VIRTUE_IDENTITY] = 0.5
    start[Dim.LEGITIMACY] = 0.5
    start[Dim.EPISTEMIC] = 0.5

    observations = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            proposed = float(row["ProposedAmount"])
            # Convert to percentage
            proposed_pct = (proposed / stake) * 100

            # Find the closest discrete offer level
            chosen_pct = min(possible_pcts, key=lambda p: abs(p - proposed_pct))
            chosen_state = _ultimatum_state(stake, chosen_pct)

            rejected = []
            for pct in possible_pcts:
                if pct != chosen_pct:
                    rejected.append(_ultimatum_state(stake, pct))

            observations.append(ObservedChoice(
                start=start.copy(),
                chosen=chosen_state,
                rejected=rejected,
            ))

    return observations


# ---------------------------------------------------------------------------
# Public goods game: encode contributions as choices
# ---------------------------------------------------------------------------

def _public_goods_state(
    endowment: float,
    contrib_pct: float,
    n_players: int = 4,
    multiplier: float = 2.0,
) -> np.ndarray:
    """Map a public goods contribution to a 9D attribute vector."""
    contrib = endowment * contrib_pct / 100
    others_contrib = endowment * 0.5 * (n_players - 1)
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
    state[Dim.SOCIAL_IMPACT] = social
    state[Dim.VIRTUE_IDENTITY] = identity
    state[Dim.LEGITIMACY] = 0.5
    state[Dim.EPISTEMIC] = 0.5
    return state


def load_public_goods_data(
    filepath: Path | None = None,
    round_num: int = 1,
) -> Tuple[List[float], float, float]:
    """Load Fraser & Nettle public goods data.

    Returns (contributions_list, endowment, mean_contribution).
    Only loads specified round for clean measurement.
    """
    if filepath is None:
        filepath = DATA_DIR / "fraser_nettle_exp2.csv"

    endowment = 20.0
    contributions = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Round"] == str(round_num) and row["Contribution"].strip():
                contributions.append(float(row["Contribution"]))

    mean_contrib = sum(contributions) / len(contributions)
    return contributions, endowment, mean_contrib


# ---------------------------------------------------------------------------
# Ruggeri cross-cultural data
# ---------------------------------------------------------------------------

def load_ruggeri_by_country(
    filepath: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    """Load Ruggeri prospect theory data aggregated by country.

    Returns {country: {problem_N: choice_rate, ...}}.
    Choice rate = fraction choosing option 1 (the PT-predicted option).
    """
    if filepath is None:
        filepath = DATA_DIR / "ruggeri_prospect_theory.csv"

    country_data: Dict[str, Dict[str, List[float]]] = {}
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            country = row["Country"]
            if country not in country_data:
                country_data[country] = {str(p): [] for p in range(1, 18)}
            for p in range(1, 18):
                val = row[str(p)]
                if val:
                    country_data[country][str(p)].append(float(val))

    # Aggregate to means
    result = {}
    for country, problems in country_data.items():
        result[country] = {}
        for p, vals in problems.items():
            if vals:
                result[country][p] = sum(vals) / len(vals)
    return result


# ---------------------------------------------------------------------------
# Calibration and out-of-sample prediction
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Result of calibrating sigma from real data."""
    sigma: np.ndarray
    n_observations: int
    source: str

    @property
    def money_weight(self) -> float:
        """Effective weight on monetary dimension (1/sigma[0,0])."""
        return 1.0 / self.sigma[Dim.CONSEQUENCES, Dim.CONSEQUENCES]

    @property
    def fairness_weight(self) -> float:
        """Effective weight on fairness dimension."""
        return 1.0 / self.sigma[Dim.FAIRNESS, Dim.FAIRNESS]

    @property
    def weight_ratio(self) -> float:
        """Ratio of fairness weight to money weight.
        Higher = fairness matters more relative to money."""
        return self.fairness_weight / self.money_weight


@dataclass
class OutOfSampleResult:
    """Out-of-sample prediction vs. observed data."""
    game: str
    predicted_optimal_pct: float
    observed_mean_pct: float
    observed_modal_pct: float
    error_vs_mean: float
    error_vs_modal: float
    n_observed: int


def calibrate_from_ultimatum(
    observations: List[ObservedChoice] | None = None,
) -> CalibrationResult:
    """Calibrate Σ from real ultimatum game data."""
    if observations is None:
        observations = load_ultimatum_data()

    sigma = estimate_sigma(observations, regularization=0.1)

    return CalibrationResult(
        sigma=sigma,
        n_observations=len(observations),
        source="Fraser & Nettle (2020) ultimatum game",
    )


def predict_public_goods(
    sigma: np.ndarray,
    endowment: float = 20.0,
) -> Tuple[float, List[Tuple[float, float]]]:
    """Predict optimal public goods contribution using calibrated sigma.

    Returns (optimal_pct, [(pct, cost), ...] for all options).
    """
    sigma_inv = np.linalg.inv(sigma + 1e-10 * np.eye(N_DIMS))

    start = np.zeros(N_DIMS)
    start[Dim.CONSEQUENCES] = endowment
    start[Dim.RIGHTS] = 1.0
    start[Dim.FAIRNESS] = 0.5
    start[Dim.AUTONOMY] = 1.0
    start[Dim.SOCIAL_IMPACT] = 0.0
    start[Dim.VIRTUE_IDENTITY] = 0.5
    start[Dim.LEGITIMACY] = 0.5
    start[Dim.EPISTEMIC] = 0.5

    possible_pcts = list(range(0, 101, 5))
    costs = []
    for pct in possible_pcts:
        state = _public_goods_state(endowment, pct)
        cost = mahalanobis_distance(start, state, sigma_inv)
        costs.append((pct, cost))

    optimal = min(costs, key=lambda x: x[1])
    return optimal[0], costs


def predict_ultimatum(
    sigma: np.ndarray,
    stake: float = 10.0,
) -> Tuple[float, List[Tuple[float, float]]]:
    """Predict optimal ultimatum offer using given sigma.

    Returns (optimal_pct, [(pct, cost), ...] for all options).
    """
    sigma_inv = np.linalg.inv(sigma + 1e-10 * np.eye(N_DIMS))

    start = np.zeros(N_DIMS)
    start[Dim.CONSEQUENCES] = stake
    start[Dim.RIGHTS] = 1.0
    start[Dim.FAIRNESS] = 0.5
    start[Dim.AUTONOMY] = 1.0
    start[Dim.VIRTUE_IDENTITY] = 0.5
    start[Dim.LEGITIMACY] = 0.5
    start[Dim.EPISTEMIC] = 0.5

    possible_pcts = list(range(0, 51, 5))
    costs = []
    for pct in possible_pcts:
        state = _ultimatum_state(stake, pct)
        cost = mahalanobis_distance(start, state, sigma_inv)
        costs.append((pct, cost))

    optimal = min(costs, key=lambda x: x[1])
    return optimal[0], costs


@dataclass
class EmpiricalReport:
    """Full empirical validation report against real data."""
    calibration: CalibrationResult
    ultimatum_in_sample: OutOfSampleResult
    public_goods_out_of_sample: OutOfSampleResult
    default_sigma_ultimatum: OutOfSampleResult
    default_sigma_public_goods: OutOfSampleResult

    def summary(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append("EMPIRICAL CALIBRATION REPORT")
        lines.append("Calibrated on real experimental data")
        lines.append("=" * 72)

        lines.append(f"\nCalibration source: {self.calibration.source}")
        lines.append(f"N observations: {self.calibration.n_observations}")
        lines.append(f"Money variance (sigma[0,0]): {self.calibration.sigma[0,0]:.2f}")
        lines.append(f"Fairness variance (sigma[2,2]): {self.calibration.sigma[2,2]:.4f}")
        lines.append(f"Fairness/money weight ratio: {self.calibration.weight_ratio:.1f}x")

        lines.append(f"\n--- DEFAULT SIGMA (hand-tuned) ---")
        lines.append(f"  Ultimatum: predicted {self.default_sigma_ultimatum.predicted_optimal_pct:.0f}%, "
                     f"observed {self.default_sigma_ultimatum.observed_mean_pct:.1f}% "
                     f"(error: {self.default_sigma_ultimatum.error_vs_mean:.1f}%)")
        lines.append(f"  Public goods: predicted {self.default_sigma_public_goods.predicted_optimal_pct:.0f}%, "
                     f"observed {self.default_sigma_public_goods.observed_mean_pct:.1f}% "
                     f"(error: {self.default_sigma_public_goods.error_vs_mean:.1f}%)")

        lines.append(f"\n--- CALIBRATED SIGMA (from ultimatum data) ---")
        lines.append(f"  Ultimatum (IN-SAMPLE): predicted {self.ultimatum_in_sample.predicted_optimal_pct:.0f}%, "
                     f"observed {self.ultimatum_in_sample.observed_mean_pct:.1f}% "
                     f"(error: {self.ultimatum_in_sample.error_vs_mean:.1f}%)")
        lines.append(f"  Public goods (OUT-OF-SAMPLE): predicted {self.public_goods_out_of_sample.predicted_optimal_pct:.0f}%, "
                     f"observed {self.public_goods_out_of_sample.observed_mean_pct:.1f}% "
                     f"(error: {self.public_goods_out_of_sample.error_vs_mean:.1f}%)")

        cal_ult_err = abs(self.ultimatum_in_sample.error_vs_mean)
        cal_pg_err = abs(self.public_goods_out_of_sample.error_vs_mean)
        def_ult_err = abs(self.default_sigma_ultimatum.error_vs_mean)
        def_pg_err = abs(self.default_sigma_public_goods.error_vs_mean)

        lines.append(f"\n--- COMPARISON ---")
        lines.append(f"  Default sigma total error: {def_ult_err + def_pg_err:.1f}%")
        lines.append(f"  Calibrated sigma total error: {cal_ult_err + cal_pg_err:.1f}%")
        if cal_ult_err + cal_pg_err < def_ult_err + def_pg_err:
            lines.append(f"  >> Calibration IMPROVED predictions by {(def_ult_err + def_pg_err) - (cal_ult_err + cal_pg_err):.1f}% <<")

        lines.append("\n" + "=" * 72)
        return "\n".join(lines)


def run_empirical_validation() -> EmpiricalReport:
    """Run full empirical validation pipeline.

    1. Load real ultimatum data
    2. Calibrate sigma from ultimatum choices
    3. Predict public goods behavior (out-of-sample)
    4. Compare calibrated vs default sigma
    """
    # Load real data
    ult_obs = load_ultimatum_data()
    pg_contribs, pg_endow, pg_mean = load_public_goods_data(round_num=1)

    # Observed statistics
    ult_amounts = []
    for obs in ult_obs:
        offer_pct = (1 - obs.chosen[Dim.CONSEQUENCES] / 10.0) * 100
        ult_amounts.append(offer_pct)
    ult_mean = sum(ult_amounts) / len(ult_amounts)

    from collections import Counter
    ult_modal_pct = Counter([round(a / 10) * 10 for a in ult_amounts]).most_common(1)[0][0]
    pg_mean_pct = (pg_mean / pg_endow) * 100
    pg_modal = Counter([round(c / pg_endow * 4) / 4 * 100 for c in pg_contribs]).most_common(1)[0][0]

    # Calibrate sigma from ultimatum data
    cal = calibrate_from_ultimatum(ult_obs)

    # Predictions with calibrated sigma
    cal_ult_pct, cal_ult_costs = predict_ultimatum(cal.sigma)
    cal_pg_pct, cal_pg_costs = predict_public_goods(cal.sigma, endowment=pg_endow)

    # Predictions with default sigma
    def_sigma = _default_sigma()
    def_ult_pct, _ = predict_ultimatum(def_sigma)
    def_pg_pct, _ = predict_public_goods(def_sigma, endowment=pg_endow)

    return EmpiricalReport(
        calibration=cal,
        ultimatum_in_sample=OutOfSampleResult(
            game="Ultimatum (in-sample)",
            predicted_optimal_pct=cal_ult_pct,
            observed_mean_pct=ult_mean,
            observed_modal_pct=ult_modal_pct,
            error_vs_mean=cal_ult_pct - ult_mean,
            error_vs_modal=cal_ult_pct - ult_modal_pct,
            n_observed=len(ult_obs),
        ),
        public_goods_out_of_sample=OutOfSampleResult(
            game="Public Goods (out-of-sample)",
            predicted_optimal_pct=cal_pg_pct,
            observed_mean_pct=pg_mean_pct,
            observed_modal_pct=pg_modal,
            error_vs_mean=cal_pg_pct - pg_mean_pct,
            error_vs_modal=cal_pg_pct - pg_modal,
            n_observed=len(pg_contribs),
        ),
        default_sigma_ultimatum=OutOfSampleResult(
            game="Ultimatum (default sigma)",
            predicted_optimal_pct=def_ult_pct,
            observed_mean_pct=ult_mean,
            observed_modal_pct=ult_modal_pct,
            error_vs_mean=def_ult_pct - ult_mean,
            error_vs_modal=def_ult_pct - ult_modal_pct,
            n_observed=len(ult_obs),
        ),
        default_sigma_public_goods=OutOfSampleResult(
            game="Public Goods (default sigma)",
            predicted_optimal_pct=def_pg_pct,
            observed_mean_pct=pg_mean_pct,
            observed_modal_pct=pg_modal,
            error_vs_mean=def_pg_pct - pg_mean_pct,
            error_vs_modal=def_pg_pct - pg_modal,
            n_observed=len(pg_contribs),
        ),
    )
