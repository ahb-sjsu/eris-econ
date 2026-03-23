# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Joint calibration across experimental data sources.

Strategy: calibrate sigma on game data (consistent dollar units), then
validate predictions out-of-sample.

Data sources for calibration:
- Fraser & Nettle (2020) — ultimatum game (w/ rejection probability encoding)
- Fraser & Nettle (2020) — public goods game

Out-of-sample validation:
- Dictator game (Engel 2011 meta-analysis)
- Prospect theory conformity (Ruggeri et al. 2020)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from eris_econ.calibration import ObservedChoice
from eris_econ.calibration_v2 import (
    CalibratedSigma,
    bootstrap_confidence,
    cross_validate,
    estimate_diagonal_sigma,
)
from eris_econ.dimensions import Dim, N_DIMS, DIM_NAMES
from eris_econ.games_v2 import (
    predict_game,
    public_goods_state,
    rejection_probability,
    ultimatum_state,
)
from eris_econ.metrics import mahalanobis_distance

DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# Normalized encodings: consequences in [0, 1]
# ---------------------------------------------------------------------------

def _ultimatum_state_norm(
    stake: float, offer_pct: float, include_rejection: bool = True,
) -> np.ndarray:
    """Ultimatum state with consequences normalized to [0, 1]."""
    state = ultimatum_state(stake, offer_pct, include_rejection)
    state[Dim.CONSEQUENCES] = state[Dim.CONSEQUENCES] / stake
    return state


def _public_goods_state_norm(
    endowment: float, contrib_pct: float,
    n_players: int = 4, multiplier: float = 2.0,
    others_contrib_pct: float = 50.0,
) -> np.ndarray:
    """Public goods state with consequences normalized to [0, 1]."""
    state = public_goods_state(
        endowment, contrib_pct, n_players, multiplier, others_contrib_pct,
    )
    # Max possible remaining: keep all + share of others' pool
    others_max = endowment * others_contrib_pct / 100 * (n_players - 1)
    max_remaining = endowment + others_max * multiplier / n_players
    state[Dim.CONSEQUENCES] = state[Dim.CONSEQUENCES] / max_remaining
    return state


# ---------------------------------------------------------------------------
# Encode observations with normalized consequences
# ---------------------------------------------------------------------------

def encode_ultimatum_norm(
    filepath: Path | None = None,
    stake: float = 10.0,
) -> List[ObservedChoice]:
    """Encode ultimatum choices with normalized consequences."""
    if filepath is None:
        filepath = DATA_DIR / "fraser_nettle_exp1.csv"

    possible_pcts = list(range(0, 51, 5))

    start = np.zeros(N_DIMS)
    start[Dim.CONSEQUENCES] = 1.0  # normalized: full stake
    start[Dim.RIGHTS] = 1.0
    start[Dim.FAIRNESS] = 0.5
    start[Dim.AUTONOMY] = 1.0
    start[Dim.PRIVACY_TRUST] = 0.5
    start[Dim.VIRTUE_IDENTITY] = 0.5
    start[Dim.LEGITIMACY] = 0.5
    start[Dim.EPISTEMIC] = 0.8

    observations = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            proposed = float(row["ProposedAmount"])
            proposed_pct = min(50, max(0, (proposed / stake) * 100))
            chosen_pct = min(possible_pcts, key=lambda p: abs(p - proposed_pct))

            chosen_state = _ultimatum_state_norm(stake, chosen_pct)
            rejected = [
                _ultimatum_state_norm(stake, pct)
                for pct in possible_pcts if pct != chosen_pct
            ]

            observations.append(ObservedChoice(
                start=start.copy(), chosen=chosen_state, rejected=rejected,
            ))

    return observations


def encode_public_goods_norm(
    filepath: Path | None = None,
    round_num: int = 1,
    endowment: float = 20.0,
) -> List[ObservedChoice]:
    """Encode public goods choices with normalized consequences."""
    if filepath is None:
        filepath = DATA_DIR / "fraser_nettle_exp2.csv"

    possible_pcts = list(range(0, 101, 5))

    start = np.zeros(N_DIMS)
    start[Dim.CONSEQUENCES] = 1.0  # normalized: full endowment equivalent
    start[Dim.RIGHTS] = 1.0
    start[Dim.FAIRNESS] = 0.5
    start[Dim.AUTONOMY] = 1.0
    start[Dim.PRIVACY_TRUST] = 0.5
    start[Dim.SOCIAL_IMPACT] = 0.0
    start[Dim.VIRTUE_IDENTITY] = 0.5
    start[Dim.LEGITIMACY] = 0.5
    start[Dim.EPISTEMIC] = 0.5

    observations = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Round"] != str(round_num) or not row["Contribution"].strip():
                continue

            contrib = float(row["Contribution"])
            contrib_pct = min(100, max(0, (contrib / endowment) * 100))
            chosen_pct = min(possible_pcts, key=lambda p: abs(p - contrib_pct))

            chosen_state = _public_goods_state_norm(endowment, chosen_pct)
            rejected = [
                _public_goods_state_norm(endowment, pct)
                for pct in possible_pcts if pct != chosen_pct
            ]

            observations.append(ObservedChoice(
                start=start.copy(), chosen=chosen_state, rejected=rejected,
            ))

    return observations


# ---------------------------------------------------------------------------
# Prediction with normalized sigma
# ---------------------------------------------------------------------------

def predict_game_norm(
    sigma: np.ndarray,
    game: str,
    stake: float = 10.0,
    endowment: float = 20.0,
    resolution: int = 1,
) -> Tuple[float, List[Tuple[float, float]]]:
    """Predict optimal choice using normalized consequence encoding."""
    sigma_inv = np.linalg.inv(sigma + 1e-10 * np.eye(N_DIMS))

    if game.startswith("ultimatum"):
        include_rej = "no_rejection" not in game
        max_pct = 50

        start = np.zeros(N_DIMS)
        start[Dim.CONSEQUENCES] = 1.0  # normalized
        start[Dim.RIGHTS] = 1.0
        start[Dim.FAIRNESS] = 0.5
        start[Dim.AUTONOMY] = 1.0
        start[Dim.PRIVACY_TRUST] = 0.5
        start[Dim.VIRTUE_IDENTITY] = 0.5
        start[Dim.LEGITIMACY] = 0.5
        start[Dim.EPISTEMIC] = 0.8

        costs = []
        for pct in range(0, max_pct + 1, resolution):
            state = _ultimatum_state_norm(stake, pct, include_rejection=include_rej)
            cost = mahalanobis_distance(start, state, sigma_inv)
            costs.append((float(pct), cost))

    elif game == "dictator":
        max_pct = 50
        start = np.zeros(N_DIMS)
        start[Dim.CONSEQUENCES] = 1.0
        start[Dim.RIGHTS] = 1.0
        start[Dim.FAIRNESS] = 0.5
        start[Dim.AUTONOMY] = 1.0
        start[Dim.PRIVACY_TRUST] = 0.5
        start[Dim.VIRTUE_IDENTITY] = 0.5
        start[Dim.LEGITIMACY] = 0.5
        start[Dim.EPISTEMIC] = 0.5

        costs = []
        for pct in range(0, max_pct + 1, resolution):
            state = _ultimatum_state_norm(stake, pct, include_rejection=False)
            state[Dim.VIRTUE_IDENTITY] *= 0.8
            state[Dim.SOCIAL_IMPACT] *= 0.7
            cost = mahalanobis_distance(start, state, sigma_inv)
            costs.append((float(pct), cost))

    elif game == "public_goods":
        start = np.zeros(N_DIMS)
        start[Dim.CONSEQUENCES] = 1.0
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
            state = _public_goods_state_norm(endowment, pct)
            cost = mahalanobis_distance(start, state, sigma_inv)
            costs.append((float(pct), cost))

    else:
        raise ValueError(f"Unknown game: {game}")

    optimal = min(costs, key=lambda x: x[1])
    return optimal[0], costs


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class JointCalibrationResult:
    """Result of joint calibration across all data sources."""
    sigma: CalibratedSigma
    n_ultimatum: int
    n_public_goods: int
    regularization: float
    cv_scores: List[float] | None = None
    bootstrap_median: np.ndarray | None = None
    bootstrap_ci_low: np.ndarray | None = None
    bootstrap_ci_high: np.ndarray | None = None

    @property
    def total_obs(self) -> int:
        return self.n_ultimatum + self.n_public_goods

    def weights_table(self) -> str:
        diag = np.diag(self.sigma.sigma)
        weights = 1.0 / diag
        lines = []
        lines.append(f"{'Dimension':<25} {'var':>10} {'weight':>10}")
        lines.append("-" * 47)
        for d in range(N_DIMS):
            name = DIM_NAMES[Dim(d)]
            lines.append(f"{name:<25} {diag[d]:>10.4f} {weights[d]:>10.4f}")
        return "\n".join(lines)


@dataclass
class PredictionResult:
    """A game prediction vs observed data."""
    game: str
    predicted_pct: float
    observed_pct: float
    error: float
    in_sample: bool
    cost_landscape: List[Tuple[float, float]]


@dataclass
class JointReport:
    """Full report from joint calibration and prediction."""
    calibration: JointCalibrationResult
    predictions: List[PredictionResult]

    def summary(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append("JOINT CALIBRATION REPORT (Normalized Consequences)")
        lines.append("=" * 72)

        cal = self.calibration
        lines.append(f"\nCalibration data:")
        lines.append(f"  Fraser & Nettle ultimatum:  {cal.n_ultimatum:>6} obs")
        lines.append(f"  Fraser & Nettle pub. goods:  {cal.n_public_goods:>6} obs")
        lines.append(f"  Total:                      {cal.total_obs:>6} obs")
        lines.append(f"  Regularization:             {cal.regularization}")
        lines.append(f"  Converged:                  {cal.sigma.converged}")
        lines.append(f"  Log-likelihood:             {cal.sigma.log_likelihood:.2f}")
        lines.append(f"  AIC: {cal.sigma.aic:.2f}  BIC: {cal.sigma.bic:.2f}")

        lines.append(f"\nDimensional weights:")
        lines.append(cal.weights_table())

        if cal.bootstrap_median is not None:
            lines.append(f"\nBootstrap 95% CI:")
            for d in range(N_DIMS):
                name = DIM_NAMES[Dim(d)]
                lo, hi = cal.bootstrap_ci_low[d], cal.bootstrap_ci_high[d]
                med = cal.bootstrap_median[d]
                lines.append(f"  {name:<25} {med:>8.4f} [{lo:.4f}, {hi:.4f}]")

        lines.append(f"\n{'='*72}")
        lines.append("PREDICTIONS")
        lines.append("=" * 72)

        for pred in self.predictions:
            tag = "IN-SAMPLE" if pred.in_sample else "OUT-OF-SAMPLE"
            lines.append(
                f"  [{tag:>14}] {pred.game:<25} "
                f"pred={pred.predicted_pct:>5.1f}%  "
                f"obs={pred.observed_pct:>5.1f}%  "
                f"err={pred.error:>+6.1f}%"
            )

        oos = [p for p in self.predictions if not p.in_sample]
        if oos:
            oos_err = sum(abs(p.error) for p in oos)
            lines.append(f"\n  Out-of-sample MAE: {oos_err / len(oos):.1f}%")

        total_err = sum(abs(p.error) for p in self.predictions)
        lines.append(f"  Overall MAE:       {total_err / len(self.predictions):.1f}%")

        lines.append("\n" + "=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_joint_calibration(
    do_cv: bool = True,
    do_bootstrap: bool = False,
    n_bootstrap: int = 30,
    money_bounds: Tuple[float, float] = (0.01, 500.0),
    moral_bounds: Tuple[float, float] = (0.01, 50.0),
) -> JointReport:
    """Run joint calibration on game data with normalized consequences.

    Args:
        do_cv: Run cross-validation for regularization selection.
        do_bootstrap: Run bootstrap confidence intervals.
        money_bounds: Bounds for consequences variance.
        moral_bounds: Bounds for non-monetary variance.
    """
    print("Loading and encoding data...")

    ult_obs = encode_ultimatum_norm()
    print(f"  Ultimatum (normalized, w/ rejection): {len(ult_obs)} obs")

    pg_obs = encode_public_goods_norm()
    print(f"  Public goods (normalized):            {len(pg_obs)} obs")

    # Combine
    rng = np.random.default_rng(42)
    all_obs = ult_obs + pg_obs
    rng.shuffle(all_obs)
    print(f"  Combined: {len(all_obs)} obs")

    # Cross-validate
    reg = 0.01
    cv_scores = None
    if do_cv:
        print("\nCross-validating regularization...")
        reg_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        reg, cv_scores = cross_validate(
            all_obs, n_folds=5, regularization_values=reg_values, rng=rng,
        )
        print(f"  Best regularization: {reg}")
        if cv_scores:
            for r, s in zip(reg_values, cv_scores):
                print(f"    reg={r:.3f}  CV NLL={s:.4f}")

    # Estimate sigma
    print(f"\nEstimating sigma (reg={reg})...")
    sigma_cal = estimate_diagonal_sigma(
        all_obs, regularization=reg,
        money_bounds=money_bounds, moral_bounds=moral_bounds,
        init_money_var=1.0, init_moral_var=0.25,
    )
    print(f"  Converged: {sigma_cal.converged}")
    print(f"  Log-likelihood: {sigma_cal.log_likelihood:.2f}")

    diag = np.diag(sigma_cal.sigma)
    for d in range(N_DIMS):
        name = DIM_NAMES[Dim(d)]
        at_bound = ""
        if d == 0:
            if abs(diag[d] - money_bounds[0]) < 0.01 or abs(diag[d] - money_bounds[1]) < 0.01:
                at_bound = " [AT BOUND]"
        else:
            if abs(diag[d] - moral_bounds[0]) < 0.01 or abs(diag[d] - moral_bounds[1]) < 0.01:
                at_bound = " [AT BOUND]"
        print(f"  {name:<25} var={diag[d]:>8.4f}  w={1/diag[d]:>8.4f}{at_bound}")

    # Bootstrap
    boot_med = boot_lo = boot_hi = None
    if do_bootstrap:
        print(f"\nBootstrap ({n_bootstrap} samples)...")
        boot_med, boot_lo, boot_hi = bootstrap_confidence(
            all_obs, n_bootstrap=n_bootstrap, regularization=reg, rng=rng,
        )

    cal_result = JointCalibrationResult(
        sigma=sigma_cal,
        n_ultimatum=len(ult_obs),
        n_public_goods=len(pg_obs),
        regularization=reg,
        cv_scores=cv_scores,
        bootstrap_median=boot_med,
        bootstrap_ci_low=boot_lo,
        bootstrap_ci_high=boot_hi,
    )

    # Predictions
    print("\nPredictions:")
    predictions = []

    # Load observed values
    ult_amounts = []
    with open(DATA_DIR / "fraser_nettle_exp1.csv") as f:
        for row in csv.DictReader(f):
            ult_amounts.append(float(row["ProposedAmount"]))
    obs_ult = sum(ult_amounts) / len(ult_amounts) / 10.0 * 100

    pg_contribs = []
    with open(DATA_DIR / "fraser_nettle_exp2.csv") as f:
        for row in csv.DictReader(f):
            if row["Round"] == "1" and row["Contribution"].strip():
                pg_contribs.append(float(row["Contribution"]))
    obs_pg = sum(pg_contribs) / len(pg_contribs) / 20.0 * 100

    obs_dic = 28.35  # Engel (2011)

    # Ultimatum with rejection
    pct, costs = predict_game_norm(sigma_cal.sigma, "ultimatum", stake=10.0)
    predictions.append(PredictionResult(
        "Ultimatum (w/ rejection)", pct, obs_ult, pct - obs_ult, True, costs))

    # Ultimatum without rejection
    pct, costs = predict_game_norm(sigma_cal.sigma, "ultimatum_no_rejection", stake=10.0)
    predictions.append(PredictionResult(
        "Ultimatum (no rejection)", pct, obs_ult, pct - obs_ult, True, costs))

    # Dictator (out-of-sample)
    pct, costs = predict_game_norm(sigma_cal.sigma, "dictator", stake=10.0)
    predictions.append(PredictionResult(
        "Dictator", pct, obs_dic, pct - obs_dic, False, costs))

    # Public goods
    pct, costs = predict_game_norm(sigma_cal.sigma, "public_goods", endowment=20.0)
    predictions.append(PredictionResult(
        "Public Goods", pct, obs_pg, pct - obs_pg, True, costs))

    for p in predictions:
        tag = "IS" if p.in_sample else "OOS"
        print(f"  [{tag}] {p.game:<30} pred={p.predicted_pct:>5.1f}%  "
              f"obs={p.observed_pct:>5.1f}%  err={p.error:>+6.1f}%")

    return JointReport(calibration=cal_result, predictions=predictions)


# ---------------------------------------------------------------------------
# Sensitivity analysis: sweep money weight
# ---------------------------------------------------------------------------

def sweep_money_weight() -> None:
    """Sweep money variance to show how predictions change."""
    print("Loading data...")
    ult_obs = encode_ultimatum_norm()
    pg_obs = encode_public_goods_norm()
    all_obs = ult_obs + pg_obs

    # Calibrate once to get moral weights
    base = estimate_diagonal_sigma(
        all_obs, regularization=0.01,
        money_bounds=(0.01, 50.0), moral_bounds=(0.01, 50.0),
    )
    base_diag = np.diag(base.sigma).copy()

    print(f"\nBase calibrated sigma diagonal:")
    for d in range(N_DIMS):
        print(f"  {DIM_NAMES[Dim(d)]:<25} {base_diag[d]:.4f}")

    print(f"\n{'Money var':>10} {'Ult%':>6} {'Dic%':>6} {'PG%':>6}")
    print("-" * 35)

    for money_var in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]:
        test_diag = base_diag.copy()
        test_diag[0] = money_var
        sigma = np.diag(test_diag)

        ult_pct, _ = predict_game_norm(sigma, "ultimatum", stake=10.0)
        dic_pct, _ = predict_game_norm(sigma, "dictator", stake=10.0)
        pg_pct, _ = predict_game_norm(sigma, "public_goods", endowment=20.0)

        print(f"  {money_var:>8.2f}  {ult_pct:>5.1f}  {dic_pct:>5.1f}  {pg_pct:>5.1f}")

    print(f"\nObserved:           48.3   28.4   45.7")


if __name__ == "__main__":
    report = run_joint_calibration(do_cv=True, do_bootstrap=False)
    print("\n")
    print(report.summary())
    print("\n\n")
    sweep_money_weight()
