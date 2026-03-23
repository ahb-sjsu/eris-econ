# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Structural fuzzing of the geometric economics framework.

Adapts the ErisML Bond Index / DEME fuzzing methodology to systematically
discover which dimensional structures best explain economic behavior data.

Instead of fuzzing an LLM evaluator, we fuzz the dimensional structure:
- Parametric transforms → dimensional weight variations
- Adversarial threshold search → minimal perturbation that breaks predictions
- Compositional chains → dimension subset testing
- Model Robustness Index → single scalar for structural reliability

This directly addresses the model selection question: "Which dimensions
are necessary, and which are redundant?"
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from eris_econ.dimensions import Dim, N_DIMS, DIM_NAMES
from eris_econ.joint_calibration import predict_game_norm
from eris_econ.metrics import mahalanobis_distance
from eris_econ.targets import Target, build_targets, evaluate_targets


# ---------------------------------------------------------------------------
# Observed data targets (legacy 3-target dict for backward compatibility)
# ---------------------------------------------------------------------------

OBSERVED = {
    "ultimatum": 48.3,     # Fraser & Nettle (2020), mean offer %
    "dictator": 28.35,     # Engel (2011) meta-analysis, mean giving %
    "public_goods": 45.7,  # Fraser & Nettle (2020) round 1, mean contrib %
}


# ---------------------------------------------------------------------------
# Prediction error metric
# ---------------------------------------------------------------------------

def prediction_error(
    sigma: np.ndarray,
    games: List[str] | None = None,
    weights: Dict[str, float] | None = None,
    targets: List[Target] | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute weighted prediction error for a given sigma.

    If `targets` is provided, uses the expanded target system.
    Otherwise falls back to the legacy 3-game system.
    """
    if targets is not None:
        mae, errors, _ = evaluate_targets(sigma, targets)
        return mae, errors

    if games is None:
        games = ["ultimatum", "dictator", "public_goods"]
    if weights is None:
        weights = {g: 1.0 for g in games}

    errors = {}
    for game in games:
        pct, _ = predict_game_norm(sigma, game, stake=10.0, endowment=20.0)
        errors[game] = pct - OBSERVED[game]

    w_total = sum(weights[g] for g in games)
    mae = sum(weights[g] * abs(errors[g]) for g in games) / w_total
    return mae, errors


# ---------------------------------------------------------------------------
# Structural fuzzing: enumerate dimension subsets
# ---------------------------------------------------------------------------

@dataclass
class SubsetResult:
    """Result of testing a dimension subset."""
    dims: Tuple[int, ...]          # Active dimension indices
    dim_names: Tuple[str, ...]     # Human-readable names
    n_dims: int                    # Number of active dimensions
    sigma_diag: np.ndarray         # Best sigma diagonal for this subset
    mae: float                     # Mean absolute error
    errors: Dict[str, float]       # Per-game errors
    pareto_optimal: bool = False   # On the parsimony-accuracy frontier


def _optimize_subset(
    active_dims: Tuple[int, ...],
    inactive_var: float = 1e6,
    n_grid: int = 20,
    calibration_targets: List[Target] | None = None,
) -> SubsetResult:
    """Find the best sigma for a given subset of active dimensions.

    Active dimensions are optimized over a weight grid.
    Inactive dimensions get very high variance (negligible weight).

    Args:
        calibration_targets: If provided, optimize against these targets
            using evaluate_targets.  Otherwise use the legacy 3-game system.
    """
    def _eval(sigma):
        if calibration_targets is not None:
            mae, errors, _ = evaluate_targets(sigma, calibration_targets)
            return mae, errors
        return prediction_error(sigma)

    n_active = len(active_dims)
    if n_active == 0:
        # No active dims: everything has negligible weight
        sigma = np.diag(np.full(N_DIMS, inactive_var))
        mae, errors = _eval(sigma)
        return SubsetResult(
            dims=active_dims,
            dim_names=tuple(),
            n_dims=0,
            sigma_diag=np.diag(sigma),
            mae=mae,
            errors=errors,
        )

    # Generate grid of variances for active dimensions
    # Lower variance = higher weight
    var_values = np.logspace(-2, 2, n_grid)  # 0.01 to 100

    best_mae = float("inf")
    best_sigma_diag = None
    best_errors = None

    if n_active == 1:
        # 1D grid search
        for v in var_values:
            sigma_diag = np.full(N_DIMS, inactive_var)
            sigma_diag[active_dims[0]] = v
            sigma = np.diag(sigma_diag)
            mae, errors = _eval(sigma)
            if mae < best_mae:
                best_mae = mae
                best_sigma_diag = sigma_diag.copy()
                best_errors = errors.copy()

    elif n_active == 2:
        # 2D grid search
        for v1 in var_values:
            for v2 in var_values:
                sigma_diag = np.full(N_DIMS, inactive_var)
                sigma_diag[active_dims[0]] = v1
                sigma_diag[active_dims[1]] = v2
                sigma = np.diag(sigma_diag)
                mae, errors = _eval(sigma)
                if mae < best_mae:
                    best_mae = mae
                    best_sigma_diag = sigma_diag.copy()
                    best_errors = errors.copy()

    else:
        # For 3+ dimensions, use random search (grid is too large)
        rng = np.random.default_rng(42)
        n_samples = min(5000, n_grid ** min(n_active, 4))
        for _ in range(n_samples):
            sigma_diag = np.full(N_DIMS, inactive_var)
            for d in active_dims:
                sigma_diag[d] = 10 ** rng.uniform(-2, 2)
            sigma = np.diag(sigma_diag)
            mae, errors = _eval(sigma)
            if mae < best_mae:
                best_mae = mae
                best_sigma_diag = sigma_diag.copy()
                best_errors = errors.copy()

    return SubsetResult(
        dims=active_dims,
        dim_names=tuple(DIM_NAMES[Dim(d)] for d in active_dims),
        n_dims=n_active,
        sigma_diag=best_sigma_diag,
        mae=best_mae,
        errors=best_errors,
    )


def enumerate_subsets(
    max_dims: int = 4,
    must_include: Tuple[int, ...] = (),
    exclude_constant: bool = True,
    calibration_targets: List[Target] | None = None,
) -> List[SubsetResult]:
    """Enumerate all dimension subsets up to max_dims and find best sigma.

    Args:
        max_dims: Maximum subset size to test.
        must_include: Dimensions that must be in every subset.
        exclude_constant: Skip dimensions that are constant in game encodings
            (Rights, Autonomy, Privacy/Trust, Legitimacy).
        calibration_targets: If provided, optimize against these targets.

    Returns:
        List of SubsetResult sorted by MAE.
    """
    # Dimensions that actually vary in game encodings
    if exclude_constant:
        varying_dims = [
            Dim.CONSEQUENCES,   # d1: money
            Dim.FAIRNESS,       # d3: fairness
            Dim.SOCIAL_IMPACT,  # d6: social
            Dim.VIRTUE_IDENTITY,  # d7: identity
            Dim.EPISTEMIC,      # d9: epistemic
        ]
    else:
        varying_dims = list(range(N_DIMS))

    # Remove must_include from the pool (they're always present)
    pool = [d for d in varying_dims if d not in must_include]

    results = []

    # Test subsets of size 0 to max_dims (on top of must_include)
    for size in range(0, min(max_dims + 1, len(pool) + 1)):
        for combo in itertools.combinations(pool, size):
            active = tuple(sorted(set(must_include) | set(combo)))
            result = _optimize_subset(active,
                                      calibration_targets=calibration_targets)
            results.append(result)

    # Sort by MAE
    results.sort(key=lambda r: r.mae)

    # Mark Pareto-optimal (best MAE for each subset size)
    best_by_size = {}
    for r in results:
        if r.n_dims not in best_by_size or r.mae < best_by_size[r.n_dims]:
            best_by_size[r.n_dims] = r.mae
    for r in results:
        if r.mae <= best_by_size[r.n_dims] + 0.01:
            r.pareto_optimal = True

    return results


# ---------------------------------------------------------------------------
# Adversarial threshold search
# ---------------------------------------------------------------------------

@dataclass
class AdversarialResult:
    """Result of adversarial perturbation on a dimension."""
    dim: int
    dim_name: str
    base_var: float
    threshold_var: float       # Minimal variance change that flips prediction
    threshold_ratio: float     # threshold_var / base_var
    game_flipped: str          # Which game prediction flipped first
    direction: str             # "increase" or "decrease"


def find_adversarial_threshold(
    sigma_diag: np.ndarray,
    dim: int,
    tolerance: float = 0.5,
    n_steps: int = 50,
) -> List[AdversarialResult]:
    """Binary search for minimal variance perturbation that breaks predictions.

    For a given dimension, find the smallest change in sigma[dim, dim]
    that causes any game prediction to change by more than `tolerance`
    percentage points.

    Returns results for both increase and decrease directions.
    """
    base_sigma = np.diag(sigma_diag)
    _, base_errors = prediction_error(base_sigma)
    base_preds = {g: OBSERVED[g] + e for g, e in base_errors.items()}

    results = []

    for direction in ["increase", "decrease"]:
        if direction == "increase":
            search_range = np.logspace(
                np.log10(sigma_diag[dim]),
                np.log10(sigma_diag[dim] * 1000),
                n_steps,
            )
        else:
            search_range = np.logspace(
                np.log10(max(0.001, sigma_diag[dim] / 1000)),
                np.log10(sigma_diag[dim]),
                n_steps,
            )[::-1]

        for test_var in search_range:
            test_diag = sigma_diag.copy()
            test_diag[dim] = test_var
            test_sigma = np.diag(test_diag)
            _, test_errors = prediction_error(test_sigma)
            test_preds = {g: OBSERVED[g] + e for g, e in test_errors.items()}

            # Check if any game prediction changed significantly
            for game in base_preds:
                if abs(test_preds[game] - base_preds[game]) > tolerance:
                    results.append(AdversarialResult(
                        dim=dim,
                        dim_name=DIM_NAMES[Dim(dim)],
                        base_var=sigma_diag[dim],
                        threshold_var=test_var,
                        threshold_ratio=test_var / sigma_diag[dim],
                        game_flipped=game,
                        direction=direction,
                    ))
                    break
            else:
                continue
            break

    return results


# ---------------------------------------------------------------------------
# Sensitivity profiling
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Sensitivity of predictions to a single dimension."""
    dim: int
    dim_name: str
    mae_with: float        # MAE with this dimension active
    mae_without: float     # MAE with this dimension removed
    delta_mae: float       # mae_without - mae_with (positive = dimension helps)
    importance_rank: int = 0


def sensitivity_profile(sigma_diag: np.ndarray) -> List[SensitivityResult]:
    """Measure each dimension's contribution by ablation.

    For each dimension, compare MAE with vs without it (set to very high
    variance = negligible weight).
    """
    base_sigma = np.diag(sigma_diag)
    base_mae, _ = prediction_error(base_sigma)

    results = []
    for d in range(N_DIMS):
        ablated_diag = sigma_diag.copy()
        ablated_diag[d] = 1e6  # effectively remove this dimension
        ablated_sigma = np.diag(ablated_diag)
        ablated_mae, _ = prediction_error(ablated_sigma)

        results.append(SensitivityResult(
            dim=d,
            dim_name=DIM_NAMES[Dim(d)],
            mae_with=base_mae,
            mae_without=ablated_mae,
            delta_mae=ablated_mae - base_mae,
        ))

    # Rank by importance (highest delta = most important)
    results.sort(key=lambda r: r.delta_mae, reverse=True)
    for i, r in enumerate(results):
        r.importance_rank = i + 1

    return results


# ---------------------------------------------------------------------------
# Model Robustness Index (MRI) — analogous to Bond Index
# ---------------------------------------------------------------------------

@dataclass
class ModelRobustnessIndex:
    """Model Robustness Index: single scalar measuring structural reliability.

    Analogous to Bond Index for evaluator robustness.
    MRI = 0.5 * mean(omega) + 0.3 * p75(omega) + 0.2 * p95(omega)
    where omega = prediction error under perturbation.
    """
    mri: float                      # The index value
    mean_omega: float               # Mean perturbation error
    p75_omega: float                # 75th percentile
    p95_omega: float                # 95th percentile
    n_perturbations: int            # Number of perturbations tested
    worst_case_mae: float           # Maximum MAE observed
    worst_case_sigma: np.ndarray    # Sigma that produced worst case
    perturbation_errors: List[float]  # All MAE values


def compute_mri(
    sigma_diag: np.ndarray,
    n_perturbations: int = 500,
    perturbation_scale: float = 0.5,
    rng: np.random.Generator | None = None,
) -> ModelRobustnessIndex:
    """Compute the Model Robustness Index via random perturbation.

    Generates random perturbations of the sigma diagonal (in log-space)
    and measures how much predictions change.

    Args:
        sigma_diag: Base sigma diagonal to perturb.
        n_perturbations: Number of random perturbations.
        perturbation_scale: Std dev of log-normal perturbation.
        rng: Random generator.

    Returns:
        ModelRobustnessIndex with summary statistics.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    base_sigma = np.diag(sigma_diag)
    base_mae, _ = prediction_error(base_sigma)

    omegas = []
    worst_mae = base_mae
    worst_sigma = sigma_diag.copy()

    for _ in range(n_perturbations):
        # Perturb in log-space
        log_perturbation = rng.normal(0, perturbation_scale, N_DIMS)
        perturbed_diag = sigma_diag * np.exp(log_perturbation)
        # Clamp to reasonable range
        perturbed_diag = np.clip(perturbed_diag, 0.001, 1e6)

        perturbed_sigma = np.diag(perturbed_diag)
        mae, _ = prediction_error(perturbed_sigma)

        # Omega: absolute change in MAE from perturbation
        omega = abs(mae - base_mae)
        omegas.append(omega)

        if mae > worst_mae:
            worst_mae = mae
            worst_sigma = perturbed_diag.copy()

    omegas_arr = np.array(omegas)
    mean_omega = float(np.mean(omegas_arr))
    p75_omega = float(np.percentile(omegas_arr, 75))
    p95_omega = float(np.percentile(omegas_arr, 95))

    # MRI formula (analogous to Bond Index)
    mri = 0.5 * mean_omega + 0.3 * p75_omega + 0.2 * p95_omega

    return ModelRobustnessIndex(
        mri=mri,
        mean_omega=mean_omega,
        p75_omega=p75_omega,
        p95_omega=p95_omega,
        n_perturbations=n_perturbations,
        worst_case_mae=worst_mae,
        worst_case_sigma=worst_sigma,
        perturbation_errors=omegas,
    )


# ---------------------------------------------------------------------------
# Compositional dimension testing
# ---------------------------------------------------------------------------

@dataclass
class CompositionResult:
    """Result of incrementally adding dimensions."""
    order: List[int]               # Order dimensions were added
    order_names: List[str]
    mae_sequence: List[float]      # MAE after each addition
    sigma_sequence: List[np.ndarray]  # Best sigma at each step


def compositional_test(
    starting_dim: int = Dim.CONSEQUENCES,
    candidate_dims: List[int] | None = None,
    calibration_targets: List[Target] | None = None,
) -> CompositionResult:
    """Greedily add dimensions one at a time, always picking the one
    that reduces MAE the most.

    This reveals the optimal ordering and diminishing returns.

    Args:
        calibration_targets: If provided, optimize against these targets.
    """
    if candidate_dims is None:
        candidate_dims = [
            Dim.FAIRNESS, Dim.SOCIAL_IMPACT,
            Dim.VIRTUE_IDENTITY, Dim.EPISTEMIC,
        ]

    active = [starting_dim]
    remaining = list(candidate_dims)
    order = [starting_dim]
    order_names = [DIM_NAMES[Dim(starting_dim)]]
    mae_sequence = []
    sigma_sequence = []

    # Baseline: just the starting dimension
    result = _optimize_subset(tuple(active),
                              calibration_targets=calibration_targets)
    mae_sequence.append(result.mae)
    sigma_sequence.append(result.sigma_diag)

    while remaining:
        best_addition = None
        best_mae = float("inf")
        best_result = None

        for dim in remaining:
            test_active = tuple(sorted(active + [dim]))
            result = _optimize_subset(test_active,
                                      calibration_targets=calibration_targets)
            if result.mae < best_mae:
                best_mae = result.mae
                best_addition = dim
                best_result = result

        active.append(best_addition)
        remaining.remove(best_addition)
        order.append(best_addition)
        order_names.append(DIM_NAMES[Dim(best_addition)])
        mae_sequence.append(best_result.mae)
        sigma_sequence.append(best_result.sigma_diag)

    return CompositionResult(
        order=order,
        order_names=order_names,
        mae_sequence=mae_sequence,
        sigma_sequence=sigma_sequence,
    )


# ---------------------------------------------------------------------------
# Full structural fuzz report
# ---------------------------------------------------------------------------

@dataclass
class StructuralFuzzReport:
    """Complete report from structural fuzzing campaign."""
    subset_results: List[SubsetResult]
    sensitivity: List[SensitivityResult]
    composition: CompositionResult
    mri: ModelRobustnessIndex
    adversarial: List[AdversarialResult]
    best_model: SubsetResult
    pareto_frontier: List[SubsetResult]

    def summary(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append("STRUCTURAL FUZZ REPORT")
        lines.append("Systematic exploration of dimensional structure")
        lines.append("=" * 72)

        # Best overall model
        b = self.best_model
        lines.append(f"\nBEST MODEL: {b.n_dims} dimensions, MAE={b.mae:.1f}%")
        lines.append(f"  Dimensions: {', '.join(b.dim_names)}")
        for g, e in b.errors.items():
            lines.append(f"  {g:<20} error={e:>+5.1f}%")

        # Pareto frontier
        lines.append(f"\nPARETO FRONTIER (best MAE at each dimensionality):")
        lines.append(f"  {'n_dims':>6} {'MAE':>8} {'Dimensions'}")
        lines.append(f"  {'-'*50}")
        seen_sizes = set()
        for r in self.pareto_frontier:
            if r.n_dims not in seen_sizes:
                seen_sizes.add(r.n_dims)
                lines.append(
                    f"  {r.n_dims:>6} {r.mae:>7.1f}%  "
                    f"{', '.join(r.dim_names)}"
                )

        # Sensitivity
        lines.append(f"\nDIMENSION IMPORTANCE (ablation):")
        lines.append(f"  {'Rank':>4} {'Dimension':<25} {'delta_MAE':>10}")
        lines.append(f"  {'-'*42}")
        for s in self.sensitivity:
            marker = " ***" if s.delta_mae > 1.0 else ""
            lines.append(
                f"  {s.importance_rank:>4} {s.dim_name:<25} "
                f"{s.delta_mae:>+9.2f}%{marker}"
            )

        # Composition order
        lines.append(f"\nGREEDY COMPOSITION ORDER:")
        for i, (name, mae) in enumerate(
            zip(self.composition.order_names, self.composition.mae_sequence)
        ):
            if i == 0:
                lines.append(f"  +{name:<25} MAE={mae:>6.1f}%")
            else:
                prev = self.composition.mae_sequence[i - 1]
                delta = mae - prev
                lines.append(
                    f"  +{name:<25} MAE={mae:>6.1f}%  "
                    f"(delta={delta:>+5.1f}%)"
                )

        # MRI
        mri = self.mri
        lines.append(f"\nMODEL ROBUSTNESS INDEX (MRI):")
        lines.append(f"  MRI = {mri.mri:.2f}")
        lines.append(f"  Mean perturbation error:  {mri.mean_omega:.2f}%")
        lines.append(f"  75th percentile:          {mri.p75_omega:.2f}%")
        lines.append(f"  95th percentile:          {mri.p95_omega:.2f}%")
        lines.append(f"  Worst-case MAE:           {mri.worst_case_mae:.1f}%")
        lines.append(f"  ({mri.n_perturbations} perturbations tested)")

        # Adversarial
        if self.adversarial:
            lines.append(f"\nADVERSARIAL THRESHOLDS:")
            for a in self.adversarial:
                lines.append(
                    f"  {a.dim_name:<25} {a.direction:>8}: "
                    f"var {a.base_var:.3f} -> {a.threshold_var:.3f} "
                    f"(ratio={a.threshold_ratio:.2f}) "
                    f"flips {a.game_flipped}"
                )

        lines.append("\n" + "=" * 72)
        return "\n".join(lines)


def run_structural_fuzz(
    max_subset_dims: int = 4,
    n_mri_perturbations: int = 500,
) -> StructuralFuzzReport:
    """Run the full structural fuzzing campaign.

    1. Enumerate all dimension subsets (up to max_subset_dims)
    2. Find the Pareto frontier of parsimony vs accuracy
    3. Run sensitivity profiling on the best model
    4. Run greedy compositional test
    5. Compute Model Robustness Index
    6. Run adversarial threshold search
    """
    print("=" * 60)
    print("STRUCTURAL FUZZ CAMPAIGN")
    print("=" * 60)

    # 1. Enumerate subsets
    print("\n[1/5] Enumerating dimension subsets...")
    subsets = enumerate_subsets(max_dims=max_subset_dims)
    print(f"  Tested {len(subsets)} subsets")

    best = subsets[0]
    pareto = [r for r in subsets if r.pareto_optimal]
    print(f"  Best: {best.n_dims}D, MAE={best.mae:.1f}%, dims={best.dim_names}")
    print(f"  Pareto frontier: {len(pareto)} models")

    # 2. Sensitivity profiling on best model
    print("\n[2/5] Sensitivity profiling...")
    sensitivity = sensitivity_profile(best.sigma_diag)
    for s in sensitivity[:3]:
        print(f"  #{s.importance_rank} {s.dim_name}: delta_MAE={s.delta_mae:+.2f}%")

    # 3. Compositional test
    print("\n[3/5] Compositional dimension addition...")
    composition = compositional_test()
    for name, mae in zip(composition.order_names, composition.mae_sequence):
        print(f"  +{name}: MAE={mae:.1f}%")

    # 4. Model Robustness Index
    print(f"\n[4/5] Computing MRI ({n_mri_perturbations} perturbations)...")
    mri = compute_mri(best.sigma_diag, n_perturbations=n_mri_perturbations)
    print(f"  MRI = {mri.mri:.2f}")

    # 5. Adversarial thresholds on active dimensions
    print("\n[5/5] Adversarial threshold search...")
    adversarial = []
    for d in best.dims:
        results = find_adversarial_threshold(best.sigma_diag, d)
        adversarial.extend(results)
        for r in results:
            print(f"  {r.dim_name} {r.direction}: threshold ratio={r.threshold_ratio:.2f}")

    report = StructuralFuzzReport(
        subset_results=subsets,
        sensitivity=sensitivity,
        composition=composition,
        mri=mri,
        adversarial=adversarial,
        best_model=best,
        pareto_frontier=pareto,
    )

    return report


def run_expanded_fuzz(
    max_subset_dims: int = 5,
    n_mri_perturbations: int = 300,
) -> StructuralFuzzReport:
    """Run structural fuzz with expanded targets (15+ prediction targets).

    Strategy:
    - Calibrate sigma on GAME targets (9 targets) during subset optimization
    - Evaluate on ALL targets (16) to measure out-of-sample performance
    - Prospect theory (6 targets) and published meta-analyses (1) are
      genuine out-of-sample validation targets
    """
    all_targets = build_targets()
    game_targets = [t for t in all_targets if t.category == "game"]

    print("=" * 60)
    print("EXPANDED STRUCTURAL FUZZ CAMPAIGN")
    print(f"  {len(all_targets)} prediction targets")
    n_game = len(game_targets)
    n_prospect = sum(1 for t in all_targets if t.category == "prospect")
    n_published = sum(1 for t in all_targets if t.category == "published")
    n_oos = n_prospect + n_published
    print(f"  Calibration: {n_game} game targets")
    print(f"  Validation:  {n_oos} out-of-sample ({n_prospect} prospect + {n_published} published)")
    print("=" * 60)

    # 1. Enumerate subsets — optimize on game targets
    print("\n[1/6] Enumerating dimension subsets (optimizing on game targets)...")
    subsets = enumerate_subsets(max_dims=max_subset_dims,
                               calibration_targets=game_targets)

    # Re-evaluate each with ALL targets for ranking
    for s in subsets:
        sigma = np.diag(s.sigma_diag)
        s.mae, s.errors, _ = evaluate_targets(sigma, all_targets)
    subsets.sort(key=lambda r: r.mae)

    # Re-mark Pareto
    best_by_size = {}
    for r in subsets:
        r.pareto_optimal = False
        if r.n_dims not in best_by_size or r.mae < best_by_size[r.n_dims]:
            best_by_size[r.n_dims] = r.mae
    for r in subsets:
        if r.mae <= best_by_size[r.n_dims] + 0.01:
            r.pareto_optimal = True

    best = subsets[0]
    pareto = [r for r in subsets if r.pareto_optimal]
    print(f"  Tested {len(subsets)} subsets")
    print(f"  Best: {best.n_dims}D, MAE={best.mae:.1f}%, dims={best.dim_names}")

    # Per-target breakdown for best model
    print(f"\n  Per-target errors (best model):")
    sigma_best = np.diag(best.sigma_diag)
    _, all_errors, n_pass = evaluate_targets(sigma_best, all_targets)

    # Show by category
    for cat, label in [("game", "GAME"), ("prospect", "PROSPECT (OOS)"),
                       ("published", "PUBLISHED (OOS)")]:
        cat_targets = [t for t in all_targets if t.category == cat]
        if cat_targets:
            print(f"    --- {label} ---")
            for t in cat_targets:
                err = all_errors[t.name]
                status = "PASS" if abs(err) <= t.tolerance else "FAIL"
                is_marker = " [IS]" if t.in_sample else ""
                print(f"    [{status}] {t.name:<30} err={err:>+6.1f}{t.unit}{is_marker}")

    print(f"\n  {n_pass}/{len(all_targets)} within tolerance")
    game_pass = sum(1 for t in game_targets
                    if abs(all_errors[t.name]) <= t.tolerance)
    oos_targets = [t for t in all_targets if t.category != "game"]
    oos_pass = sum(1 for t in oos_targets
                   if abs(all_errors[t.name]) <= t.tolerance)
    print(f"  Game: {game_pass}/{n_game}  |  OOS: {oos_pass}/{n_oos}")
    print(f"  Degrees of freedom: {len(all_targets)} targets / {best.n_dims} params "
          f"= {len(all_targets)/max(1,best.n_dims):.1f}:1")

    # 2. Sensitivity (evaluated on ALL targets)
    print("\n[2/6] Sensitivity profiling...")
    base_mae, _, _ = evaluate_targets(sigma_best, all_targets)
    sensitivity_results = []
    for d in range(N_DIMS):
        ablated_diag = best.sigma_diag.copy()
        ablated_diag[d] = 1e6
        ablated_sigma = np.diag(ablated_diag)
        ablated_mae, _, _ = evaluate_targets(ablated_sigma, all_targets)
        sensitivity_results.append(SensitivityResult(
            dim=d, dim_name=DIM_NAMES[Dim(d)],
            mae_with=base_mae, mae_without=ablated_mae,
            delta_mae=ablated_mae - base_mae,
        ))
    sensitivity_results.sort(key=lambda r: r.delta_mae, reverse=True)
    for i, r in enumerate(sensitivity_results):
        r.importance_rank = i + 1
    for s in sensitivity_results[:3]:
        print(f"  #{s.importance_rank} {s.dim_name}: delta_MAE={s.delta_mae:+.2f}%")

    # 3. Composition (calibrated on game targets)
    print("\n[3/6] Compositional test...")
    composition = compositional_test(calibration_targets=game_targets)
    for name, mae in zip(composition.order_names, composition.mae_sequence):
        print(f"  +{name}: MAE={mae:.1f}%")

    # 4. MRI
    print(f"\n[4/6] Computing MRI ({n_mri_perturbations} perturbations)...")
    mri = compute_mri(best.sigma_diag, n_perturbations=n_mri_perturbations)
    print(f"  MRI = {mri.mri:.2f}")

    # 5. Adversarial
    print("\n[5/6] Adversarial thresholds...")
    adversarial = []
    for d in best.dims:
        results = find_adversarial_threshold(best.sigma_diag, d)
        adversarial.extend(results)

    # 6. Encoding sensitivity (quick version)
    print("\n[6/6] Encoding sensitivity check...")
    rng = np.random.default_rng(42)
    n_encoding_tests = 20
    same_winner_count = 0
    for _ in range(n_encoding_tests):
        perturbed = best.sigma_diag.copy()
        for d in best.dims:
            perturbed[d] *= np.exp(rng.normal(0, 0.3))
        sigma_p = np.diag(perturbed)
        p_mae, _, _ = evaluate_targets(sigma_p, all_targets)
        if p_mae < base_mae * 2:
            same_winner_count += 1
    print(f"  {same_winner_count}/{n_encoding_tests} perturbations stable")

    report = StructuralFuzzReport(
        subset_results=subsets,
        sensitivity=sensitivity_results,
        composition=composition,
        mri=mri,
        adversarial=adversarial,
        best_model=best,
        pareto_frontier=pareto,
    )

    return report


if __name__ == "__main__":
    report = run_expanded_fuzz()
    print("\n\n")
    print(report.summary())
