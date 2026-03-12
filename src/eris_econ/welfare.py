# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Multi-dimensional welfare analysis.

Classical welfare economics measures Pareto optimality on d_1 (monetary).
Geometric welfare computes Pareto optimality on the full 9D manifold.
A state that is Pareto-optimal on d_1 alone may be Pareto-dominated
on the full manifold (when non-monetary dimensions are included).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from eris_econ.dimensions import N_DIMS, Dim


def pareto_dominates(
    state_a: np.ndarray,
    state_b: np.ndarray,
    dims: List[int] | None = None,
) -> bool:
    """Check if state_a Pareto-dominates state_b on specified dimensions.

    a dominates b iff: a[i] >= b[i] for all i, and a[j] > b[j] for some j.
    Higher values are better on all dimensions.

    Args:
        state_a, state_b: attribute vectors
        dims: which dimensions to check (default: all 9)
    """
    if dims is None:
        dims = list(range(N_DIMS))

    a = state_a[dims]
    b = state_b[dims]

    return bool(np.all(a >= b) and np.any(a > b))


def pareto_frontier(
    states: Dict[str, np.ndarray],
    dims: List[int] | None = None,
) -> List[str]:
    """Find the Pareto frontier (non-dominated states).

    Args:
        states: {state_id: attribute_vector}
        dims: which dimensions to consider

    Returns:
        List of state_ids on the Pareto frontier.
    """
    ids = list(states.keys())
    vectors = [states[sid] for sid in ids]
    n = len(ids)
    dominated = [False] * n

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if pareto_dominates(vectors[j], vectors[i], dims):
                dominated[i] = True
                break

    return [ids[i] for i in range(n) if not dominated[i]]


def monetary_pareto_frontier(states: Dict[str, np.ndarray]) -> List[str]:
    """Pareto frontier on d_1 (consequences) only — classical welfare."""
    return pareto_frontier(states, dims=[Dim.CONSEQUENCES])


def full_pareto_frontier(states: Dict[str, np.ndarray]) -> List[str]:
    """Pareto frontier on all 9 dimensions — geometric welfare."""
    return pareto_frontier(states)


def welfare_gap(
    states: Dict[str, np.ndarray],
) -> Tuple[List[str], List[str]]:
    """Find states that are monetary-Pareto-optimal but full-Pareto-dominated.

    These are the states where classical welfare analysis is misleading:
    they look optimal on money alone but are dominated when all dimensions
    are considered.

    Returns:
        (monetary_only, gap_states) where gap_states ⊆ monetary_only
        are states on the monetary frontier but NOT on the full frontier.
    """
    monetary = set(monetary_pareto_frontier(states))
    full = set(full_pareto_frontier(states))

    gap = monetary - full
    return sorted(monetary), sorted(gap)


def social_welfare(
    agent_states: List[np.ndarray],
    method: str = "utilitarian",
    weights: np.ndarray | None = None,
) -> float:
    """Compute social welfare across agents on the full manifold.

    Args:
        agent_states: list of attribute vectors, one per agent
        method: "utilitarian" (sum), "rawlsian" (min), "prioritarian" (weighted)
        weights: dimension weights for aggregation (default: equal)

    Returns:
        Scalar social welfare measure.
    """
    if weights is None:
        weights = np.ones(N_DIMS) / N_DIMS

    agent_scores = [float(weights @ state) for state in agent_states]

    if method == "utilitarian":
        return sum(agent_scores) / len(agent_scores)
    elif method == "rawlsian":
        return min(agent_scores)
    elif method == "prioritarian":
        # Concave transform: prioritize worse-off agents
        return sum(np.sqrt(max(0, s)) for s in agent_scores) / len(agent_scores)
    else:
        raise ValueError(f"Unknown method: {method}")
