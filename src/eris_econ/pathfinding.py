# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""A* pathfinding on the Economic Decision Complex.

The Bond Geodesic is the optimal path from current state to goal region,
minimizing total edge weight (Mahalanobis distance + boundary penalties).

The heuristic h(n) corresponds to System 1 (fast, automatic moral intuition).
The accumulated cost g(n) corresponds to System 2 (deliberate calculation).
Together they implement the dual-system decision process as A* search.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from eris_econ.manifold import EconomicDecisionComplex


@dataclass
class PathResult:
    """Result of A* pathfinding — the Bond Geodesic."""

    path: List[str]  # Sequence of vertex ids from start to goal
    total_cost: float  # Total path weight (g-cost of goal)
    explored: int  # Number of vertices explored
    found: bool  # Whether a path was found

    @property
    def n_steps(self) -> int:
        return max(0, len(self.path) - 1)


@dataclass(order=True)
class _Node:
    """Priority queue entry for A* search."""

    f_score: float
    vertex_id: str = field(compare=False)
    g_score: float = field(compare=False)


def zero_heuristic(vid: str, graph: EconomicDecisionComplex) -> float:
    """Trivial heuristic (Dijkstra). Always admissible."""
    return 0.0


def euclidean_heuristic(
    goal_ids: Set[str],
) -> Callable:
    """Euclidean distance to nearest goal state as heuristic.

    Admissible when Σ is identity (underestimates Mahalanobis distance).
    """

    def h(vid: str, graph: EconomicDecisionComplex) -> float:
        state = np.array(graph.get_state(vid).values)
        min_dist = float("inf")
        for gid in goal_ids:
            goal_state = np.array(graph.get_state(gid).values)
            d = np.linalg.norm(state - goal_state)
            min_dist = min(min_dist, d)
        return min_dist

    return h


def moral_heuristic(
    goal_ids: Set[str],
    boundary_probs: Dict[str, float],
    boundary_penalties: Dict[str, float],
) -> Callable:
    """Moral heuristic: h_M(n) = Σ_k β_k · P(cross boundary k from n).

    This is the Heuristic Truncation Theorem (Theorem 20.1):
    moral heuristics are admissible A* heuristics when β_k ≤ β_k*.

    boundary_probs maps boundary name → probability of crossing.
    """

    def h(vid: str, graph: EconomicDecisionComplex) -> float:
        cost = 0.0
        for name, prob in boundary_probs.items():
            beta = boundary_penalties.get(name, 0.0)
            if not np.isinf(beta):
                cost += beta * prob
        return cost

    return h


def astar(
    graph: EconomicDecisionComplex,
    start: str,
    goals: Set[str],
    heuristic: Optional[Callable] = None,
    max_explored: int = 100000,
) -> PathResult:
    """A* search for the Bond Geodesic.

    Args:
        graph: The Economic Decision Complex (must have compute_weights called)
        start: Starting vertex id
        goals: Set of goal vertex ids
        heuristic: h(vid, graph) → float. Default: zero (Dijkstra).
        max_explored: Safety limit on explored vertices.

    Returns:
        PathResult with optimal path, cost, and diagnostics.
    """
    if heuristic is None:
        heuristic = zero_heuristic

    if start not in graph.vertices:
        raise KeyError(f"Start vertex '{start}' not in graph")
    for g in goals:
        if g not in graph.vertices:
            raise KeyError(f"Goal vertex '{g}' not in graph")

    # A* initialization
    g_scores: Dict[str, float] = {start: 0.0}
    came_from: Dict[str, str] = {}
    open_set: List[_Node] = []
    closed_set: Set[str] = set()

    h_start = heuristic(start, graph)
    heapq.heappush(open_set, _Node(f_score=h_start, vertex_id=start, g_score=0.0))

    explored = 0

    while open_set and explored < max_explored:
        current = heapq.heappop(open_set)
        vid = current.vertex_id

        if vid in closed_set:
            continue
        closed_set.add(vid)
        explored += 1

        # Goal check
        if vid in goals:
            path = _reconstruct_path(came_from, vid)
            return PathResult(
                path=path,
                total_cost=current.g_score,
                explored=explored,
                found=True,
            )

        # Expand neighbors
        for edge in graph.neighbors(vid):
            neighbor = edge.target
            if neighbor in closed_set:
                continue

            tentative_g = current.g_score + edge.weight

            # Skip infinite-weight edges (sacred boundary violations)
            if np.isinf(tentative_g):
                continue

            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                came_from[neighbor] = vid
                h = heuristic(neighbor, graph)
                f = tentative_g + h
                heapq.heappush(
                    open_set,
                    _Node(f_score=f, vertex_id=neighbor, g_score=tentative_g),
                )

    # No path found
    return PathResult(path=[], total_cost=float("inf"), explored=explored, found=False)


def _reconstruct_path(came_from: Dict[str, str], current: str) -> List[str]:
    """Trace back from goal to start."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
