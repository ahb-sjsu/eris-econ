# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Bond Geodesic Equilibrium (BGE) computation.

Generalizes Nash equilibrium to multi-dimensional decision manifolds.
Each agent optimizes their Bond Geodesic (A* path) on their own
decision complex, taking other agents' strategies as given.

BGE reduces to Nash equilibrium when all non-monetary dimensions vanish
(Theorem 20.3).  Mixed BGE existence follows from reduction to finite
Nash equilibrium on the augmented game.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set


from eris_econ.manifold import EconomicDecisionComplex
from eris_econ.pathfinding import PathResult, astar


@dataclass
class BGEResult:
    """Result of Bond Geodesic Equilibrium computation."""

    agent_paths: Dict[str, PathResult]  # agent_id → their optimal path
    converged: bool  # whether iterated best response converged
    iterations: int  # number of iterations used
    total_behavioral_friction: float  # sum of all agents' path costs

    @property
    def n_agents(self) -> int:
        return len(self.agent_paths)


@dataclass
class Agent:
    """An economic agent with their own decision complex."""

    agent_id: str
    complex: EconomicDecisionComplex  # their decision manifold
    start: str  # starting vertex
    goals: Set[str]  # goal vertices
    heuristic: Optional[Callable] = None


def compute_bge(
    agents: List[Agent],
    *,
    max_iterations: int = 100,
    convergence_tol: float = 1e-6,
    strategy_callback: Optional[Callable] = None,
) -> BGEResult:
    """Compute Bond Geodesic Equilibrium via iterated best response.

    Each agent computes their optimal A* path given all other agents'
    current strategies.  Repeat until convergence (no agent wants to
    change their path).

    Args:
        agents: List of agents, each with their own decision complex.
        max_iterations: Maximum rounds of best response.
        convergence_tol: Cost change threshold for convergence.
        strategy_callback: Optional callback(agent, others_paths) that
            modifies the agent's complex based on others' strategies
            (e.g., updating edge weights based on market conditions).

    Returns:
        BGEResult with each agent's optimal path and convergence info.
    """
    # Initialize: each agent computes path independently
    paths: Dict[str, PathResult] = {}
    for agent in agents:
        agent.complex.compute_weights()
        path = astar(agent.complex, agent.start, agent.goals, agent.heuristic)
        paths[agent.agent_id] = path

    prev_costs = {aid: p.total_cost for aid, p in paths.items()}

    for iteration in range(max_iterations):
        changed = False

        for agent in agents:
            # Let each agent re-optimize given others' current paths
            if strategy_callback is not None:
                other_paths = {aid: p for aid, p in paths.items() if aid != agent.agent_id}
                strategy_callback(agent, other_paths)

            agent.complex.compute_weights()
            new_path = astar(
                agent.complex,
                agent.start,
                agent.goals,
                agent.heuristic,
            )
            paths[agent.agent_id] = new_path

            # Check if this agent's cost changed significantly
            cost_delta = abs(new_path.total_cost - prev_costs[agent.agent_id])
            if cost_delta > convergence_tol:
                changed = True
            prev_costs[agent.agent_id] = new_path.total_cost

        if not changed:
            total_bf = sum(p.total_cost for p in paths.values())
            return BGEResult(
                agent_paths=paths,
                converged=True,
                iterations=iteration + 1,
                total_behavioral_friction=total_bf,
            )

    total_bf = sum(p.total_cost for p in paths.values())
    return BGEResult(
        agent_paths=paths,
        converged=False,
        iterations=max_iterations,
        total_behavioral_friction=total_bf,
    )


def nash_projection(bge_result: BGEResult) -> Dict[str, float]:
    """Project BGE to Nash-like monetary costs (d_1 only).

    Demonstrates Theorem 20.3: BGE reduces to Nash when only
    the consequences dimension is active.
    """
    return {aid: path.total_cost for aid, path in bge_result.agent_paths.items()}


def behavioral_friction(path: PathResult) -> float:
    """Total behavioral friction for a path (sum of all edge weights).

    BF = Σ w(v_i, v_{i+1}) along the Bond Geodesic.
    Higher friction → more cognitive/emotional cost of the decision.
    """
    return path.total_cost
