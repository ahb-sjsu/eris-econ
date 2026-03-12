# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

import numpy as np

from eris_econ.dimensions import EconomicState, N_DIMS
from eris_econ.equilibrium import Agent, compute_bge, behavioral_friction
from eris_econ.manifold import EconomicDecisionComplex


def _zero():
    return EconomicState(tuple([0.0] * N_DIMS))


def _state(*vals):
    full = [0.0] * N_DIMS
    for i, v in enumerate(vals):
        full[i] = v
    return EconomicState(tuple(full))


class TestBGE:
    def test_single_agent(self):
        """Single agent BGE should converge immediately."""
        E = EconomicDecisionComplex(np.eye(N_DIMS))
        E.add_vertex("start", _zero())
        E.add_vertex("goal", _state(1.0))
        E.add_edge("start", "goal")

        agent = Agent(agent_id="A", complex=E, start="start", goals={"goal"})
        result = compute_bge([agent])
        assert result.converged
        assert result.n_agents == 1
        assert result.agent_paths["A"].found

    def test_two_agents_converge(self):
        """Two independent agents should converge."""
        agents = []
        for name in ["A", "B"]:
            E = EconomicDecisionComplex(np.eye(N_DIMS))
            E.add_vertex("s", _zero())
            E.add_vertex("g", _state(1.0))
            E.add_edge("s", "g")
            agents.append(Agent(agent_id=name, complex=E, start="s", goals={"g"}))

        result = compute_bge(agents)
        assert result.converged
        assert result.n_agents == 2


class TestBehavioralFriction:
    def test_path_cost(self):
        E = EconomicDecisionComplex(np.eye(N_DIMS))
        E.add_vertex("s", _zero())
        E.add_vertex("g", _state(1.0))
        E.add_edge("s", "g")
        E.compute_weights()

        from eris_econ.pathfinding import astar

        path = astar(E, "s", {"g"})
        bf = behavioral_friction(path)
        assert bf > 0
