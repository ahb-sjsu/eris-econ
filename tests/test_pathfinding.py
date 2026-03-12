# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

import numpy as np
import pytest

from eris_econ.dimensions import EconomicState, N_DIMS
from eris_econ.manifold import EconomicDecisionComplex
from eris_econ.pathfinding import astar, euclidean_heuristic


def _zero_state():
    return EconomicState(tuple([0.0] * N_DIMS))


def _unit_state(dim=0, val=1.0):
    vals = [0.0] * N_DIMS
    vals[dim] = val
    return EconomicState(tuple(vals))


class TestAstar:
    def _simple_graph(self):
        """A → B → C with identity sigma."""
        E = EconomicDecisionComplex(np.eye(N_DIMS))
        E.add_vertex("A", _zero_state())
        E.add_vertex("B", _unit_state(0, 1.0))
        E.add_vertex("C", _unit_state(0, 2.0))
        E.add_edge("A", "B")
        E.add_edge("B", "C")
        E.add_edge("A", "C")  # Direct but longer
        E.compute_weights()
        return E

    def test_finds_path(self):
        E = self._simple_graph()
        result = astar(E, "A", {"C"})
        assert result.found
        assert result.path[0] == "A"
        assert result.path[-1] == "C"

    def test_shortest_path(self):
        """Direct A→C has distance 2, A→B→C has distance 1+1=2.
        Both are equally optimal for identity sigma."""
        E = self._simple_graph()
        result = astar(E, "A", {"C"})
        assert result.found
        assert result.total_cost <= 2.0 + 1e-6

    def test_no_path(self):
        E = EconomicDecisionComplex(np.eye(N_DIMS))
        E.add_vertex("A", _zero_state())
        E.add_vertex("B", _unit_state(0))
        # No edge from A to B
        E.compute_weights()
        result = astar(E, "A", {"B"})
        assert not result.found

    def test_start_is_goal(self):
        E = EconomicDecisionComplex(np.eye(N_DIMS))
        E.add_vertex("A", _zero_state())
        E.compute_weights()
        result = astar(E, "A", {"A"})
        assert result.found
        assert result.total_cost == 0.0
        assert result.path == ["A"]

    def test_with_heuristic(self):
        E = self._simple_graph()
        h = euclidean_heuristic({"C"})
        result = astar(E, "A", {"C"}, heuristic=h)
        assert result.found

    def test_avoids_infinite_weight(self):
        """Paths through sacred-value boundaries should be avoided."""
        E = EconomicDecisionComplex(
            np.eye(N_DIMS),
            boundaries={"theft": float("inf")},
        )
        s0 = EconomicState(tuple([0.0] * N_DIMS))
        s_theft = list([0.0] * N_DIMS)
        s_theft[1] = -0.5  # Rights go negative
        s1 = EconomicState(tuple(s_theft))
        s_good = list([0.0] * N_DIMS)
        s_good[0] = 5.0  # Just get monetary gain
        s2 = EconomicState(tuple(s_good))

        E.add_vertex("start", s0)
        E.add_vertex("steal", s1)
        E.add_vertex("earn", s2)
        E.add_vertex("goal", _unit_state(0, 10.0))

        E.add_edge("start", "steal")
        E.add_edge("start", "earn")
        E.add_edge("steal", "goal")
        E.add_edge("earn", "goal")
        E.compute_weights()

        result = astar(E, "start", {"goal"})
        assert result.found
        assert "steal" not in result.path

    def test_missing_start(self):
        E = EconomicDecisionComplex(np.eye(N_DIMS))
        E.add_vertex("A", _zero_state())
        E.compute_weights()
        with pytest.raises(KeyError):
            astar(E, "X", {"A"})
