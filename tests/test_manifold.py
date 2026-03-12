# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

import numpy as np
import pytest

from eris_econ.dimensions import EconomicState, N_DIMS
from eris_econ.manifold import EconomicDecisionComplex


class TestEconomicDecisionComplex:
    def _make_sigma(self):
        return np.eye(N_DIMS)

    def test_add_vertex(self):
        E = EconomicDecisionComplex(self._make_sigma())
        s = EconomicState(tuple(range(N_DIMS)))
        E.add_vertex("v0", s)
        assert E.n_vertices == 1

    def test_add_edge(self):
        E = EconomicDecisionComplex(self._make_sigma())
        s1 = EconomicState(tuple([0.0] * N_DIMS))
        s2 = EconomicState(tuple([1.0] * N_DIMS))
        E.add_vertex("a", s1)
        E.add_vertex("b", s2)
        E.add_edge("a", "b", label="trade")
        assert E.n_edges == 1

    def test_edge_missing_vertex(self):
        E = EconomicDecisionComplex(self._make_sigma())
        s = EconomicState(tuple([0.0] * N_DIMS))
        E.add_vertex("a", s)
        with pytest.raises(KeyError):
            E.add_edge("a", "nonexistent")

    def test_compute_weights(self):
        E = EconomicDecisionComplex(self._make_sigma())
        s1 = EconomicState(tuple([0.0] * N_DIMS))
        s2 = EconomicState(tuple([1.0] * N_DIMS))
        E.add_vertex("a", s1)
        E.add_vertex("b", s2)
        e = E.add_edge("a", "b")
        E.compute_weights()
        assert e.weight > 0

    def test_bidirectional(self):
        E = EconomicDecisionComplex(self._make_sigma())
        s1 = EconomicState(tuple([0.0] * N_DIMS))
        s2 = EconomicState(tuple([1.0] * N_DIMS))
        E.add_vertex("a", s1)
        E.add_vertex("b", s2)
        E.add_bidirectional("a", "b")
        assert E.n_edges == 2

    def test_neighbors(self):
        E = EconomicDecisionComplex(self._make_sigma())
        s1 = EconomicState(tuple([0.0] * N_DIMS))
        s2 = EconomicState(tuple([1.0] * N_DIMS))
        E.add_vertex("a", s1)
        E.add_vertex("b", s2)
        E.add_edge("a", "b")
        assert len(E.neighbors("a")) == 1
        assert len(E.neighbors("b")) == 0

    def test_sigma_wrong_shape(self):
        with pytest.raises(ValueError):
            EconomicDecisionComplex(np.eye(5))
