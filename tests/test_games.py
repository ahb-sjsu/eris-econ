# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

from eris_econ.games import (
    ultimatum_game,
    dictator_game,
    prisoners_dilemma,
    public_goods_game,
)
from eris_econ.pathfinding import astar


class TestUltimatumGame:
    def test_builds_graph(self):
        E = ultimatum_game(stake=10.0)
        assert E.n_vertices > 1
        assert E.n_edges > 0

    def test_prefers_fair_split(self):
        """Optimal path should NOT be the most selfish offer."""
        E = ultimatum_game(stake=10.0)
        result = astar(
            E, "start", {"offer_0", "offer_10", "offer_20", "offer_30", "offer_40", "offer_50"}
        )
        assert result.found
        # Should not choose 0% (maximally unfair)
        assert result.path[-1] != "offer_0"


class TestDictatorGame:
    def test_builds_graph(self):
        E = dictator_game()
        assert E.n_vertices > 1


class TestPrisonersDilemma:
    def test_builds_both(self):
        A, B = prisoners_dilemma()
        assert A.n_vertices == 3
        assert B.n_vertices == 3


class TestPublicGoodsGame:
    def test_builds_graph(self):
        E = public_goods_game()
        assert E.n_vertices > 1
