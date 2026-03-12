# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

import numpy as np

from eris_econ.dimensions import N_DIMS
from eris_econ.welfare import (
    pareto_dominates,
    pareto_frontier,
    full_pareto_frontier,
    welfare_gap,
    social_welfare,
)


class TestParetoDominates:
    def test_dominates(self):
        a = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=float)
        b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
        assert pareto_dominates(a, b)

    def test_not_dominates_equal(self):
        a = np.ones(N_DIMS)
        assert not pareto_dominates(a, a)

    def test_not_dominates_tradeoff(self):
        a = np.array([2, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
        b = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1], dtype=float)
        assert not pareto_dominates(a, b)


class TestParetoFrontier:
    def test_simple(self):
        states = {
            "best": np.array([3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=float),
            "worst": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
        }
        frontier = pareto_frontier(states)
        assert frontier == ["best"]


class TestWelfareGap:
    def test_gap_exists(self):
        """A state can be monetary-optimal but full-manifold-dominated."""
        states = {
            "rich_unfair": np.array([10, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
            "moderate_fair": np.array([7, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
        }
        monetary, gap = welfare_gap(states)
        # rich_unfair is on monetary frontier
        assert "rich_unfair" in monetary
        # But moderate_fair dominates on full manifold
        full = full_pareto_frontier(states)
        assert "moderate_fair" in full


class TestSocialWelfare:
    def test_utilitarian(self):
        agents = [np.ones(N_DIMS), np.ones(N_DIMS) * 2]
        w = social_welfare(agents, method="utilitarian")
        assert w > 0

    def test_rawlsian(self):
        agents = [np.ones(N_DIMS), np.ones(N_DIMS) * 0.1]
        w = social_welfare(agents, method="rawlsian")
        # Should be determined by worst-off agent
        w2 = social_welfare([np.ones(N_DIMS) * 0.1], method="rawlsian")
        assert abs(w - w2) < 1e-6
