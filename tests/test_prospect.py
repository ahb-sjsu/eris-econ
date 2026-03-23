# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for prospect theory module: KT problem encodings."""

import numpy as np
import pytest

from eris_econ.dimensions import Dim, N_DIMS
from eris_econ.prospect import (
    KT_PROBLEMS,
    KTProblem,
    Prospect,
    prospect_to_state,
)


class TestProspect:
    """Tests for the Prospect dataclass."""

    def test_ev_calculation(self):
        p = Prospect([(100, 0.5), (0, 0.5)])
        assert p.ev == 50.0

    def test_ev_certain(self):
        p = Prospect([(100, 1.0)])
        assert p.ev == 100.0

    def test_is_certain_true(self):
        p = Prospect([(100, 1.0)])
        assert p.is_certain

    def test_is_certain_false(self):
        p = Prospect([(100, 0.5), (0, 0.5)])
        assert not p.is_certain

    def test_max_loss_with_loss(self):
        p = Prospect([(-100, 0.5), (0, 0.5)])
        assert p.max_loss == -100

    def test_max_loss_no_loss(self):
        p = Prospect([(100, 0.5), (50, 0.5)])
        assert p.max_loss == 0.0

    def test_variance(self):
        p = Prospect([(100, 0.5), (0, 0.5)])
        assert p.variance == 2500.0  # 0.5*(100-50)^2 + 0.5*(0-50)^2


class TestKTProblems:
    """Tests for the 17 KT problem definitions."""

    def test_17_problems(self):
        assert len(KT_PROBLEMS) == 17

    def test_ruggeri_ids_1_to_17(self):
        ids = [p.ruggeri_id for p in KT_PROBLEMS]
        assert set(ids) == set(range(1, 18))

    def test_all_have_two_options(self):
        for p in KT_PROBLEMS:
            assert isinstance(p.option_a, Prospect)
            assert isinstance(p.option_b, Prospect)

    def test_domains(self):
        domains = {p.domain for p in KT_PROBLEMS}
        assert "gain" in domains
        assert "loss" in domains

    def test_phenomena(self):
        phenomena = {p.phenomenon for p in KT_PROBLEMS}
        assert "certainty" in phenomena
        assert "reflection" in phenomena
        assert "isolation" in phenomena

    def test_problem_1_allais_paradox(self):
        """Problem 1: (2500, 0.33; 2400, 0.66; 0, 0.01) vs (2400, 1.0)."""
        p = KT_PROBLEMS[0]
        assert p.ruggeri_id == 1
        assert abs(p.option_a.ev - 2409) < 1
        assert p.option_b.ev == 2400
        assert p.option_b.is_certain

    def test_problem_7_reflection(self):
        """Problem 7: loss version of problem 3."""
        p = KT_PROBLEMS[6]
        assert p.ruggeri_id == 7
        assert p.domain == "loss"
        assert p.option_a.max_loss < 0

    def test_endowment_problems(self):
        """Problems 12 and 13 have endowments."""
        p12 = [p for p in KT_PROBLEMS if p.ruggeri_id == 12][0]
        p13 = [p for p in KT_PROBLEMS if p.ruggeri_id == 13][0]
        assert p12.endowment == 2000
        assert p13.endowment == 4000


class TestProspectToState:
    """Tests for the prospect -> 9D state mapping."""

    def test_output_shape(self):
        p = Prospect([(100, 1.0)])
        state = prospect_to_state(p)
        assert state.shape == (N_DIMS,)

    def test_consequences_proportional_to_ev(self):
        p1 = Prospect([(1000, 1.0)])
        p2 = Prospect([(2000, 1.0)])
        s1 = prospect_to_state(p1)
        s2 = prospect_to_state(p2)
        assert s2[Dim.CONSEQUENCES] > s1[Dim.CONSEQUENCES]

    def test_certain_prospect_high_trust(self):
        p = Prospect([(1000, 1.0)])
        state = prospect_to_state(p)
        assert state[Dim.PRIVACY_TRUST] > 0.8

    def test_risky_prospect_lower_trust(self):
        p = Prospect([(2000, 0.5), (0, 0.5)])
        state = prospect_to_state(p)
        assert state[Dim.PRIVACY_TRUST] < 0.8

    def test_certain_prospect_high_epistemic(self):
        p = Prospect([(1000, 1.0)])
        state = prospect_to_state(p)
        assert state[Dim.EPISTEMIC] > 0.8

    def test_loss_reduces_rights(self):
        p_loss = Prospect([(-1000, 1.0)])
        state = prospect_to_state(p_loss)
        assert state[Dim.RIGHTS] < 1.0

    def test_gain_preserves_rights(self):
        p_gain = Prospect([(1000, 1.0)])
        state = prospect_to_state(p_gain)
        assert state[Dim.RIGHTS] == 1.0

    def test_certain_prospect_prudent_identity(self):
        p = Prospect([(1000, 1.0)])
        state = prospect_to_state(p)
        assert state[Dim.VIRTUE_IDENTITY] > 0.6

    def test_endowment_shifts_consequences(self):
        p = Prospect([(0, 1.0)])
        s_no_endow = prospect_to_state(p, endowment=0)
        s_endow = prospect_to_state(p, endowment=1000)
        assert s_endow[Dim.CONSEQUENCES] > s_no_endow[Dim.CONSEQUENCES]
