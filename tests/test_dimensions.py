# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

import pytest

from eris_econ.dimensions import (
    Dim,
    EconomicState,
    N_DIMS,
    TRANSFERABLE_DIMS,
    EVALUATIVE_DIMS,
)


class TestDimensions:
    def test_nine_dimensions(self):
        assert N_DIMS == 9
        assert len(Dim) == 9

    def test_transferable_evaluative_disjoint(self):
        assert TRANSFERABLE_DIMS & EVALUATIVE_DIMS == frozenset()

    def test_all_dims_classified(self):
        from eris_econ.dimensions import PARTIALLY_TRANSFERABLE

        all_classified = TRANSFERABLE_DIMS | EVALUATIVE_DIMS | PARTIALLY_TRANSFERABLE
        assert len(all_classified) == N_DIMS


class TestEconomicState:
    def test_create(self):
        s = EconomicState((1, 2, 3, 4, 5, 6, 7, 8, 9))
        assert s[Dim.CONSEQUENCES] == 1
        assert s[Dim.EPISTEMIC] == 9

    def test_wrong_dims(self):
        with pytest.raises(ValueError):
            EconomicState((1, 2, 3))

    def test_monetary(self):
        s = EconomicState((42, 0, 0, 0, 0, 0, 0, 0, 0))
        assert s.monetary() == 42

    def test_frozen(self):
        s = EconomicState((1, 2, 3, 4, 5, 6, 7, 8, 9))
        # Should be hashable (frozen dataclass)
        {s: "test"}
