# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Table VII of Bond (2026, IEEE TCSS) is reproduced exactly by the package.

Every number below is transcribed from the published table and the
surrounding text, at the precision printed there.  If any of these tests
fails, the package no longer reproduces the paper.
"""

import json

import numpy as np
import pytest

from eris_econ.dimensions import Dim
from eris_econ.paper import (
    INACTIVE_VARIANCE,
    SELECTED_VARIANCES,
    main,
    round_half_up,
    selected_sigma,
    table_vii,
    to_latex,
    to_markdown,
    to_text,
)

# Table VII, columns: target, observed (%), predicted (%), error (pp), tolerance (pp).
TABLE_VII = [
    ("Ultimatum mean offer", 48.3, 48.0, -0.3, 5),
    ("Ultimatum modal offer", 50.0, 48.0, -2.0, 5),
    ("Dictator mean giving", 28.35, 32.0, 3.7, 5),
    ("Responder MAO", 34.0, 34.0, 0.0, 5),
    ("PG round 1", 45.7, 50.0, 4.3, 5),
    ("PG round 3", 50.0, 48.0, -2.0, 5),
    ("PG round 5", 48.7, 46.0, -2.7, 5),
    ("PG round 8", 44.6, 43.0, -1.6, 5),
    ("PG round 10", 39.0, 40.0, 1.0, 5),
    ("PT P1 Allais (certainty)", 25.5, 26.6, 1.1, 10),
    ("PT P3 Certainty strong", 12.8, 15.4, 2.6, 10),
    ("PT P7 Reflection", 79.4, 84.2, 4.8, 10),
    ("PT P11 Isolation", 16.1, 15.4, -0.7, 10),
    ("PT P16 Small-prob gain", 57.4, 50.7, -6.7, 10),
    ("PT P17 Small-prob loss", 42.8, 50.6, 7.8, 10),
    ("Guth (1982) ultimatum", 37.0, 35.0, -2.0, 10),
]


@pytest.fixture(scope="module")
def table():
    return table_vii()


def test_selected_sigma_matches_section_vi_a():
    sigma = selected_sigma()
    assert sigma.shape == (9, 9)
    assert np.count_nonzero(sigma - np.diag(np.diag(sigma))) == 0
    assert sigma[Dim.SOCIAL_IMPACT, Dim.SOCIAL_IMPACT] == 78.26
    assert sigma[Dim.VIRTUE_IDENTITY, Dim.VIRTUE_IDENTITY] == 32.28
    assert sigma[Dim.EPISTEMIC, Dim.EPISTEMIC] == 0.01262
    inactive = [d for d in Dim if d not in SELECTED_VARIANCES]
    assert len(inactive) == 6
    assert all(sigma[d, d] == INACTIVE_VARIANCE for d in inactive)


def test_sixteen_rows_in_table_order(table):
    assert [r.target for r in table.rows] == [row[0] for row in TABLE_VII]
    assert [r.index for r in table.rows] == list(range(1, 17))


@pytest.mark.parametrize("name,observed,predicted,error,tolerance", TABLE_VII)
def test_each_row_reproduces_the_paper(table, name, observed, predicted, error, tolerance):
    row = next(r for r in table.rows if r.target == name)
    assert row.observed == observed
    assert round_half_up(row.predicted, 1) == predicted
    assert round_half_up(row.error, 1) == error
    assert row.tolerance == tolerance
    assert row.passed


def test_summary_figures_quoted_in_the_text(table):
    s = table.summary
    assert s["passed"] == 16 and s["n_targets"] == 16
    assert s["games_passed"] == 9
    assert s["lotteries_passed"] == 6
    assert round_half_up(s["overall_mae"], 2) == 2.70  # Table VII, "Overall"
    assert round_half_up(s["lottery_mae"], 2) == 3.95  # Table VII, "Ruggeri lottery subset"
    assert round_half_up(s["game_mae"], 2) == 1.95  # Section VI.F
    assert round_half_up(s["non_game_mae"], 2) == 3.67  # Section VI.F
    assert round_half_up(s["non_game_to_game_ratio"], 2) == 1.88  # Section VI.F
    assert round_half_up(s["ranking_only_mae"], 1) == 5.0  # Section VI.F, ranking-only lotteries


def test_p11_shares_the_encoding_of_p3(table):
    """Section IV.D: P11 is encoded as the reduced second stage, identical to P3."""
    by = {r.target: r for r in table.rows}
    assert by["PT P11 Isolation"].predicted == by["PT P3 Certainty strong"].predicted


def test_game_predictions_lie_on_the_integer_grid(table):
    """Section IV: game predictions are cost minima on an integer percentage grid."""
    for r in table.rows:
        if r.is_game or r.target.startswith("Guth"):
            assert float(r.predicted).is_integer()


def test_round_half_up_matches_the_printed_table():
    assert round_half_up(3.65, 1) == 3.7
    assert round_half_up(26.55, 1) == 26.6
    assert round_half_up(-6.718, 1) == -6.7
    assert round_half_up(2.7024, 2) == 2.70


def test_renderers_contain_every_target(table):
    for render in (to_text, to_markdown, to_latex):
        out = render(table)
        for name, *_ in TABLE_VII:
            assert name in out
    assert "16/16 pass" in to_text(table)


def test_cli_json_round_trip(capsys):
    assert main(["--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["rows"]) == 16
    assert payload["summary"]["passed"] == 16
