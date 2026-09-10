# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Reproduce Table VII of Bond (2026), IEEE Transactions on Computational Social Systems.

Table VII lists the sixteen benchmark targets of *Geometric Prediction of
Economic Behavior: Cross-Domain Validation Across Game Theory and Prospect
Theory*, the prediction of the selected model for each, the signed error, the
tolerance, and the pass mark, together with the overall and subset error
figures quoted in the paper.  Everything here is computed from the same code
path as the paper (:mod:`eris_econ.targets`) under the paper's selected
diagonal covariance.  Nothing is stored; the numbers are recomputed on every
call.

One command::

    eris-econ-table7            # or: python -m eris_econ.paper

Programmatic use::

    from eris_econ.paper import table_vii
    t = table_vii()
    t.summary["overall_mae"]    # 2.70 in the paper
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Sequence

import numpy as np

from eris_econ.dimensions import N_DIMS, Dim
from eris_econ.targets import Target, build_targets

CITATION = (
    'A. H. Bond, "Geometric prediction of economic behavior: cross-domain validation '
    'across game theory and prospect theory," IEEE Trans. Comput. Social Syst., 2026, '
    "Table VII."
)

#: The selected model of the paper: three active variances on social impact
#: (d6), virtue and identity (d7), and epistemic status (d9), found by the
#: structural-fuzzing search of Section V.  Inactive dimensions are pinned to
#: a variance large enough to contribute negligible cost (Section V.A).
SELECTED_VARIANCES: Dict[Dim, float] = {
    Dim.SOCIAL_IMPACT: 78.26,
    Dim.VIRTUE_IDENTITY: 32.28,
    Dim.EPISTEMIC: 0.01262,
}
INACTIVE_VARIANCE = 1.0e6

#: What each target was used for (Table VII, "Role" column, and Section V.D).
#: Every target entered the ranking of candidate active sets; the roles below
#: record the additional use, if any.
ROLES: Dict[str, str] = {
    "Ultimatum mean offer": "Game, calibration",
    "Ultimatum modal offer": "Game, calibration",
    "Dictator mean giving": "Game, calibration",
    "Responder MAO": "Game, calibration",
    "PG round 1": "Game, calibration",
    "PG round 3": "Game, calibration",
    "PG round 5": "Game, calibration",
    "PG round 8": "Game, calibration",
    "PG round 10": "Game, calibration",
    "PT P1 Allais (certainty)": "Lottery, sets T",
    "PT P3 Certainty strong": "Lottery, sets T",
    "PT P7 Reflection": "Lottery, ranking only",
    "PT P11 Isolation": "Lottery, ranking only",
    "PT P16 Small-prob gain": "Lottery, ranking only",
    "PT P17 Small-prob loss": "Lottery, ranking only",
    "Guth (1982) ultimatum": "Historical, sets r9",
}

#: The four lottery targets that entered only the ranking step (Section V.D).
RANKING_ONLY = (
    "PT P7 Reflection",
    "PT P11 Isolation",
    "PT P16 Small-prob gain",
    "PT P17 Small-prob loss",
)


def selected_sigma() -> np.ndarray:
    """Return the paper's selected diagonal covariance as a 9 x 9 matrix."""
    diag = np.full(N_DIMS, INACTIVE_VARIANCE)
    for dim, variance in SELECTED_VARIANCES.items():
        diag[dim] = variance
    return np.diag(diag)


def round_half_up(x: float, places: int) -> float:
    """Round as the paper's tables do (half away from zero), not banker's rounding.

    ``round(3.65, 1)`` in Python gives 3.6 because 3.65 is stored as
    3.64999...; the paper prints 3.7.  The value is first quantized to six
    decimals to remove floating-point noise, then rounded half-up.
    """
    d = Decimal(repr(x)).quantize(Decimal("1e-6"), rounding=ROUND_HALF_UP)
    return float(d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class Row:
    """One line of Table VII."""

    index: int
    target: str
    role: str
    observed: float
    predicted: float
    error: float
    tolerance: float
    passed: bool

    @property
    def is_game(self) -> bool:
        return self.role.startswith("Game")

    @property
    def is_lottery(self) -> bool:
        return self.role.startswith("Lottery")


@dataclass(frozen=True)
class TableVII:
    """Table VII of the paper: sixteen rows and the summary figures quoted in the text."""

    rows: List[Row]
    summary: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "citation": CITATION,
            "rows": [asdict(r) for r in self.rows],
            "summary": self.summary,
        }


def _mean_abs(rows: Sequence[Row]) -> float:
    return float(np.mean([abs(r.error) for r in rows]))


def table_vii(targets: Sequence[Target] | None = None) -> TableVII:
    """Recompute Table VII from the reference implementation.

    Args:
        targets: the benchmark targets; defaults to :func:`eris_econ.targets.build_targets`.

    Returns:
        A :class:`TableVII` whose ``summary`` holds, in percent unless noted:
        ``overall_mae`` (unweighted, sixteen targets), ``game_mae`` (nine),
        ``lottery_mae`` (six Ruggeri items, the "Ruggeri lottery subset" line),
        ``non_game_mae`` (seven lottery and historical targets),
        ``non_game_to_game_ratio``, ``ranking_only_mae`` (P7, P11, P16, P17),
        ``calibration_objective`` (the weighted MAE over the nine game targets
        that the variance search minimized), ``ranking_score`` (the weighted
        MAE over all sixteen used to rank candidate active sets), and the pass
        counts ``passed``, ``games_passed``, ``lotteries_passed``.
    """
    targets = list(build_targets() if targets is None else targets)
    sigma = selected_sigma()
    rows: List[Row] = []
    for i, t in enumerate(targets, start=1):
        predicted = float(t.predict_fn(sigma))
        error = predicted - t.observed
        rows.append(
            Row(
                index=i,
                target=t.name,
                role=ROLES.get(t.name, t.category),
                observed=t.observed,
                predicted=predicted,
                error=error,
                tolerance=t.tolerance,
                passed=abs(error) <= t.tolerance,
            )
        )

    games = [r for r in rows if r.is_game]
    lotteries = [r for r in rows if r.is_lottery]
    non_games = [r for r in rows if not r.is_game]
    ranking_only = [r for r in rows if r.target in RANKING_ONLY]

    def weighted(subset: Sequence[Row]) -> float:
        by_name = {t.name: t for t in targets}
        num = sum(by_name[r.target].weight * abs(r.error) for r in subset)
        den = sum(by_name[r.target].weight for r in subset)
        return float(num / den)

    summary = {
        "overall_mae": _mean_abs(rows),
        "game_mae": _mean_abs(games),
        "lottery_mae": _mean_abs(lotteries),
        "non_game_mae": _mean_abs(non_games),
        "non_game_to_game_ratio": _mean_abs(non_games) / _mean_abs(games),
        "ranking_only_mae": _mean_abs(ranking_only),
        "calibration_objective": weighted(games),
        "ranking_score": weighted(rows),
        "passed": sum(r.passed for r in rows),
        "games_passed": sum(r.passed for r in games),
        "lotteries_passed": sum(r.passed for r in lotteries),
        "n_targets": len(rows),
    }
    return TableVII(rows=rows, summary=summary)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_error(e: float) -> str:
    v = round_half_up(e, 1)
    return f"{v:+.1f}" if v != 0 else " 0.0"


def _fmt_observed(x: float) -> str:
    """One decimal, or two where the source reports two (the Engel dictator mean, 28.35)."""
    s = f"{x:.2f}"
    return (s[:-1] if s.endswith("0") else s) + "%"


def _cells(r: Row) -> List[str]:
    return [
        str(r.index),
        r.target,
        r.role,
        _fmt_observed(r.observed),
        f"{round_half_up(r.predicted, 1):.1f}%",
        _fmt_error(r.error),
        f"+/-{r.tolerance:.0f} pp",
        "pass" if r.passed else "FAIL",
    ]


_HEADER = ["#", "Target", "Role", "Observed", "Predicted", "Error", "Tol.", "Pass"]


def _summary_lines(t: TableVII) -> List[str]:
    s = t.summary
    return [
        f"Overall MAE = {s['overall_mae']:.2f}%   {s['passed']}/{s['n_targets']} pass",
        f"Ruggeri lottery subset MAE = {s['lottery_mae']:.2f}%   {s['lotteries_passed']}/6 pass",
        f"Game targets MAE = {s['game_mae']:.2f}%   non-game MAE = {s['non_game_mae']:.2f}%   "
        f"ratio = {s['non_game_to_game_ratio']:.2f}",
        f"Ranking-only lotteries (P7, P11, P16, P17) MAE = {s['ranking_only_mae']:.1f}%",
        f"Calibration objective (weighted, nine games) = {s['calibration_objective']:.3f}   "
        f"ranking score (weighted, sixteen) = {s['ranking_score']:.3f}",
    ]


def to_text(t: TableVII) -> str:
    table = [_HEADER] + [_cells(r) for r in t.rows]
    widths = [max(len(row[c]) for row in table) for c in range(len(_HEADER))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in table]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join([CITATION, ""] + lines + [""] + _summary_lines(t))


def to_markdown(t: TableVII) -> str:
    lines = ["| " + " | ".join(_HEADER) + " |", "|" + "---|" * len(_HEADER)]
    lines += ["| " + " | ".join(_cells(r)) + " |" for r in t.rows]
    return "\n".join(lines + [""] + [f"- {line}" for line in _summary_lines(t)])


def to_latex(t: TableVII) -> str:
    lines = [
        "\\begin{tabular}{@{}r l l r r r r c@{}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h.replace('#', chr(92) + '#')}}}" for h in _HEADER) + " \\\\",
        "\\midrule",
    ]
    for r in t.rows:
        c = _cells(r)
        c[5] = c[5].replace("+", "$+$").replace("-", "$-$")
        c[6] = f"$\\pm${r.tolerance:.0f}~pp"
        c[7] = "\\checkmark" if r.passed else "$\\times$"
        lines.append(" & ".join(c).replace("%", "\\%") + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def to_json(t: TableVII) -> str:
    return json.dumps(t.to_dict(), indent=2)


RENDERERS = {"text": to_text, "markdown": to_markdown, "latex": to_latex, "json": to_json}


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: print Table VII in the requested format."""
    parser = argparse.ArgumentParser(
        prog="eris-econ-table7",
        description="Recompute Table VII of Bond (2026, IEEE TCSS) from the reference implementation.",
    )
    parser.add_argument("--format", choices=sorted(RENDERERS), default="text")
    args = parser.parse_args(argv)
    sys.stdout.write(RENDERERS[args.format](table_vii()) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
