"""Dependency-light, regime-aware analysis for PINN evaluation rows."""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable, Optional, Sequence

import numpy as np


def _nested(row: dict[str, Any], key: str) -> Any:
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(values: Sequence[float], targets: Sequence[float]) -> Optional[float]:
    """Return pooled Spearman correlation, or ``None`` when undefined."""

    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("correlation inputs must be same-length vectors")
    if len(x) < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    result = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
    return result if math.isfinite(result) else None


def kendall_tau_b(
    values: Sequence[float],
    targets: Sequence[float],
) -> Optional[float]:
    """Return Kendall's tau-b with explicit tie handling."""

    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("correlation inputs must be same-length vectors")
    if len(x) < 2:
        return None
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for left in range(len(x) - 1):
        for right in range(left + 1, len(x)):
            dx = np.sign(x[left] - x[right])
            dy = np.sign(y[left] - y[right])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )
    if denominator == 0.0:
        return None
    return (concordant - discordant) / denominator


def _pairs(
    rows: Iterable[dict[str, Any]],
    *,
    score: str,
    target: str,
) -> tuple[list[float], list[float]]:
    values: list[float] = []
    targets: list[float] = []
    for row in rows:
        value = _finite(_nested(row, score))
        target_value = _finite(_nested(row, target))
        if value is None or target_value is None:
            continue
        values.append(value)
        targets.append(target_value)
    return values, targets


def _correlation(
    rows: Sequence[dict[str, Any]],
    *,
    score: str,
    target: str,
) -> dict[str, Any]:
    values, targets = _pairs(rows, score=score, target=target)
    return {
        "count": len(values),
        "spearman": spearman(values, targets),
        "kendall_tau_b": kendall_tau_b(values, targets),
    }


def _between_cells(
    rows: Sequence[dict[str, Any]],
    *,
    score: str,
    target: str,
) -> dict[str, Any]:
    cells: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        nu = _finite(row.get("nu"))
        rho = _finite(row.get("rho"))
        if nu is None or rho is None:
            continue
        cells.setdefault((nu, rho), []).append(row)
    values: list[float] = []
    targets: list[float] = []
    for cell_rows in cells.values():
        cell_values, cell_targets = _pairs(
            cell_rows,
            score=score,
            target=target,
        )
        if not cell_values:
            continue
        values.append(float(median(cell_values)))
        targets.append(float(median(cell_targets)))
    return {
        "cell_count": len(values),
        "spearman_cell_medians": spearman(values, targets),
    }


def _within_cells(
    rows: Sequence[dict[str, Any]],
    *,
    score: str,
    target: str,
) -> dict[str, Any]:
    cells: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        nu = _finite(row.get("nu"))
        rho = _finite(row.get("rho"))
        if nu is None or rho is None:
            continue
        cells.setdefault((nu, rho), []).append(row)
    correlations: list[float] = []
    for cell_rows in cells.values():
        values, targets = _pairs(cell_rows, score=score, target=target)
        value = kendall_tau_b(values, targets)
        if value is not None:
            correlations.append(value)
    return {
        "informative_cell_count": len(correlations),
        "median_within_cell_kendall_tau_b": (
            float(median(correlations)) if correlations else None
        ),
    }


def analyze_rows(
    rows: Sequence[dict[str, Any]],
    *,
    scores: Sequence[str],
    target: str = "relative_error",
) -> dict[str, Any]:
    """Analyze all, interpolating, and noninterpolating checkpoint regimes."""

    successful = [row for row in rows if row.get("success") is True]

    def estimand(row: dict[str, Any]) -> str:
        group = row.get("estimand_group")
        if isinstance(group, str):
            return group
        declared = row.get("constraint_estimand")
        if declared in {"nu_zero", "nu_positive"}:
            return declared
        nu = _finite(row.get("nu"))
        return "nu_zero" if nu == 0.0 else "nu_positive"

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in successful:
        grouped.setdefault(estimand(row), []).append(row)

    group_reports: dict[str, Any] = {}
    for name, group_rows in sorted(grouped.items()):
        interpolating = [
            row for row in group_rows if row.get("interpolation_valid") is True
        ]
        noninterpolating = [
            row for row in group_rows if row.get("interpolation_valid") is False
        ]
        unknown = [
            row
            for row in group_rows
            if row.get("interpolation_valid") not in {True, False}
        ]
        score_reports: dict[str, Any] = {}
        for score in scores:
            score_reports[score] = {
                "all_evaluated": _correlation(
                    group_rows,
                    score=score,
                    target=target,
                ),
                "interpolating": _correlation(
                    interpolating,
                    score=score,
                    target=target,
                ),
                "noninterpolating": _correlation(
                    noninterpolating,
                    score=score,
                    target=target,
                ),
                "between_problem": _between_cells(
                    group_rows,
                    score=score,
                    target=target,
                ),
                "within_problem": _within_cells(
                    group_rows,
                    score=score,
                    target=target,
                ),
            }
        group_reports[name] = {
            "counts": {
                "successful_evaluations": len(group_rows),
                "interpolating": len(interpolating),
                "noninterpolating": len(noninterpolating),
                "interpolation_status_unknown": len(unknown),
                "hard_theory_valid": sum(
                    row.get("hard_score_theory_valid") is True
                    for row in group_rows
                ),
            },
            "scores": score_reports,
        }
    return {
        "schema_version": 1,
        "target": target,
        "estimand_policy": (
            "nu=0 and nu>0 are analyzed separately because periodic derivative "
            "matching is included only for nu>0, either in the constraint map "
            "or in the boundary regularizer"
        ),
        "interpretation": {
            "all_evaluated": (
                "Descriptive candidate-score analysis; interpolation assumptions "
                "are not imposed."
            ),
            "interpolating": "Rows satisfying the recorded interpolation gate.",
            "noninterpolating": (
                "Failure-regime rows; hard-score values are numerical candidates, "
                "not theory-valid hard IIC unless separately flagged."
            ),
        },
        "counts": {
            "input_rows": len(rows),
            "successful_evaluations": len(successful),
            "estimand_group_count": len(group_reports),
        },
        "by_estimand": group_reports,
    }
