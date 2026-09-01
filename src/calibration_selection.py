"""Deterministic, process-aware calibration sample selection."""

from __future__ import annotations

import math

import numpy as np

__all__ = ["select_calibration_samples", "coverage_rows"]


def _percentile_ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=float)
    ranks[order] = np.arange(array.size, dtype=float)
    return ranks / max(1, array.size - 1)


def select_calibration_samples(rows: list[dict], *, fraction: float,
                               minimum: int, factors: list[str],
                               weights: dict[str, float]) -> list[dict]:
    if not rows:
        return []
    target = min(len(rows), max(minimum, int(math.ceil(len(rows) * fraction))))
    contrast = _percentile_ranks([float(r["q50_minus_q05"]) for r in rows])
    edge = _percentile_ranks([float(r["edge_energy"]) for r in rows])
    valid = _percentile_ranks([float(r["selected_valid_fraction"]) for r in rows])
    candidates = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["calibration_rank_score"] = float(
            weights["contrast_pctile"] * contrast[index]
            + weights["edge_energy_pctile"] * edge[index]
            + weights["valid_fraction_pctile"] * valid[index])
        item["_levels"] = {(factor, str(row[factor])) for factor in factors}
        candidates.append(item)

    chosen: list[dict] = []
    covered: set[tuple[str, str]] = set()
    while len(chosen) < target:
        remaining = [row for row in candidates if row not in chosen]
        best = max(
            remaining,
            key=lambda row: (
                len(row["_levels"] - covered),
                row["calibration_rank_score"],
                -int(row["sample_id"]),
            ),
        )
        gain = len(best["_levels"] - covered)
        best["selection_reason"] = (
            "new_process_level_coverage" if gain else "within_session_high_snr")
        covered |= best["_levels"]
        chosen.append(best)

    for row in chosen:
        row.pop("_levels", None)
    return sorted(chosen, key=lambda row: int(row["sample_id"]))


def coverage_rows(all_rows: list[dict], selected: list[dict],
                  factors: list[str], session_id: str) -> list[dict]:
    selected_ids = {int(row["sample_id"]) for row in selected}
    output = []
    for factor in factors:
        levels = sorted({str(row[factor]) for row in all_rows})
        for level in levels:
            total = sum(str(row[factor]) == level for row in all_rows)
            count = sum(str(row[factor]) == level
                        and int(row["sample_id"]) in selected_ids
                        for row in all_rows)
            output.append({
                "session_id": session_id, "factor": factor, "level": level,
                "samples_total": total, "samples_selected": count,
                "coverage_fraction": count / total if total else 0.0,
                "level_represented": int(count > 0),
            })
    return output
