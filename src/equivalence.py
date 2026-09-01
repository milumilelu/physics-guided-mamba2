"""Independent comparison rules for CAG and KEYENCE height exports.

The comparison deliberately separates three questions:

* is the matrix geometry and calibration the same;
* are CAG heights identical to the official export on CAG-valid pixels;
* is there independent evidence for the validity mask.

An ImageDataCsv file normally has no mask.  Therefore a perfect height match
may pass the height sub-gate while the overall equivalence gate remains STOP.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .data_contracts import HeightMap

__all__ = ["compare_height_maps"]


def _transform_candidates(array: np.ndarray) -> dict[str, np.ndarray]:
    candidates = {
        "identity": array,
        "flip_x": array[:, ::-1],
        "flip_y": array[::-1, :],
        "flip_xy": array[::-1, ::-1],
    }
    if array.shape[0] == array.shape[1]:
        candidates.update({
            "transpose": array.T,
            "transpose_flip_x": array.T[:, ::-1],
            "transpose_flip_y": array.T[::-1, :],
            "transpose_flip_xy": array.T[::-1, ::-1],
        })
    return candidates


def _sampled_rmse(reference: HeightMap, candidate_z: np.ndarray,
                  stride: int = 16) -> float:
    if candidate_z.shape != reference.shape:
        return float("inf")
    ref = reference.z[::stride, ::stride]
    cand = candidate_z[::stride, ::stride]
    valid = reference.valid_mask[::stride, ::stride] & np.isfinite(cand)
    if not valid.any():
        return float("inf")
    delta = ref[valid] - cand[valid]
    return float(np.sqrt(np.mean(delta * delta)))


def _fixed_pixel_checks(cag: HeightMap, csv_map: HeightMap,
                        tolerance_um: float) -> list[dict]:
    height, width = cag.shape
    requested = [
        (0, 0), (0, width - 1), (height - 1, 0),
        (height - 1, width - 1), (height // 2, width // 2),
        (height // 4, width // 4), (3 * height // 4, 3 * width // 4),
        (height // 4, 3 * width // 4), (3 * height // 4, width // 4),
    ]
    checks: list[dict] = []
    for row, col in requested:
        if not cag.valid_mask[row, col] or not np.isfinite(csv_map.z[row, col]):
            continue
        difference = float(abs(cag.z[row, col] - csv_map.z[row, col]))
        checks.append({
            "row": row,
            "column": col,
            "cag_um": float(cag.z[row, col]),
            "csv_um": float(csv_map.z[row, col]),
            "abs_difference_um": difference,
            "passed": difference <= tolerance_um,
        })
    return checks


def compare_height_maps(cag: HeightMap, csv_map: HeightMap, *,
                        pitch_tolerance_um: float = 5e-7,
                        height_tolerance_um: float = 5e-12,
                        require_mask_evidence: bool = True,
                        allow_all_valid_mask_case: bool = False) -> dict:
    """Compare one decoded CAG measurement with an independent CSV export.

    Heights are compared only where the CAG says the measurement is valid.
    This is necessary because ImageDataCsv may contain software-filled values
    at positions that are sentinel-invalid in the CAG.
    """
    shape_match = cag.shape == csv_map.shape
    dx_difference = abs(cag.dx_um - csv_map.dx_um)
    dy_difference = abs(cag.dy_um - csv_map.dy_um)
    pitch_pass = (dx_difference <= pitch_tolerance_um
                  and dy_difference <= pitch_tolerance_um)

    result: dict = {
        "shape_cag": list(cag.shape),
        "shape_csv": list(csv_map.shape),
        "shape_match": shape_match,
        "dx_cag_um": cag.dx_um,
        "dx_csv_um": csv_map.dx_um,
        "dy_cag_um": cag.dy_um,
        "dy_csv_um": csv_map.dy_um,
        "dx_abs_difference_um": dx_difference,
        "dy_abs_difference_um": dy_difference,
        "pitch_tolerance_um": pitch_tolerance_um,
        "pitch_pass": pitch_pass,
        "height_tolerance_um": height_tolerance_um,
        "cag_valid_pixels": cag.n_valid,
        "cag_invalid_pixels": cag.n_invalid,
    }

    if not shape_match:
        result.update({
            "compared_pixels": 0,
            "height_mismatch_pixels": None,
            "max_abs_difference_um": None,
            "median_abs_difference_um": None,
            "rmse_um": None,
            "height_pass": False,
            "orientation_best_transform": None,
            "orientation_pass": False,
            "fixed_pixel_checks": [],
        })
    else:
        comparable = cag.valid_mask & np.isfinite(csv_map.z)
        delta = np.abs(cag.z[comparable] - csv_map.z[comparable])
        mismatch = delta > height_tolerance_um
        result.update({
            "compared_pixels": int(comparable.sum()),
            "height_mismatch_pixels": int(mismatch.sum()),
            "max_abs_difference_um": (float(delta.max())
                                      if delta.size else None),
            "median_abs_difference_um": (float(np.median(delta))
                                         if delta.size else None),
            "rmse_um": (float(np.sqrt(np.mean(delta * delta)))
                        if delta.size else None),
            "height_pass": bool(delta.size and not mismatch.any()
                                and comparable.sum() == cag.n_valid),
        })

        orientation_scores = {
            name: _sampled_rmse(cag, transformed)
            for name, transformed in _transform_candidates(csv_map.z).items()
        }
        best_transform = min(orientation_scores, key=orientation_scores.get)
        fixed_checks = _fixed_pixel_checks(cag, csv_map,
                                           height_tolerance_um)
        result.update({
            "orientation_rmse_um": orientation_scores,
            "orientation_best_transform": best_transform,
            "orientation_pass": best_transform == "identity",
            "fixed_pixel_checks": fixed_checks,
            "fixed_pixel_checks_pass": bool(
                len(fixed_checks) >= 5
                and all(item["passed"] for item in fixed_checks)),
        })

    mask_comparable = not csv_map.mask_is_fabricated
    all_valid_case = bool(
        shape_match and cag.n_invalid == 0
        and np.isfinite(csv_map.z).all()
    )
    if shape_match and mask_comparable:
        mask_mismatch = int(np.count_nonzero(
            cag.valid_mask != csv_map.valid_mask))
        mask_pass: bool | None = mask_mismatch == 0
        mask_decision = "PASS" if mask_pass else "FAIL"
        mask_scope = "explicit_independent_mask"
    elif allow_all_valid_mask_case and all_valid_case:
        # There is no missing-data behaviour to test in this measurement:
        # CAG has no sentinel and the official export supplies a numeric value
        # at every pixel.  This closes the gate only for the observed all-valid
        # case; it is not evidence for how the software represents a sentinel.
        mask_mismatch = 0
        mask_pass = True
        mask_decision = "PASS_ALL_VALID_CASE"
        mask_scope = "all_pixels_valid_no_sentinel_behaviour_observed"
    else:
        mask_mismatch = None
        mask_pass = None
        mask_decision = "UNAVAILABLE"
        mask_scope = "unavailable"

    result.update({
        "csv_mask_source": csv_map.metadata.get("mask_source", "unknown"),
        "mask_comparable": mask_comparable,
        "mask_mismatch_pixels": mask_mismatch,
        "mask_pass": mask_pass,
        "mask_decision": mask_decision,
        "mask_evidence_scope": mask_scope,
        "all_valid_case": all_valid_case,
        "height_decision": "PASS" if (
            result["shape_match"] and result["pitch_pass"]
            and result["height_pass"] and result["orientation_pass"]
            and result.get("fixed_pixel_checks_pass", False)
        ) else "FAIL",
    })
    mask_gate_pass = bool(mask_pass) if require_mask_evidence else True
    result["require_mask_evidence"] = require_mask_evidence
    result["overall_decision"] = (
        "PASS" if result["height_decision"] == "PASS" and mask_gate_pass
        else "STOP"
    )
    return result
