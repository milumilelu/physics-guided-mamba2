"""Robust reference-plane estimation for raw height maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .data_contracts import HeightMap

__all__ = ["PlaneFitResult", "fit_outer_reference_plane"]


@dataclass(frozen=True)
class PlaneFitResult:
    a: float
    b: float
    c: float
    rmse_um: float
    candidate_points: int
    retained_points: int
    reference_valid_fraction: float
    retained_fraction: float
    quadrant_count: int
    x_span_um: float
    y_span_um: float
    iterations: int
    status: str
    warning: str

    def to_dict(self) -> dict:
        return asdict(self)


def _fit(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    matrix = np.column_stack((x, y, np.ones_like(x)))
    coefficients, *_ = np.linalg.lstsq(matrix, z, rcond=None)
    return coefficients


def fit_outer_reference_plane(
        hm: HeightMap, *, frame_width_um: float = 50.0,
        sigma_low: float = 2.0, sigma_high: float = 3.0,
        max_iterations: int = 10, minimum_reference_valid_fraction: float = 0.20,
        max_fit_points: int = 200_000) -> PlaneFitResult:
    """Fit `z = a*x + b*y + c` on a robust outer reference frame."""
    # Centre from the coordinate endpoints rather than assuming coordinates
    # start at zero.  Raw KEYENCE maps do start near zero, while canonical
    # H_reg maps are already centred around zero.
    x = hm.x_um - (hm.x_um[0] + hm.x_um[-1]) / 2.0
    y = hm.y_um - (hm.y_um[0] + hm.y_um[-1]) / 2.0
    if not 0 < frame_width_um < min(hm.width_um, hm.height_um) / 2.0:
        raise ValueError(f"invalid outer frame width {frame_width_um}")
    frame = ((np.abs(x[None, :]) >= hm.width_um / 2.0 - frame_width_um)
             | (np.abs(y[:, None]) >= hm.height_um / 2.0 - frame_width_um))
    candidate = frame & hm.valid_mask
    candidate_points = int(frame.sum())
    valid_points = int(candidate.sum())
    valid_fraction = valid_points / candidate_points if candidate_points else 0.0
    if valid_fraction < minimum_reference_valid_fraction or valid_points < 100:
        return PlaneFitResult(
            *(float("nan"),) * 4, candidate_points, 0, valid_fraction, 0.0,
            0, 0.0, 0.0, 0, "STOP", "insufficient valid outer reference")

    rows, columns = np.nonzero(candidate)
    stride = max(1, int(np.ceil(valid_points / max_fit_points)))
    rows = rows[::stride]
    columns = columns[::stride]
    xf = x[columns]
    yf = y[rows]
    zf = hm.z[rows, columns]
    keep = np.ones(zf.size, dtype=bool)
    coefficients = _fit(xf, yf, zf)
    completed = 0
    for iteration in range(max_iterations):
        completed = iteration + 1
        residual = zf - (coefficients[0] * xf + coefficients[1] * yf
                         + coefficients[2])
        centre = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - centre)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            break
        new_keep = ((residual - centre >= -sigma_low * scale)
                    & (residual - centre <= sigma_high * scale))
        if new_keep.sum() < 100:
            return PlaneFitResult(
                *(float("nan"),) * 4, candidate_points, int(new_keep.sum()),
                valid_fraction, float(new_keep.mean()), 0, 0.0, 0.0,
                completed, "STOP", "robust clipping retained too few points")
        coefficients = _fit(xf[new_keep], yf[new_keep], zf[new_keep])
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    residual = zf[keep] - (coefficients[0] * xf[keep]
                           + coefficients[1] * yf[keep] + coefficients[2])
    rmse = float(np.sqrt(np.mean(residual * residual)))
    quadrants = {
        (int(xv >= 0), int(yv >= 0))
        for xv, yv in zip(xf[keep], yf[keep])
    }
    x_span = float(np.ptp(xf[keep]))
    y_span = float(np.ptp(yf[keep]))
    status = "PASS" if len(quadrants) == 4 else "STOP"
    warning = "" if status == "PASS" else "reference does not cover four quadrants"
    return PlaneFitResult(
        a=float(coefficients[0]), b=float(coefficients[1]),
        c=float(coefficients[2]), rmse_um=rmse,
        candidate_points=candidate_points, retained_points=int(keep.sum()),
        reference_valid_fraction=valid_fraction,
        retained_fraction=float(keep.mean()), quadrant_count=len(quadrants),
        x_span_um=x_span, y_span_um=y_span, iterations=completed,
        status=status, warning=warning)
