"""Free square fits for calibration samples and session-level angle pooling.

The routines in this module estimate only the continuous angle modulo 90
degrees.  They intentionally do not choose a D4 quadrant or mirror state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .data_contracts import HeightMap

__all__ = [
    "FreeSquareFit", "fit_free_square", "pool_session_angle",
]


@dataclass(frozen=True)
class FreeSquareFit:
    theta_deg: float
    theta_threshold_mad_deg: float
    center_x_um: float
    center_y_um: float
    width_um: float
    height_um: float
    area_um2: float
    edge_completeness: float
    fit_residual_um: float
    area_consistency: float
    contrast_um: float
    quality_score: float
    successful_thresholds: int
    status: str
    warning: str

    def to_dict(self) -> dict:
        return asdict(self)


def _fold_angle(angle_deg: float) -> float:
    return float((angle_deg + 45.0) % 90.0 - 45.0)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2.0)])


def _minimum_box(points: np.ndarray, *, quantile: float,
                 step_deg: float) -> dict[str, float]:
    angles = np.arange(-45.0, 45.0, step_deg, dtype=float)
    radians = np.deg2rad(angles)
    x = points[:, 0]
    y = points[:, 1]
    best: dict[str, float] | None = None
    for angle, rad in zip(angles, radians):
        cosine, sine = np.cos(rad), np.sin(rad)
        u = x * cosine + y * sine
        v = -x * sine + y * cosine
        u0, u1 = np.quantile(u, [quantile, 1.0 - quantile])
        v0, v1 = np.quantile(v, [quantile, 1.0 - quantile])
        width, height = float(u1 - u0), float(v1 - v0)
        score = width * height
        if best is None or score < best["score"]:
            best = {
                "theta_deg": float(angle), "score": score,
                "u0": float(u0), "u1": float(u1),
                "v0": float(v0), "v1": float(v1),
                "width_um": width, "height_um": height,
            }
    assert best is not None
    rad = np.deg2rad(best["theta_deg"])
    uc = (best["u0"] + best["u1"]) / 2.0
    vc = (best["v0"] + best["v1"]) / 2.0
    best["center_x_um"] = float(uc * np.cos(rad) - vc * np.sin(rad))
    best["center_y_um"] = float(uc * np.sin(rad) + vc * np.cos(rad))

    u = x * np.cos(rad) + y * np.sin(rad)
    v = -x * np.sin(rad) + y * np.cos(rad)
    distances = np.minimum.reduce((
        np.abs(u - best["u0"]), np.abs(u - best["u1"]),
        np.abs(v - best["v0"]), np.abs(v - best["v1"])))
    best["fit_residual_um"] = float(np.median(distances))

    occupied = 0
    for coordinate, low, high, side_distance in (
        (v, best["v0"], best["v1"], np.minimum(np.abs(u-best["u0"]), np.abs(u-best["u1"]))),
        (u, best["u0"], best["u1"], np.minimum(np.abs(v-best["v0"]), np.abs(v-best["v1"]))),
    ):
        close = side_distance <= max(2.0, 3.0 * best["fit_residual_um"])
        bins = np.floor(20 * (coordinate[close] - low) / max(high - low, 1e-9)).astype(int)
        occupied += len(set(np.clip(bins, 0, 19).tolist())) * 2
    best["edge_completeness"] = float(min(1.0, occupied / 80.0))
    return best


def fit_free_square(
        hm: HeightMap, *, plane: tuple[float, float, float],
        center_search: tuple[float, float, float, float],
        nominal_size_um: float = 200.0, crop_margin_um: float = 35.0,
        threshold_fractions: tuple[float, ...] = (0.35, 0.50, 0.65),
        component_area_ratio: tuple[float, float] = (0.35, 1.75),
        boundary_quantile: float = 0.005, angle_grid_step_deg: float = 0.05,
        minimum_successful_thresholds: int = 2,
        quality_weights: dict[str, float] | None = None) -> FreeSquareFit:
    """Fit a depressed nominal square inside a pre-declared centre domain."""
    weights = quality_weights or {
        "edge_completeness": 0.35, "fit_residual": 0.25,
        "area_consistency": 0.20, "contrast": 0.20,
    }
    x = hm.x_um - hm.width_um / 2.0
    y = hm.y_um - hm.height_um / 2.0
    a, b, c = plane
    z = hm.z - (a * x[None, :] + b * y[:, None] + c)
    xmin, xmax, ymin, ymax = center_search
    half = nominal_size_um / 2.0 + crop_margin_um
    columns = (x >= xmin - half) & (x <= xmax + half)
    rows = (y >= ymin - half) & (y <= ymax + half)
    zz = z[np.ix_(rows, columns)]
    valid = hm.valid_mask[np.ix_(rows, columns)]
    xx, yy = x[columns], y[rows]
    if valid.sum() < 1000:
        return _failed("insufficient valid data in calibration crop")
    smooth = ndimage.gaussian_filter(np.where(valid, zz, np.nanmedian(zz[valid])), 1.25)
    values = smooth[valid]
    low, high = np.quantile(values, [0.10, 0.80])
    contrast = float(high - low)
    if not np.isfinite(contrast) or contrast <= 0:
        return _failed("non-positive calibration contrast")

    fits: list[dict[str, float]] = []
    masks: list[np.ndarray] = []
    pixel_area = hm.dx_um * hm.dy_um
    expected_area = nominal_size_um ** 2
    for fraction in threshold_fractions:
        mask = valid & (smooth < low + float(fraction) * contrast)
        mask = ndimage.binary_closing(mask, iterations=2)
        mask = ndimage.binary_opening(mask, iterations=1)
        labels, count = ndimage.label(mask)
        best_label = None
        best_cost = float("inf")
        for label in range(1, count + 1):
            rr, cc = np.nonzero(labels == label)
            if rr.size == 0:
                continue
            area = rr.size * pixel_area
            ratio = area / expected_area
            cx, cy = float(np.mean(xx[cc])), float(np.mean(yy[rr]))
            if not (component_area_ratio[0] <= ratio <= component_area_ratio[1]
                    and xmin <= cx <= xmax and ymin <= cy <= ymax):
                continue
            cost = abs(np.log(max(ratio, 1e-12)))
            if cost < best_cost:
                best_cost, best_label = cost, label
        if best_label is None:
            continue
        component = ndimage.binary_fill_holes(labels == best_label)
        boundary = component & ~ndimage.binary_erosion(component)
        rr, cc = np.nonzero(boundary)
        if rr.size < 100:
            continue
        points = np.column_stack((xx[cc], yy[rr]))
        if points.shape[0] > 12000:
            points = points[::int(np.ceil(points.shape[0] / 12000))]
        fit = _minimum_box(points, quantile=boundary_quantile,
                           step_deg=angle_grid_step_deg)
        fit["area_um2"] = float(component.sum() * pixel_area)
        fits.append(fit)
        masks.append(component)

    if len(fits) < minimum_successful_thresholds:
        return _failed(f"only {len(fits)} successful segmentation thresholds")
    angles = np.array([fit["theta_deg"] for fit in fits])
    theta = float(np.median(angles))
    deviations = np.abs([_fold_angle(value - theta) for value in angles])
    angle_mad = float(np.median(deviations))
    representative = fits[int(np.argmin(deviations))]
    area_consistency = float(np.exp(-abs(representative["area_um2"] /
                                         expected_area - 1.0) / 0.35))
    residual_quality = float(np.exp(-representative["fit_residual_um"] / 3.0))
    contrast_quality = float(contrast / (contrast + max(np.median(np.abs(
        values - np.median(values))), 1e-9)))
    quality = (
        weights["edge_completeness"] * representative["edge_completeness"]
        + weights["fit_residual"] * residual_quality
        + weights["area_consistency"] * area_consistency
        + weights["contrast"] * contrast_quality)
    return FreeSquareFit(
        theta_deg=_fold_angle(theta), theta_threshold_mad_deg=angle_mad,
        center_x_um=representative["center_x_um"],
        center_y_um=representative["center_y_um"],
        width_um=representative["width_um"],
        height_um=representative["height_um"],
        area_um2=representative["area_um2"],
        edge_completeness=representative["edge_completeness"],
        fit_residual_um=representative["fit_residual_um"],
        area_consistency=area_consistency, contrast_um=contrast,
        quality_score=float(quality), successful_thresholds=len(fits),
        status="PASS", warning="")


def _failed(message: str) -> FreeSquareFit:
    return FreeSquareFit(
        *(float("nan"),) * 12, 0, "STOP", message)


def pool_session_angle(rows: list[dict], *, warning_mad_deg: float = 0.3,
                       review_mad_deg: float = 0.8,
                       histogram_bin_deg: float = 0.5) -> dict:
    """Pool passed modulo-90 calibration angles without selecting D4."""
    passed = [row for row in rows if row.get("status") == "PASS"]
    if not passed:
        return {"status": "STOP", "warning": "no successful angle fits"}
    angles = np.asarray([float(row["theta_deg"]) for row in passed])
    weights = np.asarray([max(float(row["quality_score"]), 1e-9)
                          for row in passed])
    theta = _weighted_median(angles, weights)
    deviations = np.abs([_fold_angle(value - theta) for value in angles])
    mad = _weighted_median(np.asarray(deviations), weights)

    edges = np.arange(-45.0, 45.0 + histogram_bin_deg, histogram_bin_deg)
    counts, _ = np.histogram(angles, bins=edges)
    peaks = np.argsort(counts)[::-1]
    multimodal = False
    if len(peaks) >= 2 and counts[peaks[1]] >= max(2, int(np.ceil(0.2 * len(angles)))):
        centres = (edges[:-1] + edges[1:]) / 2.0
        separation = abs(_fold_angle(centres[peaks[0]] - centres[peaks[1]]))
        multimodal = separation >= max(1.0, 2.0 * histogram_bin_deg)
    status = "STOP" if multimodal or mad > review_mad_deg else "PASS"
    warning = ""
    if multimodal:
        warning = "angle distribution is multimodal"
    elif mad > review_mad_deg:
        warning = "session angle MAD exceeds hard-review threshold"
    elif mad > warning_mad_deg:
        warning = "session angle MAD exceeds warning threshold"
    return {
        "theta_session_deg": theta,
        "theta_session_mad_deg": mad,
        "successful_samples": len(passed),
        "failed_samples": len(rows) - len(passed),
        "angle_multimodality_warning": multimodal,
        "status": status,
        "warning": warning,
    }
