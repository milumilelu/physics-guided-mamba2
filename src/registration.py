"""Constrained translation-only registration with frozen session geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .data_contracts import HeightMap

__all__ = ["TranslationFit", "initial_center_from_component",
           "register_fixed_square"]


@dataclass(frozen=True)
class TranslationFit:
    center_x_um: float
    center_y_um: float
    region_score: float
    edge_score: float
    objective_score: float
    center_search_boundary_hit: bool
    registration_unstable: bool
    sensitivity_span_um: float
    sensitivity_centers: str
    initialization: str
    status: str
    warning: str

    def to_dict(self) -> dict:
        return asdict(self)


def _robust_scale(values: np.ndarray) -> float:
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    return max(float(scale), 1e-12)


def initial_center_from_component(
        z: np.ndarray, valid: np.ndarray, x: np.ndarray,
        y: np.ndarray, domain: tuple[float, float, float, float],
        nominal_size_um: float) -> tuple[float, float, str]:
    xmin, xmax, ymin, ymax = domain
    half = nominal_size_um / 2.0 + 35.0
    columns = (x >= xmin-half) & (x <= xmax+half)
    rows = (y >= ymin-half) & (y <= ymax+half)
    zz = z[np.ix_(rows, columns)]
    vv = valid[np.ix_(rows, columns)]
    xx, yy = x[columns], y[rows]
    values = zz[vv]
    low, high = np.quantile(values, [0.10, 0.80])
    mask = vv & (ndimage.gaussian_filter(
        np.where(vv, zz, np.nanmedian(values)), 1.25) < low + 0.5*(high-low))
    mask = ndimage.binary_opening(ndimage.binary_closing(mask, iterations=2))
    labels, count = ndimage.label(mask)
    expected = nominal_size_um ** 2
    pixel_area = ((x[1]-x[0]) * (y[1]-y[0]))
    candidates: list[tuple[float, float, float]] = []
    for label in range(1, count+1):
        rr, cc = np.nonzero(labels == label)
        if not len(rr):
            continue
        cx, cy = float(np.mean(xx[cc])), float(np.mean(yy[rr]))
        ratio = len(rr) * pixel_area / expected
        if xmin <= cx <= xmax and ymin <= cy <= ymax and 0.30 <= ratio <= 1.80:
            candidates.append((abs(np.log(max(ratio, 1e-12))), cx, cy))
    if candidates:
        _, cx, cy = min(candidates)
        return cx, cy, "fixed-domain_component"
    return (xmin+xmax)/2.0, (ymin+ymax)/2.0, "domain_midpoint_fallback"


def _candidate_scores(z: np.ndarray, valid: np.ndarray,
                      x: np.ndarray, y: np.ndarray,
                      gx: np.ndarray, gy: np.ndarray,
                      candidates: list[tuple[float, float]], theta_deg: float,
                      nominal_size_um: float, edge_band_um: float,
                      sample_step_um: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    stride_y = max(1, int(round(sample_step_um / (y[1]-y[0]))))
    stride_x = max(1, int(round(sample_step_um / (x[1]-x[0]))))
    xs = x[::stride_x]
    ys = y[::stride_y]
    zz = z[::stride_y, ::stride_x]
    vv = valid[::stride_y, ::stride_x]
    gxx = gx[::stride_y, ::stride_x]
    gyy = gy[::stride_y, ::stride_x]
    xx, yy = np.meshgrid(xs, ys)
    rad = np.deg2rad(theta_deg)
    cosine, sine = np.cos(rad), np.sin(rad)
    gu = np.abs(gxx*cosine + gyy*sine)
    gv = np.abs(-gxx*sine + gyy*cosine)
    outside_gradient_scale = _robust_scale(np.concatenate((gxx[vv], gyy[vv])))
    half = nominal_size_um/2.0
    region_scores = np.full(len(candidates), np.nan)
    edge_scores = np.full(len(candidates), np.nan)
    for index, (cx, cy) in enumerate(candidates):
        u = (xx-cx)*cosine + (yy-cy)*sine
        v = -(xx-cx)*sine + (yy-cy)*cosine
        inside = vv & (np.abs(u) <= half-edge_band_um) & (np.abs(v) <= half-edge_band_um)
        radius = np.maximum(np.abs(u), np.abs(v))
        outside = vv & (radius >= half+edge_band_um) & (radius <= half+2*edge_band_um)
        vertical = vv & (np.abs(np.abs(u)-half) <= edge_band_um) & (np.abs(v) <= half+edge_band_um)
        horizontal = vv & (np.abs(np.abs(v)-half) <= edge_band_um) & (np.abs(u) <= half+edge_band_um)
        if inside.sum() < 100 or outside.sum() < 100:
            continue
        outside_values = zz[outside]
        denom = _robust_scale(outside_values)
        region_scores[index] = ((np.median(outside_values)-np.median(zz[inside])) / denom)
        sigma = edge_band_um / 2.0
        vertical_weights = np.exp(
            -0.5*(np.abs(np.abs(u[vertical])-half)/sigma)**2)
        horizontal_weights = np.exp(
            -0.5*(np.abs(np.abs(v[horizontal])-half)/sigma)**2)
        numerator = (np.sum(gu[vertical]*vertical_weights)
                     + np.sum(gv[horizontal]*horizontal_weights))
        denominator = vertical_weights.sum() + horizontal_weights.sum()
        # The Gaussian tolerance band rewards a nearby edge continuously; it
        # does not impose a brittle exact-pixel boundary at +/-100 um.
        edge_scores[index] = (float(numerator/denominator)/outside_gradient_scale
                              if denominator > 0 else np.nan)
    return region_scores, edge_scores


def _normalize(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.full_like(values, -np.inf)
    if finite.any():
        result[finite] = (values[finite]-np.median(values[finite])) / _robust_scale(values[finite])
    return result


def _grid(center: tuple[float, float], domain: tuple[float, float, float, float],
          radius: float, step: float) -> list[tuple[float, float]]:
    xmin, xmax, ymin, ymax = domain
    xs = np.arange(max(xmin, center[0]-radius), min(xmax, center[0]+radius)+step/2, step)
    ys = np.arange(max(ymin, center[1]-radius), min(ymax, center[1]+radius)+step/2, step)
    return [(float(x), float(y)) for y in ys for x in xs]


def register_fixed_square(
        hm: HeightMap, *, plane: tuple[float, float, float], theta_deg: float,
        center_search: tuple[float, float, float, float],
        nominal_size_um: float = 200.0, edge_band_um: float = 15.0,
        primary_weights: tuple[float, float] = (0.5, 0.5),
        sensitivity_weights: tuple[tuple[float, float], ...] =
        ((0.25, 0.75), (0.5, 0.5), (0.75, 0.25)),
        unstable_shift_um: float = 3.0,
        coarse_radius_um: float = 12.0, coarse_grid_step_um: float = 2.0,
        coarse_score_sampling_um: float = 2.0,
        fine_radius_um: float = 2.0, fine_grid_step_um: float = 0.5,
        fine_score_sampling_um: float = 1.0) -> TranslationFit:
    """Register one square, optimizing only its centre in a frozen domain."""
    x = hm.x_um - hm.width_um/2.0
    y = hm.y_um - hm.height_um/2.0
    a, b, c = plane
    z = hm.z - (a*x[None, :] + b*y[:, None] + c)
    filled = np.where(hm.valid_mask, z, np.nanmedian(z[hm.valid_mask]))
    smooth = ndimage.gaussian_filter(filled, 1.0)
    gy, gx = np.gradient(smooth, hm.dy_um, hm.dx_um)
    initial_x, initial_y, initialization = initial_center_from_component(
        z, hm.valid_mask, x, y, center_search, nominal_size_um)

    coarse = _grid((initial_x, initial_y), center_search,
                   coarse_radius_um, coarse_grid_step_um)
    rs, es = _candidate_scores(z, hm.valid_mask, x, y, gx, gy, coarse,
                               theta_deg, nominal_size_um, edge_band_um,
                               sample_step_um=coarse_score_sampling_um)
    objective = primary_weights[0]*_normalize(rs) + primary_weights[1]*_normalize(es)
    if not np.isfinite(objective).any():
        return TranslationFit(
            *(float("nan"),)*5, False, True, float("nan"), "",
            initialization, "STOP", "no finite coarse registration score")
    coarse_best = coarse[int(np.nanargmax(objective))]

    fine = _grid(coarse_best, center_search, fine_radius_um, fine_grid_step_um)
    rs, es = _candidate_scores(z, hm.valid_mask, x, y, gx, gy, fine,
                               theta_deg, nominal_size_um, edge_band_um,
                               sample_step_um=fine_score_sampling_um)
    rn, en = _normalize(rs), _normalize(es)
    chosen: list[tuple[float, float]] = []
    primary_index = 0
    for weight_index, weights in enumerate(sensitivity_weights):
        scores = weights[0]*rn + weights[1]*en
        best = int(np.nanargmax(scores))
        chosen.append(fine[best])
        if tuple(map(float, weights)) == tuple(map(float, primary_weights)):
            primary_index = weight_index
    primary_center = chosen[primary_index]
    final_scores = primary_weights[0]*rn + primary_weights[1]*en
    best_index = fine.index(primary_center)
    span = max(np.hypot(x1-x2, y1-y2) for x1, y1 in chosen for x2, y2 in chosen)
    xmin, xmax, ymin, ymax = center_search
    boundary = (abs(primary_center[0]-xmin) < 0.51 or abs(primary_center[0]-xmax) < 0.51
                or abs(primary_center[1]-ymin) < 0.51 or abs(primary_center[1]-ymax) < 0.51)
    unstable = bool(span > unstable_shift_um)
    status = "REVIEW" if boundary or unstable or initialization.endswith("fallback") else "PASS"
    warnings = []
    if boundary:
        warnings.append("center search boundary hit")
    if unstable:
        warnings.append("weight sensitivity exceeds threshold")
    if initialization.endswith("fallback"):
        warnings.append("segmentation initializer unavailable")
    centres_text = ";".join(
        f"{wr:.2f}/{we:.2f}:{cx:.3f},{cy:.3f}"
        for (wr, we), (cx, cy) in zip(sensitivity_weights, chosen))
    return TranslationFit(
        center_x_um=primary_center[0], center_y_um=primary_center[1],
        region_score=float(rs[best_index]), edge_score=float(es[best_index]),
        objective_score=float(final_scores[best_index]),
        center_search_boundary_hit=boundary, registration_unstable=unstable,
        sensitivity_span_um=float(span), sensitivity_centers=centres_text,
        initialization=initialization, status=status,
        warning="; ".join(warnings))
