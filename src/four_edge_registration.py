"""v3 constrained centre estimation from four signed robust edge profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .data_contracts import HeightMap
from .registration import initial_center_from_component
from .resampling import resample_to_canonical

__all__ = ["FourEdgeFit", "fit_constrained_four_edges"]


@dataclass(frozen=True)
class FourEdgeFit:
    center_x_um: float
    center_y_um: float
    left_edge_um: float
    right_edge_um: float
    top_edge_um: float
    bottom_edge_um: float
    observed_width_x_um: float
    observed_width_y_um: float
    left_snr: float
    right_snr: float
    top_snr: float
    bottom_snr: float
    minimum_edge_snr: float
    constrained_edge_rmse_um: float
    constrained_edge_max_abs_um: float
    center_search_boundary_hit: bool
    initialization: str
    status: str
    warning: str

    def to_dict(self) -> dict:
        return asdict(self)


def _robust_scale(values: np.ndarray) -> float:
    centre = float(np.median(values))
    scale = 1.4826*float(np.median(np.abs(values-centre)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    return max(scale, 1e-12)


def _locate_signed_edge(axis: np.ndarray, profile: np.ndarray,
                        expected_um: float, search_halfwidth_um: float,
                        centroid_halfwidth_um: float) -> tuple[float, float]:
    search = np.abs(axis-expected_um) <= search_halfwidth_um
    background = ~search
    baseline = float(np.median(profile[background])) if background.any() else 0.0
    scale = _robust_scale(profile[background] if background.any() else profile)
    indices = np.flatnonzero(search)
    if not indices.size:
        return float("nan"), 0.0
    peak_index = indices[int(np.argmax(profile[search]))]
    peak = float(profile[peak_index])
    snr = (peak-baseline)/scale
    near = np.abs(axis-axis[peak_index]) <= centroid_halfwidth_um
    weights = np.maximum(profile[near]-baseline, 0.0)
    position = (float(np.sum(axis[near]*weights)/weights.sum())
                if weights.sum() > 0 else float(axis[peak_index]))
    return position, float(snr)


def fit_constrained_four_edges(
        hm: HeightMap, *, plane: tuple[float, float, float], theta_deg: float,
        center_search: tuple[float, float, float, float],
        nominal_size_um: float = 200.0, local_canvas_um: float = 270.0,
        profile_strip_halfwidth_um: float = 70.0,
        smoothing_sigma_um: float = 1.0,
        edge_search_halfwidth_um: float = 20.0,
        peak_centroid_halfwidth_um: float = 3.0,
        minimum_edge_snr: float = 3.0, review_edge_snr: float = 5.0,
        width_range_um: tuple[float, float] = (180.0, 220.0),
        review_residual_um: float = 4.0, hard_residual_um: float = 10.0,
        boundary_tolerance_um: float = 0.5) -> FourEdgeFit:
    """Estimate only centre; theta, D4 and nominal size remain frozen."""
    x = hm.x_um-hm.width_um/2.0
    y = hm.y_um-hm.height_um/2.0
    a, b, c = plane
    coarse = hm.z-(a*x[None, :]+b*y[:, None]+c)
    initial_x, initial_y, initialization = initial_center_from_component(
        coarse, hm.valid_mask, x, y, center_search, nominal_size_um)
    pixels = int(np.floor(local_canvas_um/max(hm.dx_um, hm.dy_um)))
    local = resample_to_canonical(
        hm, plane=plane, center_x_um=initial_x, center_y_um=initial_y,
        theta_deg=theta_deg, length_um=local_canvas_um, pixels=pixels,
        minimum_mask_weight=0.99, order=1,
        metadata={"purpose": "v3_four_edge_local_fit"})
    filled = np.where(local.valid_mask, local.z,
                      np.nanmedian(local.z[local.valid_mask]))
    sigma_pixels = smoothing_sigma_um/local.dx_um
    smooth = ndimage.gaussian_filter(filled, sigma_pixels)
    gradient_v, gradient_u = np.gradient(smooth, local.dy_um, local.dx_um)
    central = np.abs(local.x_um) <= profile_strip_halfwidth_um
    central_v = np.abs(local.y_um) <= profile_strip_halfwidth_um
    # Median aggregation suppresses internal scan lines and isolated debris.
    profile_u = np.median(gradient_u[central_v, :], axis=0)
    profile_v = np.median(gradient_v[:, central], axis=1)
    half = nominal_size_um/2.0
    left, left_snr = _locate_signed_edge(
        local.x_um, -profile_u, -half, edge_search_halfwidth_um,
        peak_centroid_halfwidth_um)
    right, right_snr = _locate_signed_edge(
        local.x_um, profile_u, half, edge_search_halfwidth_um,
        peak_centroid_halfwidth_um)
    top, top_snr = _locate_signed_edge(
        local.y_um, -profile_v, -half, edge_search_halfwidth_um,
        peak_centroid_halfwidth_um)
    bottom, bottom_snr = _locate_signed_edge(
        local.y_um, profile_v, half, edge_search_halfwidth_um,
        peak_centroid_halfwidth_um)
    positions = np.array([left, right, top, bottom], dtype=float)
    snrs = np.array([left_snr, right_snr, top_snr, bottom_snr], dtype=float)
    if not np.all(np.isfinite(positions)):
        return _failed(initialization, "one or more edge positions are non-finite")

    # Each observed edge implies a centre under the frozen 200 um model.
    implied_u = np.array([left+half, right-half])
    implied_v = np.array([top+half, bottom-half])
    weights_u = np.clip(snrs[:2], 0.1, 20.0)
    weights_v = np.clip(snrs[2:], 0.1, 20.0)
    delta_u = float(np.average(implied_u, weights=weights_u))
    delta_v = float(np.average(implied_v, weights=weights_v))
    predicted = np.array([delta_u-half, delta_u+half,
                          delta_v-half, delta_v+half])
    residuals = positions-predicted
    residual_rmse = float(np.sqrt(np.mean(residuals**2)))
    residual_max = float(np.max(np.abs(residuals)))
    theta = np.deg2rad(theta_deg)
    center_x = initial_x+delta_u*np.cos(theta)-delta_v*np.sin(theta)
    center_y = initial_y+delta_u*np.sin(theta)+delta_v*np.cos(theta)
    xmin, xmax, ymin, ymax = center_search
    boundary = (abs(center_x-xmin) <= boundary_tolerance_um
                or abs(center_x-xmax) <= boundary_tolerance_um
                or abs(center_y-ymin) <= boundary_tolerance_um
                or abs(center_y-ymax) <= boundary_tolerance_um)
    width_x = float(right-left)
    width_y = float(bottom-top)
    minimum_snr = float(np.min(snrs))
    hard_failures = []
    if minimum_snr < minimum_edge_snr:
        hard_failures.append("edge SNR below minimum")
    if not (width_range_um[0] <= width_x <= width_range_um[1]):
        hard_failures.append("observed x width outside hard range")
    if not (width_range_um[0] <= width_y <= width_range_um[1]):
        hard_failures.append("observed y width outside hard range")
    if residual_max > hard_residual_um:
        hard_failures.append("constrained edge residual above hard maximum")
    reviews = []
    if minimum_snr < review_edge_snr:
        reviews.append("edge SNR below review threshold")
    if residual_max > review_residual_um:
        reviews.append("constrained edge residual above review threshold")
    if boundary:
        reviews.append("center search boundary hit")
    if initialization.endswith("fallback"):
        reviews.append("segmentation initializer unavailable")
    status = "STOP" if hard_failures else "REVIEW" if reviews else "PASS"
    warning = "; ".join(hard_failures+reviews)
    return FourEdgeFit(
        center_x_um=float(center_x), center_y_um=float(center_y),
        left_edge_um=left, right_edge_um=right, top_edge_um=top,
        bottom_edge_um=bottom, observed_width_x_um=width_x,
        observed_width_y_um=width_y, left_snr=left_snr,
        right_snr=right_snr, top_snr=top_snr, bottom_snr=bottom_snr,
        minimum_edge_snr=minimum_snr,
        constrained_edge_rmse_um=residual_rmse,
        constrained_edge_max_abs_um=residual_max,
        center_search_boundary_hit=boundary, initialization=initialization,
        status=status, warning=warning)


def _failed(initialization: str, warning: str) -> FourEdgeFit:
    return FourEdgeFit(
        *(float("nan"),)*15, False, initialization, "STOP", warning)
