"""v5 multi-band integrated step-contrast registration with block bootstrap."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .data_contracts import HeightMap
from .registration import initial_center_from_component
from .resampling import resample_to_canonical

__all__ = ["StepContrastBootstrapFit", "fit_step_contrast_bootstrap"]


@dataclass(frozen=True)
class StepContrastBootstrapFit:
    center_x_um: float
    center_y_um: float
    delta_u_um: float
    delta_v_um: float
    left_evidence: float
    right_evidence: float
    top_evidence: float
    bottom_evidence: float
    joint_evidence_total: float
    x_pair_evidence: float
    y_pair_evidence: float
    outer_reference_scale_um: float
    bootstrap_u_mad_um: float
    bootstrap_v_mad_um: float
    bootstrap_u_ci_span_um: float
    bootstrap_v_ci_span_um: float
    bootstrap_u_q025_um: float
    bootstrap_u_q975_um: float
    bootstrap_v_q025_um: float
    bootstrap_v_q975_um: float
    bootstrap_replicates: int
    u_multimodal: bool
    v_multimodal: bool
    center_search_boundary_hit: bool
    initialization: str
    status: str
    warning: str

    def to_dict(self) -> dict:
        return asdict(self)


def _robust_scale(values: np.ndarray) -> float:
    centre = float(np.median(values))
    value = 1.4826*float(np.median(np.abs(values-centre)))
    if not np.isfinite(value) or value <= 1e-12:
        value = float(np.std(values))
    return max(value, 1e-12)


def _interval_medians(profile: np.ndarray, axis: np.ndarray,
                      lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    """Return exact inclusive interval medians without a Python candidate loop."""
    starts = np.searchsorted(axis, lows, side="left")
    ends = np.searchsorted(axis, highs, side="right")
    lengths = ends-starts
    width = int(lengths.max(initial=0))
    if width == 0:
        return np.full(lows.shape, np.nan)
    offsets = np.arange(width)[None, :]
    indices = starts[:, None]+offsets
    selected = offsets < lengths[:, None]
    indices = np.minimum(indices, len(profile)-1)
    values = np.where(selected, profile[indices], np.nan)
    return np.nanmedian(values, axis=1)


def _pair_curve(profile: np.ndarray, axis: np.ndarray, candidates: np.ndarray,
                half: float, bandwidth: float, gap: float,
                scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    negative_edge = candidates-half
    positive_edge = candidates+half
    # Left/top: outside lies on the negative-coordinate side.
    first = (
        _interval_medians(profile, axis, negative_edge-bandwidth,
                          negative_edge-gap)
        - _interval_medians(profile, axis, negative_edge+gap,
                            negative_edge+bandwidth))/scale
    # Right/bottom: outside lies on the positive-coordinate side.
    second = (
        _interval_medians(profile, axis, positive_edge+gap,
                          positive_edge+bandwidth)
        - _interval_medians(profile, axis, positive_edge-bandwidth,
                            positive_edge-gap))/scale
    first = np.maximum(first, 0.0)
    second = np.maximum(second, 0.0)
    return first+second, first, second


def _pair_centre(profile: np.ndarray, axis: np.ndarray, candidates: np.ndarray,
                 half: float, bandwidth: float, gap: float,
                 scale: float) -> tuple[float, float, float]:
    score, first, second = _pair_curve(
        profile, axis, candidates, half, bandwidth, gap, scale)
    index = int(np.argmax(score))
    return float(candidates[index]), float(first[index]), float(second[index])


def _block_profiles(z: np.ndarray, x: np.ndarray, y: np.ndarray,
                    strip_halfwidth: float, blocks: int) -> tuple[np.ndarray, np.ndarray]:
    row_groups = np.array_split(np.flatnonzero(np.abs(y)<=strip_halfwidth), blocks)
    column_groups = np.array_split(np.flatnonzero(np.abs(x)<=strip_halfwidth), blocks)
    profiles_u = np.stack([np.median(z[group, :], axis=0) for group in row_groups])
    profiles_v = np.stack([np.median(z[:, group], axis=1) for group in column_groups])
    return profiles_u, profiles_v


def _mad(values: np.ndarray) -> float:
    return float(np.median(np.abs(values-np.median(values))))


def _multimodal(values: np.ndarray, bin_um: float,
                secondary_fraction: float, separation_um: float) -> bool:
    low = np.floor(values.min()/bin_um)*bin_um
    high = np.ceil(values.max()/bin_um)*bin_um+bin_um
    counts, edges = np.histogram(values, bins=np.arange(low, high+bin_um/2, bin_um))
    if counts.size < 2:
        return False
    order = np.argsort(counts)[::-1]
    first, second = int(order[0]), int(order[1])
    centres = (edges[:-1]+edges[1:])/2
    return bool(counts[second]/len(values)>=secondary_fraction
                and abs(centres[first]-centres[second])>=separation_um)


def fit_step_contrast_bootstrap(
        hm: HeightMap, *, plane: tuple[float, float, float], theta_deg: float,
        center_search: tuple[float, float, float, float],
        nominal_size_um: float = 200.0, local_canvas_um: float = 270.0,
        edge_search_halfwidth_um: float = 20.0,
        center_grid_step_um: float = 0.25,
        profile_strip_halfwidth_um: float = 70.0,
        smoothing_sigma_um: float = 0.75,
        contrast_bandwidths_um: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0),
        boundary_gap_um: float = 0.75, tangent_blocks: int = 8,
        bootstrap_replicates_per_band: int = 64,
        random_seed: int = 20260831,
        hard_minimum_total: float = 8.0, review_below_total: float = 12.0,
        hard_minimum_per_axis: float = 3.0,
        ci_quantiles: tuple[float, float] = (0.025, 0.975),
        review_ci_span_um: float = 6.0, hard_ci_span_um: float = 12.0,
        review_mad_um: float = 1.5, hard_mad_um: float = 3.0,
        histogram_bin_um: float = 1.0, minimum_secondary_fraction: float = 0.15,
        minimum_mode_separation_um: float = 4.0,
        boundary_tolerance_um: float = 0.5) -> StepContrastBootstrapFit:
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
        metadata={"purpose": "v5_step_contrast_bootstrap"})
    filled = np.where(local.valid_mask, local.z,
                      np.nanmedian(local.z[local.valid_mask]))
    smooth = ndimage.gaussian_filter(filled, smoothing_sigma_um/local.dx_um)
    outer = ((np.abs(local.x_um[None, :])>=120.0)
             | (np.abs(local.y_um[:, None])>=120.0)) & local.valid_mask
    reference_scale = _robust_scale(smooth[outer])
    profiles_u, profiles_v = _block_profiles(
        smooth, local.x_um, local.y_um, profile_strip_halfwidth_um,
        tangent_blocks)
    candidates = np.arange(-edge_search_halfwidth_um,
                           edge_search_halfwidth_um+center_grid_step_um/2,
                           center_grid_step_um)
    half = nominal_size_um/2.0
    rng = np.random.default_rng(random_seed)
    bootstrap_u: list[float] = []
    bootstrap_v: list[float] = []
    full_profiles_u = np.median(profiles_u, axis=0)
    full_profiles_v = np.median(profiles_v, axis=0)
    for bandwidth in contrast_bandwidths_um:
        for _ in range(bootstrap_replicates_per_band):
            selection = rng.integers(0, tangent_blocks, size=tangent_blocks)
            profile_u = np.median(profiles_u[selection], axis=0)
            profile_v = np.median(profiles_v[selection], axis=0)
            du, _, _ = _pair_centre(
                profile_u, local.x_um, candidates, half, bandwidth,
                boundary_gap_um, reference_scale)
            dv, _, _ = _pair_centre(
                profile_v, local.y_um, candidates, half, bandwidth,
                boundary_gap_um, reference_scale)
            bootstrap_u.append(du)
            bootstrap_v.append(dv)
    bu, bv = np.asarray(bootstrap_u), np.asarray(bootstrap_v)
    delta_u, delta_v = float(np.median(bu)), float(np.median(bv))
    evidence = []
    for bandwidth in contrast_bandwidths_um:
        x_score, left, right = _pair_curve(
            full_profiles_u, local.x_um, np.array([delta_u]), half,
            bandwidth, boundary_gap_um, reference_scale)
        y_score, top, bottom = _pair_curve(
            full_profiles_v, local.y_um, np.array([delta_v]), half,
            bandwidth, boundary_gap_um, reference_scale)
        evidence.append((left[0], right[0], top[0], bottom[0]))
    left, right, top, bottom = map(float, np.median(evidence, axis=0))
    x_pair, y_pair = left+right, top+bottom
    total = x_pair+y_pair
    u_low, u_high = np.quantile(bu, ci_quantiles)
    v_low, v_high = np.quantile(bv, ci_quantiles)
    u_span, v_span = float(u_high-u_low), float(v_high-v_low)
    u_mad, v_mad = _mad(bu), _mad(bv)
    u_multi = _multimodal(bu, histogram_bin_um,
                          minimum_secondary_fraction,
                          minimum_mode_separation_um)
    v_multi = _multimodal(bv, histogram_bin_um,
                          minimum_secondary_fraction,
                          minimum_mode_separation_um)
    theta = np.deg2rad(theta_deg)
    center_x = initial_x+delta_u*np.cos(theta)-delta_v*np.sin(theta)
    center_y = initial_y+delta_u*np.sin(theta)+delta_v*np.cos(theta)
    xmin, xmax, ymin, ymax = center_search
    boundary = (abs(center_x-xmin)<=boundary_tolerance_um
                or abs(center_x-xmax)<=boundary_tolerance_um
                or abs(center_y-ymin)<=boundary_tolerance_um
                or abs(center_y-ymax)<=boundary_tolerance_um)
    hard = []
    if total<hard_minimum_total:
        hard.append("joint evidence below hard minimum")
    if min(x_pair,y_pair)<hard_minimum_per_axis:
        hard.append("axis-pair evidence below hard minimum")
    if max(u_span,v_span)>hard_ci_span_um:
        hard.append("bootstrap CI span above hard maximum")
    if max(u_mad,v_mad)>hard_mad_um:
        hard.append("bootstrap MAD above hard maximum")
    if u_multi or v_multi:
        hard.append("bootstrap centre distribution is multimodal")
    reviews = []
    if total<review_below_total:
        reviews.append("joint evidence below review threshold")
    if max(u_span,v_span)>review_ci_span_um:
        reviews.append("bootstrap CI span above review threshold")
    if max(u_mad,v_mad)>review_mad_um:
        reviews.append("bootstrap MAD above review threshold")
    if boundary:
        reviews.append("center search boundary hit")
    if initialization.endswith("fallback"):
        reviews.append("segmentation initializer unavailable")
    status = "STOP" if hard else "REVIEW" if reviews else "PASS"
    return StepContrastBootstrapFit(
        center_x_um=float(center_x),center_y_um=float(center_y),
        delta_u_um=delta_u,delta_v_um=delta_v,left_evidence=left,
        right_evidence=right,top_evidence=top,bottom_evidence=bottom,
        joint_evidence_total=float(total),x_pair_evidence=float(x_pair),
        y_pair_evidence=float(y_pair),outer_reference_scale_um=reference_scale,
        bootstrap_u_mad_um=u_mad,bootstrap_v_mad_um=v_mad,
        bootstrap_u_ci_span_um=u_span,bootstrap_v_ci_span_um=v_span,
        bootstrap_u_q025_um=float(u_low),bootstrap_u_q975_um=float(u_high),
        bootstrap_v_q025_um=float(v_low),bootstrap_v_q975_um=float(v_high),
        bootstrap_replicates=len(bu),u_multimodal=u_multi,v_multimodal=v_multi,
        center_search_boundary_hit=boundary,initialization=initialization,
        status=status,warning="; ".join(hard+reviews))
