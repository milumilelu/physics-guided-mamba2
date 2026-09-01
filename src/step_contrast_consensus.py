"""v6 robust block/band consensus for constrained four-edge registration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .data_contracts import HeightMap
from .registration import initial_center_from_component
from .resampling import resample_to_canonical
from .step_contrast_bootstrap import (
    _block_profiles,
    _mad,
    _multimodal,
    _pair_curve,
    _robust_scale,
)

__all__ = ["StepContrastConsensusFit", "fit_step_contrast_consensus"]


@dataclass(frozen=True)
class StepContrastConsensusFit:
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
    influence_u_mad_um: float
    influence_v_mad_um: float
    influence_u_ci_span_um: float
    influence_v_ci_span_um: float
    influence_u_q025_um: float
    influence_u_q975_um: float
    influence_v_q025_um: float
    influence_v_q975_um: float
    influence_replicates: int
    u_multimodal: bool
    v_multimodal: bool
    local_search_boundary_hit: bool
    global_search_boundary_hit: bool
    initialization: str
    status: str
    warning: str

    def to_dict(self) -> dict:
        return asdict(self)


def _curve_cube(profiles: np.ndarray, axis: np.ndarray,
                candidates: np.ndarray, half: float,
                bandwidths: tuple[float, ...], gap: float,
                scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = []
    first_sides = []
    second_sides = []
    for bandwidth in bandwidths:
        band_scores = []
        band_first = []
        band_second = []
        for profile in profiles:
            score, first, second = _pair_curve(
                profile, axis, candidates, half, bandwidth, gap, scale
            )
            band_scores.append(score)
            band_first.append(first)
            band_second.append(second)
        scores.append(band_scores)
        first_sides.append(band_first)
        second_sides.append(band_second)
    return np.asarray(scores), np.asarray(first_sides), np.asarray(second_sides)


def _aggregate(scores: np.ndarray) -> np.ndarray:
    return np.median(np.median(scores, axis=1), axis=0)


def _estimate_axis(scores: np.ndarray, candidates: np.ndarray) -> tuple[float, np.ndarray]:
    centre = float(candidates[int(np.argmax(_aggregate(scores)))])
    deleted = []
    for block in range(scores.shape[1]):
        deleted.append(float(candidates[int(np.argmax(_aggregate(
            np.delete(scores, block, axis=1)
        ))) ]))
    for band in range(scores.shape[0]):
        deleted.append(float(candidates[int(np.argmax(_aggregate(
            np.delete(scores, band, axis=0)
        ))) ]))
    return centre, np.asarray(deleted)


def _estimate_balanced_axis(first: np.ndarray, second: np.ndarray,
                            candidates: np.ndarray) -> tuple[float, np.ndarray]:
    def constrained_midpoint(one: np.ndarray, two: np.ndarray) -> float:
        first_peak = candidates[int(np.argmax(_aggregate(one)))]
        second_peak = candidates[int(np.argmax(_aggregate(two)))]
        midpoint = (first_peak+second_peak)/2.0
        return float(candidates[int(np.argmin(np.abs(candidates-midpoint)))])

    centre = constrained_midpoint(first, second)
    deleted = []
    for block in range(first.shape[1]):
        deleted.append(constrained_midpoint(
            np.delete(first, block, axis=1), np.delete(second, block, axis=1)))
    for band in range(first.shape[0]):
        deleted.append(constrained_midpoint(
            np.delete(first, band, axis=0), np.delete(second, band, axis=0)))
    return centre, np.asarray(deleted)


def _side_evidence(first: np.ndarray, second: np.ndarray,
                   candidate_index: int) -> tuple[float, float]:
    first_at = first[:, :, candidate_index]
    second_at = second[:, :, candidate_index]
    return (
        float(np.median(np.median(first_at, axis=1))),
        float(np.median(np.median(second_at, axis=1))),
    )


def fit_step_contrast_consensus(
        hm: HeightMap, *, plane: tuple[float, float, float], theta_deg: float,
        center_search: tuple[float, float, float, float],
        nominal_size_um: float = 200.0, local_canvas_um: float = 340.0,
        edge_search_halfwidth_um: float = 55.0,
        center_grid_step_um: float = 0.25,
        profile_strip_halfwidth_um: float = 70.0,
        smoothing_sigma_um: float = 0.75,
        contrast_bandwidths_um: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0),
        boundary_gap_um: float = 0.75, tangent_blocks: int = 8,
        hard_minimum_total: float = 8.0, review_below_total: float = 12.0,
        hard_minimum_per_axis: float = 3.0,
        ci_quantiles: tuple[float, float] = (0.025, 0.975),
        review_ci_span_um: float = 6.0, hard_ci_span_um: float = 12.0,
        review_mad_um: float = 1.5, hard_mad_um: float = 3.0,
        histogram_bin_um: float = 1.0,
        minimum_secondary_fraction: float = 0.15,
        minimum_mode_separation_um: float = 4.0,
        local_boundary_tolerance_um: float = 0.5,
        global_boundary_tolerance_um: float = 0.5,
        balanced_opposing_edges: bool = False) -> StepContrastConsensusFit:
    x = hm.x_um-hm.width_um/2.0
    y = hm.y_um-hm.height_um/2.0
    a, b, c = plane
    coarse = hm.z-(a*x[None, :]+b*y[:, None]+c)
    initial_x, initial_y, initialization = initial_center_from_component(
        coarse, hm.valid_mask, x, y, center_search, nominal_size_um
    )
    pixels = int(np.floor(local_canvas_um/max(hm.dx_um, hm.dy_um)))
    local = resample_to_canonical(
        hm, plane=plane, center_x_um=initial_x, center_y_um=initial_y,
        theta_deg=theta_deg, length_um=local_canvas_um, pixels=pixels,
        minimum_mask_weight=0.99, order=1,
        metadata={"purpose": "v6_step_contrast_consensus"}
    )
    filled = np.where(
        local.valid_mask, local.z, np.nanmedian(local.z[local.valid_mask])
    )
    smooth = ndimage.gaussian_filter(filled, smoothing_sigma_um/local.dx_um)
    outer = (
        ((np.abs(local.x_um[None, :]) >= 130.0)
         | (np.abs(local.y_um[:, None]) >= 130.0))
        & local.valid_mask
    )
    reference_scale = _robust_scale(smooth[outer])
    profiles_u, profiles_v = _block_profiles(
        smooth, local.x_um, local.y_um, profile_strip_halfwidth_um,
        tangent_blocks
    )
    candidates = np.arange(
        -edge_search_halfwidth_um,
        edge_search_halfwidth_um+center_grid_step_um/2,
        center_grid_step_um
    )
    half = nominal_size_um/2.0
    u_score, left_cube, right_cube = _curve_cube(
        profiles_u, local.x_um, candidates, half,
        contrast_bandwidths_um, boundary_gap_um, reference_scale
    )
    v_score, top_cube, bottom_cube = _curve_cube(
        profiles_v, local.y_um, candidates, half,
        contrast_bandwidths_um, boundary_gap_um, reference_scale
    )
    if balanced_opposing_edges:
        delta_u, influence_u = _estimate_balanced_axis(
            left_cube, right_cube, candidates)
        delta_v, influence_v = _estimate_balanced_axis(
            top_cube, bottom_cube, candidates)
    else:
        delta_u, influence_u = _estimate_axis(u_score, candidates)
        delta_v, influence_v = _estimate_axis(v_score, candidates)
    u_index = int(np.argmin(np.abs(candidates-delta_u)))
    v_index = int(np.argmin(np.abs(candidates-delta_v)))
    left, right = _side_evidence(left_cube, right_cube, u_index)
    top, bottom = _side_evidence(top_cube, bottom_cube, v_index)
    x_pair, y_pair = left+right, top+bottom
    total = x_pair+y_pair
    u_low, u_high = np.quantile(influence_u, ci_quantiles)
    v_low, v_high = np.quantile(influence_v, ci_quantiles)
    u_span, v_span = float(u_high-u_low), float(v_high-v_low)
    u_mad, v_mad = _mad(influence_u), _mad(influence_v)
    u_multi = _multimodal(
        influence_u, histogram_bin_um, minimum_secondary_fraction,
        minimum_mode_separation_um
    )
    v_multi = _multimodal(
        influence_v, histogram_bin_um, minimum_secondary_fraction,
        minimum_mode_separation_um
    )
    theta = np.deg2rad(theta_deg)
    center_x = initial_x+delta_u*np.cos(theta)-delta_v*np.sin(theta)
    center_y = initial_y+delta_u*np.sin(theta)+delta_v*np.cos(theta)
    local_boundary = (
        abs(abs(delta_u)-edge_search_halfwidth_um) <= local_boundary_tolerance_um
        or abs(abs(delta_v)-edge_search_halfwidth_um) <= local_boundary_tolerance_um
    )
    xmin, xmax, ymin, ymax = center_search
    global_boundary = (
        abs(center_x-xmin) <= global_boundary_tolerance_um
        or abs(center_x-xmax) <= global_boundary_tolerance_um
        or abs(center_y-ymin) <= global_boundary_tolerance_um
        or abs(center_y-ymax) <= global_boundary_tolerance_um
    )
    hard = []
    if total < hard_minimum_total:
        hard.append("joint evidence below hard minimum")
    if min(x_pair, y_pair) < hard_minimum_per_axis:
        hard.append("axis-pair evidence below hard minimum")
    if max(u_span, v_span) > hard_ci_span_um:
        hard.append("delete-one CI span above hard maximum")
    if max(u_mad, v_mad) > hard_mad_um:
        hard.append("delete-one MAD above hard maximum")
    if u_multi or v_multi:
        hard.append("delete-one centre distribution is multimodal")
    if local_boundary:
        hard.append("local center search boundary hit")
    reviews = []
    if total < review_below_total:
        reviews.append("joint evidence below review threshold")
    if max(u_span, v_span) > review_ci_span_um:
        reviews.append("delete-one CI span above review threshold")
    if max(u_mad, v_mad) > review_mad_um:
        reviews.append("delete-one MAD above review threshold")
    if global_boundary:
        reviews.append("global center search boundary hit")
    if initialization.endswith("fallback"):
        reviews.append("segmentation initializer unavailable")
    status = "STOP" if hard else "REVIEW" if reviews else "PASS"
    return StepContrastConsensusFit(
        center_x_um=float(center_x), center_y_um=float(center_y),
        delta_u_um=delta_u, delta_v_um=delta_v,
        left_evidence=left, right_evidence=right,
        top_evidence=top, bottom_evidence=bottom,
        joint_evidence_total=float(total), x_pair_evidence=float(x_pair),
        y_pair_evidence=float(y_pair), outer_reference_scale_um=reference_scale,
        influence_u_mad_um=u_mad, influence_v_mad_um=v_mad,
        influence_u_ci_span_um=u_span, influence_v_ci_span_um=v_span,
        influence_u_q025_um=float(u_low), influence_u_q975_um=float(u_high),
        influence_v_q025_um=float(v_low), influence_v_q975_um=float(v_high),
        influence_replicates=len(influence_u), u_multimodal=u_multi,
        v_multimodal=v_multi, local_search_boundary_hit=local_boundary,
        global_search_boundary_hit=global_boundary,
        initialization=initialization, status=status,
        warning="; ".join(hard+reviews)
    )
