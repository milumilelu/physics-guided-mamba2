"""v4 multi-scale joint four-edge centre estimation with block bootstrap."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .data_contracts import HeightMap
from .registration import initial_center_from_component
from .resampling import resample_to_canonical

__all__ = ["JointEdgeBootstrapFit", "fit_joint_edge_bootstrap"]


@dataclass(frozen=True)
class JointEdgeBootstrapFit:
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


def _scale(values: np.ndarray) -> float:
    centre = float(np.median(values))
    scale = 1.4826*float(np.median(np.abs(values-centre)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    return max(scale, 1e-12)


def _standardized(profile: np.ndarray, axis: np.ndarray,
                  expected: float, search_halfwidth: float) -> np.ndarray:
    background = np.abs(axis-expected) > search_halfwidth
    values = profile[background] if background.any() else profile
    baseline = float(np.median(values))
    return (profile-baseline)/_scale(values)


def _pair_centre(axis: np.ndarray, negative_profile: np.ndarray,
                 positive_profile: np.ndarray, candidates: np.ndarray,
                 half: float, search_halfwidth: float) -> tuple[float, float, float]:
    left_z = _standardized(-negative_profile, axis, -half, search_halfwidth)
    right_z = _standardized(positive_profile, axis, half, search_halfwidth)
    left = np.maximum(np.interp(candidates-half, axis, left_z), 0.0)
    right = np.maximum(np.interp(candidates+half, axis, right_z), 0.0)
    score = left+right
    index = int(np.argmax(score))
    return float(candidates[index]), float(left[index]), float(right[index])


def _side_evidence(axis: np.ndarray, profile: np.ndarray, delta: float,
                   half: float, search_halfwidth: float) -> tuple[float, float]:
    left_z = _standardized(-profile, axis, -half, search_halfwidth)
    right_z = _standardized(profile, axis, half, search_halfwidth)
    left = max(float(np.interp(delta-half, axis, left_z)), 0.0)
    right = max(float(np.interp(delta+half, axis, right_z)), 0.0)
    return left, right


def _blocks(gradient_u: np.ndarray, gradient_v: np.ndarray,
            x: np.ndarray, y: np.ndarray, strip_halfwidth: float,
            count: int) -> tuple[np.ndarray, np.ndarray]:
    row_indices = np.flatnonzero(np.abs(y) <= strip_halfwidth)
    column_indices = np.flatnonzero(np.abs(x) <= strip_halfwidth)
    row_blocks = np.array_split(row_indices, count)
    column_blocks = np.array_split(column_indices, count)
    profiles_u = np.stack([
        np.median(gradient_u[indices, :], axis=0) for indices in row_blocks])
    profiles_v = np.stack([
        np.median(gradient_v[:, indices], axis=1) for indices in column_blocks])
    return profiles_u, profiles_v


def _mad(values: np.ndarray) -> float:
    centre = np.median(values)
    return float(np.median(np.abs(values-centre)))


def _multimodal(values: np.ndarray, *, bin_um: float,
                secondary_fraction: float, separation_um: float) -> bool:
    low = np.floor(values.min()/bin_um)*bin_um
    high = np.ceil(values.max()/bin_um)*bin_um+bin_um
    counts, edges = np.histogram(values, bins=np.arange(low, high+bin_um/2, bin_um))
    if counts.size < 2:
        return False
    order = np.argsort(counts)[::-1]
    first, second = int(order[0]), int(order[1])
    centres = (edges[:-1]+edges[1:])/2.0
    return bool(counts[second]/len(values) >= secondary_fraction
                and abs(centres[first]-centres[second]) >= separation_um)


def fit_joint_edge_bootstrap(
        hm: HeightMap, *, plane: tuple[float, float, float], theta_deg: float,
        center_search: tuple[float, float, float, float],
        nominal_size_um: float = 200.0, local_canvas_um: float = 270.0,
        edge_search_halfwidth_um: float = 20.0,
        center_grid_step_um: float = 0.25,
        profile_strip_halfwidth_um: float = 70.0,
        smoothing_scales_um: tuple[float, ...] = (0.75, 1.0, 1.5),
        tangent_blocks: int = 8, bootstrap_replicates_per_scale: int = 64,
        random_seed: int = 20260831,
        hard_minimum_total: float = 8.0, review_below_total: float = 12.0,
        hard_minimum_per_axis: float = 3.0,
        ci_quantiles: tuple[float, float] = (0.025, 0.975),
        review_ci_span_um: float = 6.0, hard_ci_span_um: float = 12.0,
        review_mad_um: float = 1.5, hard_mad_um: float = 3.0,
        histogram_bin_um: float = 1.0, minimum_secondary_fraction: float = 0.15,
        minimum_mode_separation_um: float = 4.0,
        boundary_tolerance_um: float = 0.5) -> JointEdgeBootstrapFit:
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
        metadata={"purpose": "v4_joint_edge_bootstrap"})
    filled = np.where(local.valid_mask, local.z,
                      np.nanmedian(local.z[local.valid_mask]))
    candidates = np.arange(-edge_search_halfwidth_um,
                           edge_search_halfwidth_um+center_grid_step_um/2,
                           center_grid_step_um)
    half = nominal_size_um/2.0
    rng = np.random.default_rng(random_seed)
    bootstrap_u: list[float] = []
    bootstrap_v: list[float] = []
    full_profiles: list[tuple[np.ndarray, np.ndarray]] = []
    for smoothing_um in smoothing_scales_um:
        smooth = ndimage.gaussian_filter(filled, smoothing_um/local.dx_um)
        gradient_v, gradient_u = np.gradient(smooth, local.dy_um, local.dx_um)
        profiles_u, profiles_v = _blocks(
            gradient_u, gradient_v, local.x_um, local.y_um,
            profile_strip_halfwidth_um, tangent_blocks)
        full_u = np.median(profiles_u, axis=0)
        full_v = np.median(profiles_v, axis=0)
        full_profiles.append((full_u, full_v))
        for _ in range(bootstrap_replicates_per_scale):
            selection = rng.integers(0, tangent_blocks, size=tangent_blocks)
            profile_u = np.median(profiles_u[selection], axis=0)
            profile_v = np.median(profiles_v[selection], axis=0)
            delta_u, _, _ = _pair_centre(
                local.x_um, profile_u, profile_u, candidates, half,
                edge_search_halfwidth_um)
            delta_v, _, _ = _pair_centre(
                local.y_um, profile_v, profile_v, candidates, half,
                edge_search_halfwidth_um)
            bootstrap_u.append(delta_u)
            bootstrap_v.append(delta_v)
    bu = np.asarray(bootstrap_u)
    bv = np.asarray(bootstrap_v)
    delta_u = float(np.median(bu))
    delta_v = float(np.median(bv))
    evidence_by_scale = []
    for full_u, full_v in full_profiles:
        left, right = _side_evidence(
            local.x_um, full_u, delta_u, half, edge_search_halfwidth_um)
        top, bottom = _side_evidence(
            local.y_um, full_v, delta_v, half, edge_search_halfwidth_um)
        evidence_by_scale.append((left, right, top, bottom))
    side_evidence = np.median(np.asarray(evidence_by_scale), axis=0)
    left, right, top, bottom = map(float, side_evidence)
    x_pair, y_pair = left+right, top+bottom
    total = x_pair+y_pair
    u_low, u_high = np.quantile(bu, ci_quantiles)
    v_low, v_high = np.quantile(bv, ci_quantiles)
    u_span, v_span = float(u_high-u_low), float(v_high-v_low)
    u_mad, v_mad = _mad(bu), _mad(bv)
    u_multi = _multimodal(
        bu, bin_um=histogram_bin_um,
        secondary_fraction=minimum_secondary_fraction,
        separation_um=minimum_mode_separation_um)
    v_multi = _multimodal(
        bv, bin_um=histogram_bin_um,
        secondary_fraction=minimum_secondary_fraction,
        separation_um=minimum_mode_separation_um)
    theta = np.deg2rad(theta_deg)
    center_x = initial_x+delta_u*np.cos(theta)-delta_v*np.sin(theta)
    center_y = initial_y+delta_u*np.sin(theta)+delta_v*np.cos(theta)
    xmin, xmax, ymin, ymax = center_search
    boundary = (abs(center_x-xmin) <= boundary_tolerance_um
                or abs(center_x-xmax) <= boundary_tolerance_um
                or abs(center_y-ymin) <= boundary_tolerance_um
                or abs(center_y-ymax) <= boundary_tolerance_um)
    hard = []
    if total < hard_minimum_total:
        hard.append("joint evidence below hard minimum")
    if min(x_pair, y_pair) < hard_minimum_per_axis:
        hard.append("axis-pair evidence below hard minimum")
    if max(u_span, v_span) > hard_ci_span_um:
        hard.append("bootstrap CI span above hard maximum")
    if max(u_mad, v_mad) > hard_mad_um:
        hard.append("bootstrap MAD above hard maximum")
    if u_multi or v_multi:
        hard.append("bootstrap centre distribution is multimodal")
    reviews = []
    if total < review_below_total:
        reviews.append("joint evidence below review threshold")
    if max(u_span, v_span) > review_ci_span_um:
        reviews.append("bootstrap CI span above review threshold")
    if max(u_mad, v_mad) > review_mad_um:
        reviews.append("bootstrap MAD above review threshold")
    if boundary:
        reviews.append("center search boundary hit")
    if initialization.endswith("fallback"):
        reviews.append("segmentation initializer unavailable")
    status = "STOP" if hard else "REVIEW" if reviews else "PASS"
    return JointEdgeBootstrapFit(
        center_x_um=float(center_x), center_y_um=float(center_y),
        delta_u_um=delta_u, delta_v_um=delta_v,
        left_evidence=left, right_evidence=right,
        top_evidence=top, bottom_evidence=bottom,
        joint_evidence_total=float(total), x_pair_evidence=float(x_pair),
        y_pair_evidence=float(y_pair), bootstrap_u_mad_um=u_mad,
        bootstrap_v_mad_um=v_mad, bootstrap_u_ci_span_um=u_span,
        bootstrap_v_ci_span_um=v_span, bootstrap_u_q025_um=float(u_low),
        bootstrap_u_q975_um=float(u_high), bootstrap_v_q025_um=float(v_low),
        bootstrap_v_q975_um=float(v_high), bootstrap_replicates=len(bu),
        u_multimodal=u_multi, v_multimodal=v_multi,
        center_search_boundary_hit=boundary, initialization=initialization,
        status=status, warning="; ".join(hard+reviews))
