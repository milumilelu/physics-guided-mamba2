"""Pure geometry and table helpers for blinded single-line range annotation.

The single-line dataset (``氧化锆/120组直线.cag``) stores one machined line per
CAG group inside a narrow strip.  Annotators mark the machined range with an
elongated rectangle in the rotated canonical view.  This module holds the
stateless helpers shared by the table builder and the interactive annotator:
the robust reference-plane fit and depth-weighted line-axis estimate follow the
frozen pilot conventions (``fit_reference_plane`` / ``weighted_line_axes`` in
the deleted pilot script, thresholds ``threshold_k * sigma_ref``), and the
plane coordinates match ``resampling.resample_to_canonical`` exactly, so a fit
computed here can be handed straight to the canonical resampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .manual_four_edge_annotation import ANNOTATION_FIELDS, canonical_box_record

#: The canonical four-edge box fields plus the elongation descriptors that a
#: single-line range adds.  ``long_axis_um`` / ``short_axis_um`` are derived
#: from ``width_um`` / ``height_um`` at save time so QC can filter thin boxes.
RANGE_FIELDS = ANNOTATION_FIELDS + ("long_axis_um", "short_axis_um", "aspect_ratio")

#: A machined single line is expected to be markedly elongated (theory:
#: 200 um length against ~10-20 um width).  Below this aspect ratio the
#: annotator only warns; the box stays savable because real lines vary.
DEFAULT_MINIMUM_ASPECT = 3.0

#: Padding added around the raw strip so the rotated square view keeps the
#: whole measurement inside the sampled window.
CROP_MARGIN_UM = 2.0

#: Signal pixels below this make the depth-weighted orientation estimate
#: unreliable; the builder then flags the row for review and falls back to 0.
MINIMUM_ORIENTATION_PIXELS = 100

#: Upper bound on the canonical view side in pixels (same cap as the
#: four-edge annotator).
VIEW_PIXEL_CAP = 1000


@dataclass(frozen=True)
class PlaneFit:
    """Robust reference plane ``z ~ a*x + b*y + c`` on centred coordinates."""

    a: float
    b: float
    c: float
    rmse_um: float
    sigma_ref_um: float
    n_inliers: int


@dataclass(frozen=True)
class OrientationEstimate:
    """Depth-weighted major axis of the ablation signal.

    ``theta_deg`` is the rotation that makes the line horizontal in the
    canonical view (identical sign convention to ``session_geometry.csv``).
    """

    theta_deg: float
    center_x_um: float
    center_y_um: float
    threshold_um: float
    signal_pixels: int
    confident: bool


def elongated_box_record(*, left_local_um: float, right_local_um: float,
                         top_local_um: float, bottom_local_um: float,
                         display_center_x_um: float,
                         display_center_y_um: float,
                         theta_deg: float) -> dict[str, float | str]:
    """Canonical box record of a line range plus long/short/aspect fields."""
    record = canonical_box_record(
        left_local_um=left_local_um, right_local_um=right_local_um,
        top_local_um=top_local_um, bottom_local_um=bottom_local_um,
        display_center_x_um=display_center_x_um,
        display_center_y_um=display_center_y_um,
        theta_deg=theta_deg,
    )
    long_axis, short_axis = sorted((
        float(record["width_um"]), float(record["height_um"])), reverse=True)
    record["long_axis_um"] = long_axis
    record["short_axis_um"] = short_axis
    record["aspect_ratio"] = long_axis/short_axis if short_axis > 0 else ""
    return record


def elongation_is_suspicious(record: dict, *,
                             minimum_aspect: float = DEFAULT_MINIMUM_ASPECT
                             ) -> bool:
    """True when a saved box is too square to describe a single line."""
    aspect = record.get("aspect_ratio", "")
    try:
        return float(aspect) < float(minimum_aspect)
    except (TypeError, ValueError):
        return False


def annotation_table_columns(annotator: str) -> list[str]:
    """Column layout of a single-line range annotation table."""
    prefix = f"annotator_{annotator.lower()}_"
    return ["session_id", "sample_id", "measurement_id",
            "roi_within_measurement"] + [prefix+field for field in RANGE_FIELDS]


def fit_reference_plane(
    z: np.ndarray, valid: np.ndarray, dx: float, dy: float, *,
    negative_clip_sigma: float = 2.5, positive_clip_sigma: float = 4.0,
    max_iter: int = 12, minimum_inliers: int = 100,
    minimum_inlier_fraction: float = 0.15,
) -> PlaneFit:
    """Asymmetric-clipping robust plane fit on centred physical coordinates.

    The centring ``(col - (cols-1)/2) * dx`` matches the ``x_centered`` /
    ``y_centered`` frame that ``resample_to_canonical`` subtracts the plane
    in, so the returned coefficients can be reused without conversion.
    """
    rows, cols = z.shape
    y, x = np.indices(z.shape, dtype=float)
    x = (x-(cols-1)/2.0)*dx
    y = (y-(rows-1)/2.0)*dy
    flat_valid = (valid.ravel() & np.isfinite(z.ravel()))
    design = np.column_stack([x.ravel(), y.ravel(), np.ones(z.size)])
    target = z.ravel()
    inliers = flat_valid.copy()
    if inliers.sum() < 3:
        raise ValueError("reference plane fit needs at least 3 valid pixels")
    beta = np.linalg.lstsq(design[inliers], target[inliers], rcond=None)[0]
    for _ in range(max_iter):
        residual = target-design@beta
        center = float(np.median(residual[inliers]))
        scale = 1.4826*float(np.median(np.abs(residual[inliers]-center)))
        if not np.isfinite(scale) or scale < 1e-9:
            break
        new_inliers = flat_valid & (
            residual-center >= -negative_clip_sigma*scale) & (
            residual-center <= positive_clip_sigma*scale)
        if new_inliers.sum() < max(minimum_inliers,
                                   int(minimum_inlier_fraction*flat_valid.sum())):
            break
        new_beta = np.linalg.lstsq(
            design[new_inliers], target[new_inliers], rcond=None)[0]
        if np.array_equal(new_inliers, inliers) and np.max(
                np.abs(new_beta-beta)) < 1e-10:
            beta = new_beta
            inliers = new_inliers
            break
        beta = new_beta
        inliers = new_inliers
    residual = target-design@beta
    reference = residual[inliers]
    rmse = float(np.sqrt(np.mean(reference**2)))
    center = float(np.median(reference))
    sigma_ref = 1.4826*float(np.median(np.abs(reference-center)))
    return PlaneFit(a=float(beta[0]), b=float(beta[1]), c=float(beta[2]),
                    rmse_um=rmse, sigma_ref_um=sigma_ref,
                    n_inliers=int(inliers.sum()))


def plane_depth(z: np.ndarray, valid: np.ndarray, dx: float, dy: float,
                fit: PlaneFit) -> np.ndarray:
    """Removal depth (positive = material removed) with NaN outside the mask.

    ``depth = plane - z`` keeps the frozen pilot height sign convention.
    """
    rows, cols = z.shape
    y, x = np.indices(z.shape, dtype=float)
    x = (x-(cols-1)/2.0)*dx
    y = (y-(rows-1)/2.0)*dy
    depth = (fit.a*x+fit.b*y+fit.c)-z
    depth[~valid] = np.nan
    return depth


def estimate_line_orientation(
    depth: np.ndarray, valid: np.ndarray, dx: float, dy: float, *,
    sigma_ref_um: float, threshold_k: float = 4.0,
    minimum_signal_pixels: int = MINIMUM_ORIENTATION_PIXELS,
) -> OrientationEstimate:
    """Depth-weighted major-axis PCA of the above-threshold ablation signal.

    Mirrors the pilot ``weighted_line_axes`` estimate; used only to rotate the
    annotator display so the elongated rectangle can stay axis-aligned.  The
    annotator itself never shows any automatic line boundary.
    """
    rows, cols = depth.shape
    threshold = float(threshold_k*sigma_ref_um)
    signal = valid & np.isfinite(depth) & (depth > threshold)
    signal_pixels = int(signal.sum())
    if signal_pixels < minimum_signal_pixels:
        return OrientationEstimate(
            theta_deg=0.0, center_x_um=0.0, center_y_um=0.0,
            threshold_um=threshold, signal_pixels=signal_pixels,
            confident=False)
    rr, cc = np.nonzero(signal)
    x = (cc-(cols-1)/2.0)*dx
    y = (rr-(rows-1)/2.0)*dy
    weights = np.maximum(depth[rr, cc], 1e-12)
    center_x = float(np.average(x, weights=weights))
    center_y = float(np.average(y, weights=weights))
    centered = np.column_stack([x-center_x, y-center_y])
    covariance = (centered*weights[:, None]).T@centered/weights.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    parallel = eigenvectors[:, int(np.argmax(eigenvalues))]
    theta = float(np.degrees(np.arctan2(parallel[1], parallel[0])))
    while theta <= -90.0:
        theta += 180.0
    while theta > 90.0:
        theta -= 180.0
    return OrientationEstimate(
        theta_deg=theta, center_x_um=center_x, center_y_um=center_y,
        threshold_um=threshold, signal_pixels=signal_pixels, confident=True)


def rotated_crop_length_um(width_um: float, height_um: float, theta_deg: float,
                           margin_um: float = CROP_MARGIN_UM) -> float:
    """Square canonical view side that covers the whole rotated strip."""
    theta = np.deg2rad(theta_deg)
    reach = (abs(width_um*np.cos(theta))+abs(height_um*np.sin(theta)))
    perpendicular = (abs(width_um*np.sin(theta))+abs(height_um*np.cos(theta)))
    return float(max(reach, perpendicular)+2.0*margin_um)


def canonical_view_pixels(length_um: float, dx_um: float,
                          cap: int = VIEW_PIXEL_CAP) -> int:
    """Canonical view pixel count, capped like the four-edge annotator."""
    if length_um <= 0 or dx_um <= 0:
        raise ValueError("length and pixel pitch must be positive")
    return int(min(cap, int(np.floor(length_um/dx_um))))
