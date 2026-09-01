"""Conservative 2-D repair of compact downward confocal dropouts.

This module is deliberately independent of the older single-line patent
pipeline.  It accepts only compact components, rejects elongated scan
grooves, preserves the input, and returns an explicit repair mask.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

__all__ = ["ConicalDropoutConfig", "repair_compact_dropouts"]


@dataclass(frozen=True)
class ConicalDropoutConfig:
    background_radius_um: float = 4.0
    seed_sigma: float = 8.0
    grow_sigma: float = 4.0
    minimum_seed_depth_um: float = 0.8
    minimum_grow_depth_um: float = 0.35
    minimum_pixels: int = 3
    maximum_span_um: float = 20.0
    maximum_aspect_ratio: float = 2.8
    minimum_bbox_fill: float = 0.12
    boundary_protection_um: float = 5.0
    ring_width_um: float = 3.0


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    centre = np.median(values)
    return float(1.4826 * np.median(np.abs(values - centre)))


def _local_plane(z: np.ndarray, valid: np.ndarray, component: np.ndarray,
                 ring_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    grown = ndimage.binary_dilation(component, iterations=max(1, ring_px))
    ring = grown & ~component & valid
    rr, cc = np.nonzero(ring)
    if rr.size < 12:
        return None
    target_r, target_c = np.nonzero(component)
    r0, c0 = float(np.mean(target_r)), float(np.mean(target_c))
    scale = max(float(np.ptp(target_r)), float(np.ptp(target_c)), 2.0)

    def design(r: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.column_stack(((c-c0)/scale, (r-r0)/scale,
                                np.ones_like(r, dtype=float)))

    matrix = design(rr.astype(float), cc.astype(float))
    values = z[rr, cc]
    keep = np.ones(values.size, dtype=bool)
    beta = None
    for _ in range(4):
        beta, *_ = np.linalg.lstsq(matrix[keep], values[keep], rcond=None)
        residual = values - matrix @ beta
        sigma = _mad(residual[keep])
        if not np.isfinite(sigma) or sigma <= 1e-12:
            break
        new_keep = np.abs(residual-np.median(residual[keep])) <= 3.5*sigma
        if new_keep.sum() < 8 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if beta is None:
        return None
    prediction = design(target_r.astype(float), target_c.astype(float)) @ beta
    return target_r, target_c, prediction


def repair_compact_dropouts(
        z: np.ndarray, valid: np.ndarray, *, dx_um: float, dy_um: float,
        config: ConicalDropoutConfig | None = None,
        allowed_mask: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Return ``(repaired, repair_mask, components, metrics)``.

    The input is never modified.  Only compact downward components whose
    bounding-box aspect ratio and physical span pass the frozen gates are
    replaced by a robust plane fitted to their surrounding ring.
    """
    cfg = config or ConicalDropoutConfig()
    z = np.asarray(z, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if z.shape != valid.shape or z.ndim != 2:
        raise ValueError("z and valid must be same-shape 2-D arrays")
    if allowed_mask is None:
        allowed = valid.copy()
    else:
        allowed = np.asarray(allowed_mask, dtype=bool) & valid
        if allowed.shape != z.shape:
            raise ValueError("allowed_mask shape mismatch")

    # Median background is used only to detect compact deficits.  Invalid
    # pixels are filled for filtering but remain forbidden candidates.
    finite = z[valid]
    if not finite.size:
        return z.copy(), np.zeros_like(valid), [], {"status": "EMPTY"}
    fill = float(np.median(finite))
    work = np.where(valid, z, fill)
    radius_px = max(2, int(round(cfg.background_radius_um/min(dx_um, dy_um))))
    size = 2*radius_px+1
    background = ndimage.median_filter(work, size=size, mode="nearest")
    deficit = np.maximum(background-work, 0.0)
    noise = _mad((work-background)[valid])
    if not np.isfinite(noise):
        noise = 0.0
    seed_threshold = max(cfg.minimum_seed_depth_um, cfg.seed_sigma*noise)
    grow_threshold = max(cfg.minimum_grow_depth_um, cfg.grow_sigma*noise)

    protection_y = int(np.ceil(cfg.boundary_protection_um/dy_um))
    protection_x = int(np.ceil(cfg.boundary_protection_um/dx_um))
    if protection_y:
        allowed[:protection_y] = False
        allowed[-protection_y:] = False
    if protection_x:
        allowed[:, :protection_x] = False
        allowed[:, -protection_x:] = False
    seed = allowed & (deficit >= seed_threshold)
    grow = allowed & (deficit >= grow_threshold)
    labels, count = ndimage.label(grow, structure=np.ones((3, 3), dtype=int))

    repaired = z.copy()
    repair_mask = np.zeros_like(valid)
    records: list[dict] = []
    ring_px = max(2, int(round(cfg.ring_width_um/min(dx_um, dy_um))))
    for label_id in range(1, count+1):
        component = labels == label_id
        if not np.any(component & seed):
            continue
        rr, cc = np.nonzero(component)
        pixels = int(rr.size)
        row_span = int(np.ptp(rr))+1
        col_span = int(np.ptp(cc))+1
        span_y = row_span*dy_um
        span_x = col_span*dx_um
        aspect = max(span_x, span_y)/max(min(span_x, span_y), 1e-12)
        bbox_fill = pixels/(row_span*col_span)
        if (pixels < cfg.minimum_pixels
                or max(span_x, span_y) > cfg.maximum_span_um
                or aspect > cfg.maximum_aspect_ratio
                or bbox_fill < cfg.minimum_bbox_fill):
            continue
        fit = _local_plane(work, valid, component, ring_px)
        if fit is None:
            continue
        target_r, target_c, prediction = fit
        original = repaired[target_r, target_c]
        replacement = np.maximum(original, prediction)
        changed = replacement > original
        if not np.any(changed):
            continue
        cr, cc2 = target_r[changed], target_c[changed]
        correction = replacement[changed]-original[changed]
        repaired[cr, cc2] = replacement[changed]
        repair_mask[cr, cc2] = True
        records.append({
            "artifact": len(records)+1,
            "pixel_count": int(changed.sum()),
            "row_min": int(rr.min()), "row_max": int(rr.max()),
            "col_min": int(cc.min()), "col_max": int(cc.max()),
            "span_x_um": float(span_x), "span_y_um": float(span_y),
            "aspect_ratio": float(aspect), "bbox_fill": float(bbox_fill),
            "max_correction_um": float(np.max(correction)),
            "mean_correction_um": float(np.mean(correction)),
        })

    repaired[~valid] = np.nan
    metrics = {
        "status": "PASS",
        "noise_mad_um": noise,
        "seed_threshold_um": seed_threshold,
        "grow_threshold_um": grow_threshold,
        "candidate_components": int(count),
        "accepted_components": len(records),
        "repaired_pixels": int(repair_mask.sum()),
        "repaired_fraction": float(repair_mask.mean()),
        "config": asdict(cfg),
    }
    return repaired, repair_mask, records, metrics
