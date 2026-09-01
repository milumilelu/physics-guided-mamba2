"""Read-only diagnostics used before any registration is attempted."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .data_contracts import HeightMap

__all__ = [
    "compute_height_diagnostics",
    "compute_invalid_components",
    "build_sample_search_regions",
]


def _finite_quantiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {name: None for name in ("q01", "q05", "q50", "q95", "q99")}
    q = np.quantile(values, [0.01, 0.05, 0.50, 0.95, 0.99])
    return dict(zip(("q01", "q05", "q50", "q95", "q99"), map(float, q)))


def compute_invalid_components(hm: HeightMap) -> dict[str, float | int]:
    invalid = ~hm.valid_mask
    if not invalid.any():
        return {"invalid_component_count": 0,
                "invalid_component_max_area_um2": 0.0}
    labels, count = ndimage.label(invalid, structure=np.ones((3, 3), dtype=int))
    areas_px = np.bincount(labels.ravel())[1:]
    return {
        "invalid_component_count": int(count),
        "invalid_component_max_area_um2": (
            float(areas_px.max() * hm.dx_um * hm.dy_um)
            if areas_px.size else 0.0),
    }


def compute_height_diagnostics(hm: HeightMap,
                               region: tuple[float, float, float, float] | None = None
                               ) -> dict[str, float | int | None]:
    """Robust diagnostics for a full measurement or a physical subregion.

    Region coordinates are centred physical coordinates `(xmin, xmax, ymin,
    ymax)`.  No height threshold is used to declare individual pixels valid.
    """
    x = hm.x_um - hm.width_um / 2.0
    y = hm.y_um - hm.height_um / 2.0
    z = hm.z
    selected = hm.valid_mask
    if region is not None:
        xmin, xmax, ymin, ymax = region
        columns = (x >= xmin) & (x <= xmax)
        rows = (y >= ymin) & (y <= ymax)
        if not rows.any() or not columns.any():
            raise ValueError(f"diagnostic region {region} does not intersect field")
        z = z[np.ix_(rows, columns)]
        selected = selected[np.ix_(rows, columns)]
    values = z[selected]
    out = _finite_quantiles(values)
    q01, q05, q50, q95, q99 = (
        out["q01"], out["q05"], out["q50"], out["q95"], out["q99"])

    # Edge energy is the median absolute first difference on pairs for which
    # both pixels are measured.  It is a diagnostic, not a registration score.
    valid_x = selected[:, 1:] & selected[:, :-1]
    valid_y = selected[1:, :] & selected[:-1, :]
    gx = np.abs(np.diff(z, axis=1))[valid_x]
    gy = np.abs(np.diff(z, axis=0))[valid_y]
    gradients = np.concatenate((gx, gy)) if gx.size or gy.size else np.array([])

    median = float(q50) if q50 is not None else None
    mad = (float(np.median(np.abs(values - median)))
           if values.size and median is not None else None)
    threshold = (median - 3.0 * 1.4826 * mad
                 if mad is not None and mad > 0 else None)
    modified_fraction = (float(np.mean(values < threshold))
                         if threshold is not None else None)
    out.update({
        "selected_valid_pixels": int(values.size),
        "selected_valid_fraction": float(selected.mean()),
        "q50_minus_q05": (float(q50 - q05)
                          if q50 is not None and q05 is not None else None),
        "negative_tail_amplitude": (float(q50 - q01)
                                    if q50 is not None and q01 is not None else None),
        "robust_range_q99_q01": (float(q99 - q01)
                                 if q99 is not None and q01 is not None else None),
        "iqr_proxy_q95_q05": (float(q95 - q05)
                              if q95 is not None and q05 is not None else None),
        "median_absolute_deviation": mad,
        "edge_energy": (float(np.median(gradients))
                        if gradients.size else None),
        "candidate_modified_fraction": modified_fraction,
    })
    return out


def build_sample_search_regions(*, width_um: float, height_um: float,
                                sample_ids: list[int],
                                single_halfwidth_um: float = 50.0,
                                nominal_halfwidth_um: float = 100.0,
                                paired_guard_band_um: float = 20.0
                                ) -> list[dict]:
    """Create ordered centre-search domains without inspecting morphology."""
    y_limit = height_um / 2.0 - nominal_halfwidth_um
    if y_limit <= 0:
        raise ValueError("field is too short to contain a nominal 200 um region")
    if len(sample_ids) == 1:
        half = min(single_halfwidth_um, width_um / 2.0 - nominal_halfwidth_um,
                   y_limit)
        if half <= 0:
            raise ValueError("field is too small for the single-sample search prior")
        return [{
            "sample_id": sample_ids[0], "slot": "single",
            "center_search_x_min_um": -half,
            "center_search_x_max_um": half,
            "center_search_y_min_um": -half,
            "center_search_y_max_um": half,
        }]
    if len(sample_ids) != 2:
        raise ValueError(f"expected one or two samples, got {sample_ids}")

    half_gap = paired_guard_band_um / 2.0
    outer = width_um / 2.0 - nominal_halfwidth_um
    inner = half_gap + nominal_halfwidth_um
    if outer <= inner:
        raise ValueError(
            "field is too narrow for two non-overlapping nominal rectangles")
    return [
        {
            "sample_id": sample_ids[0], "slot": "slot_1",
            "center_search_x_min_um": -outer,
            "center_search_x_max_um": -inner,
            "center_search_y_min_um": -y_limit,
            "center_search_y_max_um": y_limit,
        },
        {
            "sample_id": sample_ids[1], "slot": "slot_2",
            "center_search_x_min_um": inner,
            "center_search_x_max_um": outer,
            "center_search_y_min_um": -y_limit,
            "center_search_y_max_um": y_limit,
        },
    ]
