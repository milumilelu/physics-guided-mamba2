"""Mask-aware physical-coordinate resampling for canonical height maps."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .data_contracts import HeightMap

__all__ = ["canonical_grid", "resample_to_canonical", "resample_center_crop"]


def canonical_grid(length_um: float, pixels: int) -> tuple[np.ndarray, float]:
    if length_um <= 0 or pixels < 2:
        raise ValueError("canonical grid requires positive length and >=2 pixels")
    pixel_um = float(length_um/pixels)
    coordinates = ((np.arange(pixels, dtype=float)+0.5)*pixel_um
                   - length_um/2.0)
    return coordinates, pixel_um


def _normalized_map(z: np.ndarray, mask: np.ndarray,
                    row_coordinates: np.ndarray, column_coordinates: np.ndarray,
                    *, order: int, minimum_mask_weight: float) -> tuple[np.ndarray, np.ndarray]:
    numerator = ndimage.map_coordinates(
        np.where(mask, z, 0.0), [row_coordinates, column_coordinates],
        order=order, mode="constant", cval=0.0, prefilter=order > 1)
    weight = ndimage.map_coordinates(
        mask.astype(float), [row_coordinates, column_coordinates],
        order=order, mode="constant", cval=0.0, prefilter=order > 1)
    valid = weight >= minimum_mask_weight
    output = np.full(weight.shape, np.nan, dtype=float)
    output[valid] = numerator[valid]/weight[valid]
    return output, valid


def resample_to_canonical(
        hm: HeightMap, *, plane: tuple[float, float, float],
        center_x_um: float, center_y_um: float, theta_deg: float,
        length_um: float, pixels: int, minimum_mask_weight: float = 0.99,
        order: int = 1, metadata: dict | None = None) -> HeightMap:
    """Rotate/translate raw data onto a centred canonical square grid."""
    axis, pixel = canonical_grid(length_um, pixels)
    u, v = np.meshgrid(axis, axis)
    theta = np.deg2rad(theta_deg)
    raw_x_centered = (center_x_um + u*np.cos(theta)-v*np.sin(theta))
    raw_y_centered = (center_y_um + u*np.sin(theta)+v*np.cos(theta))
    raw_x_absolute = raw_x_centered + hm.width_um/2.0
    raw_y_absolute = raw_y_centered + hm.height_um/2.0
    columns = (raw_x_absolute-hm.x_um[0])/hm.dx_um
    rows = (raw_y_absolute-hm.y_um[0])/hm.dy_um

    x_centered = hm.x_um-hm.width_um/2.0
    y_centered = hm.y_um-hm.height_um/2.0
    a, b, c = plane
    coarse = hm.z-(a*x_centered[None, :]+b*y_centered[:, None]+c)
    z, valid = _normalized_map(
        coarse, hm.valid_mask, rows, columns, order=order,
        minimum_mask_weight=minimum_mask_weight)
    details = dict(metadata or {})
    details.update({
        "object": "H_reg_coarse", "center_x_um": center_x_um,
        "center_y_um": center_y_um, "theta_deg": theta_deg,
        "interpolation_order": order,
        "minimum_mask_weight": minimum_mask_weight,
        "normalized_interpolation": True,
    })
    return HeightMap(z=z, valid_mask=valid, dx_um=pixel, dy_um=pixel,
                     x_um=axis, y_um=axis, metadata=details)


def resample_center_crop(hm: HeightMap, *, length_um: float, pixels: int,
                         minimum_mask_weight: float = 0.99) -> HeightMap:
    """Resample the centred H_reg to an exact-size central H_200 grid."""
    axis, pixel = canonical_grid(length_um, pixels)
    xx, yy = np.meshgrid(axis, axis)
    columns = (xx-hm.x_um[0])/hm.dx_um
    rows = (yy-hm.y_um[0])/hm.dy_um
    z, valid = _normalized_map(
        hm.z, hm.valid_mask, rows, columns, order=1,
        minimum_mask_weight=minimum_mask_weight)
    metadata = dict(hm.metadata)
    metadata.update({"object": "H_200", "source_object": "H_reg",
                     "length_um": length_um})
    return HeightMap(z=z, valid_mask=valid, dx_um=pixel, dy_um=pixel,
                     x_um=axis, y_um=axis, metadata=metadata)
