"""Common-field geometry for registered, non-extrapolated square canvases."""

from __future__ import annotations

import math

__all__ = ["available_centered_square_um", "resolve_registered_grid"]


def available_centered_square_um(*, fov_width_um: float, fov_height_um: float,
                                 center_x_um: float, center_y_um: float,
                                 theta_deg: float) -> float:
    """Largest centred canonical square wholly inside a rotated raw FOV."""
    x_margin = min(fov_width_um/2.0-center_x_um,
                   center_x_um+fov_width_um/2.0)
    y_margin = min(fov_height_um/2.0-center_y_um,
                   center_y_um+fov_height_um/2.0)
    if x_margin <= 0 or y_margin <= 0:
        return 0.0
    theta = math.radians(theta_deg)
    projection = abs(math.cos(theta))+abs(math.sin(theta))
    return 2.0*min(x_margin, y_margin)/projection


def resolve_registered_grid(*, common_fov_um: float, preferred_size_um: float,
                            minimum_size_um: float,
                            coarsest_input_pixel_um: float) -> dict:
    if common_fov_um < minimum_size_um:
        return {
            "status": "STOP", "registered_fov_um": None,
            "grid_pixels": None, "pixel_um": None,
            "warning": "common FOV is below configured minimum",
        }
    length = min(common_fov_um, preferred_size_um)
    pixels = math.floor(length/coarsest_input_pixel_um)
    pixel = length/pixels
    return {
        "status": "PASS", "registered_fov_um": length,
        "grid_pixels": pixels, "pixel_um": pixel, "warning": "",
    }
