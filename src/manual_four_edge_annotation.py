"""Pure geometry and table helpers for blinded four-edge annotation."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

ANNOTATION_FIELDS = (
    "left_u_um", "right_u_um", "top_v_um", "bottom_v_um",
    "center_u_um", "center_v_um", "center_x_um", "center_y_um",
    "width_um", "height_um", "state", "timestamp_utc", "comment",
)


def canonical_box_record(*, left_local_um: float, right_local_um: float,
                         top_local_um: float, bottom_local_um: float,
                         display_center_x_um: float,
                         display_center_y_um: float,
                         theta_deg: float) -> dict[str, float | str]:
    """Convert a box in the rotated local view to global canonical/raw coordinates."""
    left, right = sorted((float(left_local_um), float(right_local_um)))
    top, bottom = sorted((float(top_local_um), float(bottom_local_um)))
    local_u = (left+right)/2.0
    local_v = (top+bottom)/2.0
    theta = np.deg2rad(theta_deg)
    cosine, sine = float(np.cos(theta)), float(np.sin(theta))
    origin_u = display_center_x_um*cosine+display_center_y_um*sine
    origin_v = -display_center_x_um*sine+display_center_y_um*cosine
    center_u, center_v = origin_u+local_u, origin_v+local_v
    center_x = center_u*cosine-center_v*sine
    center_y = center_u*sine+center_v*cosine
    return {
        "left_u_um": origin_u+left,
        "right_u_um": origin_u+right,
        "top_v_um": origin_v+top,
        "bottom_v_um": origin_v+bottom,
        "center_u_um": center_u,
        "center_v_um": center_v,
        "center_x_um": center_x,
        "center_y_um": center_y,
        "width_um": right-left,
        "height_um": bottom-top,
        "state": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "comment": "",
    }


def local_extents_from_record(record: dict, *, display_center_x_um: float,
                              display_center_y_um: float,
                              theta_deg: float) -> tuple[float, float, float, float] | None:
    """Restore saved global canonical edges into a local rotated display."""
    required = ("left_u_um", "right_u_um", "top_v_um", "bottom_v_um")
    if any(record.get(name, "") in ("", None) for name in required):
        return None
    theta = np.deg2rad(theta_deg)
    origin_u = display_center_x_um*np.cos(theta)+display_center_y_um*np.sin(theta)
    origin_v = -display_center_x_um*np.sin(theta)+display_center_y_um*np.cos(theta)
    return (
        float(record["left_u_um"])-origin_u,
        float(record["right_u_um"])-origin_u,
        float(record["top_v_um"])-origin_v,
        float(record["bottom_v_um"])-origin_v,
    )


def first_incomplete_index(records: list[dict], annotator: str) -> int:
    field = f"annotator_{annotator.lower()}_state"
    for index, record in enumerate(records):
        if str(record.get(field, "")).strip() not in {"complete", "unusable"}:
            return index
    return max(0, len(records)-1)


def annotation_is_complete(records: list[dict], annotator: str) -> bool:
    field = f"annotator_{annotator.lower()}_state"
    return bool(records) and all(
        str(record.get(field, "")).strip() in {"complete", "unusable"}
        for record in records
    )


def assign_annotation_values(table, index: int, prefix: str,
                             values: dict[str, float | str]) -> None:
    """Assign numeric values safely even when pandas inferred blank columns as str."""
    for field, value in values.items():
        column = prefix+field
        if table[column].dtype != object:
            table[column] = table[column].astype(object)
        table.at[index, column] = value
