#!/usr/bin/env python3
"""Extract pilot single-line geometry directly from a KEYENCE CAG container.

The script never modifies the source CAG or DOE file.  It joins CAG Path 1..120
to the DOE column ``加工顺序``, decodes the embedded VK4 height calibration, fits
an asymmetric robust reference plane, and extracts continuous-line or discrete-
crater geometry with one fixed parameter set.

This is a research pilot, not a substitute for manual review.  Every processed
group receives a diagnostic PNG and explicit quality-control fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import xml.etree.ElementTree as ET
import zipfile
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


HEIGHT_KEY = "4d137b4a-bf22-49d5-96a8-9b07b3fc5d02"
LIGHT_KEY = "e4eec84e-b9fd-4898-8a44-79d6eae57fb4"
ERROR_KEY = "2c7fd1a8-b42a-41ff-9baa-56760304e826"
MEASURE_CONDITION_KEY = "573c2040-d262-4dd4-a2c9-71ad6d004f11"
VK4_KEYS = [
    "meas_conds", "color_peak", "color_light", "light", "unknown_4",
    "unknown_5", "height", "unknown_7", "unknown_8", "color_peak_thumb",
    "color_thumb", "light_thumb", "height_thumb", "assembly_info",
    "line_measure", "line_thickness", "string_data", "reserved",
]


@dataclass(frozen=True)
class Config:
    power_w: float = 5.333
    line_length_um: float = 200.0
    threshold_k: float = 4.0
    central_fraction: float = 0.70
    profile_step_um: float = 1.0
    profile_strip_half_width_um: float = 0.65
    plane_negative_clip_sigma: float = 2.5
    plane_positive_clip_sigma: float = 4.0
    plane_max_iter: int = 12
    cone_repair_enabled: bool = True
    cone_half_window_px: int = 12
    cone_seed_sigma: float = 6.0
    cone_grow_sigma: float = 1.5
    cone_min_seed_depth_um: float = 0.8
    cone_max_component_span_px: int = 36
    cone_cut_corridor_padding_px: int = 3
    stable_core_fraction: float = 0.40
    stable_min_fraction: float = 0.35
    stable_rolling_window: int = 9
    stable_depth_sigma: float = 3.0
    stable_depth_relative_tolerance: float = 0.12
    stable_depth_min_tolerance_um: float = 0.25
    min_component_pixels: int = 5
    min_profile_points: int = 8


def mad_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def robust_median_sigma(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.median(arr)), mad_scale(arr)


def parse_groups(text: str) -> list[int]:
    groups: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(x.strip()) for x in token.split("-", 1))
            groups.extend(range(start, end + 1))
        else:
            groups.append(int(token))
    return sorted(set(groups))


def choose_pilot_groups(design: pd.DataFrame, count: int = 15) -> list[int]:
    """Deterministic maximin selection in the four-factor design space."""
    cols = ["脉宽_fs", "频率_kHz", "重复扫描次数", "速度_mm/s"]
    x = design[cols].astype(float).to_numpy()
    lo = x.min(axis=0)
    span = np.where(x.max(axis=0) > lo, x.max(axis=0) - lo, 1.0)
    z = (x - lo) / span
    ids = design["加工顺序"].astype(int).to_numpy()
    selected = [int(np.where(ids == 60)[0][0])] if 60 in ids else [0]
    while len(selected) < min(count, len(design)):
        remaining = [i for i in range(len(design)) if i not in selected]
        dmin = []
        for idx in remaining:
            distance = np.sqrt(((z[selected] - z[idx]) ** 2).sum(axis=1))
            dmin.append(float(distance.min()))
        selected.append(remaining[int(np.argmax(dmin))])
    return sorted(int(ids[i]) for i in selected)


class CagReader:
    def __init__(self, path: Path):
        self.path = path
        self.archive = zipfile.ZipFile(path)
        self.entries = self.archive.infolist()
        self._direct: dict[int, dict[str, zipfile.ZipInfo]] = {}
        self._vk4: dict[int, zipfile.ZipInfo] = {}
        self._timestamps: dict[int, str] = {}
        self._original_names: dict[int, str] = {}
        self._index()

    def close(self) -> None:
        self.archive.close()

    def __enter__(self) -> "CagReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _index(self) -> None:
        measurement_xml: bytes | None = None
        for info in self.entries:
            parts = info.filename.split("/")
            if len(parts) == 5 and parts[1].isdigit() and parts[3] in {
                HEIGHT_KEY, LIGHT_KEY, ERROR_KEY, MEASURE_CONDITION_KEY,
            } and info.file_size > 1:
                self._direct.setdefault(int(parts[1]), {})[parts[3]] = info
            if info.file_size == 568500 and len(parts) >= 3 and parts[1].isdigit():
                with self.archive.open(info) as stream:
                    if stream.read(4) == b"VK4_":
                        self._vk4[int(parts[1])] = info
            if info.file_size > 1000:
                with self.archive.open(info) as stream:
                    head = stream.read(128)
                if b"MeasurementDataMap" in head:
                    measurement_xml = self.archive.read(info)
        if measurement_xml is None:
            raise ValueError("CAG contains no MeasurementDataMap")
        root = ET.fromstring(measurement_xml.decode("utf-8-sig"))
        for item in root.findall("MeasurementData"):
            group = int(item.findtext("Path", "0"))
            name = item.findtext("OriginalFileName", "")
            self._original_names[group] = name
            stem = Path(name).stem
            stamp = stem.removeprefix("MeasureData")
            if len(stamp) == 14 and stamp.isdigit():
                self._timestamps[group] = (
                    f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]} "
                    f"{stamp[8:10]}:{stamp[10:12]}:{stamp[12:14]}"
                )
            else:
                self._timestamps[group] = ""
        expected = set(range(1, 121))
        if set(self._vk4) != expected or set(self._direct) != expected:
            raise ValueError(
                f"Expected complete groups 1..120; VK4={len(self._vk4)}, direct={len(self._direct)}"
            )

    def read_group(self, group: int) -> dict[str, object]:
        entries = self._direct[group]
        vk4 = self.archive.read(self._vk4[group])
        offsets = dict(zip(VK4_KEYS, struct.unpack_from("<18I", vk4, 12)))
        measure_offset = offsets["meas_conds"]
        # Positions follow the documented VK4 measurement-condition layout.
        year, month, day, hour, minute, second = struct.unpack_from(
            "<6I", vk4, measure_offset + 4
        )
        x_pitch_pm, y_pitch_pm, z_step_pm = struct.unpack_from(
            "<3I", vk4, measure_offset + 42 * 4
        )
        height_offset = offsets["height"]
        width, height, bit_depth, compression, data_bytes = struct.unpack_from(
            "<5I", vk4, height_offset
        )
        # KEYENCE writes a non-zero implementation marker in the field commonly
        # labelled "compression" by public VK4 readers.  For this CAG series the
        # authoritative checks are 32-bit samples, the declared byte count, and
        # the direct CAG block length; all 120 direct blocks were independently
        # verified against their embedded VK4 samples.
        if bit_depth != 32 or data_bytes != width * height * 4:
            raise ValueError(
                f"Unsupported height layout in group {group}: "
                f"bit={bit_depth}, marker={compression}, bytes={data_bytes}"
            )
        direct_height = np.frombuffer(
            self.archive.read(entries[HEIGHT_KEY]), dtype="<u4"
        ).reshape(height, width)
        direct_light = np.frombuffer(
            self.archive.read(entries[LIGHT_KEY]), dtype="<u2"
        ).reshape(height, width)
        error_mask = np.frombuffer(
            self.archive.read(entries[ERROR_KEY]), dtype=np.uint8
        ).reshape(height, width)
        z_um = direct_height.astype(np.float64) * (z_step_pm * 1e-6)
        return {
            "group": group,
            "z_um": z_um,
            "light": direct_light,
            "valid": error_mask == 0,
            "dx_um": x_pitch_pm * 1e-6,
            "dy_um": y_pitch_pm * 1e-6,
            "z_step_um": z_step_pm * 1e-6,
            "timestamp": self._timestamps.get(group, ""),
            "original_filename": self._original_names.get(group, ""),
            "vk4_timestamp": f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}",
            "data_bytes": data_bytes,
        }


def fit_reference_plane(
    z: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    rows, cols = z.shape
    y, x = np.indices(z.shape, dtype=float)
    x = (x - (cols - 1) / 2.0) * dx
    y = (y - (rows - 1) / 2.0) * dy
    flat_valid = valid.ravel() & np.isfinite(z.ravel())
    a = np.column_stack([x.ravel(), y.ravel(), np.ones(z.size)])
    target = z.ravel()
    inliers = flat_valid.copy()
    beta = np.linalg.lstsq(a[inliers], target[inliers], rcond=None)[0]
    for _ in range(cfg.plane_max_iter):
        residual = target - a @ beta
        center = float(np.median(residual[inliers]))
        scale = mad_scale(residual[inliers])
        if not np.isfinite(scale) or scale < 1e-9:
            break
        new_inliers = flat_valid & (
            residual - center >= -cfg.plane_negative_clip_sigma * scale
        ) & (
            residual - center <= cfg.plane_positive_clip_sigma * scale
        )
        if new_inliers.sum() < max(100, int(0.15 * flat_valid.sum())):
            break
        new_beta = np.linalg.lstsq(a[new_inliers], target[new_inliers], rcond=None)[0]
        if np.array_equal(new_inliers, inliers) and np.max(np.abs(new_beta - beta)) < 1e-10:
            beta = new_beta
            inliers = new_inliers
            break
        beta = new_beta
        inliers = new_inliers
    plane = (a @ beta).reshape(z.shape)
    depth = plane - z
    ref_depth = depth.ravel()[inliers]
    sigma_ref = mad_scale(ref_depth)
    rmse = float(np.sqrt(np.mean(ref_depth**2)))
    return plane, depth, sigma_ref, rmse, inliers.reshape(z.shape)


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ]
    for row, col in np.argwhere(mask):
        if seen[row, col]:
            continue
        queue = deque([(int(row), int(col))])
        seen[row, col] = True
        points: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            points.append((r, c))
            for dr, dc in neighbors:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        components.append(np.asarray(points, dtype=int))
    return components


def _max_filter_rows(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, ((0, 0), (radius, radius)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, 2 * radius + 1, axis=1
    )
    return np.max(windows, axis=-1)


def _min_filter_rows(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, ((0, 0), (radius, radius)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, 2 * radius + 1, axis=1
    )
    return np.min(windows, axis=-1)


def _dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        source = result
        expanded = source.copy()
        expanded[1:, :] |= source[:-1, :]
        expanded[:-1, :] |= source[1:, :]
        expanded[:, 1:] |= source[:, :-1]
        expanded[:, :-1] |= source[:, 1:]
        expanded[1:, 1:] |= source[:-1, :-1]
        expanded[1:, :-1] |= source[:-1, 1:]
        expanded[:-1, 1:] |= source[1:, :-1]
        expanded[:-1, :-1] |= source[1:, 1:]
        result = expanded
    return result


def _fit_local_quadratic_base(
    z: np.ndarray,
    component_mask: np.ndarray,
    excluded_mask: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray | None:
    ring = _dilate_mask(component_mask, 4) & ~excluded_mask & valid
    if int(ring.sum()) < 18:
        ring = _dilate_mask(component_mask, 8) & ~excluded_mask & valid
    rr, cc = np.nonzero(ring)
    if len(rr) < 18:
        return None
    target_rr, target_cc = np.nonzero(component_mask)
    center_r = float(np.mean(target_rr))
    center_c = float(np.mean(target_cc))
    scale = max(float(np.ptp(target_rr)), float(np.ptp(target_cc)), 4.0)

    def design(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        y = (rows - center_r) / scale
        x = (cols - center_c) / scale
        return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])

    matrix = design(rr.astype(float), cc.astype(float))
    values = z[rr, cc]
    inliers = np.ones(len(values), dtype=bool)
    beta: np.ndarray | None = None
    for _ in range(5):
        if int(inliers.sum()) < 12:
            break
        beta, *_ = np.linalg.lstsq(matrix[inliers], values[inliers], rcond=None)
        residual = values - matrix @ beta
        center = float(np.median(residual[inliers]))
        sigma = mad_scale(residual[inliers])
        if not np.isfinite(sigma) or sigma <= 1e-12:
            break
        new_inliers = np.abs(residual - center) <= 3.5 * sigma
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
    if beta is None:
        return None
    prediction = design(target_rr.astype(float), target_cc.astype(float)) @ beta
    lower = float(np.min(values[inliers]))
    upper = float(np.max(values[inliers]))
    return np.clip(prediction, lower, upper)


def repair_conical_dropouts(
    z: np.ndarray,
    valid: np.ndarray,
    cfg: Config,
    allowed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, float]]:
    """Replace narrow, compact downward dropouts with their local base surface.

    The laser line runs along image columns in this CAG series.  A 1-D grayscale
    closing along each scan row detects narrow valleys without flattening the much
    broader line cross-section.  Accepted regions are covered by a robust local
    quadratic fitted to the surrounding base ring, not by the closing surface.
    """
    empty = pd.DataFrame(
        columns=[
            "artifact", "pixel_count", "row_min", "row_max", "col_min", "col_max",
            "centroid_row", "centroid_col", "max_correction_um", "mean_correction_um",
        ]
    )
    if not cfg.cone_repair_enabled:
        return z.copy(), np.zeros_like(valid), empty, {
            "noise_second_difference_um": float("nan"),
            "seed_threshold_um": float("nan"),
            "grow_threshold_um": float("nan"),
        }

    radius = cfg.cone_half_window_px
    if radius < 2 or z.shape[1] <= 2 * radius + 1:
        raise ValueError("Invalid cone repair window for height-grid width")

    second_difference = np.diff(z, n=2, axis=1)
    noise_second = mad_scale(second_difference[np.isfinite(second_difference)])
    seed_threshold = max(
        cfg.cone_min_seed_depth_um,
        cfg.cone_seed_sigma * noise_second,
    )
    grow_threshold = max(0.1, cfg.cone_grow_sigma * noise_second)

    closed = _min_filter_rows(_max_filter_rows(z, radius), radius)
    deficit = np.maximum(closed - z, 0.0)
    deficit[~valid] = 0.0
    seed_mask = deficit >= seed_threshold
    grow_mask = deficit >= grow_threshold
    if allowed_mask is not None:
        if allowed_mask.shape != z.shape:
            raise ValueError("Cone-repair allowed mask has the wrong shape")
        seed_mask &= allowed_mask
        grow_mask &= allowed_mask

    accepted: list[np.ndarray] = []
    records: list[dict[str, float | int]] = []
    rows, cols = z.shape
    for component in connected_components(grow_mask):
        rr = component[:, 0]
        cc = component[:, 1]
        if not np.any(seed_mask[rr, cc]):
            continue
        row_min, row_max = int(rr.min()), int(rr.max())
        col_min, col_max = int(cc.min()), int(cc.max())
        row_span = row_max - row_min + 1
        col_span = col_max - col_min + 1
        # The base estimate uses left/right shoulders, so touching a Y boundary
        # is acceptable.  Touching an X boundary is not: one shoulder is absent.
        if col_min == 0 or col_max == cols - 1:
            continue
        per_row_spans = [
            int(np.ptp(cc[rr == row])) + 1 for row in np.unique(rr)
        ]
        if max(per_row_spans) > cfg.cone_max_component_span_px:
            continue
        accepted.append(component)
        records.append({
            "artifact": len(records) + 1,
            "pixel_count": len(component),
            "row_min": row_min,
            "row_max": row_max,
            "col_min": col_min,
            "col_max": col_max,
            "centroid_row": float(np.mean(rr)),
            "centroid_col": float(np.mean(cc)),
            "max_correction_um": float("nan"),
            "mean_correction_um": float("nan"),
        })

    corrected = z.copy()
    repair_mask = np.zeros_like(valid, dtype=bool)
    for index, component in enumerate(accepted):
        component_mask = np.zeros_like(valid, dtype=bool)
        component_mask[component[:, 0], component[:, 1]] = True
        prediction = _fit_local_quadratic_base(
            z, component_mask, grow_mask, valid
        )
        rr, cc = component[:, 0], component[:, 1]
        if prediction is None:
            prediction = closed[rr, cc]
        replacement = np.maximum(z[rr, cc], prediction)
        changed = replacement > z[rr, cc]
        corrected[rr[changed], cc[changed]] = replacement[changed]
        repair_mask[rr[changed], cc[changed]] = True
        correction = replacement[changed] - z[rr[changed], cc[changed]]
        records[index]["pixel_count"] = int(changed.sum())
        records[index]["max_correction_um"] = (
            float(np.max(correction)) if correction.size else 0.0
        )
        records[index]["mean_correction_um"] = (
            float(np.mean(correction)) if correction.size else 0.0
        )
    metrics = {
        "noise_second_difference_um": noise_second,
        "seed_threshold_um": seed_threshold,
        "grow_threshold_um": grow_threshold,
    }
    return corrected, repair_mask, pd.DataFrame.from_records(records, columns=empty.columns), metrics


def select_signal_components(
    components: list[np.ndarray], depth: np.ndarray, threshold: float, cfg: Config
) -> list[np.ndarray]:
    eligible: list[tuple[np.ndarray, int, float, float]] = []
    for comp in components:
        if len(comp) < cfg.min_component_pixels:
            continue
        values = depth[comp[:, 0], comp[:, 1]]
        eligible.append((comp, len(comp), float(values.max()), float(values.sum())))
    if not eligible:
        return []
    largest = max(item[1] for item in eligible)
    deepest = max(item[2] for item in eligible)
    kept = [
        item[0] for item in eligible
        if item[1] >= max(cfg.min_component_pixels, int(math.ceil(0.03 * largest)))
        or item[2] >= max(2.0 * threshold, 0.30 * deepest)
    ]
    return kept


def locate_cut_corridor(
    z: np.ndarray,
    valid: np.ndarray,
    dx: float,
    dy: float,
    cfg: Config,
) -> tuple[np.ndarray, dict[str, float]]:
    """Locate the planned laser-cut band before any cone repair is applied."""
    _, coarse_depth, sigma_ref, _, _ = fit_reference_plane(z, valid, dx, dy, cfg)
    threshold = cfg.threshold_k * sigma_ref
    components = [
        comp for comp in connected_components(valid & (coarse_depth > threshold))
        if len(comp) >= cfg.min_component_pixels
    ]
    if not components:
        raise ValueError("Cannot locate a coarse cut corridor")

    def longitudinal_score(component: np.ndarray) -> float:
        col_span = int(np.ptp(component[:, 1])) + 1
        return float(col_span * math.sqrt(len(component)))

    dominant = max(components, key=longitudinal_score)
    rr, cc = dominant[:, 0], dominant[:, 1]
    pad = cfg.cone_cut_corridor_padding_px
    row_low = max(0, int(rr.min()) - pad)
    row_high = min(z.shape[0] - 1, int(rr.max()) + pad)
    dominant_span = int(np.ptp(cc)) + 1
    nominal_pixels = cfg.line_length_um / dx
    if dominant_span >= 0.35 * nominal_pixels:
        center_col = 0.5 * (float(cc.min()) + float(cc.max()))
    else:
        center_col = (z.shape[1] - 1) / 2.0
    half_nominal = 0.5 * nominal_pixels + pad
    col_low = max(0, int(math.floor(center_col - half_nominal)))
    col_high = min(z.shape[1] - 1, int(math.ceil(center_col + half_nominal)))

    corridor = np.zeros_like(valid, dtype=bool)
    corridor[row_low:row_high + 1, col_low:col_high + 1] = True
    corridor &= valid
    return corridor, {
        "coarse_sigma_ref_um": sigma_ref,
        "coarse_threshold_um": threshold,
        "row_low": row_low,
        "row_high": row_high,
        "col_low": col_low,
        "col_high": col_high,
        "corridor_pixel_ratio": float(corridor.mean()),
        "dominant_component_col_span_px": dominant_span,
    }


def weighted_line_axes(
    selected_mask: np.ndarray, depth: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = depth.shape
    rr, cc = np.nonzero(selected_mask)
    if len(rr) < 3:
        raise ValueError("Too few ablation pixels for line-axis estimation")
    x = (cc - (cols - 1) / 2.0) * dx
    y = (rr - (rows - 1) / 2.0) * dy
    points = np.column_stack([x, y])
    weights = np.maximum(depth[rr, cc], 1e-12)
    center = np.average(points, axis=0, weights=weights)
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    parallel = eigenvectors[:, int(np.argmax(eigenvalues))]
    if parallel[0] < 0:
        parallel = -parallel
    perpendicular = np.array([-parallel[1], parallel[0]])
    all_y, all_x = np.indices(depth.shape, dtype=float)
    all_points = np.column_stack([
        ((all_x.ravel() - (cols - 1) / 2.0) * dx),
        ((all_y.ravel() - (rows - 1) / 2.0) * dy),
    ])
    local = all_points - center
    s = (local @ parallel).reshape(depth.shape)
    xi = (local @ perpendicular).reshape(depth.shape)
    return center, parallel, perpendicular, s, xi


def threshold_interval(xi: np.ndarray, values: np.ndarray, threshold: float) -> tuple[int, int] | None:
    above = np.isfinite(values) & (values > threshold)
    if not above.any():
        return None
    indices = np.flatnonzero(above)
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for idx in indices[1:]:
        idx = int(idx)
        if idx > previous + 1:
            runs.append((start, previous))
            start = idx
        previous = idx
    runs.append((start, previous))
    center_index = int(np.nanargmin(np.abs(xi)))
    containing = [run for run in runs if run[0] <= center_index <= run[1]]
    if containing:
        return containing[0]
    return max(runs, key=lambda run: float(np.nansum(values[run[0]:run[1] + 1])))


def crossing_position(
    x0: float, y0: float, x1: float, y1: float, threshold: float
) -> float:
    if not np.isfinite(y0) or not np.isfinite(y1) or abs(y1 - y0) < 1e-15:
        return float(x1)
    fraction = (threshold - y0) / (y1 - y0)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return float(x0 + fraction * (x1 - x0))


def profile_metrics(
    xi: np.ndarray, values: np.ndarray, threshold: float
) -> dict[str, float] | None:
    order = np.argsort(xi)
    xi = np.asarray(xi[order], dtype=float)
    values = np.asarray(values[order], dtype=float)
    finite = np.isfinite(xi) & np.isfinite(values)
    xi, values = xi[finite], values[finite]
    if len(xi) < 4:
        return None
    interval = threshold_interval(xi, values, threshold)
    if interval is None:
        return None
    left, right = interval
    left_x = float(xi[left])
    right_x = float(xi[right])
    left_clipped = left == 0
    right_clipped = right == len(xi) - 1
    if left > 0:
        left_x = crossing_position(
            float(xi[left - 1]), float(values[left - 1]),
            float(xi[left]), float(values[left]), threshold,
        )
    if right < len(xi) - 1:
        right_x = crossing_position(
            float(xi[right]), float(values[right]),
            float(xi[right + 1]), float(values[right + 1]), threshold,
        )
    segment_x = xi[left:right + 1]
    segment_d = np.maximum(values[left:right + 1], 0.0)
    if segment_x.size < 2 or right_x <= left_x:
        return None
    width = right_x - left_x
    depth_p95 = float(np.percentile(segment_d, 95))
    area = float(np.trapezoid(segment_d, segment_x))
    return {
        "width_um": float(width),
        "depth_p95_um": depth_p95,
        "area_um2": area,
        "left_um": left_x,
        "right_um": right_x,
        "edge_clipped": float(left_clipped or right_clipped),
    }


def extract_profiles(
    depth: np.ndarray,
    valid: np.ndarray,
    s: np.ndarray,
    xi: np.ndarray,
    threshold: float,
    cfg: Config,
    longitudinal_fraction: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fraction = cfg.central_fraction if longitudinal_fraction is None else longitudinal_fraction
    half_valid = cfg.line_length_um * fraction / 2.0
    targets = np.arange(-half_valid, half_valid + 0.5 * cfg.profile_step_um, cfg.profile_step_um)
    metrics: list[dict[str, float]] = []
    profiles: list[pd.DataFrame] = []
    xi_grid = np.linspace(float(np.nanmin(xi)), float(np.nanmax(xi)), depth.shape[0])
    for target in targets:
        strip = valid & (np.abs(s - target) <= cfg.profile_strip_half_width_um)
        if strip.sum() < cfg.min_profile_points:
            continue
        strip_xi = xi[strip]
        strip_depth = depth[strip]
        bins = np.linspace(xi_grid.min(), xi_grid.max(), len(xi_grid) + 1)
        index = np.clip(np.digitize(strip_xi, bins) - 1, 0, len(xi_grid) - 1)
        aggregated = np.full(len(xi_grid), np.nan)
        for idx in np.unique(index):
            aggregated[idx] = float(np.median(strip_depth[index == idx]))
        finite = np.isfinite(aggregated)
        if finite.sum() < cfg.min_profile_points:
            continue
        aggregated = np.interp(xi_grid, xi_grid[finite], aggregated[finite])
        metric = profile_metrics(xi_grid, aggregated, threshold)
        if metric is None:
            continue
        metric["s_um"] = float(target)
        metrics.append(metric)
        profiles.append(pd.DataFrame({
            "s_um": target,
            "xi_um": xi_grid,
            "depth_um": aggregated,
        }))
    metrics_df = pd.DataFrame(metrics)
    profiles_df = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
    return metrics_df, profiles_df


def select_stable_profile_region(
    metrics: pd.DataFrame,
    raw_profiles: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    """Keep the central contiguous depth plateau and reject both transition ends."""
    if metrics.empty:
        return metrics, raw_profiles, {
            "status": "failed_no_profiles",
            "s_start_um": float("nan"),
            "s_end_um": float("nan"),
            "effective_length_um": 0.0,
            "reference_depth_um": float("nan"),
            "depth_tolerance_um": float("nan"),
        }
    ordered = metrics.sort_values("s_um").reset_index(drop=True)
    s_values = ordered["s_um"].to_numpy(dtype=float)
    depths = ordered["depth_p95_um"].to_numpy(dtype=float)
    core_half = cfg.line_length_um * cfg.stable_core_fraction / 2.0
    core = np.abs(s_values) <= core_half
    if int(core.sum()) < 5:
        core = np.argsort(np.abs(s_values))[: min(9, len(s_values))]
        core_mask = np.zeros(len(s_values), dtype=bool)
        core_mask[core] = True
        core = core_mask
    reference = float(np.median(depths[core]))
    core_sigma = mad_scale(depths[core])
    tolerance = max(
        cfg.stable_depth_min_tolerance_um,
        cfg.stable_depth_sigma * core_sigma,
        cfg.stable_depth_relative_tolerance * max(reference, 0.0),
    )
    window = max(3, int(cfg.stable_rolling_window) | 1)
    smooth = (
        pd.Series(depths)
        .rolling(window=window, center=True, min_periods=max(2, window // 3))
        .median()
        .to_numpy(dtype=float)
    )
    stable = np.abs(smooth - reference) <= tolerance

    # Bridge at most two isolated profile failures inside an otherwise stable run.
    index = 0
    while index < len(stable):
        if stable[index]:
            index += 1
            continue
        end = index
        while end < len(stable) and not stable[end]:
            end += 1
        if index > 0 and end < len(stable) and end - index <= 2:
            stable[index:end] = True
        index = end

    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(stable):
        if not stable[index]:
            index += 1
            continue
        end = index
        while (
            end + 1 < len(stable)
            and stable[end + 1]
            and s_values[end + 1] - s_values[end] <= 1.5 * cfg.profile_step_um
        ):
            end += 1
        runs.append((index, end))
        index = end + 1

    minimum_length = cfg.stable_min_fraction * cfg.line_length_um
    status = "adaptive"
    eligible = [
        run for run in runs
        if s_values[run[1]] - s_values[run[0]] >= minimum_length
    ]
    if eligible:
        start, end = min(
            eligible,
            key=lambda run: (
                0.0 if s_values[run[0]] <= 0.0 <= s_values[run[1]]
                else min(abs(s_values[run[0]]), abs(s_values[run[1]])),
                -(s_values[run[1]] - s_values[run[0]]),
            ),
        )
        s_start, s_end = float(s_values[start]), float(s_values[end])
    else:
        status = "fallback_fixed_central_fraction"
        half = cfg.line_length_um * cfg.central_fraction / 2.0
        s_start, s_end = -half, half

    kept_metrics = ordered.loc[
        (ordered["s_um"] >= s_start) & (ordered["s_um"] <= s_end)
    ].copy()
    kept_raw = raw_profiles.loc[
        (raw_profiles["s_um"] >= s_start) & (raw_profiles["s_um"] <= s_end)
    ].copy() if not raw_profiles.empty else raw_profiles
    return kept_metrics, kept_raw, {
        "status": status,
        "s_start_um": s_start,
        "s_end_um": s_end,
        "effective_length_um": max(0.0, s_end - s_start),
        "reference_depth_um": reference,
        "depth_sigma_core_um": core_sigma,
        "depth_tolerance_um": tolerance,
    }


def median_standard_profile(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty:
        return pd.DataFrame(columns=["xi_um", "median_depth_um"])
    pivot = profiles.pivot_table(index="xi_um", columns="s_um", values="depth_um", aggfunc="median")
    return pd.DataFrame({
        "xi_um": pivot.index.to_numpy(dtype=float),
        "median_depth_um": np.nanmedian(pivot.to_numpy(dtype=float), axis=1),
    })


def component_metrics(
    components: list[np.ndarray], depth: np.ndarray, s: np.ndarray, xi: np.ndarray, dx: float, dy: float
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for number, comp in enumerate(components, start=1):
        rr, cc = comp[:, 0], comp[:, 1]
        values = np.maximum(depth[rr, cc], 0.0)
        rows.append({
            "component": number,
            "area_pixels": len(comp),
            "center_s_um": float(np.average(s[rr, cc], weights=np.maximum(values, 1e-12))),
            "center_xi_um": float(np.average(xi[rr, cc], weights=np.maximum(values, 1e-12))),
            "width_perpendicular_um": float(np.ptp(xi[rr, cc]) + min(dx, dy)),
            "length_parallel_um": float(np.ptp(s[rr, cc]) + min(dx, dy)),
            "depth_p95_um": float(np.percentile(values, 95)),
            "depth_max_um": float(values.max()),
            "volume_um3": float(values.sum() * dx * dy),
        })
    return pd.DataFrame(rows).sort_values("center_s_um").reset_index(drop=True)


def occupancy_continuity(
    selected_mask: np.ndarray, s: np.ndarray, dx: float, line_length_um: float
) -> tuple[float, float]:
    values = s[selected_mask]
    if values.size == 0:
        return 0.0, 0.0
    low = max(float(values.min()), -line_length_um / 2)
    high = min(float(values.max()), line_length_um / 2)
    if high <= low:
        return 0.0, 0.0
    step = max(dx, 0.25)
    edges = np.arange(-line_length_um / 2, line_length_um / 2 + step, step)
    occupied = np.histogram(values, bins=edges)[0] > 0
    ablated_length = float(occupied.sum() * step)
    visible_ratio = float(np.clip((high - low) / line_length_um, 0.0, 1.0))
    return float(np.clip(ablated_length / line_length_um, 0.0, 1.0)), visible_ratio


def extract_one(
    raw: dict[str, object], design_row: pd.Series, cfg: Config
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    group = int(raw["group"])
    raw_z = np.asarray(raw["z_um"], dtype=float)
    light = np.asarray(raw["light"])
    valid = np.asarray(raw["valid"], dtype=bool)
    dx, dy = float(raw["dx_um"]), float(raw["dy_um"])
    cut_corridor, corridor_metrics = locate_cut_corridor(raw_z, valid, dx, dy, cfg)
    z, cone_mask, cone_table, cone_metrics = repair_conical_dropouts(
        raw_z, valid, cfg, allowed_mask=cut_corridor
    )
    plane, depth, sigma_ref, plane_rmse, ref_mask = fit_reference_plane(z, valid, dx, dy, cfg)
    threshold = cfg.threshold_k * sigma_ref
    initial_mask = valid & np.isfinite(depth) & (depth > threshold)
    components_all = connected_components(initial_mask)
    components = select_signal_components(components_all, depth, threshold, cfg)
    selected_mask = np.zeros_like(initial_mask)
    for comp in components:
        selected_mask[comp[:, 0], comp[:, 1]] = True
    if selected_mask.sum() < 3:
        result = {
            "加工顺序": group,
            "status": "failed_no_signal",
            "sigma_ref_um": sigma_ref,
            "RMSE_plane_um": plane_rmse,
            "threshold_um": threshold,
            "valid_pixel_ratio": float(valid.mean()),
            "N_conical_artifacts_repaired": len(cone_table),
            "n_conical_pixels_repaired": int(cone_mask.sum()),
        }
        return result, pd.DataFrame(), pd.DataFrame(), {
            "depth": depth, "mask": selected_mask, "light": light,
            "raw_z": raw_z, "corrected_z": z, "cone_mask": cone_mask,
        }
    center, parallel, perpendicular, s, xi = weighted_line_axes(selected_mask, depth, dx, dy)
    all_profiles, all_raw_profiles = extract_profiles(
        depth, valid, s, xi, threshold, cfg, longitudinal_fraction=1.0
    )
    profiles, raw_profiles, stable_info = select_stable_profile_region(
        all_profiles, all_raw_profiles, cfg
    )
    stable_start = float(stable_info["s_start_um"])
    stable_end = float(stable_info["s_end_um"])
    if not np.isfinite(stable_start) or not np.isfinite(stable_end):
        stable_start = -cfg.line_length_um * cfg.central_fraction / 2.0
        stable_end = cfg.line_length_um * cfg.central_fraction / 2.0
        stable_info["effective_length_um"] = stable_end - stable_start
    effective_mask = selected_mask & (s >= stable_start) & (s <= stable_end)
    if int(effective_mask.sum()) < 3:
        stable_start = -cfg.line_length_um * cfg.central_fraction / 2.0
        stable_end = cfg.line_length_um * cfg.central_fraction / 2.0
        stable_info["status"] = "fallback_no_effective_signal"
        stable_info["effective_length_um"] = stable_end - stable_start
        effective_mask = selected_mask & (s >= stable_start) & (s <= stable_end)
        profiles = all_profiles.loc[
            (all_profiles["s_um"] >= stable_start) & (all_profiles["s_um"] <= stable_end)
        ].copy() if not all_profiles.empty else all_profiles
        raw_profiles = all_raw_profiles.loc[
            (all_raw_profiles["s_um"] >= stable_start)
            & (all_raw_profiles["s_um"] <= stable_end)
        ].copy() if not all_raw_profiles.empty else all_raw_profiles
    if int(effective_mask.sum()) < 3:
        stable_info["status"] = "not_applicable_discrete_or_fragmented"
        effective_mask = selected_mask.copy()
        stable_start = float(np.min(s[selected_mask]))
        stable_end = float(np.max(s[selected_mask]))
        stable_info["effective_length_um"] = stable_end - stable_start
        profiles = all_profiles
        raw_profiles = all_raw_profiles
    standard_profile = median_standard_profile(raw_profiles)
    crater_table_all = component_metrics(components, depth, s, xi, dx, dy)
    crater_table = crater_table_all.loc[
        (crater_table_all["center_s_um"] >= stable_start)
        & (crater_table_all["center_s_um"] <= stable_end)
    ].copy() if not crater_table_all.empty else crater_table_all
    continuity, visible_ratio = occupancy_continuity(selected_mask, s, dx, cfg.line_length_um)
    largest_fraction = float(max(len(c) for c in components) / max(1, selected_mask.sum()))
    if continuity >= 0.70 and largest_fraction >= 0.45:
        mode = "continuous"
    elif len(components) >= 2 and continuity < 0.70:
        mode = "discrete"
    else:
        mode = "uncertain"

    if mode == "continuous" and not profiles.empty:
        w_line, sigma_w = robust_median_sigma(profiles["width_um"])
        d_line, sigma_d = robust_median_sigma(profiles["depth_p95_um"])
        a_cs, _ = robust_median_sigma(profiles["area_um2"])
    elif not crater_table.empty:
        w_line, sigma_w = robust_median_sigma(crater_table["width_perpendicular_um"])
        d_line, sigma_d = robust_median_sigma(crater_table["depth_p95_um"])
        a_cs = float("nan")
    else:
        w_line = sigma_w = d_line = sigma_d = a_cs = float("nan")

    rr, cc = np.nonzero(effective_mask)
    dmax = float(depth[rr, cc].max())
    pixel_area = dx * dy
    v_rem_full = float(np.maximum(depth[selected_mask], 0.0).sum() * pixel_area)
    v_rem = float(np.maximum(depth[effective_mask], 0.0).sum() * pixel_area)
    s_values = s[effective_mask]
    xi_values = xi[effective_mask]
    if s_values.size:
        half_width = max(float(np.max(np.abs(xi_values))) + 2.0, 3.0)
        roi = valid & (s >= stable_start) & (s <= stable_end) & (np.abs(xi) <= half_width)
        v_pile = float(np.maximum(-depth[roi] - threshold, 0.0).sum() * pixel_area)
    else:
        v_pile = float("nan")
    measured_spacing = float("nan")
    if mode == "discrete" and len(crater_table) >= 2:
        measured_spacing = float(np.median(np.diff(crater_table["center_s_um"])))
    edge_mask = np.zeros_like(selected_mask)
    edge_mask[[0, -1], :] = True
    edge_mask[:, [0, -1]] = True
    edge_clipped = bool(np.any(selected_mask & edge_mask))
    if not profiles.empty and profiles["edge_clipped"].mean() > 0.05:
        edge_clipped = True
    orientation_deg = float(math.degrees(math.atan2(parallel[1], parallel[0])))
    signal_snr = d_line / sigma_ref if np.isfinite(d_line) and sigma_ref > 0 else float("nan")
    confidence = 0.0
    confidence += 0.35 * min(1.0, max(0.0, (signal_snr if np.isfinite(signal_snr) else 0.0) / 10.0))
    confidence += 0.25 * min(1.0, selected_mask.sum() / 500.0)
    confidence += 0.20 * min(1.0, len(profiles) / 50.0)
    confidence += 0.20 * (0.0 if edge_clipped else 1.0)

    frequency = float(design_row["频率_kHz"])
    speed = float(design_row["速度_mm/s"])
    repeats = int(design_row["重复扫描次数"])
    pulse_spacing = speed / frequency
    pulse_energy_mj = cfg.power_w / frequency
    line_energy_density = cfg.power_w * repeats / speed
    line_input_energy = line_energy_density * (cfg.line_length_um / 1000.0)
    result: dict[str, object] = {
        "加工顺序": group,
        "脉宽_fs": int(design_row["脉宽_fs"]),
        "频率_kHz": int(design_row["频率_kHz"]),
        "重复扫描次数": repeats,
        "速度_mm/s": int(design_row["速度_mm/s"]),
        "实际功率_W": cfg.power_w,
        "理论线长_um": cfg.line_length_um,
        "单脉冲能量_mJ": pulse_energy_mj,
        "理论脉冲间距_um": pulse_spacing,
        "理论单遍脉冲数估计": cfg.line_length_um / pulse_spacing,
        "累计线能量密度_J_mm": line_energy_density,
        "单线输入能量_J": line_input_energy,
        "cag_timestamp": raw["timestamp"],
        "cag_original_filename": raw["original_filename"],
        "status": "ok",
        "processing_mode": mode,
        "W_line_um": w_line,
        "D_line_um": d_line,
        "D_max_um": dmax,
        "sigma_W_um": sigma_w,
        "sigma_D_um": sigma_d,
        "A_cs_median_um2": a_cs,
        "V_rem_um3": v_rem,
        "V_rem_full_detected_um3": v_rem_full,
        "V_rem_per_visible_length_um2": v_rem / max(float(stable_info["effective_length_um"]), dx),
        "C_shape": a_cs / (w_line * d_line) if np.isfinite(a_cs * w_line * d_line) and w_line * d_line > 0 else float("nan"),
        "V_pile_um3": v_pile,
        "V_pile_V_rem_ratio": v_pile / v_rem if np.isfinite(v_pile) and v_rem > 0 else float("nan"),
        "N_crater_components": len(crater_table),
        "measured_crater_spacing_um": measured_spacing,
        "C_continuity": continuity,
        "sigma_ref_um": sigma_ref,
        "RMSE_plane_um": plane_rmse,
        "threshold_k": cfg.threshold_k,
        "threshold_um": threshold,
        "valid_pixel_ratio": float(valid.mean()),
        "reference_pixel_ratio": float(ref_mask.mean()),
        "line_detection_confidence": confidence,
        "visible_length_ratio": visible_ratio,
        "edge_clipped": int(edge_clipped),
        "n_valid_profiles": len(profiles),
        "n_signal_pixels": int(effective_mask.sum()),
        "n_signal_pixels_full_detected": int(selected_mask.sum()),
        "n_components_all": len(components_all),
        "n_components_selected": len(components),
        "largest_component_fraction": largest_fraction,
        "line_orientation_deg": orientation_deg,
        "height_grid_width": z.shape[1],
        "height_grid_height": z.shape[0],
        "dx_um": dx,
        "dy_um": dy,
        "z_step_um": raw["z_step_um"],
        "doe_cag_mapping": "confirmed: 加工顺序 equals CAG Path",
        "cone_preprocess_enabled": int(cfg.cone_repair_enabled),
        "N_conical_artifacts_repaired": len(cone_table),
        "n_conical_pixels_repaired": int(cone_mask.sum()),
        "max_conical_correction_um": (
            float(np.max(z[cone_mask] - raw_z[cone_mask])) if cone_mask.any() else 0.0
        ),
        "cone_noise_second_difference_um": cone_metrics["noise_second_difference_um"],
        "cone_seed_threshold_um": cone_metrics["seed_threshold_um"],
        "cone_grow_threshold_um": cone_metrics["grow_threshold_um"],
        "cone_repair_restricted_to_cut_corridor": 1,
        "cut_corridor_pixel_ratio": corridor_metrics["corridor_pixel_ratio"],
        "cut_corridor_row_low": int(corridor_metrics["row_low"]),
        "cut_corridor_row_high": int(corridor_metrics["row_high"]),
        "cut_corridor_col_low": int(corridor_metrics["col_low"]),
        "cut_corridor_col_high": int(corridor_metrics["col_high"]),
        "stable_region_status": stable_info["status"],
        "stable_s_start_um": stable_start,
        "stable_s_end_um": stable_end,
        "effective_stable_length_um": float(stable_info["effective_length_um"]),
        "stable_reference_depth_um": float(stable_info["reference_depth_um"]),
        "stable_depth_sigma_core_um": float(stable_info.get("depth_sigma_core_um", float("nan"))),
        "stable_depth_tolerance_um": float(stable_info["depth_tolerance_um"]),
    }
    standard_profile.insert(0, "加工顺序", group)
    crater_table.insert(0, "加工顺序", group)
    arrays = {
        "depth": depth,
        "plane": plane,
        "mask": effective_mask,
        "mask_full": selected_mask,
        "cut_corridor": cut_corridor,
        "light": light,
        "s": s,
        "xi": xi,
        "raw_z": raw_z,
        "corrected_z": z,
        "cone_mask": cone_mask,
        "longitudinal_s_um": all_profiles["s_um"].to_numpy(dtype=float),
        "longitudinal_depth_um": all_profiles["depth_p95_um"].to_numpy(dtype=float),
        "longitudinal_width_um": all_profiles["width_um"].to_numpy(dtype=float),
        "longitudinal_included": (
            (all_profiles["s_um"].to_numpy(dtype=float) >= stable_start)
            & (all_profiles["s_um"].to_numpy(dtype=float) <= stable_end)
        ),
    }
    return result, standard_profile, crater_table, arrays


def colorize_depth(depth: np.ndarray, mask: np.ndarray) -> Image.Image:
    finite = depth[np.isfinite(depth)]
    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 98))
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    r = (255 * normalized).astype(np.uint8)
    b = (255 * (1.0 - normalized)).astype(np.uint8)
    g = (255 * (1.0 - np.abs(normalized - 0.5) * 2.0)).astype(np.uint8)
    rgb = np.dstack([r, g, b])
    rgb[mask] = np.array([255, 255, 255], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((1024, 256), Image.Resampling.NEAREST)


def draw_profile_panel(
    canvas: Image.Image,
    profile: pd.DataFrame,
    threshold: float,
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    ox, oy = origin
    width, height = size
    draw.rectangle([ox, oy, ox + width, oy + height], outline=(80, 80, 80), width=2)
    if profile.empty:
        draw.text((ox + 10, oy + 10), "No standard profile", fill=(20, 20, 20))
        return
    x = profile["xi_um"].to_numpy(dtype=float)
    y = profile["median_depth_um"].to_numpy(dtype=float)
    xmin, xmax = float(x.min()), float(x.max())
    ymin = min(float(np.nanmin(y)), 0.0)
    ymax = max(float(np.nanmax(y)), threshold, ymin + 1e-6)
    px = ox + 20 + (x - xmin) / max(xmax - xmin, 1e-12) * (width - 40)
    py = oy + height - 20 - (y - ymin) / max(ymax - ymin, 1e-12) * (height - 40)
    points = [(int(a), int(b)) for a, b in zip(px, py)]
    if len(points) >= 2:
        draw.line(points, fill=(0, 70, 160), width=3)
    threshold_y = int(oy + height - 20 - (threshold - ymin) / max(ymax - ymin, 1e-12) * (height - 40))
    draw.line([(ox + 20, threshold_y), (ox + width - 20, threshold_y)], fill=(200, 40, 40), width=2)
    draw.text((ox + 24, oy + 8), f"Median cross-section; threshold={threshold:.3f} um", fill=(20, 20, 20))
    draw.text((ox + 24, oy + height - 18), f"xi: {xmin:.2f} .. {xmax:.2f} um", fill=(20, 20, 20))


def draw_longitudinal_panel(
    canvas: Image.Image,
    result: dict[str, object],
    arrays: dict[str, np.ndarray],
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    ox, oy = origin
    width, height = size
    draw.rectangle([ox, oy, ox + width, oy + height], outline=(80, 80, 80), width=2)
    s = arrays.get("longitudinal_s_um", np.array([]))
    depth = arrays.get("longitudinal_depth_um", np.array([]))
    included = arrays.get("longitudinal_included", np.array([], dtype=bool))
    if len(s) < 2:
        draw.text((ox + 10, oy + 10), "No longitudinal profiles", fill=(20, 20, 20))
        return
    xmin, xmax = float(np.min(s)), float(np.max(s))
    ymin, ymax = float(np.min(depth)), float(np.max(depth))
    if ymax <= ymin:
        ymax = ymin + 1.0
    px = ox + 24 + (s - xmin) / max(xmax - xmin, 1e-12) * (width - 48)
    py = oy + height - 24 - (depth - ymin) / (ymax - ymin) * (height - 48)
    draw.line([(int(a), int(b)) for a, b in zip(px, py)], fill=(150, 150, 150), width=2)
    stable_points = [
        (int(px[index]), int(py[index])) for index in range(len(s)) if included[index]
    ]
    if len(stable_points) >= 2:
        draw.line(stable_points, fill=(0, 70, 170), width=3)
    for boundary in (
        float(result.get("stable_s_start_um", xmin)),
        float(result.get("stable_s_end_um", xmax)),
    ):
        bx = int(ox + 24 + (boundary - xmin) / max(xmax - xmin, 1e-12) * (width - 48))
        draw.line([(bx, oy + 20), (bx, oy + height - 20)], fill=(210, 30, 30), width=2)
    draw.text(
        (ox + 15, oy + 7),
        "Longitudinal P95 depth: gray=excluded; blue=stable; red=boundaries",
        fill=(20, 20, 20),
    )
    draw.text(
        (ox + 15, oy + height - 18),
        f"s: {xmin:.1f} .. {xmax:.1f} um; stable={result.get('stable_region_status', '')}",
        fill=(20, 20, 20),
    )


def save_diagnostic(
    output: Path,
    result: dict[str, object],
    standard_profile: pd.DataFrame,
    arrays: dict[str, np.ndarray],
) -> None:
    canvas = Image.new("RGB", (1120, 1010), "white")
    draw = ImageDraw.Draw(canvas)
    heat = colorize_depth(arrays["depth"], arrays["mask"])
    canvas.paste(heat, (48, 70))
    title = (
        f"Group {int(result['加工顺序']):03d} | {result.get('processing_mode', 'failed')} | "
        f"W={result.get('W_line_um', float('nan')):.3f} um | "
        f"D={result.get('D_line_um', float('nan')):.3f} um"
    )
    draw.text((48, 25), title, fill=(0, 0, 0))
    draw.text(
        (48, 335),
        "Depth map: blue=above/reference, red=deeper removal, white=selected signal",
        fill=(20, 20, 20),
    )
    draw_longitudinal_panel(
        canvas,
        result,
        arrays,
        (48, 385),
        (1024, 200),
    )
    draw_profile_panel(
        canvas,
        standard_profile,
        float(result.get("threshold_um", 0.0)),
        (48, 625),
        (1024, 300),
    )
    footer = (
        f"sigma_ref={result.get('sigma_ref_um', float('nan')):.4f} um; "
        f"confidence={result.get('line_detection_confidence', float('nan')):.3f}; "
        f"edge_clipped={result.get('edge_clipped', '')}; "
        f"profiles={result.get('n_valid_profiles', 0)}"
    )
    draw.text((48, 960), footer, fill=(0, 0, 0))
    canvas.save(output)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cag", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", default="pilot", help="pilot, all, or comma/range list")
    parser.add_argument("--pilot-count", type=int, default=15)
    parser.add_argument("--power-w", type=float, default=5.333)
    parser.add_argument("--line-length-um", type=float, default=200.0)
    parser.add_argument("--threshold-k", type=float, default=4.0)
    args = parser.parse_args()

    design = pd.read_csv(args.design, encoding="gb18030")
    required = ["加工顺序", "脉宽_fs", "频率_kHz", "重复扫描次数", "速度_mm/s"]
    missing = [column for column in required if column not in design.columns]
    if missing:
        raise ValueError(f"DOE missing columns: {missing}")
    if len(design) != 120 or sorted(design["加工顺序"].astype(int)) != list(range(1, 121)):
        raise ValueError("DOE must contain unique 加工顺序 1..120")
    if args.groups == "pilot":
        groups = choose_pilot_groups(design, args.pilot_count)
    elif args.groups == "all":
        groups = list(range(1, 121))
    else:
        groups = parse_groups(args.groups)
    if not groups or min(groups) < 1 or max(groups) > 120:
        raise ValueError(f"Invalid groups: {groups}")

    cfg = Config(
        power_w=args.power_w,
        line_length_um=args.line_length_um,
        threshold_k=args.threshold_k,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = args.output_dir / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    results: list[dict[str, object]] = []
    profiles: list[pd.DataFrame] = []
    craters: list[pd.DataFrame] = []
    longitudinal_profiles: list[pd.DataFrame] = []
    sensitivity: list[dict[str, object]] = []

    with CagReader(args.cag) as reader:
        for group in groups:
            row = design.loc[design["加工顺序"] == group].iloc[0]
            raw = reader.read_group(group)
            result, profile, crater, arrays = extract_one(raw, row, cfg)
            results.append(result)
            if not profile.empty:
                profiles.append(profile)
            if not crater.empty:
                craters.append(crater)
            if len(arrays.get("longitudinal_s_um", np.array([]))):
                longitudinal_profiles.append(pd.DataFrame({
                    "加工顺序": group,
                    "s_um": arrays["longitudinal_s_um"],
                    "depth_p95_um": arrays["longitudinal_depth_um"],
                    "width_um": arrays["longitudinal_width_um"],
                    "included_in_stable_region": arrays["longitudinal_included"].astype(int),
                }))
            save_diagnostic(
                diagnostics / f"group_{group:03d}_diagnostic.png",
                result,
                profile,
                arrays,
            )
            for k in (3.0, 4.0, 5.0):
                alt_cfg = Config(
                    power_w=args.power_w,
                    line_length_um=args.line_length_um,
                    threshold_k=k,
                )
                alt_result, _, _, _ = extract_one(raw, row, alt_cfg)
                sensitivity.append({
                    "加工顺序": group,
                    "threshold_k": k,
                    "status": alt_result.get("status"),
                    "processing_mode": alt_result.get("processing_mode"),
                    "W_line_um": alt_result.get("W_line_um"),
                    "D_line_um": alt_result.get("D_line_um"),
                    "C_continuity": alt_result.get("C_continuity"),
                    "n_signal_pixels": alt_result.get("n_signal_pixels"),
                    "edge_clipped": alt_result.get("edge_clipped"),
                })
            print(
                f"group={group:03d} status={result.get('status')} "
                f"mode={result.get('processing_mode')} "
                f"W={result.get('W_line_um', float('nan')):.3f} "
                f"D={result.get('D_line_um', float('nan')):.3f}"
            )

    result_frame = pd.DataFrame(results).sort_values("加工顺序")
    sensitivity_frame = pd.DataFrame(sensitivity)
    sensitivity_summary: list[dict[str, object]] = []
    for group, subset in sensitivity_frame.groupby("加工顺序"):
        subset = subset.sort_values("threshold_k")
        nominal = subset.loc[subset["threshold_k"] == cfg.threshold_k]
        if nominal.empty:
            nominal = subset.iloc[[len(subset) // 2]]
        nominal_w = float(nominal.iloc[0]["W_line_um"])
        nominal_d = float(nominal.iloc[0]["D_line_um"])
        width_values = pd.to_numeric(subset["W_line_um"], errors="coerce")
        depth_values = pd.to_numeric(subset["D_line_um"], errors="coerce")
        sensitivity_summary.append({
            "加工顺序": int(group),
            "threshold_W_relative_range": (
                float((width_values.max() - width_values.min()) / nominal_w)
                if np.isfinite(nominal_w) and nominal_w > 0 else float("nan")
            ),
            "threshold_D_relative_range": (
                float((depth_values.max() - depth_values.min()) / nominal_d)
                if np.isfinite(nominal_d) and nominal_d > 0 else float("nan")
            ),
            "threshold_mode_count": int(subset["processing_mode"].nunique(dropna=True)),
        })
    result_frame = result_frame.merge(
        pd.DataFrame(sensitivity_summary), on="加工顺序", how="left"
    )
    result_frame["relative_sigma_W"] = result_frame["sigma_W_um"] / result_frame["W_line_um"]
    result_frame["relative_sigma_D"] = result_frame["sigma_D_um"] / result_frame["D_line_um"]
    qc_status: list[str] = []
    qc_flags: list[str] = []
    for _, row in result_frame.iterrows():
        flags: list[str] = []
        if row.get("status") != "ok":
            flags.append("extraction_failed")
        if int(row.get("edge_clipped", 0)):
            flags.append("edge_clipped")
        if float(row.get("line_detection_confidence", 0.0)) < 0.75:
            flags.append("low_detection_confidence")
        if row.get("processing_mode") == "discrete":
            flags.append("discrete_mode_requires_review")
        if row.get("stable_region_status") != "adaptive":
            flags.append("stable_region_not_identified")
        if (
            float(row.get("threshold_W_relative_range", 0.0)) > 0.25
            or float(row.get("threshold_D_relative_range", 0.0)) > 0.15
            or int(row.get("threshold_mode_count", 1)) > 1
        ):
            flags.append("threshold_sensitive")
        if row.get("processing_mode") == "continuous" and int(row.get("n_valid_profiles", 0)) < 50:
            flags.append("too_few_profiles")
        if float(row.get("reference_pixel_ratio", 0.0)) < 0.35:
            flags.append("insufficient_reference_surface")
        if float(row.get("relative_sigma_D", 0.0)) > 0.50:
            flags.append("high_depth_variability")
        qc_status.append("pass" if not flags else "review")
        qc_flags.append(";".join(flags))
    result_frame["qc_status"] = qc_status
    result_frame["qc_flags"] = qc_flags
    write_csv(args.output_dir / "pilot_single_line_features.csv", result_frame)
    write_csv(
        args.output_dir / "pilot_standard_profiles.csv",
        pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame(),
    )
    write_csv(
        args.output_dir / "pilot_crater_components.csv",
        pd.concat(craters, ignore_index=True) if craters else pd.DataFrame(),
    )
    write_csv(
        args.output_dir / "pilot_threshold_sensitivity.csv",
        sensitivity_frame,
    )
    write_csv(
        args.output_dir / "pilot_longitudinal_profiles.csv",
        pd.concat(longitudinal_profiles, ignore_index=True)
        if longitudinal_profiles else pd.DataFrame(),
    )
    protocol = {
        "source_cag": str(args.cag),
        "source_design": str(args.design),
        "groups": groups,
        "config": asdict(cfg),
        "mapping": "User-confirmed: DOE 加工顺序 equals CAG Path",
        "conical_dropout_preprocessing": {
            "detection": (
                "row-wise grayscale closing residual; adaptive seed/grow thresholds "
                "from the robust second-difference noise scale"
            ),
            "acceptance": (
                "candidate must contain a strong seed, have bounded per-row width, "
                "and retain both X-direction shoulders"
            ),
            "replacement": (
                "robust local quadratic fitted to the surrounding base ring; only "
                "upward corrections to raw Z are allowed"
            ),
            "regularization": "none",
        },
        "height_sign": "removal depth = robust reference plane - measured height",
        "longitudinal_end_trimming": {
            "reference": "median P95 depth in the central configured core",
            "smoothing": "centered rolling median of longitudinal profile depth",
            "acceptance": (
                "central contiguous run within the maximum of 3*MAD, 12% of "
                "reference depth, and 0.25 um"
            ),
            "outputs": "stable start/end, effective length, and full-versus-effective volume",
            "fallback": "flag for review; fragmented/discrete lines are not treated as stable plateaus",
        },
        "threshold": "D > threshold_k * 1.4826 * MAD(reference residual)",
        "width": "median threshold-crossing width; crater perpendicular span in discrete mode",
        "depth": "median of per-profile/per-crater P95 removal depth",
        "warning": "Pilot outputs require visual review before 120-group batch processing.",
        "qc_rules": {
            "edge_clipped": "review",
            "line_detection_confidence_below": 0.75,
            "discrete_mode": "review",
            "threshold_width_relative_range_above": 0.25,
            "threshold_depth_relative_range_above": 0.15,
            "threshold_mode_change": "review",
            "continuous_profiles_below": 50,
            "reference_surface_ratio_below": 0.35,
            "relative_depth_MAD_above": 0.50,
        },
    }
    (args.output_dir / "pilot_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote pilot outputs to {args.output_dir}")
    print(f"Pilot groups: {groups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
