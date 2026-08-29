#!/usr/bin/env python3
"""Render the uncorrected height samples of group 1 directly from the CAG."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from extract_zro2_single_line import CagReader, HEIGHT_KEY, VK4_KEYS


CAG = Path(r"C:\Users\RZF\Desktop\博士课题资料\光机所实验原始数据\氧化锆\120组直线.cag")
OUTPUT_DIR = Path(r"C:\Users\RZF\Desktop\专利\outputs\cag_raw_verification")


def viridis(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Small dependency-free interpolation of the standard viridis palette."""
    anchors = np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=float,
    )
    t = np.clip((values - lo) / max(hi - lo, np.finfo(float).eps), 0.0, 1.0)
    scaled = t * (len(anchors) - 1)
    left = np.floor(scaled).astype(int)
    right = np.minimum(left + 1, len(anchors) - 1)
    weight = (scaled - left)[..., None]
    return np.rint(anchors[left] * (1.0 - weight) + anchors[right] * weight).astype(np.uint8)


def draw_colorbar(
    canvas: Image.Image, box: tuple[int, int, int, int], lo: float, hi: float, label: str
) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    gradient = np.linspace(hi, lo, y1 - y0 + 1)[:, None]
    colors = viridis(gradient, lo, hi)
    bar = Image.fromarray(np.repeat(colors, x1 - x0 + 1, axis=1), mode="RGB")
    canvas.paste(bar, (x0, y0))
    draw.rectangle(box, outline="black", width=1)
    draw.text((x1 + 8, y0 - 5), f"{hi:.4f}", fill="black")
    draw.text((x1 + 8, y1 - 10), f"{lo:.4f}", fill="black")
    draw.text((x0 - 10, y1 + 10), label, fill="black")


def render_physical_map(
    path: Path, z: np.ndarray, x_span: float, y_span: float
) -> None:
    lo, hi = float(np.min(z)), float(np.max(z))
    native = Image.fromarray(viridis(z, lo, hi), mode="RGB")
    # dx == dy, so equal enlargement preserves the native physical X:Y ratio.
    plot = native.resize((1536, 96), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (1800, 280), "white")
    canvas.paste(plot, (80, 80))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((80, 80, 1615, 175), outline="black", width=2)
    draw.text((80, 22), "CAG group 001: raw absolute height; native physical X:Y aspect", fill="black")
    draw.text((80, 182), f"X: 0 .. {x_span:.6f} um", fill="black")
    draw.text((8, 112), f"Y span\n{y_span:.6f} um", fill="black")
    draw.text(
        (80, 224),
        "No plane correction, filtering, interpolation, thresholding, or percentile clipping",
        fill="black",
    )
    draw_colorbar(canvas, (1640, 80, 1665, 175), lo, hi, "Raw Z (um)")
    canvas.save(path)


def render_point_cloud(
    path: Path, xx: np.ndarray, yy: np.ndarray, z: np.ndarray
) -> None:
    width, height = 1800, 1050
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    lo, hi = float(np.min(z)), float(np.max(z))

    xyz = np.column_stack([xx.ravel(), yy.ravel(), z.ravel()]).astype(float)
    xyz -= np.median(xyz, axis=0)
    xyz[:, 2] *= 4.0  # Explicit display exaggeration only.
    azimuth = np.deg2rad(-68.0)
    elevation = np.deg2rad(28.0)
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    up = np.array(
        [-np.cos(azimuth) * np.sin(elevation),
         -np.sin(azimuth) * np.sin(elevation),
         np.cos(elevation)]
    )
    view = np.array(
        [np.cos(azimuth) * np.cos(elevation),
         np.sin(azimuth) * np.cos(elevation),
         np.sin(elevation)]
    )
    screen_x = xyz @ right
    screen_y = xyz @ up
    depth = xyz @ view
    usable_w, usable_h = 1450.0, 780.0
    scale = min(
        usable_w / max(float(np.ptp(screen_x)), 1e-12),
        usable_h / max(float(np.ptp(screen_y)), 1e-12),
    )
    px = 790.0 + screen_x * scale
    py = 560.0 - screen_y * scale
    colors = viridis(z.ravel(), lo, hi)
    order = np.argsort(depth)
    for index in order:
        x0, y0 = int(round(px[index])), int(round(py[index]))
        color = tuple(int(channel) for channel in colors[index])
        draw.ellipse((x0 - 1, y0 - 1, x0 + 1, y0 + 1), fill=color)

    draw.text((60, 25), "CAG group 001: all 65,536 raw calibrated height points", fill="black")
    draw.text(
        (60, 58),
        "Native XYZ values; oblique screen projection; displayed Z dimension enlarged 4x for visibility",
        fill="black",
    )
    draw.text((60, 990), "X/Y units: um; Z color and coordinate units: um", fill="black")
    draw_colorbar(canvas, (1630, 180, 1660, 820), lo, hi, "Raw Z (um)")
    canvas.save(path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CagReader(CAG) as reader:
        group = reader.read_group(1)
        vk4 = reader.archive.read(reader._vk4[1])
        direct_height_bytes = reader.archive.read(reader._direct[1][HEIGHT_KEY])

    z = np.asarray(group["z_um"], dtype=np.float64)
    valid = np.asarray(group["valid"], dtype=bool)
    dx = float(group["dx_um"])
    dy = float(group["dy_um"])
    dz = float(group["z_step_um"])
    rows, cols = z.shape
    x = np.arange(cols, dtype=float) * dx
    y = np.arange(rows, dtype=float) * dy
    xx, yy = np.meshgrid(x, y)

    map_path = OUTPUT_DIR / "group_001_raw_height_map_physical_aspect.png"
    render_physical_map(map_path, z, float(x[-1] - x[0]), float(y[-1] - y[0]))

    cloud_path = OUTPUT_DIR / "group_001_raw_point_cloud_all_points.png"
    render_point_cloud(cloud_path, xx, yy, z)

    raw_digits = np.rint(z / dz).astype("<u4")
    vk4_offsets = dict(zip(VK4_KEYS, struct.unpack_from("<18I", vk4, 12)))
    vk4_height_offset = vk4_offsets["height"]
    vk4_width, vk4_height = struct.unpack_from("<2I", vk4, vk4_height_offset)
    embedded_data_offset = vk4.find(direct_height_bytes, vk4_height_offset)
    if embedded_data_offset < 0:
        raise ValueError("The direct height block was not found in the embedded VK4 dataset")
    vk4_digits = np.frombuffer(
        vk4,
        dtype="<u4",
        count=vk4_width * vk4_height,
        offset=embedded_data_offset,
    ).reshape(vk4_height, vk4_width)
    direct_vk4_equal = bool(np.array_equal(raw_digits, vk4_digits))
    metadata = {
        "source_cag": str(CAG),
        "group": 1,
        "original_filename": group["original_filename"],
        "timestamp": group["timestamp"],
        "vk4_timestamp": group["vk4_timestamp"],
        "shape_rows_cols": [rows, cols],
        "point_count": int(z.size),
        "valid_point_count": int(valid.sum()),
        "invalid_point_count": int(valid.size - valid.sum()),
        "dx_um": dx,
        "dy_um": dy,
        "z_step_um": dz,
        "x_center_span_um": float(x[-1] - x[0]),
        "y_center_span_um": float(y[-1] - y[0]),
        "z_min_um": float(np.min(z)),
        "z_max_um": float(np.max(z)),
        "z_range_um": float(np.ptp(z)),
        "z_mean_um": float(np.mean(z)),
        "z_median_um": float(np.median(z)),
        "z_std_um": float(np.std(z)),
        "z_zero_sample_count": int(np.count_nonzero(z == 0.0)),
        "z_percentiles_um": {
            "p01": float(np.percentile(z, 1)),
            "p50": float(np.percentile(z, 50)),
            "p99": float(np.percentile(z, 99)),
        },
        "raw_u32_sha256": hashlib.sha256(raw_digits.tobytes(order="C")).hexdigest(),
        "direct_height_equals_embedded_vk4": direct_vk4_equal,
        "direct_vk4_mismatch_count": int(np.count_nonzero(raw_digits != vk4_digits)),
        "embedded_vk4_height_data_offset": embedded_data_offset,
        "processing": "none; direct calibrated height = uint32 sample * z_step_um",
        "physical_map": "native X:Y aspect; no color clipping",
        "point_cloud": "all points; native XYZ values; display box Z dimension enlarged 4x",
    }
    metadata_path = OUTPUT_DIR / "group_001_raw_height_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(map_path)
    print(cloud_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
