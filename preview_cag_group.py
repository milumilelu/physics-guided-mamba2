#!/usr/bin/env python3
"""Read one CAG group and render a preview (height map + oblique point cloud).

Self-contained: reuses the proven CagReader decoding logic from
extract_zro2_single_line.py but only depends on numpy + Pillow (no pandas).
"""
from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HEIGHT_KEY = "4d137b4a-bf22-49d5-96a8-9b07b3fc5d02"
LIGHT_KEY = "e4eec84e-b9fd-4898-8a44-79d6eae57fb4"
ERROR_KEY = "2c7fd1a8-b42a-41ff-9baa-56760304e826"
VK4_KEYS = [
    "meas_conds", "color_peak", "color_light", "light", "unknown_4",
    "unknown_5", "height", "unknown_7", "unknown_8", "color_peak_thumb",
    "color_thumb", "light_thumb", "height_thumb", "assembly_info",
    "line_measure", "line_thickness", "string_data", "reserved",
]


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

    def __exit__(self, *a) -> None:
        self.close()

    def _index(self) -> None:
        measurement_xml = None
        for info in self.entries:
            parts = info.filename.split("/")
            if len(parts) == 5 and parts[1].isdigit() and parts[3] in {
                HEIGHT_KEY, LIGHT_KEY, ERROR_KEY,
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
            stamp = Path(name).stem.removeprefix("MeasureData")
            if len(stamp) == 14 and stamp.isdigit():
                self._timestamps[group] = (
                    f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]} "
                    f"{stamp[8:10]}:{stamp[10:12]}:{stamp[12:14]}"
                )
            else:
                self._timestamps[group] = ""

    def read_group(self, group: int) -> dict:
        entries = self._direct[group]
        vk4 = self.archive.read(self._vk4[group])
        offsets = dict(zip(VK4_KEYS, struct.unpack_from("<18I", vk4, 12)))
        measure_offset = offsets["meas_conds"]
        x_pitch_pm, y_pitch_pm, z_step_pm = struct.unpack_from("<3I", vk4, measure_offset + 42 * 4)
        height_offset = offsets["height"]
        width, height, bit_depth, compression, data_bytes = struct.unpack_from("<5I", vk4, height_offset)
        if bit_depth != 32 or data_bytes != width * height * 4:
            raise ValueError(f"Unsupported height layout in group {group}")
        direct_height = np.frombuffer(self.archive.read(entries[HEIGHT_KEY]), dtype="<u4").reshape(height, width)
        direct_light = np.frombuffer(self.archive.read(entries[LIGHT_KEY]), dtype="<u2").reshape(height, width)
        error_mask = np.frombuffer(self.archive.read(entries[ERROR_KEY]), dtype=np.uint8).reshape(height, width)
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
            "data_bytes": data_bytes,
        }


def viridis(values, lo, hi):
    anchors = np.array([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]], dtype=float)
    t = np.clip((values - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    scaled = t * (len(anchors) - 1)
    left = np.floor(scaled).astype(int)
    right = np.minimum(left + 1, len(anchors) - 1)
    weight = (scaled - left)[..., None]
    return np.rint(anchors[left] * (1 - weight) + anchors[right] * weight).astype(np.uint8)


def render_physical_map(path, z, x_span, y_span):
    lo, hi = float(np.min(z)), float(np.max(z))
    native = Image.fromarray(viridis(z, lo, hi), mode="RGB")
    plot = native.resize((1536, 96), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (1800, 280), "white")
    canvas.paste(plot, (80, 80))
    d = ImageDraw.Draw(canvas)
    d.rectangle((80, 80, 1615, 175), outline="black", width=2)
    d.text((80, 22), "CAG group 001: raw absolute height; native physical X:Y aspect", fill="black")
    d.text((80, 182), f"X: 0 .. {x_span:.6f} um", fill="black")
    d.text((8, 112), f"Y span\n{y_span:.6f} um", fill="black")
    d.text((80, 224), "No plane correction / filtering / interpolation / clipping", fill="black")
    canvas.save(path)


def render_point_cloud(path, xx, yy, z):
    w, h = 1800, 1050
    canvas = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(canvas)
    lo, hi = float(np.min(z)), float(np.max(z))
    xyz = np.column_stack([xx.ravel(), yy.ravel(), z.ravel()]).astype(float)
    xyz -= np.median(xyz, axis=0)
    xyz[:, 2] *= 4.0
    az = np.deg2rad(-68.0); el = np.deg2rad(28.0)
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.array([-np.cos(az)*np.sin(el), -np.sin(az)*np.sin(el), np.cos(el)])
    view = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
    sx = xyz @ right; sy = xyz @ up; depth = xyz @ view
    usable_w, usable_h = 1450.0, 780.0
    scale = min(usable_w / max(float(np.ptp(sx)), 1e-12), usable_h / max(float(np.ptp(sy)), 1e-12))
    px = 790.0 + sx * scale; py = 560.0 - sy * scale
    colors = viridis(z.ravel(), lo, hi)
    order = np.argsort(depth)
    for idx in order:
        x0, y0 = int(round(px[idx])), int(round(py[idx]))
        d.ellipse((x0-1, y0-1, x0+1, y0+1), fill=tuple(int(c) for c in colors[idx]))
    d.text((60, 25), "CAG group 001: all 65,536 raw calibrated height points", fill="black")
    d.text((60, 58), "Native XYZ; oblique projection; Z enlarged 4x for visibility", fill="black")
    d.text((60, 990), "X/Y units: um; Z color & coordinate units: um", fill="black")
    canvas.save(path)


def main() -> None:
    CAG = Path(r"C:\Users\RZF\Desktop\专利\氧化锆\120组直线.cag")
    OUT = Path(r"C:\Users\RZF\Desktop\专利\outputs\cag_preview")
    OUT.mkdir(parents=True, exist_ok=True)
    with CagReader(CAG) as reader:
        g = reader.read_group(1)
    z = np.asarray(g["z_um"], dtype=float)
    dx, dy = float(g["dx_um"]), float(g["dy_um"])
    rows, cols = z.shape
    x = np.arange(cols) * dx
    y = np.arange(rows) * dy
    xx, yy = np.meshgrid(x, y)
    map_path = OUT / "group_001_raw_height_map.png"
    cloud_path = OUT / "group_001_raw_point_cloud.png"
    render_physical_map(map_path, z, float(x[-1]-x[0]), float(y[-1]-y[0]))
    render_point_cloud(cloud_path, xx, yy, z)
    meta = {
        "source_cag": str(CAG),
        "group": 1,
        "original_filename": g["original_filename"],
        "timestamp": g["timestamp"],
        "shape_rows_cols": [rows, cols],
        "point_count": int(z.size),
        "valid_point_count": int(np.asarray(g["valid"]).sum()),
        "dx_um": dx, "dy_um": dy, "z_step_um": float(g["z_step_um"]),
        "x_span_um": float(x[-1]-x[0]), "y_span_um": float(y[-1]-y[0]),
        "z_min_um": float(np.min(z)), "z_max_um": float(np.max(z)),
        "z_mean_um": float(np.mean(z)), "z_std_um": float(np.std(z)),
        "z_zero_count": int(np.count_nonzero(z == 0.0)),
    }
    (OUT / "group_001_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(map_path)
    print(cloud_path)


if __name__ == "__main__":
    main()
