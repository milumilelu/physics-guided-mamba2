#!/usr/bin/env python3
"""Create an actual-data before/after diagnostic for group-1 cone repair."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from extract_zro2_single_line import CagReader, Config, repair_conical_dropouts
from render_cag_group1_raw import CAG, viridis


OUTPUT = Path(
    r"C:\Users\RZF\Desktop\专利\outputs\cag_raw_verification\group_001_conical_preprocess_check.png"
)
METADATA = Path(
    r"C:\Users\RZF\Desktop\专利\outputs\cag_raw_verification\group_001_conical_preprocess_check.json"
)


def map_panel(z: np.ndarray, lo: float, hi: float, mask: np.ndarray | None = None) -> Image.Image:
    rgb = viridis(z, lo, hi)
    if mask is not None:
        rgb[mask] = np.array([255, 20, 147], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((1536, 192), Image.Resampling.NEAREST)


def draw_profile(
    canvas: Image.Image,
    raw: np.ndarray,
    corrected: np.ndarray,
    row: int,
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    ox, oy = origin
    width, height = size
    draw.rectangle((ox, oy, ox + width, oy + height), outline="black", width=2)
    lo = min(float(raw[row].min()), float(corrected[row].min()))
    hi = max(float(raw[row].max()), float(corrected[row].max()))
    x = np.linspace(ox + 25, ox + width - 25, raw.shape[1])

    def points(values: np.ndarray) -> list[tuple[int, int]]:
        y = oy + height - 25 - (values - lo) / max(hi - lo, 1e-12) * (height - 50)
        return [(int(round(a)), int(round(b))) for a, b in zip(x, y)]

    draw.line(points(raw[row]), fill=(0, 60, 150), width=2)
    draw.line(points(corrected[row]), fill=(220, 30, 30), width=2)
    draw.text((ox + 15, oy + 8), f"Absolute Z cross-section at raw row {row}", fill="black")
    draw.text((ox + 15, oy + 30), "blue=raw; red=repaired local base", fill="black")
    draw.text((ox + 15, oy + height - 18), f"Z range {lo:.4f} .. {hi:.4f} um", fill="black")


def main() -> None:
    with CagReader(CAG) as reader:
        data = reader.read_group(1)
    raw = np.asarray(data["z_um"], dtype=float)
    cfg = Config()
    corrected, mask, table, metrics = repair_conical_dropouts(raw, data["valid"], cfg)
    correction = corrected - raw
    max_row, max_col = np.unravel_index(np.argmax(correction), correction.shape)
    lo, hi = float(raw.min()), float(raw.max())

    canvas = Image.new("RGB", (1740, 950), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 18), "Group 001 conical-dropout preprocessing: actual CAG data", fill="black")
    canvas.paste(map_panel(raw, lo, hi, mask), (70, 55))
    draw.text((70, 252), "Raw Z; accepted repair mask in magenta (Y display enlarged 2x)", fill="black")
    canvas.paste(map_panel(corrected, lo, hi), (70, 290))
    draw.text((70, 487), "Corrected Z, using the same un-clipped absolute color scale", fill="black")
    draw_profile(canvas, raw, corrected, int(max_row), (70, 530), (1536, 350))
    draw.text(
        (70, 912),
        f"artifacts={len(table)}; pixels={int(mask.sum())}; max correction={correction.max():.4f} um; "
        f"max location=(row {max_row}, col {max_col})",
        fill="black",
    )
    canvas.save(OUTPUT)

    payload = {
        "group": 1,
        "config": {
            "half_window_px": cfg.cone_half_window_px,
            "seed_sigma": cfg.cone_seed_sigma,
            "grow_sigma": cfg.cone_grow_sigma,
            "minimum_seed_depth_um": cfg.cone_min_seed_depth_um,
            "maximum_component_span_px": cfg.cone_max_component_span_px,
        },
        "adaptive_thresholds": metrics,
        "artifact_count": len(table),
        "repaired_pixel_count": int(mask.sum()),
        "repaired_pixel_ratio": float(mask.mean()),
        "maximum_correction_um": float(correction.max()),
        "artifacts": table.to_dict(orient="records"),
    }
    METADATA.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(OUTPUT)
    print(METADATA)


if __name__ == "__main__":
    main()
