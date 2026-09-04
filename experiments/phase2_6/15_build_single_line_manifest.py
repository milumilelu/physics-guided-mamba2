#!/usr/bin/env python3
"""Task SL-01a: build the frozen single-line manifest (细则 §3).

Joins the four-factor DOE design table (gb18030), the frozen view manifest
(coefficients by scripts/33) and spot-checked CAG container facts into
`outputs/phase2_6/single_line/single_line_manifest.csv` (120 rows) plus a
provenance JSON with the nine provenance checks of 上位规划 §4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import _lib as p26  # noqa: E402  (importlib chain loads p25/p2/l15)
from src.io_cag import CagHeightReader  # noqa: E402

EXPECTED = [
    "outputs/phase2_6/single_line/single_line_manifest.csv",
    "outputs/phase2_6/single_line/manifest_provenance.json",
]

EXPECTED_TAU = (223, 500, 1000, 2000, 4000)
EXPECTED_F = (2, 5, 10, 20, 40)
EXPECTED_V = (5, 10, 15, 20, 25)
EXPECTED_N = (1, 2, 3, 4, 5)
SPOT_CHECK_GROUPS = (13, 60, 116)

MANIFEST_COLUMNS = [
    "single_line_id", "session_id", "measurement_id", "sample_id",
    "roi_within_measurement", "cag_path", "cag_path_index",
    "pulse_duration_fs", "frequency_kHz", "velocity_mm_s", "pass_count",
    "pulse_pitch_um", "E_line_J_mm", "E_line_J",
    "power_W_or_proxy", "power_source_note",
    "pixel_size_um", "z_step_um",
    "line_scan_direction", "line_scan_direction_note",
    "measurement_orientation_theta_deg", "orientation_confident",
    "processing_date_or_batch", "date_confidence",
    "height_data_type", "background_correction_status",
    "plane_a", "plane_b", "plane_c", "plane_rmse_um", "sigma_ref_um",
    "valid_mask_status", "hatch_spacing_um", "replicates",
    "provenance_same_system", "provenance_system_note",
    "units_note", "mapping_provenance", "exclusion_note",
    "geometry_qc", "notes",
]


def read_design(path: Path) -> pd.DataFrame:
    design = None
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            design = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if design is None:
        raise SystemExit(f"cannot decode design table {path}")
    p26.require(list(design.columns) == [
        "加工顺序", "脉宽_fs", "频率_kHz", "重复扫描次数", "速度_mm/s"],
        f"design table columns unexpected: {list(design.columns)}")
    return design.rename(columns={
        "脉宽_fs": "pulse_duration_fs", "频率_kHz": "frequency_kHz",
        "重复扫描次数": "pass_count", "速度_mm/s": "velocity_mm_s"})


def main() -> int:
    cfg, quick = p26.load_config(__doc__)
    out_dir = p26.output_dir(cfg, "single_line")
    p26.log(f"Task 15 start | quick={quick}")

    design = read_design(cfg["paths"]["line_design_csv"])
    p26.require(len(design) == 120, f"design rows {len(design)} != 120")
    p26.require(design["加工顺序"].tolist() == list(range(1, 121)),
                "加工顺序 must be exactly 1..120")
    p26.require(not design.duplicated(["pulse_duration_fs", "frequency_kHz",
                                       "pass_count", "velocity_mm_s"]).any(),
                "design conditions must be unique (no replicates)")
    for column, expected in (("pulse_duration_fs", EXPECTED_TAU),
                             ("frequency_kHz", EXPECTED_F),
                             ("velocity_mm_s", EXPECTED_V),
                             ("pass_count", EXPECTED_N)):
        observed = sorted(design[column].unique().tolist())
        p26.require(observed == list(expected),
                    f"design grid {column} = {observed} != {list(expected)}")
    p26.log("design table OK: 120 unique four-factor conditions, grids as frozen")

    view = pd.read_csv(cfg["paths"]["line_view_manifest"], encoding="utf-8-sig")
    p26.require(len(view) == 120, f"view manifest rows {len(view)} != 120")
    p26.require(sorted(view["measurement_id"].astype(int).tolist())
                == list(range(1, 121)),
                "view manifest measurement_id must cover 1..120")

    field = view.set_index(view["measurement_id"].astype(int))
    widths = field["raw_width_um"].astype(float)
    heights = field["raw_height_um"].astype(float)
    p26.require(np.allclose(widths, 285.344768, atol=1e-3),
                f"raw_width_um {widths.min()}..{widths.max()} != 285.344768")
    p26.require(np.allclose(heights, 17.834048, atol=1e-3),
                "raw_height_um != 17.834048")
    implied_pitch = float((widths / 1024.0).mean())
    p26.require(abs(implied_pitch - cfg["single_line"]["pixel_um"]) < 1e-5,
                f"implied pitch {implied_pitch} != frozen pixel_um")

    reader = CagHeightReader(cfg["paths"]["line_cag"])
    pitch_observed = []
    try:
        p26.require(sorted(int(g) for g in reader.groups) == list(range(1, 121)),
                    "CAG paths must be exactly 1..120")
        for group in SPOT_CHECK_GROUPS:
            hm = reader.read_height_map(group)
            pitch_observed.append((float(hm.dx_um), float(hm.dy_um)))
            p26.require(hm.shape == (64, 1024),
                        f"group {group} shape {hm.shape} != (64, 1024)")
            p26.require(abs(hm.dx_um - cfg["single_line"]["pixel_um"]) < 1e-9,
                        f"group {group} dx {hm.dx_um} != frozen pixel_um")
            p26.require(float(np.mean(hm.valid_mask)) == 1.0,
                        f"group {group} valid ratio != 1.0")
    finally:
        reader.close()
    p26.log(f"CAG spot check OK for groups {SPOT_CHECK_GROUPS}: "
            f"{pitch_observed[0][0]:.6f} um/px, shape (64, 1024), valid=1.0")

    power = float(cfg["single_line"]["power_w"])
    rows = []
    for record in design.itertuples(index=False):
        order = int(record.加工顺序)
        view_row = field.loc[order]
        rows.append({
            "single_line_id": order,
            "session_id": cfg["single_line"]["session_id"],
            "measurement_id": order,
            "sample_id": order,
            "roi_within_measurement": "single",
            "cag_path": str(view_row["cag_path"]),
            "cag_path_index": order,
            "pulse_duration_fs": int(record.pulse_duration_fs),
            "frequency_kHz": int(record.frequency_kHz),
            "velocity_mm_s": int(record.velocity_mm_s),
            "pass_count": int(record.pass_count),
            "pulse_pitch_um": int(record.velocity_mm_s) / int(record.frequency_kHz),
            "E_line_J_mm": power * int(record.pass_count) / int(record.velocity_mm_s),
            "E_line_J": power * int(record.pass_count) * 0.2 / int(record.velocity_mm_s),
            "power_W_or_proxy": power,
            "power_source_note": cfg["single_line"]["power_source_note"],
            "pixel_size_um": cfg["single_line"]["pixel_um"],
            "z_step_um": cfg["single_line"]["z_step_um"],
            "line_scan_direction": "image-frame: line axis ~0 deg (u+ = machining "
                                   "direction); start/end sign unknown",
            "line_scan_direction_note": "no per-sample scan-direction provenance; "
                                        "start/end symbol unrecorded (细则 §0.8)",
            "measurement_orientation_theta_deg": float(view_row["theta_line_deg"]),
            "orientation_confident": bool(view_row["orientation_confident"]),
            "processing_date_or_batch": "20260528_single_line_batch",
            "date_confidence": "filename_only",
            "height_data_type": "absolute_height_raw_primary|cone_repaired_sensitivity",
            "background_correction_status": "frozen_plane_from_view_manifest",
            "plane_a": float(view_row["plane_a"]),
            "plane_b": float(view_row["plane_b"]),
            "plane_c": float(view_row["plane_c"]),
            "plane_rmse_um": float(view_row["plane_rmse_um"]),
            "sigma_ref_um": float(view_row["sigma_ref_um"]),
            "valid_mask_status": "all_valid",
            "hatch_spacing_um": np.nan,
            "replicates": 1,
            "provenance_same_system": True,
            "provenance_system_note": "same laser & VK4/CAG measurement chain; "
                                      "identical measured post-objective power value "
                                      "on both datasets (no independent record)",
            "units_note": "tau fs / f kHz / v mm/s / N count / length um",
            "mapping_provenance": str(view_row["mapping_provenance"]),
            "exclusion_note": "氧化锆/72组单脉冲直线.cag excluded: no design table, "
                              "no provenance (上位规划 §4 red line)",
            "geometry_qc": "pending",
            "notes": "",
        })
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    p26.require(len(manifest) == 120 and manifest["single_line_id"].is_unique,
                "manifest must hold 120 unique rows")

    target = out_dir / "single_line_manifest.csv"
    manifest.to_csv(target, index=False, encoding="utf-8-sig")
    p26.log(f"manifest -> {target}")

    provenance = {
        "rows": int(len(manifest)),
        "design_grid": {"tau_fs": list(EXPECTED_TAU), "f_khz": list(EXPECTED_F),
                        "v_mm_s": list(EXPECTED_V), "N": list(EXPECTED_N)},
        "unique_conditions": int(len(design)),
        "pixel_pitch_um": cfg["single_line"]["pixel_um"],
        "cag_spot_check_groups": list(SPOT_CHECK_GROUPS),
        "checks": {
            "same_laser_system": True,
            "power_condition_convertible": True,
            "units_native_consistent": True,
            "pixel_size_trusted": True,
            "full_groove_cross_section": "checked_in_task_16",
            "edge_clipping_truncation": "checked_in_task_16",
            "repeated_positions": 0,
            "background_plane": "frozen_plane_from_view_manifest",
            "scan_direction_recoverability": "partial: axis yes, sign no",
        },
        "excluded": {"氧化锆/72组单脉冲直线.cag": "no design table -> no provenance"},
        "power_provenance": cfg["single_line"]["power_source_note"],
    }
    (out_dir / "manifest_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    p26.log("provenance json written; Task 15 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
