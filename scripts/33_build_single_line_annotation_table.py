#!/usr/bin/env python3
"""Build the blinded single-line range annotation table and view manifest.

For every CAG group in the single-line dataset (``氧化锆/120组直线.cag``) this
script fits the frozen-pilot robust reference plane, estimates the depth-
weighted line orientation, and freezes a per-measurement view manifest
(``annotations/single_line_view_manifest.csv``) that the interactive
annotator (script 34) resamples from.  It also creates/refreshes the empty
annotation table ``annotations/single_line_range_annotation.csv``.

The orientation angle is a display-rotation convenience only; it is never
shown as a boundary, and the human still marks the whole elongated rectangle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as np_plot  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.io_cag import CagHeightReader  # noqa: E402
from src.manual_single_line_annotation import (  # noqa: E402
    CROP_MARGIN_UM,
    RANGE_FIELDS,
    annotation_table_columns,
    canonical_view_pixels,
    estimate_line_orientation,
    fit_reference_plane,
    plane_depth,
    rotated_crop_length_um,
)
from src.resampling import resample_to_canonical  # noqa: E402

SESSION_ID = "zro2_120_line"
MAPPING_PROVENANCE = (
    "pilot_user_confirmed: DOE 加工顺序 equals CAG Path "
    "(outputs/zro2_single_line_pilot/pilot_protocol.json)")

VIEW_FIELDS = (
    "session_id", "measurement_id", "sample_id", "roi_within_measurement",
    "cag_path", "mapping_provenance",
    "plane_a", "plane_b", "plane_c", "plane_rmse_um", "sigma_ref_um",
    "theta_line_deg", "orientation_center_x_um", "orientation_center_y_um",
    "orientation_threshold_um", "orientation_signal_pixels",
    "orientation_confident",
    "crop_center_x_um", "crop_center_y_um", "crop_length_um", "crop_pixels",
    "raw_width_um", "raw_height_um", "valid_pixel_ratio",
)

STATES = ("complete", "unusable")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def read_design_orders(design_path: Path) -> list[int]:
    column = "加工顺序"
    design = None
    for encoding in ("utf-8-sig", "gbk"):
        try:
            design = pd.read_csv(design_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if design is None:
        raise SystemExit(f"cannot decode design table {design_path}")
    if column not in design.columns:
        raise SystemExit(f"design table {design_path} lacks column {column}")
    return [int(value) for value in design[column].tolist()]


def load_existing_annotations(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame({name: pd.Series(dtype=object) for name in columns})
    existing = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    for name in columns:
        if name not in existing.columns:
            existing[name] = ""
        existing[name] = existing[name].astype(object)
    return existing[[name for name in columns if name in existing.columns]]


def build_view_rows(reader: CagHeightReader, cag_path: str, *,
                    threshold_k: float, margin_um: float,
                    limit: int | None) -> tuple[list[dict], list[str]]:
    view_rows: list[dict] = []
    warnings: list[str] = []
    groups = reader.groups
    expected = limit if limit is not None else len(groups)
    for position, group in enumerate(groups):
        if position >= expected:
            break
        hm = reader.read_height_map(group)
        fit = fit_reference_plane(hm.z, hm.valid_mask, hm.dx_um, hm.dy_um)
        depth = plane_depth(hm.z, hm.valid_mask, hm.dx_um, hm.dy_um, fit)
        orientation = estimate_line_orientation(
            depth, hm.valid_mask, hm.dx_um, hm.dy_um,
            sigma_ref_um=fit.sigma_ref_um, threshold_k=threshold_k)
        width_um = float(hm.x_um.size*hm.dx_um)
        height_um = float(hm.y_um.size*hm.dy_um)
        crop_length = rotated_crop_length_um(
            width_um, height_um, orientation.theta_deg, margin_um=margin_um)
        view_rows.append({
            "session_id": SESSION_ID,
            "measurement_id": int(group),
            "sample_id": int(group),
            "roi_within_measurement": "single",
            "cag_path": cag_path,
            "mapping_provenance": MAPPING_PROVENANCE,
            "plane_a": fit.a, "plane_b": fit.b, "plane_c": fit.c,
            "plane_rmse_um": fit.rmse_um, "sigma_ref_um": fit.sigma_ref_um,
            "theta_line_deg": orientation.theta_deg,
            "orientation_center_x_um": orientation.center_x_um,
            "orientation_center_y_um": orientation.center_y_um,
            "orientation_threshold_um": orientation.threshold_um,
            "orientation_signal_pixels": orientation.signal_pixels,
            "orientation_confident": bool(orientation.confident),
            "crop_center_x_um": 0.0, "crop_center_y_um": 0.0,
            "crop_length_um": crop_length,
            "crop_pixels": canonical_view_pixels(crop_length, hm.dx_um),
            "raw_width_um": width_um, "raw_height_um": height_um,
            "valid_pixel_ratio": float(np.mean(hm.valid_mask)),
        })
        if not orientation.confident:
            warnings.append(
                f"group {group}: only {orientation.signal_pixels} signal "
                "pixels above threshold; theta forced to 0, review the view")
        print(f"group {group:3d}: plane rmse={fit.rmse_um:.3f} um "
              f"sigma_ref={fit.sigma_ref_um:.3f} um "
              f"theta={orientation.theta_deg:+.3f} deg "
              f"signal={orientation.signal_pixels}px "
              f"crop={crop_length:.1f} um", flush=True)
    return view_rows, warnings


def render_previews(view_rows: list[dict], reader: CagHeightReader,
                    output_dir: Path, count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in view_rows[:count]:
        hm = reader.read_height_map(int(row["measurement_id"]))
        plane = (float(row["plane_a"]), float(row["plane_b"]),
                 float(row["plane_c"]))
        local = resample_to_canonical(
            hm, plane=plane,
            center_x_um=float(row["crop_center_x_um"]),
            center_y_um=float(row["crop_center_y_um"]),
            theta_deg=float(row["theta_line_deg"]),
            length_um=float(row["crop_length_um"]),
            pixels=int(row["crop_pixels"]), minimum_mask_weight=.99, order=1,
            metadata={"purpose": "single_line_annotation_view_preview"})
        depth = np.where(local.valid_mask,
                         float(np.median(local.z[local.valid_mask]))
                         - local.z, np.nan)
        finite = depth[local.valid_mask]
        lo, hi = np.percentile(finite, (2, 98))
        figure, axis = np_plot.subplots(figsize=(11, 3.2))
        axis.imshow(depth, extent=(local.x_um[0], local.x_um[-1],
                                   local.y_um[-1], local.y_um[0]),
                    cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
        axis.set_aspect("equal")
        axis.set_xlabel("canonical u (um)")
        axis.set_ylabel("canonical v (um)")
        axis.set_title(
            f"group {row['measurement_id']} preview | "
            f"theta={row['theta_line_deg']:+.3f} deg | "
            f"signal={row['orientation_signal_pixels']}px")
        figure.tight_layout()
        target = output_dir/f"group_{int(row['measurement_id']):03d}_preview.png"
        figure.savefig(target, dpi=130)
        np_plot.close(figure)
        print(f"preview -> {target}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cag", type=Path,
                        default=REPO/"氧化锆/120组直线.cag")
    parser.add_argument("--design", type=Path,
                        default=REPO/"氧化锆/氧化锆_line_design.csv")
    parser.add_argument("--table", type=Path, default=(
        REPO/"annotations/single_line_range_annotation.csv"))
    parser.add_argument("--view-manifest", type=Path, default=(
        REPO/"annotations/single_line_view_manifest.csv"))
    parser.add_argument("--annotator", default="A", choices=("A", "B", "a", "b"))
    parser.add_argument("--threshold-k", type=float, default=4.0)
    parser.add_argument("--margin-um", type=float, default=CROP_MARGIN_UM)
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke option: build only the first N groups")
    parser.add_argument("--render-preview", type=int, default=0,
                        help="render the first N canonical views as PNGs "
                             "into outputs/single_line_annotation_preview/")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even when the table already holds "
                             "completed annotations (values are preserved)")
    args = parser.parse_args()

    columns = annotation_table_columns(args.annotator)
    existing = load_existing_annotations(args.table, columns)
    if len(existing) and not args.force:
        state_column = f"annotator_{args.annotator.lower()}_state"
        finished = existing[state_column].astype(str).isin(STATES).sum()
        if finished:
            raise SystemExit(
                f"{args.table} already holds {finished} finished annotations; "
                "pass --force to rebuild (existing values are preserved)")
    design_orders = read_design_orders(args.design)
    print(f"design records: {len(design_orders)} | annotator="
          f"{args.annotator.upper()}", flush=True)
    cag_resolved = args.cag.resolve()
    try:
        cag_ref = cag_resolved.relative_to(REPO).as_posix()
    except ValueError:
        cag_ref = cag_resolved.as_posix()

    reader = CagHeightReader(args.cag)
    try:
        groups = reader.groups
        missing = sorted(set(design_orders)-set(groups))
        if missing:
            print(f"WARNING: design orders absent from the CAG: {missing}")
        view_rows, warnings = build_view_rows(
            reader, cag_ref, threshold_k=args.threshold_k,
            margin_um=args.margin_um, limit=args.limit)
        if args.render_preview:
            render_previews(view_rows, reader,
                            REPO/"outputs/single_line_annotation_preview",
                            args.render_preview)
    finally:
        reader.close()

    view_frame = pd.DataFrame(view_rows, columns=VIEW_FIELDS)
    atomic_write_csv(view_frame, args.view_manifest)
    print(f"view manifest -> {args.view_manifest} "
          f"({len(view_frame)} rows)", flush=True)

    table = pd.DataFrame({
        "session_id": [row["session_id"] for row in view_rows],
        "sample_id": [row["sample_id"] for row in view_rows],
        "measurement_id": [row["measurement_id"] for row in view_rows],
        "roi_within_measurement": [
            row["roi_within_measurement"] for row in view_rows],
    })
    for name in columns[4:]:
        table[name] = ""
    if len(existing):
        previous = existing.set_index(
            existing["measurement_id"].astype(int))
        for position, measurement in enumerate(table["measurement_id"]):
            if int(measurement) in previous.index:
                for name in columns[4:]:
                    table.at[position, name] = previous.at[
                        int(measurement), name]
    atomic_write_csv(table, args.table)
    finished = table[f"annotator_{args.annotator.lower()}_state"].astype(
        str).isin(STATES).sum()
    print(f"annotation table -> {args.table} ({len(table)} rows, "
          f"{finished} finished values preserved)", flush=True)
    for warning in warnings:
        print(f"REVIEW: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
