#!/usr/bin/env python3
"""WP7: estimate session continuous angle and enforce the D4 manual gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.io_cag import CagHeightReader  # noqa: E402
from src.session_geometry import fit_free_square, pool_session_angle  # noqa: E402


ALLOWED_D4 = {
    "identity", "rot90", "rot180", "rot270",
    "flip_x", "flip_y", "transpose", "anti_transpose",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manual_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sessions = document.get("sessions", {})
    if not isinstance(sessions, dict):
        raise ValueError("manual orientation file must contain a sessions mapping")
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("config/rectangle_registration.yaml"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/rectangle_registration"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    root = (REPO / args.output_dir).resolve()
    prior = json.loads((root / "calibration/calibration_selection_summary.json")
                       .read_text(encoding="utf-8"))
    if prior.get("decision") != "PASS":
        print("STOP: WP6 calibration selection is not PASS", file=sys.stderr)
        return 2
    sessions = read_csv(REPO / config["paths"]["session_manifest"])
    selected = read_csv(root / "calibration/calibration_sample_ids.csv")
    sample_views = read_csv(root / "inventory/sample_view_manifest.csv")
    planes = read_csv(root / "metrics/coarse_leveling_metrics.csv")
    if args.dry_run:
        print(f"READY: sessions={len(sessions)} calibration_samples={len(selected)}")
        return 0

    view_by_key = {(row["session_id"], int(row["sample_id"])): row
                   for row in sample_views}
    plane_by_key = {(row["session_id"], int(row["measurement_id"])): row
                    for row in planes}
    geometry_cfg = config["session_geometry"]
    theta_cfg = geometry_cfg["theta"]
    free_cfg = theta_cfg["free_fit"]
    fit_rows: list[dict] = []
    for session in sessions:
        sid = session["session_id"]
        session_selected = sorted(
            (row for row in selected if row["session_id"] == sid),
            key=lambda row: int(row["sample_id"]))
        with CagHeightReader(REPO / session["cag_path"]) as reader:
            cached_measurement = None
            cached_map = None
            for selected_row in session_selected:
                sample_id = int(selected_row["sample_id"])
                view = view_by_key[(sid, sample_id)]
                measurement_id = int(view["measurement_id"])
                if cached_measurement != measurement_id:
                    cached_map = reader.read_height_map(measurement_id)
                    cached_measurement = measurement_id
                plane = plane_by_key[(sid, measurement_id)]
                fit = fit_free_square(
                    cached_map,
                    plane=(float(plane["a"]), float(plane["b"]), float(plane["c"])),
                    center_search=tuple(float(view[name]) for name in (
                        "center_search_x_min_um", "center_search_x_max_um",
                        "center_search_y_min_um", "center_search_y_max_um")),
                    nominal_size_um=float(free_cfg["nominal_size_um"]),
                    crop_margin_um=float(free_cfg["crop_margin_um"]),
                    threshold_fractions=tuple(float(value) for value in
                                              free_cfg["threshold_fractions"]),
                    component_area_ratio=tuple(float(value) for value in
                                               free_cfg["component_area_ratio"]),
                    boundary_quantile=float(free_cfg["boundary_quantile"]),
                    angle_grid_step_deg=float(free_cfg["angle_grid_step_deg"]),
                    minimum_successful_thresholds=int(
                        free_cfg["minimum_successful_thresholds"]),
                    quality_weights=theta_cfg["quality_weights"],
                )
                fit_rows.append({
                    "session_id": sid, "measurement_id": measurement_id,
                    "sample_id": sample_id,
                    "roi_within_measurement": view["roi_within_measurement"],
                    **fit.to_dict(),
                })

    manual_path = REPO / config["paths"]["manual_orientation"]
    manual = manual_records(manual_path)
    session_rows: list[dict] = []
    theta_pass = True
    d4_pass = True
    for session in sessions:
        sid = session["session_id"]
        pooled = pool_session_angle(
            [row for row in fit_rows if row["session_id"] == sid],
            warning_mad_deg=float(theta_cfg["warning_mad_deg"]),
            review_mad_deg=float(theta_cfg["review_mad_deg"]),
            histogram_bin_deg=float(theta_cfg["multimodality"]["bins_deg"]),
        )
        theta_pass &= pooled["status"] == "PASS"
        record = manual.get(sid, {})
        transform = record.get("d4_transform")
        confirmed = (record.get("status") ==
                     geometry_cfg["manual_gate_A0"]["required_status"]
                     and transform in ALLOWED_D4
                     and bool(str(record.get("evidence", "")).strip()))
        d4_pass &= confirmed
        session_rows.append({
            "session_id": sid, **pooled,
            "d4_transform_session": transform or "",
            "d4_status": "confirmed" if confirmed else "requires_manual_confirmation",
            "d4_evidence": record.get("evidence", ""),
        })

    geometry_dir = root / "geometry"
    write_csv(geometry_dir / "theta_sample_distribution.csv", fit_rows)
    write_csv(geometry_dir / "session_geometry.csv", session_rows)
    template = {
        "schema_version": 1,
        "instructions": (
            "Copy this file to config/manual_orientation.yaml, replace each "
            "pending status with confirmed, choose one listed D4 transform, "
            "and record independent microscope/scan-direction evidence. "
            "Do not choose from which machined side is deeper."),
        "allowed_transforms": sorted(ALLOWED_D4),
        "sessions": {
            row["session_id"]: {
                "status": "pending",
                "d4_transform": "identity",
                "evidence": "",
            } for row in session_rows
        },
    }
    geometry_dir.mkdir(parents=True, exist_ok=True)
    (geometry_dir / "manual_orientation_template.yaml").write_text(
        yaml.safe_dump(template, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    decision = ("PASS" if theta_pass and d4_pass else
                "AWAITING_MANUAL_CONFIRMATION" if theta_pass else "STOP")
    summary = {
        "stage": "WP7_session_geometry",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "theta_gate_pass": theta_pass,
        "d4_gate_pass": d4_pass,
        "h_reg_export_allowed": theta_pass and d4_pass,
        "calibration_fits": len(fit_rows),
        "sessions": session_rows,
        "manual_confirmation_file": str(manual_path),
        "manual_template": str(geometry_dir / "manual_orientation_template.yaml"),
    }
    (geometry_dir / "session_geometry_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if decision in {"PASS", "AWAITING_MANUAL_CONFIRMATION"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
