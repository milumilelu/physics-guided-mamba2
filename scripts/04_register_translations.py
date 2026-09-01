#!/usr/bin/env python3
"""WP8: register every sample by translation with session geometry frozen."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.io_cag import CagHeightReader  # noqa: E402
from src.registration import register_fixed_square  # noqa: E402


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


def main() -> int:
    config = yaml.safe_load((REPO / "config/rectangle_registration.yaml")
                            .read_text(encoding="utf-8"))
    root = REPO / config["paths"]["outputs_root"]
    geometry_summary = json.loads(
        (root / "geometry/session_geometry_summary.json").read_text(encoding="utf-8"))
    if geometry_summary.get("decision") != "PASS":
        print("STOP: WP7 session geometry is not PASS", file=sys.stderr)
        return 2
    session_geometry = {
        row["session_id"]: row for row in read_csv(root / "geometry/session_geometry.csv")}
    if any(row["d4_transform_session"] != "identity"
           for row in session_geometry.values()):
        print("STOP: this WP8 implementation requires D4 coordinates to be applied first",
              file=sys.stderr)
        return 2
    sessions = read_csv(REPO / config["paths"]["session_manifest"])
    views = read_csv(root / "inventory/sample_view_manifest.csv")
    planes = read_csv(root / "metrics/coarse_leveling_metrics.csv")
    plane_by_key = {(row["session_id"], int(row["measurement_id"])): row
                    for row in planes}
    registration_cfg = config["registration"]
    paired_cfg = config["paired_registration"]
    numerical_cfg = registration_cfg["numerical_search"]
    nominal = float(registration_cfg["fixed"]["nominal_size_um"][0])
    metrics: list[dict] = []
    for session in sessions:
        sid = session["session_id"]
        theta = float(session_geometry[sid]["theta_session_deg"])
        session_views = sorted(
            (row for row in views if row["session_id"] == sid),
            key=lambda row: (int(row["measurement_id"]), int(row["sample_id"])))
        with CagHeightReader(REPO / session["cag_path"]) as reader:
            cached_measurement = None
            cached_map = None
            for view in session_views:
                measurement_id = int(view["measurement_id"])
                if cached_measurement != measurement_id:
                    cached_map = reader.read_height_map(measurement_id)
                    cached_measurement = measurement_id
                plane = plane_by_key[(sid, measurement_id)]
                fit = register_fixed_square(
                    cached_map,
                    plane=tuple(float(plane[key]) for key in ("a", "b", "c")),
                    theta_deg=theta,
                    center_search=tuple(float(view[key]) for key in (
                        "center_search_x_min_um", "center_search_x_max_um",
                        "center_search_y_min_um", "center_search_y_max_um")),
                    nominal_size_um=nominal,
                    edge_band_um=float(registration_cfg["scores"]["edge_score"]["band_um"]),
                    primary_weights=tuple(float(v) for v in
                                          registration_cfg["weights"]["primary"]),
                    sensitivity_weights=tuple(tuple(float(v) for v in pair) for pair in
                                              registration_cfg["weights"]["sensitivity"]),
                    unstable_shift_um=float(
                        registration_cfg["weights"]["unstable_shift_um"]),
                    coarse_radius_um=float(numerical_cfg["coarse_radius_um"]),
                    coarse_grid_step_um=float(numerical_cfg["coarse_grid_step_um"]),
                    coarse_score_sampling_um=float(
                        numerical_cfg["coarse_score_sampling_um"]),
                    fine_radius_um=float(numerical_cfg["fine_radius_um"]),
                    fine_grid_step_um=float(numerical_cfg["fine_grid_step_um"]),
                    fine_score_sampling_um=float(
                        numerical_cfg["fine_score_sampling_um"]),
                )
                metrics.append({
                    "session_id": sid, "measurement_id": measurement_id,
                    "sample_id": int(view["sample_id"]),
                    "roi_within_measurement": view["roi_within_measurement"],
                    "theta_session_deg": theta,
                    "d4_transform_session": "identity",
                    **fit.to_dict(),
                })

    by_measurement: dict[tuple[str, int], list[dict]] = {}
    for row in metrics:
        by_measurement.setdefault((row["session_id"], row["measurement_id"]), []).append(row)
    paired_conflicts = 0
    for group in by_measurement.values():
        if len(group) != 2:
            continue
        slot1 = next((row for row in group if row["roi_within_measurement"] == "slot_1"), None)
        slot2 = next((row for row in group if row["roi_within_measurement"] == "slot_2"), None)
        conflict = slot1 is None or slot2 is None
        if not conflict and slot1["status"] != "STOP" and slot2["status"] != "STOP":
            separation = float(slot2["center_x_um"])-float(slot1["center_x_um"])
            conflict = (separation < float(paired_cfg["minimum_center_separation_um"])
                        or float(slot1["center_x_um"]) >= float(slot2["center_x_um"]))
            for row in group:
                row["paired_center_separation_um"] = separation
        if conflict:
            paired_conflicts += 1
            for row in group:
                row["slot_assignment_conflict"] = True
                row["status"] = "FAILED"
                row["warning"] = (str(row.get("warning", "")) +
                                  "; paired slot order/separation conflict").strip("; ")
        else:
            for row in group:
                row["slot_assignment_conflict"] = False

    pass_count = sum(row["status"] == "PASS" for row in metrics)
    review_count = sum(row["status"] == "REVIEW" for row in metrics)
    failed_count = len(metrics)-pass_count-review_count
    decision = "STOP" if failed_count else "REVIEW" if review_count else "PASS"
    write_csv(root / "registration/translation_metrics.csv", metrics)
    summary = {
        "stage": "WP8_constrained_translation_registration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "samples": len(metrics), "pass": pass_count, "review": review_count,
        "failed": failed_count, "paired_conflict_measurements": paired_conflicts,
        "session_geometry_frozen": True,
    }
    output = root / "registration/translation_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if decision in {"PASS", "REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
