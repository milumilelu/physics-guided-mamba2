#!/usr/bin/env python3
"""WP8-v3: full constrained four-edge registration with legacy scores as QA."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.four_edge_registration import fit_constrained_four_edges  # noqa: E402
from src.io_cag import CagHeightReader  # noqa: E402


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
    sessions = read_csv(REPO / config["paths"]["session_manifest"])
    views = read_csv(root / "inventory/sample_view_manifest.csv")
    planes = read_csv(root / "metrics/coarse_leveling_metrics.csv")
    plane_by_key = {(row["session_id"], int(row["measurement_id"])): row
                    for row in planes}
    geometry = {row["session_id"]: row for row in
                read_csv(root / "geometry/session_geometry.csv")}
    legacy = {(row["session_id"], int(row["sample_id"])): row for row in
              read_csv(root / "registration/translation_metrics.csv")}
    cfg = config["registration_v3"]
    metrics: list[dict] = []
    for session in sessions:
        sid = session["session_id"]
        theta = float(geometry[sid]["theta_session_deg"])
        session_views = sorted(
            (row for row in views if row["session_id"] == sid),
            key=lambda row: (int(row["measurement_id"]),
                             0 if row["roi_within_measurement"] == "slot_1" else 1))
        with CagHeightReader(REPO / session["cag_path"]) as reader:
            cached_measurement = None
            cached_map = None
            for view in session_views:
                measurement_id = int(view["measurement_id"])
                sample_id = int(view["sample_id"])
                if cached_measurement != measurement_id:
                    cached_map = reader.read_height_map(measurement_id)
                    cached_measurement = measurement_id
                plane = plane_by_key[(sid, measurement_id)]
                fit = fit_constrained_four_edges(
                    cached_map,
                    plane=tuple(float(plane[key]) for key in ("a", "b", "c")),
                    theta_deg=theta,
                    center_search=tuple(float(view[key]) for key in (
                        "center_search_x_min_um", "center_search_x_max_um",
                        "center_search_y_min_um", "center_search_y_max_um")),
                    nominal_size_um=float(cfg["nominal_size_um"]),
                    local_canvas_um=float(cfg["local_canvas_um"]),
                    profile_strip_halfwidth_um=float(
                        cfg["profile_strip_halfwidth_um"]),
                    smoothing_sigma_um=float(cfg["smoothing_sigma_um"]),
                    edge_search_halfwidth_um=float(cfg["edge_search_halfwidth_um"]),
                    peak_centroid_halfwidth_um=float(
                        cfg["peak_centroid_halfwidth_um"]),
                    minimum_edge_snr=float(cfg["edge_snr"]["minimum"]),
                    review_edge_snr=float(cfg["edge_snr"]["review_below"]),
                    width_range_um=tuple(float(value) for value in
                                         cfg["observed_width_um"]["hard_range"]),
                    review_residual_um=float(
                        cfg["constrained_edge_residual_um"]["review_above"]),
                    hard_residual_um=float(
                        cfg["constrained_edge_residual_um"]["hard_maximum"]),
                    boundary_tolerance_um=float(cfg["center_boundary_tolerance_um"]),
                )
                old = legacy[(sid, sample_id)]
                shift = float(np.hypot(
                    fit.center_x_um-float(old["center_x_um"]),
                    fit.center_y_um-float(old["center_y_um"])))
                metrics.append({
                    "registration_method": cfg["method"],
                    "evidence_level": int(cfg["evidence_level"]),
                    "session_id": sid, "measurement_id": measurement_id,
                    "sample_id": sample_id,
                    "roi_within_measurement": view["roi_within_measurement"],
                    "theta_session_deg": theta,
                    "d4_transform_session": "identity",
                    **fit.to_dict(),
                    "qa_legacy_region_score": old["region_score"],
                    "qa_legacy_edge_score": old["edge_score"],
                    "qa_legacy_weight_sensitivity_span_um": old["sensitivity_span_um"],
                    "qa_legacy_center_x_um": old["center_x_um"],
                    "qa_legacy_center_y_um": old["center_y_um"],
                    "qa_v3_vs_legacy_center_shift_um": shift,
                })

    paired_cfg = config["paired_registration"]
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in metrics:
        grouped.setdefault((row["session_id"], row["measurement_id"]), []).append(row)
    conflicts = 0
    for group in grouped.values():
        if len(group) != 2:
            continue
        slot1 = next(row for row in group if row["roi_within_measurement"] == "slot_1")
        slot2 = next(row for row in group if row["roi_within_measurement"] == "slot_2")
        separation = float(slot2["center_x_um"])-float(slot1["center_x_um"])
        conflict = (separation < float(paired_cfg["minimum_center_separation_um"])
                    or float(slot1["center_x_um"]) >= float(slot2["center_x_um"]))
        for row in group:
            row["paired_center_separation_um"] = separation
            row["slot_assignment_conflict"] = conflict
        if conflict:
            conflicts += 1
            for row in group:
                row["status"] = "STOP"
                row["warning"] = (row["warning"]+"; paired slot conflict").strip("; ")

    pass_count = sum(row["status"] == "PASS" for row in metrics)
    review_count = sum(row["status"] == "REVIEW" for row in metrics)
    stop_count = len(metrics)-pass_count-review_count
    decision = "STOP" if stop_count else "REVIEW" if review_count else "PASS"
    output = REPO / config["paths"]["registration_metrics"]
    write_csv(output, metrics)
    summary = {
        "stage": "WP8_v3_constrained_four_edge_registration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_frozen_before_run": "METHOD_V3_受约束四边稳健配准.md",
        "evidence_level": 3, "decision": decision,
        "samples": len(metrics), "pass": pass_count,
        "review": review_count, "stop": stop_count,
        "paired_conflict_measurements": conflicts,
    }
    (root / "registration/translation_summary_v3.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if decision in {"PASS", "REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
