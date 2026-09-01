#!/usr/bin/env python3
"""WP8-v5: full multi-band integrated step-contrast registration."""

from __future__ import annotations

import csv
import json
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.io_cag import CagHeightReader  # noqa: E402
from src.step_contrast_bootstrap import fit_step_contrast_bootstrap  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prior_rows(root: Path, filename: str) -> dict[tuple[str, int], dict]:
    return {
        (row["session_id"], int(row["sample_id"])): row
        for row in read_csv(root / "registration" / filename)
    }


def main() -> int:
    config = yaml.safe_load(
        (REPO / "config/rectangle_registration.yaml").read_text(encoding="utf-8")
    )
    root = REPO / config["paths"]["outputs_root"]
    sessions = read_csv(REPO / config["paths"]["session_manifest"])
    views = read_csv(root / "inventory/sample_view_manifest.csv")
    planes = read_csv(root / "metrics/coarse_leveling_metrics.csv")
    plane_by_key = {
        (row["session_id"], int(row["measurement_id"])): row for row in planes
    }
    geometry = {
        row["session_id"]: row
        for row in read_csv(root / "geometry/session_geometry.csv")
    }
    v2 = prior_rows(root, "translation_metrics.csv")
    v3 = prior_rows(root, "translation_metrics_v3.csv")
    v4 = prior_rows(root, "translation_metrics_v4.csv")
    cfg = config["registration_v5"]
    metrics: list[dict] = []

    for session in sessions:
        sid = session["session_id"]
        theta = float(geometry[sid]["theta_session_deg"])
        session_views = sorted(
            (row for row in views if row["session_id"] == sid),
            key=lambda row: (
                int(row["measurement_id"]),
                0 if row["roi_within_measurement"] == "slot_1" else 1,
            ),
        )
        with CagHeightReader(REPO / session["cag_path"]) as reader:
            cached_measurement = None
            cached_map = None
            for index, view in enumerate(session_views, start=1):
                measurement_id = int(view["measurement_id"])
                sample_id = int(view["sample_id"])
                if cached_measurement != measurement_id:
                    cached_map = reader.read_height_map(measurement_id)
                    cached_measurement = measurement_id
                plane = plane_by_key[(sid, measurement_id)]
                seed_offset = zlib.crc32(f"{sid}:{sample_id}".encode("utf-8"))
                fit = fit_step_contrast_bootstrap(
                    cached_map,
                    plane=tuple(float(plane[key]) for key in ("a", "b", "c")),
                    theta_deg=theta,
                    center_search=tuple(
                        float(view[key])
                        for key in (
                            "center_search_x_min_um",
                            "center_search_x_max_um",
                            "center_search_y_min_um",
                            "center_search_y_max_um",
                        )
                    ),
                    nominal_size_um=float(cfg["nominal_size_um"]),
                    local_canvas_um=float(cfg["local_canvas_um"]),
                    edge_search_halfwidth_um=float(cfg["edge_search_halfwidth_um"]),
                    center_grid_step_um=float(cfg["center_grid_step_um"]),
                    profile_strip_halfwidth_um=float(cfg["profile_strip_halfwidth_um"]),
                    smoothing_sigma_um=float(cfg["smoothing_sigma_um"]),
                    contrast_bandwidths_um=tuple(
                        float(value) for value in cfg["contrast_bandwidths_um"]
                    ),
                    boundary_gap_um=float(cfg["boundary_gap_um"]),
                    tangent_blocks=int(cfg["tangent_blocks"]),
                    bootstrap_replicates_per_band=int(
                        cfg["bootstrap_replicates_per_band"]
                    ),
                    random_seed=(int(cfg["random_seed"]) + seed_offset) % 2**32,
                    hard_minimum_total=float(
                        cfg["joint_evidence"]["hard_minimum_total"]
                    ),
                    review_below_total=float(
                        cfg["joint_evidence"]["review_below_total"]
                    ),
                    hard_minimum_per_axis=float(
                        cfg["joint_evidence"]["hard_minimum_per_axis"]
                    ),
                    ci_quantiles=tuple(float(v) for v in cfg["uncertainty"]["ci_quantiles"]),
                    review_ci_span_um=float(cfg["uncertainty"]["review_ci_span_um"]),
                    hard_ci_span_um=float(cfg["uncertainty"]["hard_ci_span_um"]),
                    review_mad_um=float(cfg["uncertainty"]["review_mad_um"]),
                    hard_mad_um=float(cfg["uncertainty"]["hard_mad_um"]),
                    histogram_bin_um=float(cfg["multimodality"]["histogram_bin_um"]),
                    minimum_secondary_fraction=float(
                        cfg["multimodality"]["minimum_secondary_fraction"]
                    ),
                    minimum_mode_separation_um=float(
                        cfg["multimodality"]["minimum_mode_separation_um"]
                    ),
                    boundary_tolerance_um=float(cfg["center_boundary_tolerance_um"]),
                )
                old2, old3, old4 = (
                    prior[(sid, sample_id)] for prior in (v2, v3, v4)
                )
                metrics.append(
                    {
                        "registration_method": cfg["method"],
                        "evidence_level": int(cfg["evidence_level"]),
                        "session_id": sid,
                        "measurement_id": measurement_id,
                        "sample_id": sample_id,
                        "roi_within_measurement": view["roi_within_measurement"],
                        "theta_session_deg": theta,
                        "d4_transform_session": "identity",
                        **fit.to_dict(),
                        "qa_v2_status": old2["status"],
                        "qa_v2_region_score": old2["region_score"],
                        "qa_v2_edge_score": old2["edge_score"],
                        "qa_v2_weight_sensitivity_span_um": old2["sensitivity_span_um"],
                        "qa_v3_status": old3["status"],
                        "qa_v4_status": old4["status"],
                        "qa_v5_vs_v2_center_shift_um": float(
                            np.hypot(
                                fit.center_x_um - float(old2["center_x_um"]),
                                fit.center_y_um - float(old2["center_y_um"]),
                            )
                        ),
                        "qa_v5_vs_v3_center_shift_um": float(
                            np.hypot(
                                fit.center_x_um - float(old3["center_x_um"]),
                                fit.center_y_um - float(old3["center_y_um"]),
                            )
                        ),
                        "qa_v5_vs_v4_center_shift_um": float(
                            np.hypot(
                                fit.center_x_um - float(old4["center_x_um"]),
                                fit.center_y_um - float(old4["center_y_um"]),
                            )
                        ),
                    }
                )
                if index % 20 == 0 or index == len(session_views):
                    print(f"{sid}: {index}/{len(session_views)}", flush=True)

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
        separation = float(slot2["center_x_um"]) - float(slot1["center_x_um"])
        conflict = (
            separation < float(paired_cfg["minimum_center_separation_um"])
            or float(slot1["center_x_um"]) >= float(slot2["center_x_um"])
        )
        for row in group:
            row["paired_center_separation_um"] = separation
            row["slot_assignment_conflict"] = conflict
            if conflict:
                row["status"] = "STOP"
                row["warning"] = (row["warning"] + "; paired slot conflict").strip("; ")
        conflicts += int(conflict)

    counts = {status: sum(row["status"] == status for row in metrics) for status in ("PASS", "REVIEW", "STOP")}
    decision = "STOP" if counts["STOP"] else "REVIEW" if counts["REVIEW"] else "PASS"
    output = REPO / config["paths"]["registration_metrics"]
    write_csv(output, metrics)
    summary = {
        "stage": "WP8_v5_integrated_step_contrast_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_frozen_before_run": "METHOD_V5_多带宽积分阶跃四边配准.md",
        "evidence_level": 3,
        "decision": decision,
        "samples": len(metrics),
        "pass": counts["PASS"],
        "review": counts["REVIEW"],
        "stop": counts["STOP"],
        "paired_conflict_measurements": conflicts,
    }
    (root / "registration/translation_summary_v5.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if decision in {"PASS", "REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
