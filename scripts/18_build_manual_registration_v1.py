#!/usr/bin/env python3
"""WP3: build the unified manual_v1 primary registration table.

Input is ONLY the WP1 frozen evidence snapshot (never the editable formal
table).  Every row's centre comes from the manual four edges; the status is
the mechanical PASS/STOP of the frozen manual geometry and paired gates and
is never influenced by any automatic version.  v6 status and manual-vs-v6
differences are attached as QA fields only.
"""

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

from src.manual_registration_evaluation import (  # noqa: E402
    check_paired_measurements, manual_v1_record, paired_failure_keys, xy_to_uv)
from src.stage_manifest import append_stage_record, sha256_of  # noqa: E402

CONFIG_PATH = REPO / "config/manual_registration_v1.yaml"
MANUAL_V1_DIR = (REPO / "outputs/rectangle_registration/registration"
                 / "manual_v1")
SESSION_GEOMETRY_CSV = (REPO / "outputs/rectangle_registration/geometry"
                        / "session_geometry.csv")
MEASUREMENT_METRICS_CSV = (REPO / "outputs/rectangle_registration/inventory"
                           / "measurement_metrics.csv")
RUN_MANIFEST = (REPO / "outputs/rectangle_registration/manual_v1"
                / "run_manifest.json")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_sha = sha256_of(config_path)
    gate_cfg = config["manual_geometry_gate"]

    freeze_manifest = json.loads(
        (MANUAL_V1_DIR / "freeze_manifest.json").read_text(encoding="utf-8"))
    if freeze_manifest["decision"] != "PASS":
        print("STOP: WP1 freeze manifest is not PASS", file=sys.stderr)
        return 2
    snapshot_path = MANUAL_V1_DIR / "manual_four_edge_validation_frozen.csv"
    snapshot_sha = sha256_of(snapshot_path)
    if snapshot_sha != freeze_manifest["snapshot"]["sha256"]:
        print("STOP: frozen snapshot hash differs from freeze manifest",
              file=sys.stderr)
        return 2
    if snapshot_sha != config["expected_source_sha256"]:
        print("STOP: frozen snapshot no longer matches the frozen baseline "
              "hash in the config", file=sys.stderr)
        return 2
    snapshot_rows = read_csv(snapshot_path)

    geometry = {row["session_id"]: row
                for row in read_csv(SESSION_GEOMETRY_CSV)}
    measurements = {(row["session_id"], int(row["measurement_id"])): row
                    for row in read_csv(MEASUREMENT_METRICS_CSV)}

    # paired gate -> per-row paired_gate_pass flags (structured, never
    # string-matched out of the human-readable messages)
    paired_issues = check_paired_measurements(
        snapshot_rows,
        minimum_separation_um=float(
            gate_cfg["minimum_paired_center_separation_um"]),
        require_slot1_left=bool(gate_cfg["require_paired_order"]))
    failed_pair_keys = paired_failure_keys(
        snapshot_rows,
        minimum_separation_um=float(
            gate_cfg["minimum_paired_center_separation_um"]),
        require_slot1_left=bool(gate_cfg["require_paired_order"]))
    if bool(paired_issues) != bool(failed_pair_keys):  # pragma: no cover
        raise RuntimeError("paired gate: message/structured-key mismatch")

    # v6 QA comparator (statuses stay frozen; never used for decisions)
    v6_rows = {(row["session_id"], row["sample_id"]): row for row in read_csv(
        REPO / "outputs/rectangle_registration/registration"
        / "translation_metrics_v6.csv")}

    records: list[dict] = []
    for row in snapshot_rows:
        session_id = row["session_id"]
        geometry_row = geometry[session_id]
        theta = float(geometry_row["theta_session_deg"])
        measurement = measurements[(session_id, int(row["measurement_id"]))]
        paired_ok = ((session_id, int(row["measurement_id"]))
                     not in failed_pair_keys)
        record = manual_v1_record(
            row, theta_deg=theta,
            d4_transform=geometry_row["d4_transform_session"],
            source_sha256=snapshot_sha, config_sha256=config_sha,
            gate_cfg=gate_cfg,
            fov_width_um=float(measurement["fov_width_um"]),
            fov_height_um=float(measurement["fov_height_um"]),
            paired_gate_ok=paired_ok)
        v6 = v6_rows[(session_id, row["sample_id"])]
        v6_u, v6_v = xy_to_uv(float(v6["center_x_um"]),
                              float(v6["center_y_um"]), theta)
        record.update({
            "qa_v6_status": v6["status"],
            "qa_v6_center_u_um": v6_u,
            "qa_v6_center_v_um": v6_v,
            "qa_v6_delta_u_um": v6_u - record["manual_center_u_um"],
            "qa_v6_delta_v_um": v6_v - record["manual_center_v_um"],
            "qa_v6_center_disagreement_um": (
                (v6_u - record["manual_center_u_um"])**2
                + (v6_v - record["manual_center_v_um"])**2) ** 0.5,
        })
        records.append(record)

    records.sort(key=lambda r: (r["session_id"], r["sample_id"]))
    write_csv(MANUAL_V1_DIR / "translation_metrics_manual_v1.csv", records)

    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record["status"]] = \
            status_counts.get(record["status"], 0) + 1
    method_values = {record["registration_method"] for record in records}
    if method_values != {"manual_four_edge_a_v1"}:
        print(f"STOP: sample-wise method mixing detected: {method_values}",
              file=sys.stderr)
        return 2

    summary = {
        "stage": "WP3_build_manual_registration_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if status_counts == {"PASS": 200} else "STOP",
        "method": "manual_four_edge_a_v1",
        "evidence_level": 3,
        "center_source": "midpoint_of_four_manual_edges",
        "config": {"path": str(config_path.relative_to(REPO)),
                   "sha256": config_sha},
        "inputs": {
            "frozen_snapshot": {"path": str(
                snapshot_path.relative_to(REPO)), "sha256": snapshot_sha},
            "session_geometry": {"path": str(
                SESSION_GEOMETRY_CSV.relative_to(REPO))},
        },
        "row_count": len(records),
        "unique_keys": len({(r["session_id"], r["sample_id"])
                            for r in records}),
        "status_counts": status_counts,
        "geometry_gate_pass": sum(r["geometry_gate_pass"] for r in records),
        "paired_gate_pass": sum(r["paired_gate_pass"] for r in records),
        "center_within_fov": sum(r["center_within_fov"] for r in records),
        "box_corner_overflow_samples": sum(
            1 for r in records if r["box_corner_margin_um"] < 0.0),
        "samplewise_fallback": False,
        "qa_v6_field_role": ("QA only; v6 statuses remain frozen and never "
                             "influence manual_v1 status"),
        "qa_v6_status_counts": {
            status: sum(1 for r in records if r["qa_v6_status"] == status)
            for status in sorted({r["qa_v6_status"] for r in records})},
        "nominal_box_um": config["nominal_box_um"],
        "manual_box_sizes_are_qa_only": True,
    }
    summary_path = MANUAL_V1_DIR / "translation_summary_manual_v1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    append_stage_record(
        RUN_MANIFEST, stage="WP3_build_manual_registration_v1",
        command=[sys.executable, str(Path(__file__).relative_to(REPO))],
        exit_code=0,
        inputs={"config": config_path, "frozen_snapshot": snapshot_path,
                "session_geometry": SESSION_GEOMETRY_CSV,
                "measurement_metrics": MEASUREMENT_METRICS_CSV,
                "v6_metrics": REPO / "outputs/rectangle_registration/"
                "registration/translation_metrics_v6.csv"},
        outputs={"translation_metrics_manual_v1_csv": MANUAL_V1_DIR /
                 "translation_metrics_manual_v1.csv",
                 "translation_summary_manual_v1_json": summary_path},
        extra={"decision": summary["decision"],
               "status_counts": status_counts})
    return 0 if summary["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
