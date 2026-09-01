#!/usr/bin/env python3
"""WP1: freeze the manual_v1 registration contract before any comparison.

Reads the formal single-annotator four-edge table, re-verifies every frozen
baseline (row counts, session counts, completeness, SHA-256, geometric
identities, paired slot order/separation, centre within FOV), and -- only if
everything passes -- writes a read-only evidence snapshot plus a freeze
manifest.  Never modifies the formal table.  Any failure is a hard STOP.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.manual_registration_evaluation import (  # noqa: E402
    ManualRegistrationError, box_corner_margin_um,
    check_paired_measurements, manual_geometry_gate, validate_manual_identity,
)
from src.stage_manifest import append_stage_record, sha256_of  # noqa: E402

CONFIG_PATH = REPO / "config/manual_registration_v1.yaml"
REGISTRATION_DIR = REPO / "outputs/rectangle_registration/registration"
MANUAL_V1_DIR = REGISTRATION_DIR / "manual_v1"
SESSION_GEOMETRY_CSV = (REPO / "outputs/rectangle_registration/geometry"
                        / "session_geometry.csv")
MEASUREMENT_METRICS_CSV = (REPO / "outputs/rectangle_registration/inventory"
                           / "measurement_metrics.csv")
RUN_MANIFEST = (REPO / "outputs/rectangle_registration/manual_v1"
                / "run_manifest.json")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_sha = sha256_of(config_path)
    gate_cfg = config["manual_geometry_gate"]
    tolerance = float(config["canonical_identity_tolerance_um"])

    source_path = REPO / config["source_annotation_table"]
    failures: list[str] = []
    warnings: list[str] = []

    # ---- hash contract ------------------------------------------------
    source_sha = sha256_of(source_path)
    if source_sha != config["expected_source_sha256"]:
        failures.append(
            f"source SHA-256 {source_sha} != frozen expected "
            f"{config['expected_source_sha256']}; STOP unless the researcher "
            "explicitly confirms an intentional revision")

    rows = read_csv(source_path)

    # ---- row-count and session contracts -------------------------------
    if len(rows) != int(config["expected_rows"]):
        failures.append(f"row count {len(rows)} != {config['expected_rows']}")
    session_counts = Counter(row["session_id"] for row in rows)
    for session_id, expected in config["expected_session_counts"].items():
        if session_counts.get(session_id, 0) != int(expected):
            failures.append(
                f"session {session_id}: {session_counts.get(session_id, 0)} "
                f"rows != {expected}")
    state_counts = Counter(row["annotator_a_state"] for row in rows)
    if state_counts.get("complete", 0) != int(config["expected_annotator_a_complete"]):
        failures.append(
            f"annotator_a_state complete = "
            f"{state_counts.get('complete', 0)} != "
            f"{config['expected_annotator_a_complete']}")
    if any(state not in {"complete"} for state in state_counts):
        failures.append(f"unexpected annotator_a states: "
                        f"{dict(state_counts)}")
    for row in rows:
        for field in row:
            if field.startswith("annotator_b_") and str(row[field]).strip():
                failures.append(
                    f"annotator_b field {field} is non-empty for "
                    f"({row['session_id']}, {row['sample_id']})")
                break

    # ---- uniqueness ----------------------------------------------------
    keys = [(row["session_id"], row["sample_id"]) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        failures.append(f"duplicate (session_id, sample_id): {duplicates[:5]}")
    roi_values = Counter(row["roi_within_measurement"] for row in rows)
    if set(roi_values) != {"single", "slot_1", "slot_2"}:
        failures.append(f"unexpected roi values: {dict(roi_values)}")
    for session_id, expected in config["expected_session_counts"].items():
        paired_expected = 0 if session_id == "zro2_120_formal" else int(expected)
        actual = sum(1 for row in rows if row["session_id"] == session_id
                     and row["roi_within_measurement"] != "single")
        if actual != paired_expected:
            failures.append(
                f"session {session_id}: {actual} paired rows != {paired_expected}")

    # ---- supporting frozen tables --------------------------------------
    geometry_rows = read_csv(SESSION_GEOMETRY_CSV)
    geometry = {row["session_id"]: row for row in geometry_rows}
    for session_id, row in geometry.items():
        if row["status"] != "PASS" or row["d4_status"] != "confirmed":
            failures.append(
                f"session {session_id}: geometry status={row['status']} "
                f"d4_status={row['d4_status']} is not the frozen confirmed state")
    measurements = {
        (row["session_id"], int(row["measurement_id"])): row
        for row in read_csv(MEASUREMENT_METRICS_CSV)}

    # ---- per-row identity, geometry and paired gates ---------------------
    paired_issues = check_paired_measurements(
        rows, minimum_separation_um=float(
            gate_cfg["minimum_paired_center_separation_um"]),
        require_slot1_left=bool(gate_cfg["require_paired_order"]))
    failures.extend(paired_issues)

    corner_violations = 0
    for row in rows:
        session_id = row["session_id"]
        theta = float(geometry[session_id]["theta_session_deg"])
        try:
            identity = validate_manual_identity(
                row, theta, tolerance_um=tolerance)
        except ManualRegistrationError as exc:
            failures.append(f"identity: {exc}")
            continue
        gate_ok, gate_reason = manual_geometry_gate(
            identity["width_um"], identity["height_um"], gate_cfg)
        if not gate_ok:
            failures.append(f"geometry gate ({session_id}, {row['sample_id']}): "
                            f"{gate_reason}")
        measurement = measurements.get((session_id, int(row["measurement_id"])))
        if measurement is None:
            failures.append(f"no measurement metrics for ({session_id}, "
                            f"{row['measurement_id']})")
            continue
        fov_width = float(measurement["fov_width_um"])
        fov_height = float(measurement["fov_height_um"])
        if not (abs(identity["center_x_um"]) <= fov_width/2.0
                and abs(identity["center_y_um"]) <= fov_height/2.0):
            failures.append(
                f"manual centre outside measured FOV ({session_id}, "
                f"{row['sample_id']})")
        margin = box_corner_margin_um(
            left_u_um=identity["left_u_um"], right_u_um=identity["right_u_um"],
            top_v_um=identity["top_v_um"], bottom_v_um=identity["bottom_v_um"],
            theta_deg=theta, fov_width_um=fov_width, fov_height_um=fov_height)
        if margin < 0.0:
            corner_violations += 1
            warnings.append(
                f"({session_id}, {row['sample_id']}): manual box corner "
                f"exceeds raw FOV by {-margin:.2f} um (edge-of-field "
                "observation, not a gate)")

    report = {
        "stage": "WP1_freeze_manual_registration_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path.relative_to(REPO)),
                   "sha256": config_sha},
        "source": {"path": str(source_path.relative_to(REPO)),
                   "sha256": source_sha,
                   "expected_sha256": config["expected_source_sha256"],
                   "hash_matches_frozen_baseline": (
                       source_sha == config["expected_source_sha256"])},
        "row_count": len(rows),
        "session_counts": dict(sorted(session_counts.items())),
        "annotator_a_state_counts": dict(state_counts),
        "annotator_b_all_empty": not any(
            str(row[field]).strip() for row in rows
            for field in row if field.startswith("annotator_b_")),
        "paired_measurements": sum(
            1 for row in rows if row["roi_within_measurement"] != "single") // 2,
        "corner_overflow_warnings": corner_violations,
        "failures": failures,
        "warnings": warnings,
    }

    if failures:
        report["decision"] = "STOP"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("WP1 FREEZE FAILED -- the formal table was NOT modified and no "
              "snapshot was written.", file=sys.stderr)
        return 2

    # ---- all checks passed: write the frozen evidence snapshot ----------
    MANUAL_V1_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = MANUAL_V1_DIR / "manual_four_edge_validation_frozen.csv"
    snapshot_reused = False
    if snapshot_path.exists():
        # idempotent re-run: an already-frozen, byte-identical snapshot is
        # re-verified rather than rewritten (it is written read-only)
        if sha256_of(snapshot_path) == source_sha:
            snapshot_reused = True
        else:
            try:
                snapshot_path.chmod(0o644)
            except OSError:  # pragma: no cover - platform dependent
                pass
    if not snapshot_reused:
        shutil.copyfile(source_path, snapshot_path)
    snapshot_sha = sha256_of(snapshot_path)
    if snapshot_sha != source_sha:
        print("STOP: snapshot copy hash mismatch", file=sys.stderr)
        return 2
    # read-only evidence: drop the write bit where the OS supports it
    try:
        snapshot_path.chmod(0o444)
    except OSError:  # pragma: no cover - platform dependent
        pass

    report["decision"] = "PASS"
    report["snapshot"] = {"path": str(snapshot_path.relative_to(REPO)),
                          "sha256": snapshot_sha,
                          "reused_existing_snapshot": snapshot_reused}
    report["session_geometry"] = {
        session_id: {
            "theta_session_deg": float(row["theta_session_deg"]),
            "d4_transform_session": row["d4_transform_session"],
            "d4_status": row["d4_status"],
        } for session_id, row in sorted(geometry.items())}
    freeze_manifest_path = MANUAL_V1_DIR / "freeze_manifest.json"
    freeze_manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    append_stage_record(
        RUN_MANIFEST, stage="WP1_freeze_manual_registration_v1",
        command=[sys.executable, str(Path(__file__).relative_to(REPO))],
        exit_code=0,
        inputs={"config": config_path, "source_annotation_table": source_path,
                "session_geometry": SESSION_GEOMETRY_CSV,
                "measurement_metrics": MEASUREMENT_METRICS_CSV},
        outputs={"frozen_snapshot": snapshot_path,
                 "freeze_manifest": freeze_manifest_path},
        extra={"decision": "PASS", "row_count": len(rows),
               "source_sha256": source_sha})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
