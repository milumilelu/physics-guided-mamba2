#!/usr/bin/env python3
"""WP9 gate: resolve each session common FOV before any resampling.

Supports versioned runs: ``--registration-metrics`` selects an alternative
frozen registration table (e.g. the manual_v1 table) and ``--output-tag``
redirects every writable artefact under ``<outputs_root>/<tag>/`` so legacy
archives are never overwritten.  Without these arguments the behaviour is
byte-compatible with the original v2 pipeline.

Exclusions are an **explicit** parameter.  Nothing is excluded unless
``--exclusion-policy <yaml>`` is passed, so the legacy default invocation can
never be changed by editing an exclusion file, and each versioned run carries
its own policy (manual_v1 uses config/phase_a_exclusions_manual_v1.yaml).
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

from src.canvas import available_centered_square_um, resolve_registered_grid  # noqa: E402
from src.manual_registration_evaluation import resolve_pipeline_paths  # noqa: E402
from src.stage_manifest import append_stage_record, sha256_of  # noqa: E402


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
    parser.add_argument(
        "--registration-metrics", default=None,
        help="alternative frozen registration table (repo-relative path); "
             "default comes from config paths.registration_metrics")
    parser.add_argument(
        "--output-tag", default=None,
        help="write all outputs under <outputs_root>/<tag>/ instead of the "
             "legacy locations (e.g. manual_v1)")
    parser.add_argument("--stage-manifest", default=None,
                        help="append a stage record to this run manifest")
    parser.add_argument(
        "--exclusion-policy", default=None,
        help="repo-relative path to an exclusion-policy YAML; if omitted NO "
             "sample is excluded (exclusions are never implicit)")
    args = parser.parse_args(argv)

    config = yaml.safe_load((REPO / "config/rectangle_registration.yaml")
                            .read_text(encoding="utf-8"))
    root = REPO / config["paths"]["outputs_root"]
    registration_path = (REPO / args.registration_metrics
                         if args.registration_metrics
                         else REPO / config["paths"]["registration_metrics"])
    paths = resolve_pipeline_paths(root, args.output_tag)
    registrations = read_csv(registration_path)
    exclusion_policy_path = (REPO / args.exclusion_policy
                             if args.exclusion_policy else None)
    excluded_sessions: set[str] = set()
    excluded_measurements: set[tuple[str, int]] = set()
    exclusion_reasons: dict[tuple[str, int], str] = {}
    exclusion_entries: list[dict] = []
    if exclusion_policy_path is not None:
        exclusions = yaml.safe_load(
            exclusion_policy_path.read_text(encoding="utf-8"))
        exclusion_entries = exclusions["exclusions"]
        for item in exclusion_entries:
            if item.get("all_samples") or item.get("all_measurements"):
                excluded_sessions.add(item["session_id"])
            for measurement_id in item.get("measurement_ids", []):
                excluded_measurements.add((item["session_id"],
                                           int(measurement_id)))
    measurements = read_csv(root / "inventory/measurement_metrics.csv")
    measurement_by_key = {
        (row["session_id"], int(row["measurement_id"])): row
        for row in measurements}
    diagnostic_rows: list[dict] = []
    for row in registrations:
        measurement = measurement_by_key[(row["session_id"], int(row["measurement_id"]))]
        available = available_centered_square_um(
            fov_width_um=float(measurement["fov_width_um"]),
            fov_height_um=float(measurement["fov_height_um"]),
            center_x_um=float(row["center_x_um"]),
            center_y_um=float(row["center_y_um"]),
            theta_deg=float(row["theta_session_deg"]),
        )
        row_session = row["session_id"]
        row_measurement = int(row["measurement_id"])
        excluded = (row_session in excluded_sessions
                    or (row_session, row_measurement) in excluded_measurements)
        reason = ""
        disposition = ""
        for item in exclusion_entries:
            if item["session_id"] != row_session:
                continue
            whole_session = bool(item.get("all_samples")
                                 or item.get("all_measurements"))
            if whole_session or row_measurement in [
                    int(value) for value in item.get("measurement_ids", [])]:
                reason = " ".join(str(item["reason"]).split())
                disposition = str(item.get("disposition", ""))
                break
        diagnostic_rows.append({
            "session_id": row["session_id"],
            "measurement_id": row["measurement_id"],
            "sample_id": row["sample_id"],
            "registration_status": row["status"],
            "included_in_phase_a": not excluded,
            "exclusion_reason": reason if excluded else "",
            "exclusion_disposition": disposition if excluded else "",
            "available_centered_square_um": available,
            "limiting_sample": False,
        })
    canvas_cfg = config["registered_canvas"]
    coarsest = max(max(float(row["dx_um"]), float(row["dy_um"]))
                   for row in measurements)
    session_rows: list[dict] = []
    for sid in sorted({row["session_id"] for row in diagnostic_rows}):
        session_samples = [row for row in diagnostic_rows
                           if row["session_id"] == sid and row["included_in_phase_a"]]
        if not session_samples:
            session_rows.append({
                "session_id": sid, "common_fov_um": None,
                "status": "EXCLUDED", "registered_fov_um": None,
                "grid_pixels": None, "pixel_um": None,
                "warning": ("session excluded by the explicit exclusion "
                            "policy"),
                "external_reference_width_um": None,
                "limiting_measurement_id": None, "limiting_sample_id": None,
            })
            continue
        limiting = min(session_samples, key=lambda row: row["available_centered_square_um"])
        limiting["limiting_sample"] = True
        common = float(limiting["available_centered_square_um"])
        grid = resolve_registered_grid(
            common_fov_um=common,
            preferred_size_um=float(canvas_cfg["preferred_size_um"][0]),
            minimum_size_um=float(canvas_cfg["minimum_size_um"][0]),
            coarsest_input_pixel_um=coarsest,
        )
        registered = grid["registered_fov_um"]
        session_rows.append({
            "session_id": sid, "common_fov_um": common,
            **grid,
            "external_reference_width_um": (
                (float(registered)-200.0)/2.0 if registered is not None else None),
            "limiting_measurement_id": limiting["measurement_id"],
            "limiting_sample_id": limiting["sample_id"],
        })
    decision = ("PASS" if all(row["status"] in {"PASS", "EXCLUDED"}
                              for row in session_rows) else "STOP")
    write_csv(paths.sample_fov_diagnostics_csv, diagnostic_rows)
    write_csv(paths.session_canvas_csv, session_rows)
    summary = {
        "stage": "WP9_common_fov_gate" + (
            f"_{args.output_tag}" if args.output_tag else ""),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision, "resampling_performed": False,
        "output_tag": args.output_tag,
        "registration_metrics": str(registration_path.relative_to(REPO)),
        "exclusion_policy": (str(exclusion_policy_path.relative_to(REPO))
                             if exclusion_policy_path else None),
        "exclusion_policy_sha256": (sha256_of(exclusion_policy_path)
                                    if exclusion_policy_path else None),
        "included_samples": sum(row["included_in_phase_a"] for row in diagnostic_rows),
        "excluded_samples": sum(not row["included_in_phase_a"] for row in diagnostic_rows),
        "coarsest_input_pixel_um": coarsest,
        "sessions": session_rows,
    }
    paths.common_fov_summary_json.parent.mkdir(parents=True, exist_ok=True)
    paths.common_fov_summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.stage_manifest:
        command = [sys.executable,
                   str(Path(__file__).relative_to(REPO)),
                   "--registration-metrics",
                   str(registration_path.relative_to(REPO)),
                   "--output-tag", str(args.output_tag)]
        manifest_inputs = {"registration_metrics": registration_path,
                           "measurement_metrics": root /
                           "inventory/measurement_metrics.csv",
                           "rectangle_registration_yaml": REPO /
                           "config/rectangle_registration.yaml"}
        if exclusion_policy_path is not None:
            command += ["--exclusion-policy",
                        str(exclusion_policy_path.relative_to(REPO))]
            manifest_inputs["exclusion_policy"] = exclusion_policy_path
        append_stage_record(
            Path(args.stage_manifest),
            stage=summary["stage"],
            command=command,
            exit_code=0 if decision == "PASS" else 2,
            inputs=manifest_inputs,
            outputs={"sample_fov_diagnostics": paths.sample_fov_diagnostics_csv,
                     "session_canvas": paths.session_canvas_csv,
                     "common_fov_summary": paths.common_fov_summary_json},
            extra={"decision": decision})
    return 0 if decision == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
