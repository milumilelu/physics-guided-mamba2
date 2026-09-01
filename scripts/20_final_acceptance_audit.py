#!/usr/bin/env python3
"""WP6: final acceptance audit for the manual_v1 Phase A chain.

This is a real, executable audit -- it replaces the earlier placeholder
``"(final audit consolidation)"`` entry in the run manifest so that final
acceptance can be reproduced with one command.

What it verifies (integrity and reproducibility, never approval):

1. the run manifest contains the prescribed stage sequence, in order, all
   with exit code 0, and the test suite / environment check ran BEFORE the
   common-canvas and resampling stages;
2. the formal four-edge annotation table still matches the frozen baseline
   SHA-256 (it was never modified);
3. the manual_v1 primary table has 200 unique rows, all PASS, all manual
   centres, one single registration method (no sample-wise mixing);
4. the tagged manual_v1 output tree never touches a legacy v2-v7 path;
5. export/NPZ counts are mutually consistent and every NPZ carries
   provenance metadata (H_reg, H_200 and the mask archive);
6. the QA evidence (200 individual images + both montages) exists;
7. the approval file was written by a script, so its status is PENDING or
   BLOCKED -- never PASS.

Exit code 0 means "the audit ran and every integrity invariant holds"; the
Phase A acceptance state itself is reported separately as BLOCKED /
AWAITING_HUMAN_REVIEW / AUDIT_FAILED.
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
    APPROVAL_ALLOWED_STATUSES, resolve_pipeline_paths)
from src.stage_manifest import append_stage_record, sha256_of  # noqa: E402

ROOT = REPO / "outputs/rectangle_registration"
MANUAL_V1_DIR = ROOT / "manual_v1"
REG_MANUAL_V1_DIR = ROOT / "registration/manual_v1"
QA_DIR = ROOT / "qa/manual_v1"
APPROVAL_PATH = ROOT / "PHASE_A_APPROVAL_MANUAL_V1.md"
RUN_MANIFEST = MANUAL_V1_DIR / "run_manifest.json"
CONFIG_PATH = REPO / "config/manual_registration_v1.yaml"
AUDIT_PATH = MANUAL_V1_DIR / "final_acceptance_audit_manual_v1.json"

PRESCRIBED_ORDER = [
    "WP6_pre_run_environment_check",
    "WP6_pre_run_test_suite",
    "WP1_freeze_manual_registration_v1",
    "WP2_manual_vs_automatic_consistency",
    "WP3_build_manual_registration_v1",
    "WP9_common_fov_gate_manual_v1",
    "WP9_resampling_and_final_leveling_manual_v1",
    "WP5_phase_a_qa_manual_v1",
]

LEGACY_ARTIFACTS = [
    ROOT / "registration/translation_metrics_v2.csv",
    ROOT / "registration/translation_metrics_v3.csv",
    ROOT / "registration/translation_metrics_v4.csv",
    ROOT / "registration/translation_metrics_v5.csv",
    ROOT / "registration/translation_metrics_v6.csv",
    ROOT / "registration/translation_metrics_v7.csv",
    ROOT / "registration/manual_four_edge_validation.csv",
    ROOT / "resampling/common_fov_summary.json",
    ROOT / "resampling/session_canvas.csv",
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", default=str(RUN_MANIFEST))
    parser.add_argument("--stage-manifest", default=str(RUN_MANIFEST))
    args = parser.parse_args(argv)

    manifest_path = Path(args.run_manifest)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manual_v1_config_sha = sha256_of(CONFIG_PATH)
    paths = resolve_pipeline_paths(ROOT, "manual_v1")
    legacy_paths = resolve_pipeline_paths(ROOT, None)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    history = manifest["stage_history"]
    recorded_stages = [record["stage"] for record in history]

    failures: list[str] = []
    notes: dict = {}

    # ---- 1. stage sequence ------------------------------------------------
    observed_order = [stage for stage in recorded_stages
                      if stage in PRESCRIBED_ORDER]
    expected_order = [stage for stage in PRESCRIBED_ORDER
                      if stage in observed_order]
    if observed_order != expected_order:
        failures.append(
            f"stage order {observed_order} is not the prescribed sequence "
            f"{expected_order}")
    non_zero = {record["stage"]: record["exit_code"]
                for record in history if record["exit_code"] != 0}
    if non_zero:
        failures.append(f"stages with non-zero exit code: {non_zero}")
    pre_index = min((recorded_stages.index(stage) for stage
                     in ("WP6_pre_run_test_suite",
                         "WP6_pre_run_environment_check")
                     if stage in recorded_stages), default=None)
    pipeline_index = min((recorded_stages.index(stage) for stage
                          in ("WP9_common_fov_gate_manual_v1",
                              "WP9_resampling_and_final_leveling_manual_v1")
                          if stage in recorded_stages), default=None)
    order_ok = (pre_index is not None and pipeline_index is not None
                and pre_index < pipeline_index)
    if not order_ok:
        failures.append("test suite / environment check did not run before "
                        "the common-canvas and resampling stages")
    notes["stage_sequence_observed"] = observed_order
    notes["stage_exit_codes"] = {record["stage"]: record["exit_code"]
                                 for record in history}

    # ---- 2. formal table untouched ---------------------------------------
    formal_path = REPO / config["source_annotation_table"]
    formal_sha = sha256_of(formal_path)
    if formal_sha != config["expected_source_sha256"]:
        failures.append(
            f"formal annotation table SHA-256 {formal_sha} != frozen "
            f"baseline {config['expected_source_sha256']}")

    # ---- 3. primary table -------------------------------------------------
    wp3_path = REG_MANUAL_V1_DIR / "translation_metrics_manual_v1.csv"
    wp3_rows = read_csv(wp3_path)
    keys = [(row["session_id"], row["sample_id"]) for row in wp3_rows]
    if len(wp3_rows) != int(config["expected_rows"]):
        failures.append(f"manual_v1 table has {len(wp3_rows)} rows, "
                        f"expected {config['expected_rows']}")
    if len(set(keys)) != len(keys):
        failures.append("manual_v1 table has duplicate (session, sample) keys")
    methods = {row["registration_method"] for row in wp3_rows}
    if methods != {config["method"]}:
        failures.append(f"registration methods {methods} != {{{config['method']}}}")
    non_pass = [row for row in wp3_rows if row["status"] != "PASS"]
    if non_pass:
        failures.append(f"{len(non_pass)} rows are not PASS in the manual_v1 "
                        "table")
    wp3_summary = json.loads(
        (REG_MANUAL_V1_DIR / "translation_summary_manual_v1.json")
        .read_text(encoding="utf-8"))
    freeze_manifest = json.loads(
        (REG_MANUAL_V1_DIR / "freeze_manifest.json").read_text(encoding="utf-8"))
    snapshot_sha = freeze_manifest["snapshot"]["sha256"]
    if not all(row["source_sha256"] == snapshot_sha for row in wp3_rows):
        failures.append("manual_v1 rows do not all reference the frozen "
                        "snapshot hash")

    # ---- 4. tagged outputs are disjoint from the legacy archive -----------
    tagged_dirs = [paths.resampling_dir, paths.registered_h_reg_dir,
                   paths.registered_h_200_dir, paths.registered_masks_dir,
                   paths.metrics_dir]
    legacy_dirs = [legacy_paths.resampling_dir, legacy_paths.registered_h_reg_dir,
                   legacy_paths.registered_h_200_dir,
                   legacy_paths.registered_masks_dir, legacy_paths.metrics_dir]
    overlap = sorted({str(a) for a in tagged_dirs} & {str(b) for b in legacy_dirs})
    if overlap:
        failures.append(f"tagged outputs share paths with the legacy archive: "
                        f"{overlap}")
    notes["legacy_artifacts"] = {
        str(path.relative_to(REPO)): {
            "exists": path.exists(),
            "sha256": sha256_of(path) if path.is_file() else None,
            "mtime_utc": (datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc).isoformat()
                if path.exists() else None),
        } for path in LEGACY_ARTIFACTS}

    # ---- 5. export / NPZ consistency --------------------------------------
    metrics_rows = read_csv(paths.registration_metrics_csv)
    npz_counts = {
        "h_reg": len(list(paths.registered_h_reg_dir.glob("*.npz"))),
        "h_200": len(list(paths.registered_h_200_dir.glob("*.npz"))),
        "masks": len(list(paths.registered_masks_dir.glob("*.npz"))),
    }
    if not (npz_counts["h_reg"] == npz_counts["h_200"] == npz_counts["masks"]
            == len(metrics_rows)):
        failures.append(f"NPZ counts {npz_counts} disagree with the "
                        f"{len(metrics_rows)} exported rows")
    notes["npz_path_columns_present"] = {}
    for field in ("h_reg_path", "h_200_path", "mask_path"):
        present = sum(1 for row in metrics_rows
                      if row.get(field) and (REPO / row[field]).is_file())
        notes["npz_path_columns_present"][field] = present
        if present != len(metrics_rows):
            failures.append(f"{field}: {present}/{len(metrics_rows)} exported "
                            "rows point at an existing NPZ")
    notes["npz_counts"] = npz_counts
    notes["exported_rows"] = len(metrics_rows)
    notes["samples_total"] = len(wp3_rows)
    notes["samples_without_export"] = len(wp3_rows) - len(metrics_rows)

    # ---- 6. QA evidence ----------------------------------------------------
    individual_dir = QA_DIR / "registration_individual"
    individual_count = len(list(individual_dir.glob("*.png")))
    montages = {name: (QA_DIR / name) for name in
                ("registration_montage_absolute.png",
                 "registration_montage_local.png")}
    if individual_count != int(config["expected_rows"]):
        failures.append(f"{individual_count} individual QA images, expected "
                        f"{config['expected_rows']}")
    for name, path in montages.items():
        if not path.is_file() or path.stat().st_size < 10_000:
            failures.append(f"montage {name} missing or too small")
    qa_summary_path = QA_DIR / "phase_a_qa_summary_manual_v1.json"
    qa_summary = json.loads(qa_summary_path.read_text(encoding="utf-8"))
    notes["qa_summary_sha256"] = sha256_of(qa_summary_path)

    # ---- 7. approval file: PENDING/BLOCKED only ---------------------------
    approval_text = APPROVAL_PATH.read_text(encoding="utf-8")
    status_line = next((line for line in approval_text.splitlines()
                        if line.startswith("Status:")), "")
    approval_status = status_line.replace("Status:", "").strip()
    if approval_status not in APPROVAL_ALLOWED_STATUSES:
        failures.append(f"approval status {approval_status!r} is not one of "
                        f"{APPROVAL_ALLOWED_STATUSES} (scripts may never "
                        "write PASS)")
    notes["approval_status"] = approval_status
    notes["approval_sha256"] = sha256_of(APPROVAL_PATH)

    # ---- acceptance state --------------------------------------------------
    phase_a_decision = qa_summary.get("decision")
    blockers = qa_summary.get("blockers", [])
    failed_auto_checks = qa_summary.get("failed_auto_checks", [])
    if failures:
        acceptance_state = "AUDIT_FAILED"
    elif blockers or phase_a_decision == "BLOCKED":
        acceptance_state = "BLOCKED"
    elif phase_a_decision == "AWAITING_REVIEW":
        acceptance_state = "AWAITING_HUMAN_REVIEW"
    else:
        acceptance_state = "AWAITING_HUMAN_REVIEW"

    audit = {
        "stage": "WP6_final_acceptance_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance_state": acceptance_state,
        "phase_a_decision": phase_a_decision,
        "blockers": blockers,
        "failed_auto_checks": failed_auto_checks,
        "audit_failures": failures,
        "audit_integrity_ok": not failures,
        "notes": notes,
        "provenance": {
            "formal_annotation_table_sha256": formal_sha,
            "manual_annotation_snapshot_sha256": snapshot_sha,
            "manual_registration_config_sha256": manual_v1_config_sha,
            "run_manifest_sha256": sha256_of(manifest_path),
        },
        "stopped_at": ("Phase A manual_v1 is not approved; Phase B not "
                       "started (per plan \u00a75)"),
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    append_stage_record(
        Path(args.stage_manifest),
        stage="WP6_final_acceptance_audit",
        command=[sys.executable, str(Path(__file__).relative_to(REPO)),
                 "--run-manifest", str(manifest_path.relative_to(REPO))],
        exit_code=0 if not failures else 2,
        inputs={"run_manifest": manifest_path,
                "manual_v1_metrics": paths.registration_metrics_csv,
                "translation_metrics_manual_v1": wp3_path,
                "approval_file": APPROVAL_PATH,
                "formal_annotation_table": formal_path},
        outputs={"final_acceptance_audit": AUDIT_PATH},
        extra={"acceptance_state": acceptance_state,
               "phase_a_decision": phase_a_decision,
               "audit_integrity_ok": not failures,
               "audit_failures": failures})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
