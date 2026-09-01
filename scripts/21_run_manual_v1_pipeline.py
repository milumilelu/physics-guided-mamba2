#!/usr/bin/env python3
"""One-command driver for the manual_v1 Phase A chain (correct order).

The acceptance flow used to be run by hand, which put the test suite and the
environment check *after* the common-canvas / resampling stages and left the
final audit as a non-executable placeholder.  This driver fixes both: it runs
every stage in the prescribed order, records each real command with its exit
code into the run manifest, and finishes with scripts/20_final_acceptance_audit.py.

Order (strict):

    1. WP6_pre_run_environment_check   scripts/verify_environment.py
    2. WP6_pre_run_test_suite          python -m unittest discover -s tests
    3. WP1 freeze                      scripts/16_freeze_manual_registration_v1.py
    4. WP2 consistency (QA only)       scripts/17_evaluate_manual_vs_automatic.py
    5. WP3 primary table               scripts/18_build_manual_registration_v1.py
    6. WP9 common FOV gate             scripts/05_resolve_common_canvas.py (tagged)
    7. WP9 resampling + final leveling scripts/06_resample_and_final_level.py (tagged)
    8. WP5 Phase A QA                  scripts/19_generate_manual_v1_phase_a_qa.py
    9. WP6 final acceptance audit      scripts/20_final_acceptance_audit.py

Stages 3-9 record themselves into the run manifest; stages 1-2 are recorded by
this driver because they are not pipeline scripts.

``--fresh-manifest`` archives the current manifest (kept, never deleted) and
starts a new one, so the accepted sequence is a clean, self-contained record.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.stage_manifest import append_stage_record, sha256_of  # noqa: E402

ROOT = REPO / "outputs/rectangle_registration"
MANUAL_V1_DIR = ROOT / "manual_v1"
REG_MANUAL_V1_DIR = ROOT / "registration/manual_v1"
RUN_MANIFEST = MANUAL_V1_DIR / "run_manifest.json"
CONFIG_PATH = REPO / "config/manual_registration_v1.yaml"

REGISTRATION_METRICS = ("outputs/rectangle_registration/registration/manual_v1"
                        "/translation_metrics_manual_v1.csv")
EXCLUSION_POLICY = "config/phase_a_exclusions_manual_v1.yaml"
TAG = "manual_v1"


def python(*arguments: str) -> list[str]:
    return [sys.executable, *arguments]


def stage_commands(snapshot_sha: str | None) -> list[tuple[str, list[str]]]:
    """(stage name, command) pairs in the prescribed order."""
    npz_metadata = json.dumps({
        "registration_method": "manual_four_edge_a_v1",
        "manual_annotation_sha256": snapshot_sha,
        "manual_registration_config_sha256": sha256_of(CONFIG_PATH),
    }, ensure_ascii=False, sort_keys=True)
    return [
        ("WP6_pre_run_environment_check",
         python("scripts/verify_environment.py")),
        ("WP6_pre_run_test_suite",
         python("-m", "unittest", "discover", "-s", "tests")),
        ("WP1_freeze_manual_registration_v1",
         python("scripts/16_freeze_manual_registration_v1.py")),
        ("WP2_manual_vs_automatic_consistency",
         python("scripts/17_evaluate_manual_vs_automatic.py")),
        ("WP3_build_manual_registration_v1",
         python("scripts/18_build_manual_registration_v1.py")),
        ("WP9_common_fov_gate_manual_v1",
         python("scripts/05_resolve_common_canvas.py",
                "--registration-metrics", REGISTRATION_METRICS,
                "--output-tag", TAG,
                "--exclusion-policy", EXCLUSION_POLICY,
                "--stage-manifest", str(RUN_MANIFEST))),
        ("WP9_resampling_and_final_leveling_manual_v1",
         python("scripts/06_resample_and_final_level.py",
                "--registration-metrics", REGISTRATION_METRICS,
                "--output-tag", TAG,
                "--npz-metadata", npz_metadata,
                "--stage-manifest", str(RUN_MANIFEST))),
        ("WP5_phase_a_qa_manual_v1",
         python("scripts/19_generate_manual_v1_phase_a_qa.py")),
        ("WP6_final_acceptance_audit",
         python("scripts/20_final_acceptance_audit.py")),
    ]


def parse_test_totals(output: str) -> tuple[int, str]:
    match = re.search(r"^Ran (\d+) tests? in", output, re.MULTILINE)
    count = int(match.group(1)) if match else 0
    result = "OK" if output.rstrip().endswith("OK") else "FAILURES/ERRORS"
    return count, result


def current_snapshot_sha() -> str | None:
    freeze_manifest_path = REG_MANUAL_V1_DIR / "freeze_manifest.json"
    if not freeze_manifest_path.exists():
        return None
    return json.loads(freeze_manifest_path.read_text(
        encoding="utf-8"))["snapshot"]["sha256"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-manifest", action="store_true",
                        help="archive the existing run manifest and start a "
                             "new one before running the chain")
    args = parser.parse_args(argv)

    if args.fresh_manifest and RUN_MANIFEST.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = MANUAL_V1_DIR / f"run_manifest_archived_{stamp}.json"
        shutil.copyfile(RUN_MANIFEST, archive)
        print(f"[driver] previous manifest archived to "
              f"{archive.relative_to(REPO)}")
        RUN_MANIFEST.unlink()

    started = time.time()
    # the snapshot hash is only final after WP1, so the resampling command is
    # built once WP1 has completed
    resampling_index = 6
    for index, (stage_name, command) in enumerate(stage_commands(None)):
        if index == resampling_index:
            command = stage_commands(current_snapshot_sha())[index][1]
        print(f"[driver] ==== {stage_name} ====", flush=True)
        print("[driver] " + " ".join(command), flush=True)
        started_stage = time.time()
        completed = subprocess.run(command, cwd=str(REPO),
                                   capture_output=True, text=True)
        duration = time.time()-started_stage
        for line in (completed.stdout or "").strip().splitlines()[-4:]:
            print(f"  | {line}")
        if completed.returncode != 0:
            print((completed.stdout or "")[-3000:], file=sys.stderr)
            print((completed.stderr or "")[-3000:], file=sys.stderr)
            print(f"[driver] STOP: {stage_name} exited "
                  f"{completed.returncode}", file=sys.stderr)
            return completed.returncode or 2
        if stage_name == "WP6_pre_run_environment_check":
            append_stage_record(
                RUN_MANIFEST, stage=stage_name, command=command, exit_code=0,
                inputs={"verify_environment": REPO /
                        "scripts/verify_environment.py"},
                extra={"result": "environment=OK",
                       "duration_seconds": round(duration, 3)})
        elif stage_name == "WP6_pre_run_test_suite":
            count, result = parse_test_totals(completed.stdout or "")
            append_stage_record(
                RUN_MANIFEST, stage=stage_name, command=command, exit_code=0,
                extra={"tests_run": count, "result": result,
                       "duration_seconds": round(duration, 3),
                       "command_note": (
                           "plan prescribed 'pytest -q'; pytest is not "
                           "installed in the offline .venv, so unittest "
                           "discovery over the same tests/ directory is "
                           "used")})
        print(f"[driver] {stage_name} OK ({duration:.1f}s)", flush=True)

    print(f"[driver] full chain finished in {time.time()-started:.1f}s")
    print(f"[driver] run manifest: {RUN_MANIFEST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
