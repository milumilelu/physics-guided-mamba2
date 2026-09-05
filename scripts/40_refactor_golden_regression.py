#!/usr/bin/env python3
"""WP1 golden-regression layer (regeneration): rerun the Phase 2.7 formal
pipelines for Tasks 22/23 into a SCRATCH tree and compare against the frozen
artifacts.  The frozen `outputs/phase2_7/` tree is READ-ONLY throughout
(v2.1 F8): writes are redirected with the `--output-root` flag added to
p27.load_config, and the script verifies the frozen tree's hashes did not
move.  Deterministic (seeded) pipelines must reproduce the frozen files
bit-for-bit; JSONs are compared value-exact.

Usage (from repo root, .venv):
    python scripts/40_refactor_golden_regression.py [--tasks 22,23]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / "outputs" / "phase2_8" / "_regression_scratch"
FROZEN = REPO / "outputs" / "phase2_7"
P27 = REPO / "experiments" / "phase2_7"
PY = sys.executable

# frozen artifacts each task must reproduce (relative to outputs/phase2_7).
# Note: envelope/forward_model_simulation.csv exists in the frozen tree but
# is NOT written by the current frozen Task 23 script (stale r1 leftover) --
# the regression targets what the frozen code actually produces.
EXPECTED = {
    22: ["hatch_ablation/hatch_ablation_cv.csv",
         "summary/gsl27_1_evaluation.json"],
    23: ["envelope/single_track_envelope.csv",
         "envelope/envelope_selection_compare.csv",
         "envelope/forward_model_diagnostic.csv",
         "envelope/bootstrap_delta_tv.csv",
         "summary/gsl27_3_evaluation.json"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def compare_json(frozen: Path, scratch: Path) -> str:
    a = json.loads(frozen.read_text(encoding="utf-8"))
    b = json.loads(scratch.read_text(encoding="utf-8"))
    return "EXACT" if a == b else "VALUE_DIFF"


def compare_csv(frozen: Path, scratch: Path) -> str:
    if frozen.read_bytes() == scratch.read_bytes():
        return "EXACT"
    import pandas as pd
    fa = pd.read_csv(frozen)
    fb = pd.read_csv(scratch)
    if fa.shape == fb.shape and list(fa.columns) == list(fb.columns):
        try:
            pd.testing.assert_frame_equal(fa, fb, check_exact=True)
            return "EXACT_AFTER_PARSE"
        except AssertionError:
            return "VALUE_DIFF"
    return "SHAPE_DIFF"


def run_task(task: int, rerun: bool = True) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True
    if rerun:
        script = P27 / ("22_hatch_ablation.py" if task == 22
                        else "23_single_track_envelope.py")
        cmd = [PY, str(script),
               "--output-root", str(SCRATCH.relative_to(REPO)).replace("\\", "/")]
        print(f"[task {task}] rerun -> {SCRATCH}", flush=True)
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if proc.returncode != 0:
            ok = False
            lines.append(f"task {task}: rerun FAILED rc={proc.returncode}")
            lines.append(proc.stdout[-2000:])
            lines.append(proc.stderr[-2000:])
            return ok, lines
    else:
        print(f"[task {task}] compare-only (existing scratch)", flush=True)
    for rel in EXPECTED[task]:
        frozen, scratch = FROZEN / rel, SCRATCH / rel
        if not scratch.exists():
            ok = False
            lines.append(f"task {task}: MISSING scratch artifact {rel}")
            continue
        verdict = (compare_json(frozen, scratch) if rel.endswith(".json")
                   else compare_csv(frozen, scratch))
        lines.append(f"task {task}: {rel} -> {verdict}")
        ok = ok and verdict in ("EXACT", "EXACT_AFTER_PARSE")
    return ok, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="22,23")
    parser.add_argument("--compare-only", action="store_true",
                        help="compare existing scratch against frozen "
                             "(skip the rerun subprocesses)")
    args = parser.parse_args()
    tasks = [int(t) for t in args.tasks.split(",")]

    frozen_before = tree_hashes(FROZEN)
    if not args.compare_only:
        if SCRATCH.exists():
            for p in sorted(SCRATCH.rglob("*"), reverse=True):
                p.unlink() if p.is_file() else p.rmdir()
        SCRATCH.mkdir(parents=True, exist_ok=True)

    report: list[str] = ["# Phase 2.8 golden regression (scratch rerun)",
                         "",
                         f"frozen tree hashed: {len(frozen_before)} files",
                         f"mode: {'compare-only' if args.compare_only else 'rerun'}",
                         ""]
    all_ok = True
    for task in tasks:
        ok, lines = run_task(task, rerun=not args.compare_only)
        all_ok = all_ok and ok
        report += lines + [""]

    frozen_after = tree_hashes(FROZEN)
    frozen_intact = frozen_before == frozen_after
    report.append(f"frozen tree unchanged: {frozen_intact}")
    all_ok = all_ok and frozen_intact

    report_path = REPO / "outputs" / "phase2_8" / "refactor_regression_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)
    print(f"report -> {report_path}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
