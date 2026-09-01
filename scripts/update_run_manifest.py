"""Append a stage record to the run manifest and reconcile the open gaps.

The manifest is the single audit trail for the pipeline: what ran, on which
commit, with which config, what was decided and what is still open.  Every work
package ends by recording itself here, so a reader can reconstruct the state of
the run without reading the code.

The manifest is never rewritten from scratch -- each call merges into what is
already there and pushes a snapshot onto ``stage_history``, so an earlier stage's
record cannot be quietly edited by a later one.

Usage
-----
    python scripts/update_run_manifest.py --stage WP2_decode_probe \\
        --set-json '{"decision":"PASS"}' --close-gaps G1,G9 \\
        --add-warning "text"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def git_state() -> dict:
    def run(*args: str) -> str:
        try:
            out = subprocess.run(["git", *args], cwd=REPO_ROOT,
                                 capture_output=True, text=True,
                                 encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return out.stdout.strip() if out.returncode == 0 else ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "commit_short": run("rev-parse", "--short", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_dirty": bool(run("status", "--porcelain")),
        "porcelain": [line for line in
                      run("status", "--porcelain").splitlines() if line],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",
                        default="outputs/rectangle_registration/run_manifest.json")
    parser.add_argument("--stage", required=True,
                        help="stage name, e.g. WP2_decode_probe")
    parser.add_argument("--set-json", default="{}",
                        help="JSON object merged into the stage record")
    parser.add_argument("--close-gaps", default="",
                        help="comma-separated gap ids that are now resolved")
    parser.add_argument("--open-gaps", default="",
                        help="comma-separated new gap descriptions to record")
    parser.add_argument("--add-warning", action="append", default=[],
                        help="warning text to append (repeatable)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    path = REPO_ROOT / args.manifest
    manifest = json.loads(path.read_text(encoding="utf-8"))

    stage = {
        "stage": args.stage,
        "recorded_at_local": dt.datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "git": git_state(),
    }
    try:
        stage.update(json.loads(args.set_json))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--set-json is not valid JSON: {exc}") from exc

    closed = [g.strip() for g in args.close_gaps.split(",") if g.strip()]
    if closed:
        stage["closes_gaps"] = closed
        manifest["open_gaps_blocking_phase0"] = [
            g for g in manifest.get("open_gaps_blocking_phase0", [])
            if not any(g.startswith(c) for c in closed)]

    opened = [g.strip() for g in args.open_gaps.split(",") if g.strip()]
    if opened:
        stage["opens_gaps"] = opened
        manifest.setdefault("open_gaps_blocking_phase0", []).extend(opened)

    if args.add_warning:
        stage["warnings_added"] = list(args.add_warning)
        manifest.setdefault("warnings", []).extend(args.add_warning)

    history = manifest.setdefault("stage_history", [])
    history.append(stage)
    manifest["stage"] = args.stage
    manifest["stage_last_updated_local"] = stage["recorded_at_local"]
    manifest["git"] = stage["git"]
    # A few stage fields also describe the current run as a whole.  Promote
    # them explicitly so the top-level summary cannot remain stale while the
    # immutable stage record still preserves exactly what was supplied.
    for key in ("decision", "next_action", "phase0_decision",
                "equivalence_decision", "blocker_codes"):
        if key in stage:
            manifest[key] = stage[key]

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"[manifest] stage={args.stage} recorded")
    print(f"[manifest] closed gaps: {closed or 'none'}")
    print(f"[manifest] open gaps remaining: "
          f"{len(manifest.get('open_gaps_blocking_phase0', []))}")
    print(f"[manifest] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
