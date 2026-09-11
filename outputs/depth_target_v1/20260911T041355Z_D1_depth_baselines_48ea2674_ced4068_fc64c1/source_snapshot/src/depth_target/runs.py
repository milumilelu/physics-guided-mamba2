"""Run creation and sealing for depth_target_v1 (mirrors task_state_v1 conventions)."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from src.task_state_learning.artifacts import ROOT, sha256, write_json

RUNS_DIR = ROOT / "outputs" / "depth_target_v1"


def new_run(experiment_id, config_path):
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       encoding="utf-8").strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}_{experiment_id}_{sha256(config_path)[:8]}_{revision[:7]}_{uuid.uuid4().hex[:6]}"
    out = RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, out / "config_source.yaml")
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain=v1"],
                                     encoding="utf-8")
    write_json(out / "git_state.json", {"repo_full_sha": revision, "repo_dirty": bool(status),
                                        "status": status})
    versions = {p: version(p) for p in ["numpy", "scipy", "pandas", "scikit-learn", "PyYAML"]}
    write_json(out / "environment.json", {"python": sys.version, "executable": sys.executable,
                                          "platform": platform.platform(), "packages": versions})
    sources = []
    for folder in ["src/depth_target", "experiments/depth_target_v1",
                   "config/depth_target_v1", "tests/depth_target_v1"]:
        base = ROOT / folder
        if not base.exists():
            continue
        for source in sorted(base.rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts:
                dest = out / "source_snapshot" / source.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
                sources.append(source)
    rows = [{"path": str(p.resolve().relative_to(ROOT)).replace("\\", "/"),
             "sha256": sha256(p)} for p in sources]
    write_json(out / "source_snapshot.json", {"files": rows})
    return out


def seal(out):
    rows = [{"path": p.relative_to(out).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(Path(out).rglob("*")) if p.is_file() and p.name != "release_manifest.json"]
    write_json(Path(out) / "release_manifest.json", {"files": rows})


def verify_seal(out):
    out = Path(out).resolve()
    if not out.is_relative_to(RUNS_DIR):
        raise ValueError("Upstream run must be within outputs/depth_target_v1")
    for row in json.loads((out / "release_manifest.json").read_text(encoding="utf-8"))["files"]:
        path = (out / row["path"]).resolve()
        if not path.is_relative_to(out) or sha256(path) != row["sha256"]:
            raise ValueError(f"Upstream artifact hash mismatch: {row['path']}")
    return out


def verify_upstream_seal_tolerant(out, allowed_parent):
    """Seal verification for historical runs, tolerant of post-hoc CRLF->LF normalization.

    Some workspace process converted historical text artifacts from CRLF (sealed)
    to LF. Parsed content is unchanged; we verify each file against its sealed
    hash either as-is or after LF->CRLF reconstruction, and report the mode per
    file. Historical files are never rewritten.
    """
    out = Path(out).resolve()
    if not out.is_relative_to(Path(allowed_parent)):
        raise ValueError(f"Upstream run must be within {allowed_parent}")
    report = {"run": str(out), "n_files": 0, "exact": 0, "crlf_reconstructed": 0,
              "mismatched": []}
    for row in json.loads((out / "release_manifest.json").read_text(encoding="utf-8"))["files"]:
        path = (out / row["path"]).resolve()
        if not path.is_relative_to(out) or not path.exists():
            report["mismatched"].append(row["path"])
            continue
        raw = path.read_bytes()
        import hashlib
        digest = hashlib.sha256(raw).hexdigest()
        report["n_files"] += 1
        if digest == row["sha256"]:
            report["exact"] += 1
            continue
        rebuilt = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        if hashlib.sha256(rebuilt).hexdigest() == row["sha256"]:
            report["crlf_reconstructed"] += 1
        else:
            report["mismatched"].append(row["path"])
    if report["mismatched"]:
        raise ValueError(f"Upstream seal failure (content-level): {report['mismatched']}")
    return out, report
