"""Exclusive run creation and reproducibility snapshots for new experiments."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], encoding="utf-8")


def input_manifest(paths):
    rows = [{"path": str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/"),
             "bytes": Path(p).stat().st_size, "sha256": sha256(p)} for p in paths]
    rows = sorted(rows, key=lambda row: row["path"])
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return rows, digest


def new_run(experiment_id, config_path):
    revision = git("rev-parse", "HEAD").strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}_{experiment_id}_{sha256(config_path)[:8]}_{revision[:7]}_{uuid.uuid4().hex[:6]}"
    out = ROOT / "outputs" / "task_state_v1" / name
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, out / "config_source.yaml")
    status = git("status", "--porcelain=v1")
    write_json(out / "git_state.json", {"repo_full_sha": revision, "repo_dirty": bool(status), "status": status})
    for staged, name in ((False, "working_tree.patch"), (True, "staged.patch")):
        args = ["git", "-C", str(ROOT), "diff", "--binary"] + (["--cached"] if staged else [])
        (out / name).write_bytes(subprocess.check_output(args))
    versions = {p: version(p) for p in ["numpy", "scipy", "pandas", "scikit-learn", "PyYAML"]}
    write_json(out / "environment.json", {"python": sys.version, "executable": sys.executable,
               "platform": platform.platform(), "packages": versions})
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], encoding="utf-8")
    (out / "environment_lock.txt").write_text(frozen, encoding="utf-8")
    # Snapshot all new implementation/config/test files, including untracked ones.
    sources = []
    for folder in ["src/task_state_learning", "experiments/task_state_v1", "config/task_state_v1", "tests/task_state_v1"]:
        for source in sorted((ROOT / folder).rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts:
                dest = out / "source_snapshot" / source.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
                sources.append(source)
    rows, digest = input_manifest(sources)
    write_json(out / "source_snapshot.json", {"sha256": digest, "files": rows})
    return out


def seal(out):
    rows = [{"path": p.relative_to(out).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "release_manifest.json"]
    write_json(out / "release_manifest.json", {"files": rows})


def verify_seal(out):
    out = Path(out).resolve()
    if not out.is_relative_to(ROOT / "outputs/task_state_v1"):
        raise ValueError("Upstream run must be within outputs/task_state_v1")
    for row in json.loads((out / "release_manifest.json").read_text(encoding="utf-8"))["files"]:
        path = (out / row["path"]).resolve()
        if not path.is_relative_to(out) or sha256(path) != row["sha256"]:
            raise ValueError(f"Upstream artifact hash mismatch: {row['path']}")
    return out
