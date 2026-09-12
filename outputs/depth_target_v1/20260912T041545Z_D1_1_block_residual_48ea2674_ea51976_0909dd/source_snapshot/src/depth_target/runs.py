"""Run creation and sealing for depth_target_v1 (mirrors task_state_v1 conventions)."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
from pathlib import Path
import hashlib
import subprocess
import sys
import uuid

from src.task_state_learning.artifacts import ROOT, sha256

RUNS_DIR = ROOT / "outputs" / "depth_target_v1"


TEXT_SUFFIXES = {".py", ".csv", ".json", ".yaml", ".yml", ".md", ".txt", ".log"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8", newline="\n")


def copy_source(source, dest):
    if source.suffix in TEXT_SUFFIXES:
        dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    else:
        dest.write_bytes(source.read_bytes())


def new_run(experiment_id, config_path):
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       encoding="utf-8").strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}_{experiment_id}_{sha256(config_path)[:8]}_{revision[:7]}_{uuid.uuid4().hex[:6]}"
    out = RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=False)
    copy_source(Path(config_path), out / "config_source.yaml")
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
                copy_source(source, dest)
                sources.append(dest)
    rows = [{"path": str(p.relative_to(out / "source_snapshot")).replace("\\", "/"),
             "sha256": sha256(p)} for p in sources]
    write_json(out / "source_snapshot.json", {"files": rows})
    return out


def seal(out):
    for path in Path(out).rglob("*"):
        if path.is_file() and path.suffix in TEXT_SUFFIXES and b"\r" in path.read_bytes():
            raise ValueError(f"New sealed text must use LF: {path}")
    rows = [{"path": p.relative_to(out).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(Path(out).rglob("*")) if p.is_file() and p.name != "release_manifest.json"]
    write_json(Path(out) / "release_manifest.json", {"format": "depth_target_lf_v1", "files": rows})



class SealError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"Upstream seal failure: {report['mismatched']}")


def file_status(path, expected, legacy=False):
    if not path.is_file():
        return "missing"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected:
        return "exact"
    if legacy and path.suffix in TEXT_SUFFIXES and b"\r" not in raw:
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            return "mismatch"
        if hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest() == expected:
            return "legacy_crlf_reconstructed"
    return "mismatch"


def seal_report(out, allowed_parent, legacy=False):
    out = Path(out).resolve()
    if not out.is_relative_to(Path(allowed_parent).resolve()):
        raise ValueError(f"Upstream run must be within {allowed_parent}")
    manifest = json.loads((out / "release_manifest.json").read_text(encoding="utf-8"))
    # New seals must survive checkout byte-for-byte.
    legacy = legacy and manifest.get("format") != "depth_target_lf_v1"
    report = {"run": str(out), "n_files": 0, "exact": 0, "crlf_reconstructed": 0,
              "mismatched": [], "files": []}
    seen = set()
    for row in manifest["files"]:
        path = (out / row["path"]).resolve()
        status = "mismatch" if not path.is_relative_to(out) or row["path"] in seen else file_status(path, row["sha256"], legacy)
        seen.add(row["path"])
        report["files"].append({"path": row["path"], "status": status})
        report["n_files"] += 1
        if status == "exact":
            report["exact"] += 1
        elif status == "legacy_crlf_reconstructed":
            report["crlf_reconstructed"] += 1
        else:
            report["mismatched"].append(row["path"])
    if report["mismatched"]:
        raise SealError(report)
    return out, report


def verify_seal(out):
    return seal_report(out, RUNS_DIR)[0]


def verify_upstream_seal_tolerant(out, allowed_parent):
    return seal_report(out, allowed_parent, legacy=True)


def verify_contract_binding(contract, config):
    """Bind D1 to D0 inputs, scientific configuration and feature implementation."""
    import csv
    import yaml
    snapshot = yaml.safe_load((contract / "protocol_snapshot.yaml").read_text(encoding="utf-8"))
    expected = dict(snapshot)
    expected.pop("audit_release_sha256", None)
    if expected != config:
        raise ValueError("Configuration drift vs D0 protocol")
    audit = ROOT / config["inputs"]["audit_run"]
    paths = {"height_package": ROOT / config["inputs"]["height_package"],
             "frozen_targets_E00": audit / "frozen_targets.csv",
             "frozen_outer_split_E00": audit / "split_manifest.csv",
             "frozen_inner_split_E00": audit / "inner_split_manifest.csv"}
    legacy = json.loads((contract / "release_manifest.json").read_text(encoding="utf-8")).get("format") != "depth_target_lf_v1"
    rows = list(csv.DictReader((contract / "input_provenance.csv").read_text(encoding="utf-8").splitlines()))
    if len(rows) != len(paths) or {r["role"] for r in rows} != set(paths):
        raise ValueError("Incomplete D0 input provenance")
    report = []
    for row in rows:
        path = paths[row["role"]].resolve()
        status = file_status(path, row["sha256"], legacy)
        if path != (ROOT / row["path"]).resolve() or status not in {"exact", "legacy_crlf_reconstructed"}:
            raise ValueError(f"Input drift vs D0: {row['role']} ({status})")
        report.append({"role": row["role"], "status": status})
    if file_status(audit / "release_manifest.json", snapshot["audit_release_sha256"], legacy) not in {"exact", "legacy_crlf_reconstructed"}:
        raise ValueError("Audit manifest drift vs D0")
    for name in ("features.py", "splits.py"):
        current = ROOT / "src/depth_target" / name
        frozen = contract / "source_snapshot/src/depth_target" / name
        if current.read_text(encoding="utf-8") != frozen.read_text(encoding="utf-8"):
            raise ValueError(f"Feature/partition implementation drift vs D0: {name}")
    return report
