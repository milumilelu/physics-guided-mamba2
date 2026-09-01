"""Append-only stage manifest for versioned pipeline runs (manual_v1).

Every work-package script records its full command, exit code, input/output
SHA-256 hashes and environment versions into a run manifest dedicated to the
tagged run, so the audit trail can be reconstructed without reading code.
Records are only ever appended; history is never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["sha256_of", "hash_artifact", "environment_versions",
           "append_stage_record"]


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def hash_artifact(path: Path) -> dict:
    """Hash a file, or inventory a directory (count + aggregate hash)."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_dir():
        entries = []
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            entries.append(f"{item.relative_to(path).as_posix()}:"
                           f"{sha256_of(item)}")
        combined = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
        return {"path": str(path), "exists": True, "type": "directory",
                "file_count": len(entries),
                "aggregate_sha256": combined.upper()}
    return {"path": str(path), "exists": True, "type": "file",
            "sha256": sha256_of(path)}


def environment_versions() -> dict:
    versions = {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
    }
    for module_name in ("numpy", "scipy", "pandas", "matplotlib", "yaml"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "?"))
        except Exception:  # pragma: no cover - environment-dependent
            versions[module_name] = "unavailable"
    return versions


def append_stage_record(manifest_path: Path, *, stage: str,
                        command: list[str], exit_code: int,
                        inputs: dict[str, Path] | None = None,
                        outputs: dict[str, Path] | None = None,
                        extra: dict | None = None) -> None:
    """Append one immutable stage record to the tagged run manifest."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "stage_history": []}
    record = {
        "stage": stage,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(part) for part in command],
        "exit_code": int(exit_code),
        "environment": environment_versions(),
        "inputs": {name: hash_artifact(path)
                   for name, path in (inputs or {}).items()},
        "outputs": {name: hash_artifact(path)
                    for name, path in (outputs or {}).items()},
    }
    if extra:
        record.update(extra)
    manifest["stage_history"].append(record)
    manifest["stage"] = stage
    manifest["stage_last_updated_utc"] = record["recorded_at_utc"]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
