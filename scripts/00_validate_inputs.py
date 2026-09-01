"""Phase 0 gate for rectangle-registration inputs.

This script inventories real files and verifies only facts available before image
registration.  It intentionally returns a non-zero exit status when CAG/CSV
equivalence or sample-to-measurement mapping has not been established.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


VK4_KEYS = [
    "meas_conds",
    "color_peak",
    "color_light",
    "light",
    "unknown_4",
    "unknown_5",
    "height",
    "unknown_7",
    "unknown_8",
    "color_peak_thumb",
    "color_thumb",
    "light_thumb",
    "height_thumb",
    "assembly_info",
    "line_measure",
    "line_thickness",
    "string_data",
    "reserved",
]


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    cag_path: str
    design_path: str
    csv_subdir: str
    input_format: str
    mapping_rule: str
    rois_per_measurement: int
    mapping_provenance: str
    expected_design_rows: int


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading large CAGs into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def read_design(path: Path) -> tuple[pd.DataFrame, str]:
    """Read a design table with a short, deterministic encoding fallback list."""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"Unable to parse design table {path}: {' | '.join(errors)}")


def locate_measurement_map(archive: zipfile.ZipFile) -> tuple[str, bytes]:
    """Locate the Keyence MeasurementDataMap XML by its document marker."""
    for info in archive.infolist():
        if not 1_000 < info.file_size < 5_000_000:
            continue
        with archive.open(info) as stream:
            head = stream.read(256)
        if b"MeasurementDataMap" in head:
            return info.filename, archive.read(info)
    raise ValueError("CAG contains no MeasurementDataMap")


def parse_vk4_header(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any]:
    """Read only the VK4 header fields needed for Phase 0 metadata checks."""
    with archive.open(info) as stream:
        prefix = stream.read(256)
    if prefix[:4] != b"VK4_":
        raise ValueError(f"Not a VK4 stream: {info.filename}")
    offsets = dict(zip(VK4_KEYS, struct.unpack_from("<18I", prefix, 12)))
    measure_offset = offsets["meas_conds"]
    height_offset = offsets["height"]
    required = max(measure_offset + 45 * 4, height_offset + 5 * 4)
    if required > len(prefix):
        with archive.open(info) as stream:
            prefix = stream.read(required)
    x_pitch_pm, y_pitch_pm, z_step_pm = struct.unpack_from(
        "<3I", prefix, measure_offset + 42 * 4
    )
    width, height, bit_depth, marker, data_bytes = struct.unpack_from(
        "<5I", prefix, height_offset
    )
    return {
        "width_px": int(width),
        "height_px": int(height),
        "bit_depth": int(bit_depth),
        "implementation_marker": int(marker),
        "declared_height_bytes": int(data_bytes),
        "dx_um": float(x_pitch_pm * 1e-6),
        "dy_um": float(y_pitch_pm * 1e-6),
        "z_step_um": float(z_step_pm * 1e-6),
        "height_layout_consistent": bool(
            bit_depth == 32 and data_bytes == width * height * 4
        ),
    }


def inspect_cag(path: Path) -> dict[str, Any]:
    """Inventory measurement IDs and embedded VK4 metadata without decoding heights."""
    with zipfile.ZipFile(path) as archive:
        xml_entry, xml_bytes = locate_measurement_map(archive)
        root = ET.fromstring(xml_bytes.decode("utf-8-sig"))
        measurements = root.findall("MeasurementData")
        paths = [int(item.findtext("Path", "0")) for item in measurements]
        original_names = [item.findtext("OriginalFileName", "") for item in measurements]

        vk4_infos: dict[int, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            parts = info.filename.split("/")
            if len(parts) < 2 or not parts[1].isdigit() or info.file_size < 1_000_000:
                continue
            with archive.open(info) as stream:
                if stream.read(4) == b"VK4_":
                    vk4_infos[int(parts[1])] = info

        metadata_rows: list[dict[str, Any]] = []
        for measurement_id in sorted(vk4_infos):
            row = {"measurement_id": measurement_id}
            row.update(parse_vk4_header(archive, vk4_infos[measurement_id]))
            metadata_rows.append(row)

    shapes = sorted(
        {
            (row["width_px"], row["height_px"], row["dx_um"], row["dy_um"], row["z_step_um"])
            for row in metadata_rows
        }
    )
    return {
        "measurement_map_entry": xml_entry,
        "measurement_count": len(measurements),
        "measurement_ids": paths,
        "measurement_ids_unique": len(set(paths)) == len(paths),
        "measurement_ids_contiguous_1_to_n": sorted(paths)
        == list(range(1, len(paths) + 1)),
        "original_filenames": original_names,
        "embedded_vk4_count": len(metadata_rows),
        "height_layouts_all_consistent": bool(metadata_rows)
        and all(row["height_layout_consistent"] for row in metadata_rows),
        "unique_grid_specs": [
            {
                "width_px": shape[0],
                "height_px": shape[1],
                "dx_um": shape[2],
                "dy_um": shape[3],
                "z_step_um": shape[4],
            }
            for shape in shapes
        ],
    }


def git_value(repo: Path, *args: str) -> str:
    """Return a Git value, or an explicit unavailable marker."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write deterministic UTF-8 CSV output, including a header for empty tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("config/session_manifest.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/rectangle_registration.yaml"),
    )
    parser.add_argument(
        "--height-manifest",
        type=Path,
        default=Path("config/height_source_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rectangle_registration/phase0"),
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Diagnostic-only shortcut; a skipped-hash run cannot pass Phase 0.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest_path = (repo / args.manifest).resolve()
    config_path = (repo / args.config).resolve()
    height_manifest_path = (repo / args.height_manifest).resolve()
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evidence_path = (repo / config["input"]["equivalence_gate"]
                     ["evidence_path"]).resolve()

    manifest_df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    specs = [
        SessionSpec(
            session_id=str(row.session_id),
            cag_path=str(row.cag_path),
            design_path=str(row.design_path),
            csv_subdir=str(row.csv_subdir),
            input_format=str(row.input_format),
            mapping_rule=str(row.mapping_rule),
            rois_per_measurement=int(row.rois_per_measurement),
            mapping_provenance=str(row.mapping_provenance),
            expected_design_rows=int(row.expected_design_rows),
        )
        for row in manifest_df.itertuples(index=False)
    ]
    with height_manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        height_source_rows = list(csv.DictReader(stream))
    height_rows_by_session = {
        spec.session_id: [row for row in height_source_rows
                          if row["session_id"] == spec.session_id]
        for spec in specs
    }
    height_csvs = [(repo / row["csv_path"]).resolve()
                   for row in height_source_rows]
    inventory_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    session_details: dict[str, Any] = {}

    for spec in specs:
        cag_path = (repo / spec.cag_path).resolve()
        design_path = (repo / spec.design_path).resolve()
        for role, path in (("cag", cag_path), ("design", design_path)):
            exists = path.is_file()
            inventory_rows.append(
                {
                    "session_id": spec.session_id,
                    "role": role,
                    "path": str(path.relative_to(repo)) if exists else str(path),
                    "exists": int(exists),
                    "bytes": path.stat().st_size if exists else "",
                    "sha256": (
                        "SKIPPED"
                        if exists and args.skip_hashes
                        else sha256_file(path) if exists else ""
                    ),
                }
            )
            if not exists:
                blockers.append(
                    {
                        "code": "missing_input_file",
                        "session_id": spec.session_id,
                        "message": f"Missing {role}: {path}",
                    }
                )

        if not cag_path.is_file() or not design_path.is_file():
            continue

        design, encoding = read_design(design_path)
        cag = inspect_cag(cag_path)
        design_id_column = "加工顺序" if "加工顺序" in design.columns else ""
        design_ids = (
            [int(value) for value in design[design_id_column].tolist()]
            if design_id_column
            else []
        )
        design_ids_valid = bool(design_ids) and sorted(design_ids) == list(
            range(1, len(design_ids) + 1)
        )
        measurement_count = int(cag["measurement_count"])
        cardinality_match = len(design) == measurement_count * spec.rois_per_measurement
        session_height_rows = height_rows_by_session.get(spec.session_id, [])
        registered_measurements = [int(row["measurement_id"])
                                   for row in session_height_rows]
        registered_sample_ids: list[int] = []
        for row in session_height_rows:
            for column in ("slot_1_sample_id", "slot_2_sample_id"):
                value = str(row.get(column, "")).strip()
                if value:
                    registered_sample_ids.append(int(value))
        height_measurements_complete = (
            sorted(registered_measurements) == list(range(1, measurement_count + 1))
            and len(set(registered_measurements)) == len(registered_measurements)
        )
        height_sample_mapping_complete = (
            sorted(registered_sample_ids) == sorted(design_ids)
            and len(set(registered_sample_ids)) == len(registered_sample_ids)
        )
        supported_mapping_rule = spec.mapping_rule in {
            "one_to_one_measurement_id",
            "paired_slot_from_cag_data_name",
        }
        mapping_resolved = bool(
            supported_mapping_rule
            and cardinality_match
            and design_ids_valid
            and height_measurements_complete
            and height_sample_mapping_complete
            and spec.mapping_provenance
        )

        if len(design) != spec.expected_design_rows:
            blockers.append(
                {
                    "code": "unexpected_design_row_count",
                    "session_id": spec.session_id,
                    "message": (
                        f"Expected {spec.expected_design_rows} design rows, found {len(design)}"
                    ),
                }
            )
        if not design_ids_valid:
            blockers.append(
                {
                    "code": "invalid_design_ids",
                    "session_id": spec.session_id,
                    "message": "加工顺序 is missing, duplicated, or not contiguous 1..N",
                }
            )
        if not cardinality_match:
            blockers.append(
                {
                    "code": "design_measurement_cardinality_mismatch",
                    "session_id": spec.session_id,
                    "message": (
                        f"Design rows={len(design)}, CAG measurements={measurement_count}, "
                        f"ROIs per measurement={spec.rois_per_measurement}"
                    ),
                }
            )
        if not height_measurements_complete:
            blockers.append(
                {
                    "code": "height_measurement_set_incomplete",
                    "session_id": spec.session_id,
                    "message": (
                        f"Height manifest measurement IDs do not uniquely cover "
                        f"1..{measurement_count}"
                    ),
                }
            )
        if not height_sample_mapping_complete:
            blockers.append(
                {
                    "code": "height_sample_mapping_incomplete",
                    "session_id": spec.session_id,
                    "message": "Height manifest slots do not uniquely cover design sample IDs",
                }
            )
        if not mapping_resolved:
            blockers.append(
                {
                    "code": "sample_measurement_mapping_unresolved",
                    "session_id": spec.session_id,
                    "message": (
                        "No explicit, verified mapping from design sample_id to CAG measurement/ROI"
                    ),
                }
            )
        if cag["embedded_vk4_count"] != cag["measurement_count"]:
            blockers.append(
                {
                    "code": "embedded_vk4_count_mismatch",
                    "session_id": spec.session_id,
                    "message": (
                        f"MeasurementDataMap={cag['measurement_count']}, embedded VK4={cag['embedded_vk4_count']}"
                    ),
                }
            )
        if not cag["height_layouts_all_consistent"]:
            blockers.append(
                {
                    "code": "invalid_vk4_height_layout",
                    "session_id": spec.session_id,
                    "message": "At least one embedded VK4 height header is inconsistent",
                }
            )

        session_rows.append(
            {
                "session_id": spec.session_id,
                "cag_path": spec.cag_path,
                "design_path": spec.design_path,
                "design_encoding": encoding,
                "design_rows": len(design),
                "cag_measurements": measurement_count,
                "embedded_vk4_count": cag["embedded_vk4_count"],
                "rois_per_measurement": spec.rois_per_measurement,
                "cardinality_match": int(cardinality_match),
                "mapping_rule": spec.mapping_rule,
                "mapping_provenance": spec.mapping_provenance,
                "mapping_resolved": int(mapping_resolved),
                "height_measurements_registered": len(registered_measurements),
                "height_samples_registered": len(registered_sample_ids),
                "unique_grid_spec_count": len(cag["unique_grid_specs"]),
                "session_status": "mapping_resolved" if mapping_resolved else "mapping_blocked",
            }
        )
        for source_row in sorted(session_height_rows,
                                 key=lambda row: int(row["measurement_id"])):
            measurement_id = int(source_row["measurement_id"])
            slots = (("slot_1_sample_id", "single" if spec.rois_per_measurement == 1
                      else "slot_1"), ("slot_2_sample_id", "slot_2"))
            for column, slot_name in slots:
                value = str(source_row.get(column, "")).strip()
                if not value:
                    continue
                sample_id = int(value)
                mapping_rows.append(
                    {
                        "session_id": spec.session_id,
                        "sample_id": sample_id,
                        "design_row_id": sample_id,
                        "cag_measurement_id": measurement_id,
                        "roi_within_measurement": slot_name,
                        "mapping_rule": spec.mapping_rule,
                        "mapping_provenance": spec.mapping_provenance,
                        "mapping_status": "resolved" if mapping_resolved else "unresolved",
                    }
                )
        session_details[spec.session_id] = {
            "spec": asdict(spec),
            "design_columns": [str(column) for column in design.columns],
            "cag": cag,
        }

    if not height_source_rows:
        blockers.append(
            {
                "code": "empty_height_source_manifest",
                "session_id": "all",
                "message": "height_source_manifest.csv contains no measurements",
            }
        )
    else:
        allowed_sources = set(config["input"]["allowed_source_types"])
        for source_row, path in zip(height_source_rows, height_csvs):
            exists = path.is_file()
            actual_hash = ""
            if exists:
                actual_hash = ("SKIPPED" if args.skip_hashes
                               else sha256_file(path))
            inventory_rows.append(
                {
                    "session_id": source_row["session_id"],
                    "role": "height_csv",
                    "path": source_row["csv_path"],
                    "exists": int(exists),
                    "bytes": path.stat().st_size if exists else "",
                    "sha256": actual_hash,
                }
            )
            if not exists:
                blockers.append({
                    "code": "missing_height_csv",
                    "session_id": source_row["session_id"],
                    "message": f"Missing registered height CSV: {source_row['csv_path']}",
                })
            if source_row["csv_source_type"] not in allowed_sources:
                blockers.append({
                    "code": "invalid_height_source_type",
                    "session_id": source_row["session_id"],
                    "message": source_row["csv_source_type"],
                })
            if source_row["provenance_status"] != "registered":
                blockers.append({
                    "code": "height_provenance_unregistered",
                    "session_id": source_row["session_id"],
                    "message": source_row["csv_path"],
                })
            if (exists and not args.skip_hashes
                    and actual_hash != source_row["csv_sha256"]):
                blockers.append({
                    "code": "height_csv_hash_mismatch",
                    "session_id": source_row["session_id"],
                    "message": source_row["csv_path"],
                })

    equivalence: dict[str, Any] = {}
    if not evidence_path.is_file():
        blockers.append({
            "code": "cag_csv_equivalence_not_established",
            "session_id": "all",
            "message": f"Missing evidence: {evidence_path}",
        })
    else:
        try:
            equivalence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            blockers.append({
                "code": "invalid_equivalence_evidence",
                "session_id": "all",
                "message": str(exc),
            })
        else:
            evidence_hashes = equivalence.get("input_hashes", {})
            expected_evidence_hashes = {
                "config": sha256_file(config_path),
                "height_source_manifest": sha256_file(height_manifest_path),
            }
            stale = [name for name, value in expected_evidence_hashes.items()
                     if evidence_hashes.get(name) != value]
            if stale:
                blockers.append({
                    "code": "stale_equivalence_evidence",
                    "session_id": "all",
                    "message": f"Input hashes changed: {', '.join(stale)}",
                })
            if equivalence.get("decision") != "PASS":
                blockers.append({
                    "code": "cag_csv_equivalence_not_established",
                    "session_id": "all",
                    "message": (
                        f"Equivalence decision is {equivalence.get('decision', 'missing')}"
                    ),
                })

    if args.skip_hashes:
        blockers.append(
            {
                "code": "input_hashes_skipped",
                "session_id": "all",
                "message": "Input SHA-256 hashes were skipped",
            }
        )

    inventory_columns = ["session_id", "role", "path", "exists", "bytes", "sha256"]
    session_columns = [
        "session_id",
        "cag_path",
        "design_path",
        "design_encoding",
        "design_rows",
        "cag_measurements",
        "embedded_vk4_count",
        "rois_per_measurement",
        "cardinality_match",
        "mapping_rule",
        "mapping_provenance",
        "mapping_resolved",
        "height_measurements_registered",
        "height_samples_registered",
        "unique_grid_spec_count",
        "session_status",
    ]
    mapping_columns = [
        "session_id",
        "sample_id",
        "design_row_id",
        "cag_measurement_id",
        "roi_within_measurement",
        "mapping_rule",
        "mapping_provenance",
        "mapping_status",
    ]
    write_csv(output_dir / "input_inventory.csv", inventory_rows, inventory_columns)
    write_csv(output_dir / "session_manifest_resolved.csv", session_rows, session_columns)
    write_csv(output_dir / "sample_design_mapping.csv", mapping_rows, mapping_columns)

    phase0_pass = not blockers
    validation = {
        "phase": "Phase 0",
        "pass": phase0_pass,
        "decision": "PASS" if phase0_pass else "STOP",
        "height_csv_count": len(height_csvs),
        "measurement_count": len(height_source_rows),
        "sample_count": len(mapping_rows),
        "session_count": len(specs),
        "equivalence_decision": equivalence.get("decision", "missing"),
        "equivalence_evidence_path": str(evidence_path.relative_to(repo)),
        "blockers": blockers,
        "warnings": warnings,
        "session_details": session_details,
    }
    (output_dir / "phase0_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    blocker_lines = "\n".join(
        f"- `{item['code']}` [{item['session_id']}]: {item['message']}"
        for item in blockers
    ) or "- 无"
    result_md = f"""# Phase 0 数据可用性验收结果

## 结论

**{validation['decision']}**

- session：{len(specs)}
- measurement：{len(height_source_rows)}
- sample mapping：{len(mapping_rows)}
- CAG–CSV 等价证据：{validation['equivalence_decision']}
- blocker：{len(blockers)}

## 阻断项

{blocker_lines}

## 规则

只有 `phase0_validation.json` 的 `decision` 为 `PASS` 且 blockers 为空，才允许进入 Phase A。
本报告由 `scripts/00_validate_inputs.py` 自动生成，不应手工修改。
"""
    (output_dir / "PHASE0_RESULT.md").write_text(result_md, encoding="utf-8")

    versions = {}
    for package in ("numpy", "pandas", "Pillow", "matplotlib", "scikit-learn",
                    "scipy", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    run_manifest_path = output_dir.parent / "run_manifest.json"
    try:
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    manifest.update({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_value(repo, "rev-parse", "HEAD"),
        "git_worktree_dirty": bool(git_value(repo, "status", "--porcelain")),
        "python_version": sys.version,
        "package_versions": versions,
        "config_path": str(config_path.relative_to(repo)),
        "config_sha256": sha256_file(config_path),
        "session_manifest_path": str(manifest_path.relative_to(repo)),
        "height_source_manifest_path": str(height_manifest_path.relative_to(repo)),
        "equivalence_evidence_path": str(evidence_path.relative_to(repo)),
        "equivalence_decision": equivalence.get("decision", "missing"),
        "input_files": inventory_rows,
        "session_definitions": [asdict(spec) for spec in specs],
        "phase0_decision": validation["decision"],
        "blocker_codes": [item["code"] for item in blockers],
        "manual_approval_status": "not_applicable_phase0_failed" if blockers else "pending",
    })
    run_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Phase 0 decision: {validation['decision']}")
    print(f"Sessions inspected: {len(specs)}")
    print(f"Registered height CSV measurements: {len(height_csvs)}")
    print(f"Blockers: {len(blockers)}")
    for blocker in blockers:
        print(f"- [{blocker['session_id']}] {blocker['code']}: {blocker['message']}")
    print(f"Outputs: {output_dir}")
    return 0 if phase0_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
