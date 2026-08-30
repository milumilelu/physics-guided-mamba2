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
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    specs = [
        SessionSpec(
            session_id=str(row.session_id),
            cag_path=str(row.cag_path),
            design_path=str(row.design_path),
            input_format=str(row.input_format),
            mapping_rule=str(row.mapping_rule),
            rois_per_measurement=int(row.rois_per_measurement),
            mapping_provenance=str(row.mapping_provenance),
            expected_design_rows=int(row.expected_design_rows),
        )
        for row in manifest_df.itertuples(index=False)
    ]

    height_csvs = sorted(
        path
        for path in repo.rglob("*_高度.csv")
        if ".venv" not in path.parts and "outputs" not in path.parts
    )
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
        supported_mapping_rule = spec.mapping_rule in {
            "one_to_one_measurement_id",
            "paired_left_to_right_odd_even",
        }
        mapping_resolved = bool(
            supported_mapping_rule
            and cardinality_match
            and design_ids_valid
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
                "unique_grid_spec_count": len(cag["unique_grid_specs"]),
                "session_status": "mapping_resolved" if mapping_resolved else "mapping_blocked",
            }
        )
        for sample_id in design_ids:
            if spec.mapping_rule == "one_to_one_measurement_id":
                measurement_id: int | str = sample_id
                roi_within_measurement = "single"
            elif spec.mapping_rule == "paired_left_to_right_odd_even":
                measurement_id = (sample_id + 1) // 2
                roi_within_measurement = "left" if sample_id % 2 else "right"
            else:
                measurement_id = ""
                roi_within_measurement = ""
            mapping_rows.append(
                {
                    "session_id": spec.session_id,
                    "sample_id": sample_id,
                    "design_row_id": sample_id,
                    "cag_measurement_id": measurement_id,
                    "roi_within_measurement": roi_within_measurement,
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

    if not height_csvs:
        blockers.append(
            {
                "code": "no_real_height_csv_fixture",
                "session_id": "all",
                "message": "No *_高度.csv exists in the workspace outside outputs/.venv",
            }
        )
        blockers.append(
            {
                "code": "cag_csv_equivalence_not_established",
                "session_id": "all",
                "message": (
                    "No same-series CSV fixtures are available for CAG height/mask equivalence testing"
                ),
            }
        )
    else:
        for path in height_csvs:
            inventory_rows.append(
                {
                    "session_id": "unassigned",
                    "role": "height_csv",
                    "path": str(path.relative_to(repo)),
                    "exists": 1,
                    "bytes": path.stat().st_size,
                    "sha256": "SKIPPED" if args.skip_hashes else sha256_file(path),
                }
            )
        blockers.append(
            {
                "code": "cag_csv_equivalence_not_established",
                "session_id": "all",
                "message": "Height CSV exists, but no frozen CAG-CSV equivalence result is registered",
            }
        )

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
        "session_count": len(specs),
        "blockers": blockers,
        "warnings": warnings,
        "session_details": session_details,
    }
    (output_dir / "phase0_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    versions = {}
    for package in ("numpy", "pandas", "Pillow", "matplotlib", "scikit-learn", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_value(repo, "rev-parse", "HEAD"),
        "git_worktree_dirty": bool(git_value(repo, "status", "--porcelain")),
        "python_version": sys.version,
        "package_versions": versions,
        "config_path": str(manifest_path.relative_to(repo)),
        "input_files": inventory_rows,
        "session_definitions": [asdict(spec) for spec in specs],
        "phase0_decision": validation["decision"],
        "blocker_codes": [item["code"] for item in blockers],
        "manual_approval_status": "not_applicable_phase0_failed" if blockers else "pending",
    }
    (output_dir.parent / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Phase 0 decision: {validation['decision']}")
    print(f"Sessions inspected: {len(specs)}")
    print(f"Real height CSV fixtures: {len(height_csvs)}")
    print(f"Blockers: {len(blockers)}")
    for blocker in blockers:
        print(f"- [{blocker['session_id']}] {blocker['code']}: {blocker['message']}")
    print(f"Outputs: {output_dir}")
    return 0 if phase0_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
