#!/usr/bin/env python3
"""Validate same-series CAG decoding against independent KEYENCE exports.

The script is intentionally a gate.  It exits 0 only when every configured
session has enough independent fixtures and both height and mask evidence pass.
Height and mask decisions are always reported separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.equivalence import compare_height_maps  # noqa: E402
from src.io_cag import CagHeightReader, container_sha256  # noqa: E402
from src.io_vk_csv import parse_vk_csv  # noqa: E402
from src.provenance import SourceType  # noqa: E402


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "session_id", "measurement_id", "cag_data_name", "csv_path",
        "csv_source_type", "shape_match", "pitch_pass",
        "height_decision", "mask_decision", "overall_decision",
        "compared_pixels", "cag_invalid_pixels", "height_mismatch_pixels",
        "max_abs_difference_um", "median_abs_difference_um", "rmse_um",
        "orientation_best_transform", "fixed_pixel_checks_pass",
        "csv_sha256", "cag_sha256", "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_qa(path: Path, cag, official, comparison: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = cag.valid_mask & np.isfinite(official.z)
    difference = np.full(cag.shape, np.nan, dtype=np.float64)
    difference[valid] = official.z[valid] - cag.z[valid]
    stride = max(1, int(max(cag.shape) / 700))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = [
        (cag.z, "CAG height (μm)", "viridis"),
        (official.z, "Official CSV height (μm)", "viridis"),
        (difference, "CSV − CAG on CAG-valid pixels (μm)", "coolwarm"),
        (cag.valid_mask.astype(float), "CAG raw valid mask", "gray"),
    ]
    for axis, (data, title, cmap) in zip(axes.flat, panels):
        image = axis.imshow(data[::stride, ::stride], cmap=cmap,
                            interpolation="nearest")
        axis.set_title(title)
        axis.set_axis_off()
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"height={comparison['height_decision']}  "
        f"mask={comparison['mask_decision']}  "
        f"max|Δ|={comparison['max_abs_difference_um']} μm"
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("config/rectangle_registration.yaml"))
    parser.add_argument("--height-manifest", type=Path,
                        default=Path("config/height_source_manifest.csv"))
    parser.add_argument("--fixture-selection", type=Path,
                        default=Path("config/equivalence_fixture_selection.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "outputs/rectangle_registration/phase0/equivalence"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-qa", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (REPO / args.config).resolve()
    manifest_path = (REPO / args.height_manifest).resolve()
    selection_path = (REPO / args.fixture_selection).resolve()
    output_dir = (REPO / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tolerance = config["equivalence"]["tolerance"]
    minimum = int(config["equivalence"]["fixtures_per_session_min"])
    pitch_tolerance = float(tolerance["dx_dy_abs_um"])
    require_mask = (config["equivalence"]["mask_semantics"]
                    ["if_csv_has_no_mask"]["decision"] == "stop")
    allow_all_valid_mask_case = bool(
        config["equivalence"]["mask_semantics"]
        .get("allow_all_valid_case", False))

    manifest_rows = read_csv(manifest_path)
    selected_rows = read_csv(selection_path)
    manifest_index = {
        (row["session_id"], int(row["measurement_id"])): row
        for row in manifest_rows
    }
    session_ids = list(config["expected_counts"]["per_session"].keys())
    selected_by_session = {
        session: [row for row in selected_rows if row["session_id"] == session]
        for session in session_ids
    }

    blockers: list[dict] = []
    metric_rows: list[dict] = []
    session_reports: dict[str, dict] = {}
    hash_cache = output_dir / ".cag_hash_cache.json"

    if args.dry_run:
        for session in session_ids:
            official = sum(
                manifest_index.get((session, int(row["measurement_id"])), {})
                .get("csv_source_type") == SourceType.OFFICIAL
                for row in selected_by_session[session]
            )
            session_reports[session] = {
                "selected": len(selected_by_session[session]),
                "official_fixtures_available": official,
                "minimum_required": minimum,
            }
        decision = "STOP" if any(
            report["official_fixtures_available"] < minimum
            for report in session_reports.values()) else "READY"
        print(json.dumps({"decision": decision, "sessions": session_reports},
                         ensure_ascii=False, indent=2))
        return 0 if decision == "READY" else 2

    with ExitStack() as stack:
        readers: dict[str, CagHeightReader] = {}
        cag_hashes: dict[str, str] = {}

        for session in session_ids:
            session_selected = selected_by_session[session]
            official_rows = []
            for selected in session_selected:
                key = (session, int(selected["measurement_id"]))
                source = manifest_index.get(key)
                if source is None:
                    blockers.append({
                        "code": "fixture_missing_from_height_manifest",
                        "session_id": session,
                        "measurement_id": key[1],
                    })
                    continue
                if source["csv_source_type"] != SourceType.OFFICIAL:
                    blockers.append({
                        "code": "independent_official_fixture_missing",
                        "session_id": session,
                        "measurement_id": key[1],
                        "found_source_type": source["csv_source_type"],
                    })
                    continue
                official_rows.append(source)

            if len(official_rows) < minimum:
                blockers.append({
                    "code": "insufficient_independent_fixtures",
                    "session_id": session,
                    "available": len(official_rows),
                    "required": minimum,
                })

            if not official_rows:
                session_reports[session] = {
                    "selected": len(session_selected),
                    "official_fixtures_compared": 0,
                    "minimum_required": minimum,
                    "height_decision": "NOT_TESTED",
                    "mask_decision": "NOT_TESTED",
                    "decision": "STOP",
                }
                continue

            cag_path = (REPO / official_rows[0]["cag_path"]).resolve()
            readers[session] = stack.enter_context(CagHeightReader(cag_path))
            cag_hashes[session] = container_sha256(cag_path, hash_cache)
            session_metrics = []

            for source in official_rows:
                measurement_id = int(source["measurement_id"])
                csv_path = (REPO / source["csv_path"]).resolve()
                base = {
                    "session_id": session,
                    "measurement_id": measurement_id,
                    "cag_data_name": source["cag_data_name"],
                    "csv_path": source["csv_path"],
                    "csv_source_type": source["csv_source_type"],
                    "cag_sha256": cag_hashes[session],
                    "error": "",
                }
                try:
                    actual_csv_hash = sha256_file(csv_path)
                    if actual_csv_hash != source["csv_sha256"]:
                        raise ValueError(
                            "CSV SHA-256 differs from height_source_manifest")
                    cag_map = readers[session].read_height_map(measurement_id)
                    csv_map = parse_vk_csv(csv_path)
                    comparison = compare_height_maps(
                        cag_map, csv_map,
                        pitch_tolerance_um=pitch_tolerance,
                        require_mask_evidence=require_mask,
                        allow_all_valid_mask_case=allow_all_valid_mask_case,
                    )
                    row = {**base, **comparison,
                           "csv_sha256": actual_csv_hash}
                    session_metrics.append(comparison)
                    if not args.no_qa:
                        save_qa(
                            output_dir / "qa" /
                            f"{session}_m{measurement_id:03d}_difference.png",
                            cag_map, csv_map, comparison)
                except Exception as exc:
                    row = {**base, "csv_sha256": "", "shape_match": False,
                           "pitch_pass": False, "height_decision": "FAIL",
                           "mask_decision": "NOT_TESTED",
                           "overall_decision": "STOP",
                           "error": f"{type(exc).__name__}: {exc}"}
                    blockers.append({
                        "code": "fixture_comparison_failed",
                        "session_id": session,
                        "measurement_id": measurement_id,
                        "message": row["error"],
                    })
                metric_rows.append(row)

            enough_fixtures = len(session_metrics) >= minimum
            compared_heights_pass = bool(
                session_metrics
                and all(item["height_decision"] == "PASS"
                        for item in session_metrics))
            compared_masks_pass = bool(
                session_metrics
                and all(item["mask_pass"] is True
                        for item in session_metrics))
            height_pass = enough_fixtures and compared_heights_pass
            mask_pass = enough_fixtures and compared_masks_pass
            if session_metrics and not compared_heights_pass:
                blockers.append({"code": "height_equivalence_failed",
                                 "session_id": session})
            if session_metrics and not compared_masks_pass:
                blockers.append({"code": "mask_equivalence_not_established",
                                 "session_id": session})
            height_decision = (
                "PASS" if height_pass else
                "INSUFFICIENT_FIXTURES" if compared_heights_pass else "FAIL"
            )
            mask_decision = (
                "PASS" if mask_pass else
                "INSUFFICIENT_FIXTURES" if compared_masks_pass else
                "UNAVAILABLE_OR_FAIL"
            )
            session_reports[session] = {
                "selected": len(session_selected),
                "official_fixtures_compared": len(session_metrics),
                "minimum_required": minimum,
                "height_decision": height_decision,
                "mask_decision": mask_decision,
                "decision": "PASS" if height_pass and mask_pass else "STOP",
                "cag_path": str(cag_path.relative_to(REPO)),
                "cag_sha256": cag_hashes[session],
            }

    decision = "PASS" if not blockers else "STOP"
    report = {
        "schema_version": 1,
        "stage": "WP3_cag_csv_equivalence",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "height_gate_all_sessions": all(
            value["height_decision"] == "PASS"
            for value in session_reports.values()),
        "mask_gate_all_sessions": all(
            value["mask_decision"] in {"PASS", "PASS_ALL_VALID_CASE"}
            for value in session_reports.values()),
        "require_mask_evidence": require_mask,
        "tolerances": {
            "pitch_abs_um": pitch_tolerance,
            "height_abs_um": 5e-12,
            "height_rule": "exact after CAG half-up rounding to 0.001 um",
        },
        "input_hashes": {
            "config": sha256_file(config_path),
            "height_source_manifest": sha256_file(manifest_path),
            "fixture_selection": sha256_file(selection_path),
        },
        "sessions": session_reports,
        "blockers": blockers,
        "notes": [
            "Heights are compared only on CAG raw-valid pixels.",
            "ImageDataCsv has no independent validity mask; an exact height "
            "match does not by itself close the mask gate.",
            "The mask sub-gate may pass with scope PASS_ALL_VALID_CASE only "
            "when CAG has zero sentinels and the official CSV is numeric at "
            "every pixel; this does not validate future sentinel behaviour.",
            "Decoder-derived CSV files are never accepted as fixtures.",
        ],
    }
    write_metrics(output_dir / "cag_csv_equivalence_metrics.csv", metric_rows)
    evidence_path = output_dir / "cag_csv_equivalence.json"
    evidence_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"CAG-CSV equivalence decision: {decision}")
    for session, value in session_reports.items():
        print(f"- {session}: fixtures={value['official_fixtures_compared']} "
              f"height={value['height_decision']} "
              f"mask={value['mask_decision']} decision={value['decision']}")
    print(f"Blockers: {len(blockers)}")
    print(f"Evidence: {evidence_path}")
    return 0 if decision == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
