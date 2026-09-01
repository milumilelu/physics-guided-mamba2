#!/usr/bin/env python3
"""WP5: inventory all measurements and create 200 morphology-free sample views."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.inventory import (  # noqa: E402
    build_sample_search_regions,
    compute_height_diagnostics,
    compute_invalid_components,
)
from src.io_cag import CagHeightReader  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("config/rectangle_registration.yaml"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/rectangle_registration/inventory"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (REPO / args.config).resolve()
    output_dir = (REPO / args.output_dir).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase0_path = (REPO / config["paths"]["outputs_root"] /
                   config["output_layout"]["phase0"] /
                   "phase0_validation.json")
    phase0 = json.loads(phase0_path.read_text(encoding="utf-8"))
    if phase0.get("decision") != "PASS" or phase0.get("blockers"):
        print("STOP: Phase 0 is not PASS", file=sys.stderr)
        return 2

    sessions = read_csv(REPO / config["paths"]["session_manifest"])
    sources = read_csv(REPO / config["paths"]["height_source_manifest"])
    expected_measurements = int(config["expected_counts"]["measurements_total"])
    expected_samples = int(config["expected_counts"]["samples_total"])
    if args.dry_run:
        print(f"READY: sessions={len(sessions)} measurements={len(sources)} "
              f"expected={expected_measurements}")
        return 0 if len(sources) == expected_measurements else 2

    measurement_rows: list[dict] = []
    sample_views: list[dict] = []
    sample_diagnostics: list[dict] = []
    errors: list[dict] = []

    for session in sessions:
        sid = session["session_id"]
        session_sources = sorted(
            (row for row in sources if row["session_id"] == sid),
            key=lambda row: int(row["measurement_id"]))
        cag_path = (REPO / session["cag_path"]).resolve()
        with CagHeightReader(cag_path) as reader:
            for source in session_sources:
                measurement_id = int(source["measurement_id"])
                try:
                    hm = reader.read_height_map(measurement_id)
                    full = compute_height_diagnostics(hm)
                    components = compute_invalid_components(hm)
                    measurement_rows.append({
                        "session_id": sid,
                        "measurement_id": measurement_id,
                        "cag_data_name": source["cag_data_name"],
                        "width_px": hm.shape[1],
                        "height_px": hm.shape[0],
                        "dx_um": hm.dx_um,
                        "dy_um": hm.dy_um,
                        "fov_width_um": hm.width_um,
                        "fov_height_um": hm.height_um,
                        "valid_pixels": hm.n_valid,
                        "invalid_pixels": hm.n_invalid,
                        "valid_fraction": hm.valid_fraction,
                        **components,
                        **full,
                        "status": "PASS",
                    })

                    sample_ids = [int(source["slot_1_sample_id"])]
                    if source["slot_2_sample_id"].strip():
                        sample_ids.append(int(source["slot_2_sample_id"]))
                    regions = build_sample_search_regions(
                        width_um=hm.width_um,
                        height_um=hm.height_um,
                        sample_ids=sample_ids,
                        single_halfwidth_um=float(
                            config["registration"]["center_search"]["halfwidth_um"]),
                        nominal_halfwidth_um=float(
                            config["nominal_programmed_region_um"][0]) / 2.0,
                        paired_guard_band_um=float(
                            config["paired_registration"]["guard_band_um"]),
                    )
                    for region in regions:
                        view = {
                            "session_id": sid,
                            "measurement_id": measurement_id,
                            "sample_id": region["sample_id"],
                            "roi_within_measurement": region["slot"],
                            "shared_height_source_id": f"{sid}:m{measurement_id:03d}",
                            **{key: value for key, value in region.items()
                               if key not in {"sample_id", "slot"}},
                        }
                        sample_views.append(view)
                        diagnostic_region = (
                            region["center_search_x_min_um"] - 100.0,
                            region["center_search_x_max_um"] + 100.0,
                            region["center_search_y_min_um"] - 100.0,
                            region["center_search_y_max_um"] + 100.0,
                        )
                        sample_diagnostics.append({
                            **{key: view[key] for key in (
                                "session_id", "measurement_id", "sample_id",
                                "roi_within_measurement", "shared_height_source_id")},
                            **compute_height_diagnostics(hm, diagnostic_region),
                        })
                except Exception as exc:
                    errors.append({
                        "session_id": sid,
                        "measurement_id": measurement_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    unique_samples = {(row["session_id"], row["sample_id"])
                      for row in sample_views}
    decision = "PASS" if (
        not errors
        and len(measurement_rows) == expected_measurements
        and len(sample_views) == expected_samples
        and len(unique_samples) == expected_samples
    ) else "STOP"
    write_csv(output_dir / "measurement_metrics.csv", measurement_rows)
    write_csv(output_dir / "sample_view_manifest.csv", sample_views)
    write_csv(output_dir / "contrast_diagnostics.csv", sample_diagnostics)
    summary = {
        "stage": "WP5_height_inventory",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "sessions": len(sessions),
        "measurements": len(measurement_rows),
        "measurements_expected": expected_measurements,
        "sample_views": len(sample_views),
        "samples_expected": expected_samples,
        "unique_session_sample_ids": len(unique_samples),
        "all_measurements_all_valid": bool(
            measurement_rows and all(row["invalid_pixels"] == 0
                                     for row in measurement_rows)),
        "errors": errors,
    }
    (output_dir / "inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if decision == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
