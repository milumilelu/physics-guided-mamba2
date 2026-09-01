#!/usr/bin/env python3
"""WP6: robust coarse leveling and process-aware calibration selection."""

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

from src.calibration_selection import (  # noqa: E402
    coverage_rows,
    select_calibration_samples,
)
from src.io_cag import CagHeightReader  # noqa: E402
from src.leveling import fit_outer_reference_plane  # noqa: E402


def read_csv(path: Path, encoding: str = "utf-8-sig") -> list[dict]:
    with path.open("r", encoding=encoding, newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("config/rectangle_registration.yaml"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/rectangle_registration"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    root = (REPO / args.output_dir).resolve()
    inventory_summary = json.loads(
        (root / "inventory/inventory_summary.json").read_text(encoding="utf-8"))
    if inventory_summary.get("decision") != "PASS":
        print("STOP: WP5 inventory is not PASS", file=sys.stderr)
        return 2
    sessions = read_csv(REPO / config["paths"]["session_manifest"])
    sources = read_csv(REPO / config["paths"]["height_source_manifest"])
    diagnostics = read_csv(root / "inventory/contrast_diagnostics.csv")
    if args.dry_run:
        print(f"READY: measurements={len(sources)} sample diagnostics={len(diagnostics)}")
        return 0

    leveling_cfg = config["coarse_leveling"]
    plane_rows: list[dict] = []
    errors: list[dict] = []
    for session in sessions:
        sid = session["session_id"]
        session_sources = sorted(
            (row for row in sources if row["session_id"] == sid),
            key=lambda row: int(row["measurement_id"]))
        with CagHeightReader(REPO / session["cag_path"]) as reader:
            for source in session_sources:
                mid = int(source["measurement_id"])
                try:
                    hm = reader.read_height_map(mid)
                    fit = fit_outer_reference_plane(
                        hm,
                        frame_width_um=float(leveling_cfg["outer_frame_width_um"]),
                        sigma_low=float(leveling_cfg["sigma_low"]),
                        sigma_high=float(leveling_cfg["sigma_high"]),
                        max_iterations=int(leveling_cfg["max_iterations"]),
                        minimum_reference_valid_fraction=float(
                            leveling_cfg["minimum_reference_valid_fraction"]),
                        max_fit_points=int(leveling_cfg["max_fit_points"]),
                    )
                    plane_rows.append({"session_id": sid, "measurement_id": mid,
                                       **fit.to_dict()})
                    if fit.status != "PASS":
                        errors.append({"session_id": sid, "measurement_id": mid,
                                       "error": fit.warning})
                except Exception as exc:
                    errors.append({"session_id": sid, "measurement_id": mid,
                                   "error": f"{type(exc).__name__}: {exc}"})

    factors = list(config["design"]["stratification_variables"])
    column_map = config["design"]["column_map"]
    selected_all: list[dict] = []
    coverage_all: list[dict] = []
    for session in sessions:
        sid = session["session_id"]
        design_rows = read_csv(REPO / session["design_path"],
                               config["design"]["encoding"])
        design_by_id = {int(row[column_map["sample_id"]]): row
                        for row in design_rows}
        candidates = []
        for diag in diagnostics:
            if diag["session_id"] != sid:
                continue
            sample_id = int(diag["sample_id"])
            row = dict(diag)
            for factor in factors:
                row[factor] = design_by_id[sample_id][column_map[factor]]
            candidates.append(row)
        selected = select_calibration_samples(
            candidates,
            fraction=float(config["calibration"]["selection_fraction"]),
            minimum=int(config["calibration"]["minimum_calibration_per_session"]),
            factors=factors,
            weights=config["calibration"]["ranking_score"],
        )
        selected_all.extend(selected)
        coverage_all.extend(coverage_rows(candidates, selected, factors, sid))

    represented = all(int(row["level_represented"]) == 1 for row in coverage_all)
    decision = "PASS" if not errors and represented else "STOP"
    write_csv(root / "metrics/coarse_leveling_metrics.csv", plane_rows)
    write_csv(root / "calibration/calibration_sample_ids.csv", selected_all)
    write_csv(root / "calibration/calibration_coverage_by_process.csv", coverage_all)
    summary = {
        "stage": "WP6_leveling_and_calibration_selection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "measurements_fitted": len(plane_rows),
        "calibration_samples": len(selected_all),
        "calibration_by_session": {
            sid: sum(row["session_id"] == sid for row in selected_all)
            for sid in {row["session_id"] for row in selected_all}},
        "all_process_levels_represented": represented,
        "errors": errors,
    }
    (root / "calibration/calibration_selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if decision == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
