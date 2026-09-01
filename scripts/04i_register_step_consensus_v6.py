#!/usr/bin/env python3
"""WP8-v6: full feasible-domain block-consensus registration."""

from __future__ import annotations

import csv
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.io_cag import CagHeightReader  # noqa: E402
from src.step_contrast_consensus import fit_step_contrast_consensus  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, data: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in data for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def load_prior(root: Path, name: str) -> dict[tuple[str, int], dict]:
    return {(r["session_id"], int(r["sample_id"])): r
            for r in read_csv(root / "registration" / name)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v6", "v7"), default="v6")
    args = parser.parse_args()
    version = args.version
    config = yaml.safe_load(
        (REPO / "config/rectangle_registration.yaml").read_text(encoding="utf-8")
    )
    root = REPO / config["paths"]["outputs_root"]
    sessions = read_csv(REPO / config["paths"]["session_manifest"])
    views = read_csv(root / "inventory/sample_view_manifest.csv")
    planes = {(r["session_id"], int(r["measurement_id"])): r
              for r in read_csv(root / "metrics/coarse_leveling_metrics.csv")}
    geometry = {r["session_id"]: r
                for r in read_csv(root / "geometry/session_geometry.csv")}
    prior_versions = ("v2", "v3", "v4", "v5") + (("v6",) if version == "v7" else ())
    priors = {
        label: load_prior(
            root, "translation_metrics.csv" if label == "v2" else f"translation_metrics_{label}.csv"
        ) for label in prior_versions
    }
    cfg = config[f"registration_{version}"]
    metrics: list[dict] = []
    for session in sessions:
        sid = session["session_id"]
        theta = float(geometry[sid]["theta_session_deg"])
        selected = sorted(
            (r for r in views if r["session_id"] == sid),
            key=lambda r: (int(r["measurement_id"]),
                           0 if r["roi_within_measurement"] == "slot_1" else 1)
        )
        with CagHeightReader(REPO / session["cag_path"]) as reader:
            cached_id = None
            hm = None
            for index, view in enumerate(selected, start=1):
                measurement_id = int(view["measurement_id"])
                sample_id = int(view["sample_id"])
                if cached_id != measurement_id:
                    hm = reader.read_height_map(measurement_id)
                    cached_id = measurement_id
                if view["roi_within_measurement"] == "single":
                    margin = 115.0
                    center_search = (
                        -hm.width_um/2+margin, hm.width_um/2-margin,
                        -hm.height_um/2+margin, hm.height_um/2-margin,
                    )
                else:
                    center_search = tuple(float(view[k]) for k in (
                        "center_search_x_min_um", "center_search_x_max_um",
                        "center_search_y_min_um", "center_search_y_max_um"
                    ))
                plane = planes[(sid, measurement_id)]
                influence = cfg["influence"]
                fit = fit_step_contrast_consensus(
                    hm,
                    plane=tuple(float(plane[k]) for k in ("a", "b", "c")),
                    theta_deg=theta, center_search=center_search,
                    nominal_size_um=float(cfg["nominal_size_um"]),
                    local_canvas_um=float(cfg["local_canvas_um"]),
                    edge_search_halfwidth_um=float(cfg["edge_search_halfwidth_um"]),
                    center_grid_step_um=float(cfg["center_grid_step_um"]),
                    profile_strip_halfwidth_um=float(cfg["profile_strip_halfwidth_um"]),
                    smoothing_sigma_um=float(cfg["smoothing_sigma_um"]),
                    contrast_bandwidths_um=tuple(float(v) for v in cfg["contrast_bandwidths_um"]),
                    boundary_gap_um=float(cfg["boundary_gap_um"]),
                    tangent_blocks=int(cfg["tangent_blocks"]),
                    hard_minimum_total=float(cfg["joint_evidence"]["hard_minimum_total"]),
                    review_below_total=float(cfg["joint_evidence"]["review_below_total"]),
                    hard_minimum_per_axis=float(cfg["joint_evidence"]["hard_minimum_per_axis"]),
                    ci_quantiles=tuple(float(v) for v in influence["ci_quantiles"]),
                    review_ci_span_um=float(influence["review_ci_span_um"]),
                    hard_ci_span_um=float(influence["hard_ci_span_um"]),
                    review_mad_um=float(influence["review_mad_um"]),
                    hard_mad_um=float(influence["hard_mad_um"]),
                    histogram_bin_um=float(cfg["multimodality"]["histogram_bin_um"]),
                    minimum_secondary_fraction=float(cfg["multimodality"]["minimum_secondary_fraction"]),
                    minimum_mode_separation_um=float(cfg["multimodality"]["minimum_mode_separation_um"]),
                    local_boundary_tolerance_um=float(cfg["local_search_boundary_tolerance_um"]),
                    global_boundary_tolerance_um=float(cfg["global_search_boundary_tolerance_um"]),
                    balanced_opposing_edges=bool(cfg.get("balanced_opposing_edges", False)),
                )
                key = (sid, sample_id)
                row = {
                    "registration_method": cfg["method"],
                    "evidence_level": int(cfg["evidence_level"]),
                    "session_id": sid, "measurement_id": measurement_id,
                    "sample_id": sample_id,
                    "roi_within_measurement": view["roi_within_measurement"],
                    "theta_session_deg": theta,
                    "d4_transform_session": "identity",
                    "initializer_search_x_min_um": center_search[0],
                    "initializer_search_x_max_um": center_search[1],
                    "initializer_search_y_min_um": center_search[2],
                    "initializer_search_y_max_um": center_search[3],
                    **fit.to_dict(),
                }
                for label, table in priors.items():
                    old = table[key]
                    row[f"qa_{label}_status"] = old["status"]
                    row[f"qa_v6_vs_{label}_center_shift_um"] = float(np.hypot(
                        fit.center_x_um-float(old["center_x_um"]),
                        fit.center_y_um-float(old["center_y_um"])
                    ))
                row["qa_v2_region_score"] = priors["v2"][key]["region_score"]
                row["qa_v2_edge_score"] = priors["v2"][key]["edge_score"]
                row["qa_v2_weight_sensitivity_span_um"] = priors["v2"][key]["sensitivity_span_um"]
                metrics.append(row)
                if index % 20 == 0 or index == len(selected):
                    print(f"{sid}: {index}/{len(selected)}", flush=True)

    groups: dict[tuple[str, int], list[dict]] = {}
    for row in metrics:
        groups.setdefault((row["session_id"], row["measurement_id"]), []).append(row)
    conflicts = 0
    minimum_separation = float(config["paired_registration"]["minimum_center_separation_um"])
    for group in groups.values():
        if len(group) != 2:
            continue
        one = next(r for r in group if r["roi_within_measurement"] == "slot_1")
        two = next(r for r in group if r["roi_within_measurement"] == "slot_2")
        separation = float(two["center_x_um"])-float(one["center_x_um"])
        conflict = separation < minimum_separation or float(one["center_x_um"]) >= float(two["center_x_um"])
        for row in group:
            row["paired_center_separation_um"] = separation
            row["slot_assignment_conflict"] = conflict
            if conflict:
                row["status"] = "STOP"
                row["warning"] = (row["warning"]+"; paired slot conflict").strip("; ")
        conflicts += int(conflict)

    counts = {s: sum(r["status"] == s for r in metrics) for s in ("PASS", "REVIEW", "STOP")}
    decision = "STOP" if counts["STOP"] else "REVIEW" if counts["REVIEW"] else "PASS"
    output = root / "registration" / f"translation_metrics_{version}.csv"
    write_csv(output, metrics)
    method_docs = {
        "v6": "METHOD_V6_全可行域多块共识四边配准.md",
        "v7": "METHOD_V7_平衡对边共识四边配准.md",
    }
    summary = {
        "stage": f"WP8_{version}_feasible_domain_block_consensus",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_frozen_before_run": method_docs[version],
        "evidence_level": 3, "decision": decision, "samples": len(metrics),
        "pass": counts["PASS"], "review": counts["REVIEW"],
        "stop": counts["STOP"], "paired_conflict_measurements": conflicts,
    }
    (root / "registration" / f"translation_summary_{version}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if decision in {"PASS", "REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
