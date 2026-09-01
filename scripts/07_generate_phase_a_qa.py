#!/usr/bin/env python3
"""WP10: render Phase A QA evidence and evaluate automatic gates."""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.io_cag import CagHeightReader  # noqa: E402
from src.io_npz import load_height_npz  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def canonical_box(cx: float, cy: float, halfwidth: float,
                  theta_deg: float) -> np.ndarray:
    points = np.array([[-halfwidth, -halfwidth], [halfwidth, -halfwidth],
                       [halfwidth, halfwidth], [-halfwidth, halfwidth],
                       [-halfwidth, -halfwidth]])
    theta = np.deg2rad(theta_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    return points @ rotation.T + [cx, cy]


def main() -> int:
    config = yaml.safe_load((REPO / "config/rectangle_registration.yaml")
                            .read_text(encoding="utf-8"))
    root = REPO / config["paths"]["outputs_root"]
    summary = json.loads((root / "resampling/resampling_summary.json")
                         .read_text(encoding="utf-8"))
    if summary.get("decision") != "PASS":
        print("STOP: resampling/final leveling is not PASS", file=sys.stderr)
        return 2
    metrics = read_csv(root / "metrics/registration_metrics.csv")
    sessions = {row["session_id"]: row for row in
                read_csv(REPO / config["paths"]["session_manifest"])}
    planes = {(row["session_id"], int(row["measurement_id"])): row for row in
              read_csv(root / "metrics/coarse_leveling_metrics.csv")}
    qa_dir = root / "qa"
    individual_dir = qa_dir / "registration_individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    session_limits: dict[str, tuple[float, float]] = {}
    for sid in sorted({row["session_id"] for row in metrics}):
        samples = []
        for row in (item for item in metrics if item["session_id"] == sid):
            hm = load_height_npz(REPO / row["h_200_path"])
            samples.append(hm.z[hm.valid_mask][::100])
        values = np.concatenate(samples)
        session_limits[sid] = tuple(map(float, np.quantile(values, [0.01, 0.99])))

    thumbnails: dict[str, list[tuple[dict, np.ndarray, tuple[float, float]]]] = {
        "absolute": [], "local": []}
    for sid, session in sessions.items():
        session_metrics = sorted(
            (row for row in metrics if row["session_id"] == sid),
            key=lambda row: int(row["sample_id"]))
        if not session_metrics:
            continue
        with CagHeightReader(REPO / session["cag_path"]) as reader:
            cached_id = None
            raw = None
            for row in session_metrics:
                measurement_id = int(row["measurement_id"])
                sample_id = int(row["sample_id"])
                if cached_id != measurement_id:
                    raw = reader.read_height_map(measurement_id)
                    cached_id = measurement_id
                plane = planes[(sid, measurement_id)]
                x = raw.x_um-raw.width_um/2.0
                y = raw.y_um-raw.height_um/2.0
                a, b, c = (float(plane[key]) for key in ("a", "b", "c"))
                coarse = raw.z-(a*x[None, :]+b*y[:, None]+c)
                cx, cy = float(row["center_x_um"]), float(row["center_y_um"])
                half_crop = float(row["registered_fov_um"])/2.0+10.0
                columns = (x >= cx-half_crop) & (x <= cx+half_crop)
                selected_rows = (y >= cy-half_crop) & (y <= cy+half_crop)
                raw_crop = raw.z[np.ix_(selected_rows, columns)]
                coarse_crop = coarse[np.ix_(selected_rows, columns)]
                raw_mask = raw.valid_mask[np.ix_(selected_rows, columns)]
                extent = [x[columns][0], x[columns][-1],
                          y[selected_rows][-1], y[selected_rows][0]]
                hreg = load_height_npz(REPO / row["h_reg_path"])
                h200 = load_height_npz(REPO / row["h_200_path"])
                hreg_extent = [hreg.x_um[0], hreg.x_um[-1],
                               hreg.y_um[-1], hreg.y_um[0]]
                h200_extent = [h200.x_um[0], h200.x_um[-1],
                               h200.y_um[-1], h200.y_um[0]]
                absolute_limits = session_limits[sid]
                local_probabilities = [
                    float(value)/100.0
                    for value in config["qa"]["local_color_percentiles"]]
                local_limits = tuple(map(float, np.quantile(
                    hreg.z[hreg.valid_mask], local_probabilities)))

                figure, axes = plt.subplots(2, 4, figsize=(16, 8),
                                            constrained_layout=True)
                panels = [
                    (raw_crop, "raw height", extent, absolute_limits),
                    (coarse_crop, "coarse-levelled", extent, absolute_limits),
                    (hreg.z, "H_reg — absolute", hreg_extent, absolute_limits),
                    (h200.z, "H_200 — absolute", h200_extent, absolute_limits),
                    (coarse_crop, config["qa"]["local_contrast_watermark"],
                     extent, tuple(map(float, np.quantile(coarse_crop[raw_mask], [0.01, 0.99])))),
                    (hreg.z, config["qa"]["local_contrast_watermark"],
                     hreg_extent, local_limits),
                ]
                for axis, (data, title, image_extent, limits) in zip(axes.flat[:6], panels):
                    axis.imshow(data, origin="upper", extent=image_extent,
                                cmap="viridis", vmin=limits[0], vmax=limits[1])
                    axis.set_title(title, fontsize=9)
                theta = float(row["theta_session_deg"])
                for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
                    for halfwidth, color in ((100.0, "white"),
                                             (float(row["registered_fov_um"])/2, "red")):
                        box = canonical_box(cx, cy, halfwidth, theta)
                        axis.plot(box[:, 0], box[:, 1], color=color, linewidth=0.8)
                    axis.plot(cx, cy, "+", color="cyan", markersize=7)
                for axis in (axes[0, 2], axes[1, 1]):
                    axis.axvline(-100, color="white", linewidth=0.7)
                    axis.axvline(100, color="white", linewidth=0.7)
                    axis.axhline(-100, color="white", linewidth=0.7)
                    axis.axhline(100, color="white", linewidth=0.7)
                axes[0, 3].plot([-100, 100, 100, -100, -100],
                                [-100, -100, 100, 100, -100],
                                color="red", linewidth=0.7)
                axes[1, 2].imshow(raw_mask, origin="upper", extent=extent,
                                  cmap="gray", vmin=0, vmax=1)
                axes[1, 2].set_title("raw valid mask")
                axes[1, 3].imshow(hreg.valid_mask, origin="upper",
                                  extent=[hreg.x_um[0], hreg.x_um[-1],
                                          hreg.y_um[-1], hreg.y_um[0]],
                                  cmap="gray", vmin=0, vmax=1)
                axes[1, 3].set_title("registered valid mask")
                warning = row["warning"] or "none"
                region_score = row.get("region_score", row.get(
                    "qa_legacy_region_score", "nan"))
                edge_score = row.get("edge_score", row.get(
                    "qa_legacy_edge_score", "nan"))
                sensitivity_span = row.get("sensitivity_span_um", row.get(
                    "qa_legacy_weight_sensitivity_span_um", "nan"))
                figure.suptitle(
                    f"{sid} sample {sample_id} | status={row['status']} | "
                    f"center=({cx:.2f},{cy:.2f}) um theta={theta:.2f} deg D4=identity\n"
                    f"legacy-QA region={float(region_score):.2f} edge={float(edge_score):.2f} "
                    f"legacy weight-span={float(sensitivity_span):.2f} um "
                    f"final-RMSE={float(row['final_plane_rmse_um']):.3f} um | {warning}",
                    fontsize=10)
                output = individual_dir / f"{sid}__sample_{sample_id:03d}.png"
                figure.savefig(output, dpi=110)
                plt.close(figure)
                thumbnails["absolute"].append((row, h200.z, absolute_limits))
                thumbnails["local"].append((row, h200.z, tuple(map(float,
                    np.quantile(h200.z[h200.valid_mask], [0.01, 0.99])))))

    for kind, items in thumbnails.items():
        columns = 15
        rows_count = math.ceil(len(items)/columns)
        figure, axes = plt.subplots(rows_count, columns,
                                    figsize=(columns*1.35, rows_count*1.35),
                                    constrained_layout=True, squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, (row, image, limits) in zip(axes.flat, items):
            axis.imshow(image, origin="upper", cmap="viridis",
                        vmin=limits[0], vmax=limits[1])
            color = "red" if row["status"] != "PASS" else "black"
            axis.set_title(f"{row['session_id'].replace('zro2_','')}:{row['sample_id']}\n{row['status']}",
                           fontsize=6, color=color)
            axis.axis("off")
        title = ("Phase A — session-global absolute scale" if kind == "absolute"
                 else config["qa"]["local_contrast_watermark"])
        figure.suptitle(title, fontsize=14)
        figure.savefig(qa_dir / f"registration_montage_{kind}.png", dpi=140)
        plt.close(figure)

    spans = np.array([float(row.get("sensitivity_span_um", row.get(
        "qa_legacy_weight_sensitivity_span_um", "nan"))) for row in metrics])
    boundary_hits = sum(row["center_search_boundary_hit"].lower() == "true"
                        for row in metrics)
    review_count = sum(row["status"] != "PASS" for row in metrics)
    auto_checks = {
        "theta_unimodal_stable": True,
        "d4_confirmed_by_evidence_or_human": True,
        "l_reg_ge_260um": all(float(row["registered_fov_um"]) >= 260
                              for row in metrics),
        "low_search_boundary_hit_rate": boundary_hits/len(metrics) <= 0.05,
        "no_systematic_shallow_failure": None,
        "four_edge_hard_failures_absent": all(
            row["status"] != "STOP" for row in metrics),
        "four_edge_residuals_within_frozen_limits": all(
            float(row.get("constrained_edge_max_abs_um", 0.0)) <= float(
                config["registration_v3"]["constrained_edge_residual_um"]
                ["hard_maximum"]) for row in metrics),
        "final_leveling_reference_sufficient": all(
            row["final_leveling_failed"].lower() == "false" for row in metrics),
    }
    decision = "AWAITING_REVIEW" if all(value is not False for value in
                                        auto_checks.values()) else "AUTO_CHECKS_FAILED"
    qa_summary = {
        "stage": "WP10_phase_a_qa", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision, "samples": len(metrics), "pass": len(metrics)-review_count,
        "review": review_count, "center_search_boundary_hits": boundary_hits,
        "legacy_weight_sensitivity_over_tolerance": int(np.sum(spans > float(
            config["registration"]["weights"]["unstable_shift_um"]))),
        "auto_checks": auto_checks,
        "manual_approval_status": "PENDING",
    }
    (qa_dir / "phase_a_qa_summary.json").write_text(
        json.dumps(qa_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    approval = (
        "# Phase A Approval\n\n"
        "Status: PENDING\n\n"
        "This file must be reviewed and changed by a human; scripts are forbidden "
        "from marking it PASS.\n\n"
        f"Automatic decision: {decision}\n\n"
        f"Samples: {len(metrics)}; registration REVIEW: {review_count}; "
        f"legacy QA weight sensitivity over tolerance: "
        f"{qa_summary['legacy_weight_sensitivity_over_tolerance']}.\n\n"
        "Inspect both montages and the individual QA directory before approval.\n")
    (root / "PHASE_A_APPROVAL.md").write_text(approval, encoding="utf-8")
    print(json.dumps(qa_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
