#!/usr/bin/env python3
"""WP5: rebuild the Phase A QA evidence for manual_v1 and write the
approval document (PENDING only -- scripts can never approve).

Renders one individual QA image for every one of the 200 manual_v1 samples
(including the 22 samples excluded from registered exports by the frozen
Phase A exclusion policy, which get explicit NOT-EXPORTED panels), two
montages, an automatic-check summary, and PHASE_A_APPROVAL_MANUAL_V1.md.
No legacy v2 montage or QA image is touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.io_cag import CagHeightReader  # noqa: E402
from src.io_npz import load_height_npz  # noqa: E402
from src.manual_registration_evaluation import (  # noqa: E402
    ManualRegistrationError, render_approval_text, uv_to_xy)
from src.stage_manifest import append_stage_record, sha256_of  # noqa: E402

ROOT = REPO / "outputs/rectangle_registration"
MANUAL_V1_DIR = ROOT / "manual_v1"
REG_MANUAL_V1_DIR = ROOT / "registration/manual_v1"
QA_DIR = ROOT / "qa/manual_v1"
APPROVAL_PATH = ROOT / "PHASE_A_APPROVAL_MANUAL_V1.md"
RUN_MANIFEST = MANUAL_V1_DIR / "run_manifest.json"
CONFIG_PATH = REPO / "config/manual_registration_v1.yaml"
SESSION_MANIFEST = REPO / "config/session_manifest.csv"
V6_METRICS = ROOT / "registration/translation_metrics_v6.csv"


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


def manual_box_polygon(left_u: float, right_u: float, top_v: float,
                       bottom_v: float, theta_deg: float) -> np.ndarray:
    corners = [(left_u, top_v), (right_u, top_v),
               (right_u, bottom_v), (left_u, bottom_v), (left_u, top_v)]
    points = np.array([
        uv_to_xy(u, v, theta_deg) for u, v in corners])
    return points


def _text_panel(axis, lines: list[str]) -> None:
    axis.axis("off")
    axis.text(0.5, 0.5, "\n".join(lines), ha="center", va="center",
              fontsize=9, color="#7a0000", wrap=True)


SHORT_EXCLUSION_LABEL = {
    "remeasurement_required": "session below the 260 um grid minimum",
    "excluded_complete_paired_measurement": "paired measurement excluded",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    config = yaml.safe_load(
        (REPO / "config/rectangle_registration.yaml")
        .read_text(encoding="utf-8"))
    manual_config = yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8"))
    watermark = config["qa"]["local_contrast_watermark"]
    local_probabilities = [float(v)/100.0 for v in
                           config["qa"]["local_color_percentiles"]]

    freeze_manifest = json.loads(
        (REG_MANUAL_V1_DIR / "freeze_manifest.json")
        .read_text(encoding="utf-8"))
    if freeze_manifest["decision"] != "PASS":
        print("STOP: WP1 freeze manifest is not PASS", file=sys.stderr)
        return 2
    wp3_summary = json.loads(
        (REG_MANUAL_V1_DIR / "translation_summary_manual_v1.json")
        .read_text(encoding="utf-8"))
    if wp3_summary["decision"] != "PASS":
        print("STOP: WP3 manual_v1 table is not PASS", file=sys.stderr)
        return 2
    resampling_summary = json.loads(
        (MANUAL_V1_DIR / "resampling/resampling_summary.json")
        .read_text(encoding="utf-8"))
    if resampling_summary.get("decision") != "PASS":
        print("STOP: WP4 resampling/final leveling is not PASS",
              file=sys.stderr)
        return 2
    canvas_summary = json.loads(
        (MANUAL_V1_DIR / "resampling/common_fov_summary.json")
        .read_text(encoding="utf-8"))

    wp3_rows = read_csv(
        REG_MANUAL_V1_DIR / "translation_metrics_manual_v1.csv")
    wp3_by_key = {(row["session_id"], row["sample_id"]): row
                  for row in wp3_rows}
    geometry_by_session = {row["session_id"]: row for row in read_csv(
        ROOT / "geometry/session_geometry.csv")}
    metrics_rows = read_csv(
        MANUAL_V1_DIR / "metrics/registration_metrics.csv")
    metrics_by_key = {(row["session_id"], row["sample_id"]): row
                      for row in metrics_rows}
    v6_by_key = {(row["session_id"], row["sample_id"]): row
                 for row in read_csv(V6_METRICS)}
    diagnostics = read_csv(
        MANUAL_V1_DIR / "resampling/sample_fov_diagnostics.csv")
    diag_by_key = {(row["session_id"], row["sample_id"]): row
                   for row in diagnostics}
    planes = {(row["session_id"], int(row["measurement_id"])): row
              for row in read_csv(
                  ROOT / "metrics/coarse_leveling_metrics.csv")}
    sessions = read_csv(SESSION_MANIFEST)

    individual_dir = QA_DIR / "registration_individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    # ---- session-global absolute colour limits (exported samples) ---------
    session_limits: dict[str, tuple[float, float]] = {}
    for sid in sorted({row["session_id"] for row in metrics_rows}):
        values = np.concatenate([
            load_height_npz(REPO / row["h_200_path"]).z[
                load_height_npz(REPO / row["h_200_path"]).valid_mask][::100]
            for row in metrics_rows if row["session_id"] == sid])
        session_limits[sid] = tuple(map(float, np.quantile(values,
                                                           [0.01, 0.99])))

    # ---- individual QA images ---------------------------------------------
    thumbnails: list[tuple[dict, np.ndarray | None, tuple | None, str]] = []
    for session in sessions:
        sid = session["session_id"]
        session_rows = sorted(
            (row for row in wp3_rows if row["session_id"] == sid),
            key=lambda row: (int(row["measurement_id"]), int(row["sample_id"])))
        if not session_rows:
            continue
        with CagHeightReader(REPO / session["cag_path"]) as reader:
            cached_id = None
            raw = None
            for row in session_rows:
                measurement_id = int(row["measurement_id"])
                sample_id = int(row["sample_id"])
                key = (sid, row["sample_id"])
                if cached_id != measurement_id:
                    raw = reader.read_height_map(measurement_id)
                    cached_id = measurement_id
                plane = planes[(sid, measurement_id)]
                x = raw.x_um - raw.width_um/2.0
                y = raw.y_um - raw.height_um/2.0
                a, b, c = (float(plane[k]) for k in ("a", "b", "c"))
                coarse = raw.z - (a*x[None, :] + b*y[:, None] + c)
                cx, cy = float(row["manual_center_x_um"]), float(
                    row["manual_center_y_um"])
                theta = float(row["theta_session_deg"])
                exported = key in metrics_by_key
                metric = metrics_by_key.get(key)
                if exported:
                    half_crop = float(metric["registered_fov_um"])/2.0 + 10.0
                else:
                    half_crop = 160.0
                columns = (x >= cx-half_crop) & (x <= cx+half_crop)
                selected_rows = (y >= cy-half_crop) & (y <= cy+half_crop)
                raw_crop = raw.z[np.ix_(selected_rows, columns)]
                coarse_crop = coarse[np.ix_(selected_rows, columns)]
                raw_mask = raw.valid_mask[np.ix_(selected_rows, columns)]
                extent = [x[columns][0], x[columns][-1],
                          y[selected_rows][-1], y[selected_rows][0]]

                manual_box = manual_box_polygon(
                    float(row["manual_left_u_um"]), float(
                        row["manual_right_u_um"]),
                    float(row["manual_top_v_um"]), float(
                        row["manual_bottom_v_um"]), theta)
                nominal_box = canonical_box(cx, cy, 100.0, theta)
                v6 = v6_by_key[key]
                v6_cx, v6_cy = float(v6["center_x_um"]), float(v6["center_y_um"])
                v6_box = canonical_box(v6_cx, v6_cy, 100.0, theta)

                # a dedicated bottom banner row keeps long exclusion reasons
                # from overlapping the QA panels
                figure = plt.figure(figsize=(17, 8.5), constrained_layout=True)
                grid = GridSpec(3, 4, figure=figure,
                                height_ratios=[1.0, 1.0, 0.30])
                axes = np.array([[figure.add_subplot(grid[row, column])
                                  for column in range(4)]
                                 for row in range(2)])
                banner_axis = figure.add_subplot(grid[2, :])
                banner_axis.axis("off")

                if sid in session_limits:
                    absolute_limits = session_limits[sid]
                else:  # excluded session: local robust limits for raw panels
                    absolute_limits = tuple(map(float, np.quantile(
                        coarse_crop[raw_mask], [0.01, 0.99])))

                def overlay_raw(axis: plt.Axes) -> None:
                    axis.plot(manual_box[:, 0], manual_box[:, 1],
                              color="cyan", linewidth=1.1, label="manual box")
                    axis.plot(nominal_box[:, 0], nominal_box[:, 1],
                              color="white", linewidth=0.8)
                    axis.plot(cx, cy, "+", color="cyan", markersize=8)
                    axis.plot(v6_cx, v6_cy, "x", color="yellow",
                              markersize=8, markeredgewidth=1.6)
                    axis.plot(v6_box[:, 0], v6_box[:, 1], color="magenta",
                              linewidth=0.7, linestyle=":")

                axes[0, 0].imshow(raw_crop, origin="upper", extent=extent,
                                  cmap="viridis", vmin=absolute_limits[0],
                                  vmax=absolute_limits[1])
                axes[0, 0].set_title("raw height", fontsize=9)
                overlay_raw(axes[0, 0])
                axes[0, 1].imshow(coarse_crop, origin="upper", extent=extent,
                                  cmap="viridis", vmin=absolute_limits[0],
                                  vmax=absolute_limits[1])
                axes[0, 1].set_title("coarse-levelled", fontsize=9)
                overlay_raw(axes[0, 1])
                axes[1, 0].imshow(
                    coarse_crop, origin="upper", extent=extent, cmap="viridis",
                    vmin=float(np.quantile(coarse_crop[raw_mask], 0.01)),
                    vmax=float(np.quantile(coarse_crop[raw_mask], 0.99)))
                axes[1, 0].set_title(watermark, fontsize=6.5, color="#7a0000")
                overlay_raw(axes[1, 0])
                axes[1, 2].imshow(raw_mask, origin="upper", extent=extent,
                                  cmap="gray", vmin=0, vmax=1)
                axes[1, 2].set_title("raw valid mask", fontsize=9)

                if exported:
                    hreg = load_height_npz(REPO / metric["h_reg_path"])
                    h200 = load_height_npz(REPO / metric["h_200_path"])
                    hreg_extent = [hreg.x_um[0], hreg.x_um[-1],
                                   hreg.y_um[-1], hreg.y_um[0]]
                    h200_extent = [h200.x_um[0], h200.x_um[-1],
                                   h200.y_um[-1], h200.y_um[0]]
                    reg_box = canonical_box(
                        cx, cy, float(metric["registered_fov_um"])/2.0, theta)
                    for axis in (axes[0, 0], axes[0, 1]):
                        axis.plot(reg_box[:, 0], reg_box[:, 1],
                                  color="red", linewidth=0.7)
                    axes[0, 2].imshow(hreg.z, origin="upper",
                                      extent=hreg_extent, cmap="viridis",
                                      vmin=absolute_limits[0],
                                      vmax=absolute_limits[1])
                    axes[0, 2].set_title("H_reg — absolute", fontsize=9)
                    axes[0, 3].imshow(h200.z, origin="upper",
                                      extent=h200_extent, cmap="viridis",
                                      vmin=absolute_limits[0],
                                      vmax=absolute_limits[1])
                    axes[0, 3].set_title("H_200 — absolute", fontsize=9)
                    axes[1, 1].imshow(
                        hreg.z, origin="upper", extent=hreg_extent,
                        cmap="viridis",
                        vmin=float(np.quantile(
                            hreg.z[hreg.valid_mask], local_probabilities[0])),
                        vmax=float(np.quantile(
                            hreg.z[hreg.valid_mask], local_probabilities[1])))
                    axes[1, 1].set_title(watermark, fontsize=6.5,
                                         color="#7a0000")
                    axes[1, 3].imshow(hreg.valid_mask, origin="upper",
                                      extent=hreg_extent, cmap="gray",
                                      vmin=0, vmax=1)
                    axes[1, 3].set_title("registered valid mask", fontsize=9)
                    for axis, colour in ((axes[0, 2], "white"),
                                         (axes[0, 3], "red"),
                                         (axes[1, 1], "white")):
                        axis.axvline(float(row["manual_left_u_um"]),
                                     color="cyan", linewidth=0.7,
                                     linestyle="--")
                        axis.axvline(float(row["manual_right_u_um"]),
                                     color="cyan", linewidth=0.7,
                                     linestyle="--")
                        axis.axhline(float(row["manual_top_v_um"]),
                                     color="cyan", linewidth=0.7,
                                     linestyle="--")
                        axis.axhline(float(row["manual_bottom_v_um"]),
                                     color="cyan", linewidth=0.7,
                                     linestyle="--")
                        axis.axvline(-100, color=colour, linewidth=0.7)
                        axis.axvline(100, color=colour, linewidth=0.7)
                        axis.axhline(-100, color=colour, linewidth=0.7)
                        axis.axhline(100, color=colour, linewidth=0.7)
                    final_rmse = float(metric["final_plane_rmse_um"])
                    valid_fraction = float(metric["h200_valid_fraction"])
                    level_line = (f"final-RMSE={final_rmse:.3f} um | "
                                  f"H_200 valid={valid_fraction:.3f}")
                    thumbnail_limits = absolute_limits
                    thumbnail = h200.z
                    banner_axis.text(
                        0.5, 0.5,
                        f"EXPORTED  L_reg={float(metric['registered_fov_um']):.2f} um"
                        f"  |  H_200 valid={valid_fraction:.3f}"
                        f"  |  final-RMSE={final_rmse:.3f} um"
                        "  |  H_reg + H_200 + mask NPZ written",
                        ha="center", va="center", fontsize=8.5,
                        color="#1a4d1a", wrap=True)
                else:
                    diag = diag_by_key[key]
                    reason = " ".join(
                        (diag.get("exclusion_reason") or "excluded").split())
                    short_label = SHORT_EXCLUSION_LABEL.get(
                        diag.get("exclusion_disposition", ""),
                        "excluded by the explicit exclusion policy")
                    for axis, title in (
                            (axes[0, 2], "H_reg"),
                            (axes[0, 3], "H_200"),
                            (axes[1, 1], "H_reg local contrast"),
                            (axes[1, 3], "registered valid mask")):
                        _text_panel(axis, [f"{title} NOT EXPORTED",
                                           short_label])
                    banner_axis.text(
                        0.5, 0.5,
                        "NOT EXPORTED (excluded from registered exports by "
                        f"the explicit exclusion policy): {reason}",
                        ha="center", va="center", fontsize=8.5,
                        color="#7a0000", wrap=True)
                    level_line = ("final-RMSE=n/a (not exported) | available "
                                  f"centred square="
                                  f"{float(diag['available_centered_square_um']):.2f}"
                                  " um")
                    thumbnail = None
                    thumbnail_limits = None

                v6_disagreement = float(row["qa_v6_center_disagreement_um"])
                warning = row["warning"] or "none"
                figure.suptitle(
                    f"{sid} sample {sample_id} | manual_v1 status="
                    f"{row['status']} | "
                    f"manual center=({cx:.2f},{cy:.2f}) um "
                    f"theta={theta:.2f} deg D4={row['d4_transform_session']}\n"
                    f"manual box {float(row['manual_width_um']):.1f} x "
                    f"{float(row['manual_height_um']):.1f} um (QA obs; "
                    "nominal stays 200 x 200 um) | v6 status="
                    f"{row['qa_v6_status']} (magenta box, yellow x), "
                    f"manual-vs-v6 disagreement {v6_disagreement:.2f} um\n"
                    f"{level_line} | warning: {warning}", fontsize=9)
                output_path = (individual_dir
                               / f"{sid}__sample_{sample_id:03d}.png")
                figure.savefig(output_path, dpi=110)
                plt.close(figure)
                thumbnails.append((row, thumbnail, thumbnail_limits,
                                   row["status"] if exported else "EXCLUDED"))

    # ---- montages -----------------------------------------------------------
    for kind in ("absolute", "local"):
        columns = 15
        rows_count = math.ceil(len(thumbnails)/columns)
        figure, axes = plt.subplots(rows_count, columns,
                                    figsize=(columns*1.35, rows_count*1.35),
                                    constrained_layout=True, squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, (row, image, limits, label) in zip(axes.flat, thumbnails):
            if image is not None:
                if kind == "local":
                    mask = ~np.isnan(image)
                    limits = tuple(map(float, np.quantile(
                        image[mask], local_probabilities))) if mask.any() \
                        else (0.0, 1.0)
                axis.imshow(image, origin="upper", cmap="viridis",
                            vmin=limits[0], vmax=limits[1])
            else:
                axis.add_patch(plt.Rectangle(
                    (0.02, 0.02), 0.96, 0.96, transform=axis.transAxes,
                    fill=True, facecolor="0.85", edgecolor="#7a0000",
                    linewidth=1.2))
                axis.text(0.5, 0.5, "EXCLUDED\nno H_200",
                          transform=axis.transAxes, ha="center", va="center",
                          fontsize=6, color="#7a0000")
            color = "red" if label != "PASS" else "black"
            axis.set_title(
                f"{row['session_id'].replace('zro2_', '')}:"
                f"{row['sample_id']}\n{label}", fontsize=6, color=color)
            axis.axis("off")
        title = ("Phase A manual_v1 — session-global absolute scale"
                 if kind == "absolute" else
                 f"Phase A manual_v1 — {watermark}")
        figure.suptitle(title, fontsize=14)
        figure.savefig(QA_DIR / f"registration_montage_{kind}.png", dpi=140)
        plt.close(figure)

    # ---- automatic checks -----------------------------------------------------
    # Every gate below is evaluated over ALL 200 manual_v1 samples.  A sample
    # without a registered export (whatever the reason) fails the gate: an
    # excluded session is a blocker, never a way to satisfy the gate.
    exported_keys = set(metrics_by_key)
    all_keys = set(wp3_by_key)
    excluded_keys = sorted(all_keys - exported_keys)
    samples_total = len(wp3_rows)
    coverage_ok = (len(metrics_rows) == samples_total and not excluded_keys)

    l_reg_ok = all(float(row["registered_fov_um"]) >= 260.0
                   for row in metrics_rows)
    final_leveling_ok = (
        resampling_summary["decision"] == "PASS"
        and all(row["final_leveling_failed"].lower() == "false"
                for row in metrics_rows))
    reference_ok = all(
        float(row["final_reference_fraction"]) >= float(
            config["final_leveling"]["minimum_reference_valid_fraction"])
        for row in metrics_rows)
    npz_counts = {
        "h_reg": len(list((MANUAL_V1_DIR / "registered/H_reg")
                          .glob("*.npz"))),
        "h_200": len(list((MANUAL_V1_DIR / "registered/H_200")
                          .glob("*.npz"))),
        "masks": len(list((MANUAL_V1_DIR / "registered/masks")
                          .glob("*.npz"))),
    }
    per_row_npz_present = all(
        all((REPO / row[field]).is_file()
            for field in ("h_reg_path", "h_200_path", "mask_path")
            if row.get(field))
        for row in metrics_rows)
    session_canvases_ok = all(session["status"] == "PASS"
                              for session in canvas_summary["sessions"])
    masks_complete = (npz_counts == {"h_reg": samples_total,
                                     "h_200": samples_total,
                                     "masks": samples_total}
                      and per_row_npz_present and session_canvases_ok)

    # no sample-wise method mixing: every row uses the manual method and the
    # exported centres are exactly the WP3 manual centres
    method_ok = ({row["registration_method"] for row in wp3_rows}
                 == {"manual_four_edge_a_v1"}
                 and {row["registration_method"] for row in metrics_rows}
                 == {"manual_four_edge_a_v1"})
    centre_ok = all(
        abs(float(metric["center_x_um"])
            - float(wp3_by_key[key]["manual_center_x_um"])) <= 1e-6
        and abs(float(metric["center_y_um"])
                - float(wp3_by_key[key]["manual_center_y_um"])) <= 1e-6
        for key, metric in metrics_by_key.items())
    samplewise_mixing_absent = method_ok and centre_ok

    # provenance hash chain: config -> freeze -> WP3 -> every exported NPZ
    # (H_reg, H_200 and the mask archive all carry metadata)
    snapshot_sha = freeze_manifest["snapshot"]["sha256"]
    config_sha = sha256_of(CONFIG_PATH)
    provenance_ok = (
        snapshot_sha == manual_config["expected_source_sha256"]
        and freeze_manifest["source"]["sha256"] == snapshot_sha
        and all(row["source_sha256"] == snapshot_sha for row in wp3_rows)
        and all(row["config_sha256"] == config_sha for row in wp3_rows))

    def npz_metadata_ok(row: dict, field: str) -> bool:
        path = REPO / row[field]
        if not path.is_file():
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                if "metadata_json" not in data.files:
                    return False
                metadata = json.loads(str(data["metadata_json"]))
        except (OSError, ValueError, KeyError):
            return False
        return (metadata.get("manual_annotation_sha256") == snapshot_sha
                and metadata.get("registration_method")
                == "manual_four_edge_a_v1"
                and metadata.get("manual_registration_config_sha256")
                == config_sha)

    npz_provenance_detail = {
        field: all(npz_metadata_ok(row, field) for row in metrics_rows)
        for field in ("h_reg_path", "h_200_path", "mask_path")}
    npz_provenance_ok = all(npz_provenance_detail.values())
    provenance_ok = provenance_ok and npz_provenance_ok and coverage_ok

    theta_d4_ok = (
        all(row["status"] == "PASS" and row["d4_status"] == "confirmed"
            for row in geometry_by_session.values())
        and all(
            abs(float(wp3_row["theta_session_deg"])
                - float(geometry_by_session[wp3_row["session_id"]]
                        ["theta_session_deg"])) <= 1e-9
            and wp3_row["d4_transform_session"]
            == geometry_by_session[wp3_row["session_id"]]
            ["d4_transform_session"]
            for wp3_row in wp3_rows))

    auto_checks = {
        "manual_geometry_gate_200_of_200": (
            wp3_summary["status_counts"].get("PASS", 0) == 200
            and samples_total == 200),
        "theta_d4_frozen_confirmed": theta_d4_ok,
        "paired_gate_all_passed": wp3_summary["paired_gate_pass"] == 200,
        "all_200_samples_have_registered_exports": coverage_ok,
        "l_reg_ge_260um_all_samples": coverage_ok and l_reg_ok,
        "final_leveling_all_pass": coverage_ok and final_leveling_ok,
        "final_leveling_reference_sufficient": coverage_ok and reference_ok,
        "registered_npz_complete_all_samples": masks_complete,
        "no_samplewise_method_mixing": samplewise_mixing_absent,
        "provenance_hash_chain_closed": provenance_ok,
    }

    # ---- blockers declared by the explicit exclusion policy -----------------
    exclusion_policy_rel = canvas_summary.get("exclusion_policy")
    exclusion_policy_sha = None
    policy_blockers: list[dict] = []
    if exclusion_policy_rel:
        policy_path = REPO / Path(str(exclusion_policy_rel).replace("\\", "/"))
        policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        exclusion_policy_sha = sha256_of(policy_path)
        rows_per_session = Counter(row["session_id"] for row in wp3_rows)
        for item in policy.get("exclusions", []):
            if not item.get("blocks_phase_a_acceptance"):
                continue
            policy_blockers.append({
                "id": f"policy:{item['session_id']}",
                "scope": item["session_id"],
                "samples": rows_per_session.get(item["session_id"], 0),
                "disposition": item.get("disposition", ""),
                "statement": " ".join(str(item["reason"]).split())})
        for blocker in policy.get("blockers", []):
            policy_blockers.append({
                "id": str(blocker.get("id", "BLK")),
                "scope": str(blocker.get("scope", "")),
                "samples": int(blocker.get("samples", 0)),
                "disposition": str(blocker.get("disposition", "blocker")),
                "statement": " ".join(
                    str(blocker.get("statement", "")).split())})

    failed_checks = [name for name, value in auto_checks.items()
                     if not value]
    if policy_blockers:
        decision = "BLOCKED"
    elif failed_checks:
        decision = "AUTO_CHECKS_FAILED"
    else:
        decision = "AWAITING_REVIEW"
    approval_status = ("BLOCKED"
                       if decision in {"BLOCKED", "AUTO_CHECKS_FAILED"}
                       else "PENDING")

    known_conditions = {
        "gate_coverage": (f"all {samples_total} manual_v1 samples are in "
                          "scope for every gate; a sample without a "
                          "registered export fails the gate, it is never "
                          "excused by the exclusion policy"),
        "samples_without_registered_export": {
            "count": len(excluded_keys),
            "keys": [f"{sid}:{sample}" for sid, sample in excluded_keys],
            "policy": exclusion_policy_rel or "none (no policy passed)",
            "policy_sha256": exclusion_policy_sha,
        },
        "zro2_60_pass_samples_53_54": (
            "re-included by config/phase_a_exclusions_manual_v1.yaml: the "
            "available centred square is 338.65 um and 276.35 um with "
            "manual centres, both at or above the 260 um minimum, so the "
            "legacy measurement-27 exclusion was not carried over"),
        "session_canvas_status": {
            session["session_id"]: session["status"]
            for session in canvas_summary["sessions"]},
        "npz_provenance_detail": npz_provenance_detail,
        "manual_box_corner_overflow_observations": (
            wp3_summary["box_corner_overflow_samples"]),
    }

    summary = {
        "stage": "WP5_phase_a_qa_manual_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "manual_approval_status": approval_status,
        "samples_total": samples_total,
        "exported_h_reg_h_200": len(metrics_rows),
        "samples_without_registered_export": len(excluded_keys),
        "failed_auto_checks": failed_checks,
        "blockers": policy_blockers,
        "exclusion_policy": exclusion_policy_rel,
        "exclusion_policy_sha256": exclusion_policy_sha,
        "individual_qa_images": len(thumbnails),
        "npz_counts": npz_counts,
        "npz_counts_expected": samples_total,
        "session_canvas": [
            {key: session.get(key) for key in
             ("session_id", "status", "common_fov_um", "registered_fov_um")}
            for session in canvas_summary["sessions"]],
        "auto_checks": auto_checks,
        "known_conditions": known_conditions,
        "qa_outputs": {
            "individual_dir": str(individual_dir.relative_to(REPO)),
            "montage_absolute": str(
                (QA_DIR / "registration_montage_absolute.png")
                .relative_to(REPO)),
            "montage_local": str(
                (QA_DIR / "registration_montage_local.png")
                .relative_to(REPO)),
            "approval_file": str(APPROVAL_PATH.relative_to(REPO)),
        },
        "provenance": {
            "manual_annotation_sha256": snapshot_sha,
            "manual_registration_config_sha256": sha256_of(CONFIG_PATH),
            "v6_metrics_sha256": sha256_of(V6_METRICS),
        },
    }
    summary_path = QA_DIR / "phase_a_qa_summary_manual_v1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # ---- approval document (PENDING/BLOCKED only, never PASS) ----------------
    body_lines = [
        f"Automatic decision: {decision}",
        "",
        (f"Samples: {samples_total} manual_v1 rows (all manual geometry "
         f"gates PASS); {len(metrics_rows)} exported H_reg/H_200; "
         f"{len(excluded_keys)} samples have no registered export and "
         "therefore fail the all-sample gates."),
        "",
        "Automatic checks (evaluated over all "
        f"{samples_total} samples):",
        *[f"- {name}: {'PASS' if value else 'FAIL'}"
          for name, value in auto_checks.items()],
    ]
    if failed_checks:
        body_lines += ["", f"Failed automatic checks: {failed_checks}"]
    if policy_blockers:
        body_lines += [
            "",
            "Blocking conditions (Phase A cannot be approved until these "
            "are resolved):",
            *[f"- [{blocker['id']}] {blocker['scope']} "
              f"({blocker['samples']} samples, "
              f"{blocker['disposition']}): {blocker['statement']}"
              for blocker in policy_blockers],
            "",
            "Required resolution: remeasure the affected session, or amend "
            "the acceptance protocol formally and re-run the full chain. Do "
            "not switch this file to PASS manually while a blocker stands.",
        ]
    body_lines += [
        "",
        "Known conditions for the reviewer:",
        *[f"- {key}: {json.dumps(value, ensure_ascii=False)}"
          for key, value in known_conditions.items()],
        "",
        ("Review both montages and the individual QA images under "
         "qa/manual_v1/ before deciding. The v6 centres/statuses shown as "
         "overlays are QA comparators only; manual_v1 uses manual centres "
         "for every sample with no per-sample fallback."),
    ]
    try:
        approval_text = render_approval_text(
            status=approval_status, decision=decision,
            body_lines=body_lines)
    except ManualRegistrationError as exc:  # pragma: no cover
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    APPROVAL_PATH.write_text(approval_text, encoding="utf-8")

    append_stage_record(
        RUN_MANIFEST, stage="WP5_phase_a_qa_manual_v1",
        command=[sys.executable, str(Path(__file__).relative_to(REPO))],
        exit_code=0,
        inputs={"manual_v1_metrics": MANUAL_V1_DIR /
                 "metrics/registration_metrics.csv",
                "translation_metrics_manual_v1": REG_MANUAL_V1_DIR /
                 "translation_metrics_manual_v1.csv",
                "freeze_manifest": REG_MANUAL_V1_DIR / "freeze_manifest.json",
                "v6_metrics": V6_METRICS,
                "resampling_summary": MANUAL_V1_DIR /
                "resampling/resampling_summary.json",
                "common_fov_summary": MANUAL_V1_DIR /
                "resampling/common_fov_summary.json",
                "exclusion_policy": (REPO / Path(
                    str(exclusion_policy_rel).replace("\\", "/"))
                    if exclusion_policy_rel else REPO)},
        outputs={"individual_qa_dir": individual_dir,
                 "montage_absolute": QA_DIR /
                 "registration_montage_absolute.png",
                 "montage_local": QA_DIR / "registration_montage_local.png",
                 "phase_a_qa_summary": summary_path,
                 "approval_file": APPROVAL_PATH},
        extra={"decision": decision,
               "manual_approval_status": approval_status,
               "auto_checks": auto_checks,
               "failed_auto_checks": failed_checks,
               "blockers": [blocker["id"] for blocker in policy_blockers]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
