#!/usr/bin/env python3
"""WP2: manual (annotator A) vs automatic (v3-v7) consistency evaluation.

Reads ONLY the WP1 frozen snapshot, merges one-to-one with each automatic
translation table, and reports per-sample and stratified consistency
statistics.  All automatic versions are QA comparators only: their frozen
statuses are never modified and no centre source is selected per sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.manual_registration_evaluation import (  # noqa: E402
    band_fractions, band_label, evaluate_pair, merge_one_to_one,
    summarize_axis, summarize_radial)
from src.stage_manifest import append_stage_record, sha256_of  # noqa: E402

CONFIG_PATH = REPO / "config/manual_registration_v1.yaml"
REGISTRATION_DIR = REPO / "outputs/rectangle_registration/registration"
MANUAL_V1_DIR = REGISTRATION_DIR / "manual_v1"
QA_DIR = REPO / "outputs/rectangle_registration/qa/manual_v1"
SESSION_GEOMETRY_CSV = (REPO / "outputs/rectangle_registration/geometry"
                        / "session_geometry.csv")
MEASUREMENT_METRICS_CSV = (REPO / "outputs/rectangle_registration/inventory"
                           / "measurement_metrics.csv")
RUN_MANIFEST = (REPO / "outputs/rectangle_registration/manual_v1"
                / "run_manifest.json")
VERSIONS = ("v3", "v4", "v5", "v6", "v7")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


STAT_FIELDS = [
    "n",
    "delta_u_median_um", "delta_u_mad_um", "delta_u_q05_um", "delta_u_q95_um",
    "delta_v_median_um", "delta_v_mad_um", "delta_v_q05_um", "delta_v_q95_um",
    "radial_median_um", "radial_q90_um", "radial_q95_um", "radial_max_um",
    "frac_le_close", "frac_close_to_moderate", "frac_above_moderate",
]


def group_stats(samples: list[dict], bands: dict) -> dict:
    stats = {"n": len(samples)}
    for axis in ("u", "v"):
        axis_stats = summarize_axis(
            [row[f"delta_{axis}_um"] for row in samples])
        for key, value in axis_stats.items():
            stats[f"delta_{axis}_{key}"] = value
    radial_stats = summarize_radial(
        [row["center_disagreement_um"] for row in samples])
    for key, value in radial_stats.items():
        stats[f"radial_{key}"] = value
    for key, value in band_fractions(
            [row["center_disagreement_um"] for row in samples],
            close_um=bands["close"], moderate_um=bands["moderate"]).items():
        stats[key] = value
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_sha = sha256_of(config_path)
    bands = config["algorithm_manual_qa_bands_um"]

    freeze_manifest = json.loads(
        (MANUAL_V1_DIR / "freeze_manifest.json").read_text(encoding="utf-8"))
    if freeze_manifest["decision"] != "PASS":
        print("STOP: WP1 freeze manifest is not PASS", file=sys.stderr)
        return 2
    snapshot_path = MANUAL_V1_DIR / "manual_four_edge_validation_frozen.csv"
    snapshot_sha = sha256_of(snapshot_path)
    if snapshot_sha != freeze_manifest["snapshot"]["sha256"]:
        print("STOP: frozen snapshot hash differs from freeze manifest",
              file=sys.stderr)
        return 2
    manual_rows = read_csv(snapshot_path)

    geometry = {row["session_id"]: row
                for row in read_csv(SESSION_GEOMETRY_CSV)}
    measurements = {(row["session_id"], int(row["measurement_id"])): row
                    for row in read_csv(MEASUREMENT_METRICS_CSV)}

    # depth stratification proxy, frozen from the inventory metrics:
    # negative_tail_amplitude = |Q01 - Q50| of the raw height histogram,
    # a robust groove-depth proxy per measurement (paired slots share it).
    depth_values = []
    for row in manual_rows:
        measurement = measurements[(row["session_id"],
                                    int(row["measurement_id"]))]
        depth_values.append(float(measurement["negative_tail_amplitude"]))
    quartile_cuts = [float(np.quantile(depth_values, q)) for q in
                     (0.25, 0.50, 0.75)]
    depth_by_key: dict[tuple, tuple[float, str]] = {}
    for row, depth in zip(manual_rows, depth_values):
        quartile = int(np.digitize(depth, quartile_cuts)) + 1
        depth_by_key[(row["session_id"], row["sample_id"])] = (
            depth, f"Q{quartile}")

    per_sample_rows: list[dict] = []
    version_stats: dict[str, dict] = {}
    for version in VERSIONS:
        auto_path = REGISTRATION_DIR / f"translation_metrics_{version}.csv"
        auto_rows = read_csv(auto_path)
        pairs = merge_one_to_one(manual_rows, auto_rows)
        version_samples: list[dict] = []
        for manual, auto in pairs:
            theta = float(geometry[manual["session_id"]]["theta_session_deg"])
            evaluation = evaluate_pair(manual, auto, theta)
            depth, quartile = depth_by_key[
                (manual["session_id"], manual["sample_id"])]
            record = {
                "version": version,
                "session_id": manual["session_id"],
                "measurement_id": manual["measurement_id"],
                "sample_id": manual["sample_id"],
                "roi_within_measurement": manual["roi_within_measurement"],
                "manual_center_u_um": float(
                    manual["annotator_a_center_u_um"]),
                "manual_center_v_um": float(
                    manual["annotator_a_center_v_um"]),
                "manual_width_um": float(manual["annotator_a_width_um"]),
                "manual_height_um": float(manual["annotator_a_height_um"]),
                "auto_center_x_um": float(auto["center_x_um"]),
                "auto_center_y_um": float(auto["center_y_um"]),
                "auto_status": auto["status"],
                "depth_proxy_um": depth,
                "depth_quartile": quartile,
                **evaluation,
                "consistency_band": band_label(
                    evaluation["center_disagreement_um"],
                    close_um=bands["close"], moderate_um=bands["moderate"]),
            }
            per_sample_rows.append(record)
            version_samples.append(record)

        stats = group_stats(version_samples, bands)
        status_counts = defaultdict(int)
        for sample in version_samples:
            status_counts[sample["auto_status"]] += 1
        stats["status_counts"] = dict(sorted(status_counts.items()))
        if version == config["primary_automatic_comparator"]:
            stats["stop_samples"] = [
                f"{s['session_id']}:{s['sample_id']}"
                for s in version_samples if s["auto_status"] == "STOP"]
            stats["review_samples"] = [
                f"{s['session_id']}:{s['sample_id']}"
                for s in version_samples if s["auto_status"] == "REVIEW"]
            for status_key in ("stop_samples", "review_samples"):
                stats[f"{status_key}_disagreement"] = {
                    "median_um": summarize_radial([
                        s["center_disagreement_um"] for s in version_samples
                        if (s["auto_status"] == "STOP"
                            if status_key == "stop_samples" else
                            s["auto_status"] == "REVIEW")])["median_um"],
                }
        version_stats[version] = stats

    # ---- stratified tables ------------------------------------------------
    def stratified(field: str) -> list[dict]:
        rows_out: list[dict] = []
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for sample in per_sample_rows:
            groups[(sample["version"], sample[field])].append(sample)
        for (version, group), samples in sorted(groups.items()):
            rows_out.append({"version": version, "group": str(group),
                             **group_stats(samples, bands)})
        return rows_out

    write_csv(MANUAL_V1_DIR / "manual_vs_automatic_per_sample.csv",
              per_sample_rows)
    write_csv(MANUAL_V1_DIR / "manual_vs_automatic_by_session.csv",
              stratified("session_id"))
    write_csv(MANUAL_V1_DIR / "manual_vs_automatic_by_status.csv",
              stratified("auto_status"))
    write_csv(MANUAL_V1_DIR / "manual_vs_automatic_by_depth_quartile.csv",
              stratified("depth_quartile"))

    summary = {
        "stage": "WP2_manual_vs_automatic_consistency",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wording_rule": (
            "single-annotator edges are Level 3 evidence; these are "
            "manual-automatic consistency observations, not absolute errors; "
            "no ground truth is claimed"),
        "decision": "QA_ONLY",
        "config": {"path": str(config_path.relative_to(REPO)),
                   "sha256": config_sha},
        "inputs": {
            "manual_frozen_snapshot": {"path": str(
                snapshot_path.relative_to(REPO)), "sha256": snapshot_sha},
            **{version: {"path": str(
                (REGISTRATION_DIR / f"translation_metrics_{version}.csv")
                .relative_to(REPO)),
                "sha256": sha256_of(
                    REGISTRATION_DIR / f"translation_metrics_{version}.csv")}
               for version in VERSIONS},
        },
        "bands_um": bands,
        "band_note": ("bands only label consistency levels; they are not "
                      "error thresholds and must not be used to pick between "
                      "automatic versions post hoc"),
        "primary_automatic_comparator": config["primary_automatic_comparator"],
        "secondary_comparators": config["secondary_comparators"],
        "forbid_samplewise_fallback": bool(config["forbid_samplewise_fallback"]),
        "forbid_absolute_error_language": bool(
            config["forbid_absolute_error_language"]),
        "depth_stratification": {
            "definition": ("per-measurement groove depth proxy "
                           "|Q01-Q50| from frozen inventory metrics; paired "
                           "slots share the measurement-level proxy"),
            "quartile_cuts_um": quartile_cuts,
        },
        "versions": version_stats,
        "v6_status_note": ("v6 statuses remain frozen (190 PASS / 2 REVIEW / "
                          "8 STOP); listing them here is QA only"),
    }
    summary_path = MANUAL_V1_DIR / "manual_vs_automatic_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    # ---- QA figures --------------------------------------------------------
    QA_DIR.mkdir(parents=True, exist_ok=True)
    _plot_scatter(per_sample_rows, VERSIONS, bands)
    _plot_ecdf(per_sample_rows, VERSIONS, bands)
    _plot_width_height(manual_rows)
    _plot_outliers(per_sample_rows, config["primary_automatic_comparator"],
                   bands)
    print(json.dumps({"stage": summary["stage"],
                      "decision": "QA_ONLY",
                      "per_sample_rows": len(per_sample_rows),
                      "versions": {version: {
                          "n": version_stats[version]["n"],
                          "radial_median_um": version_stats[version][
                              "radial_median_um"],
                          "frac_le_close": version_stats[version][
                              "frac_le_close"]}
                          for version in VERSIONS}}, ensure_ascii=False,
                     indent=2))

    append_stage_record(
        RUN_MANIFEST, stage="WP2_manual_vs_automatic_consistency",
        command=[sys.executable, str(Path(__file__).relative_to(REPO))],
        exit_code=0,
        inputs={"config": config_path, "frozen_snapshot": snapshot_path,
                **{version: REGISTRATION_DIR /
                   f"translation_metrics_{version}.csv"
                   for version in VERSIONS}},
        outputs={"per_sample_csv": MANUAL_V1_DIR /
                 "manual_vs_automatic_per_sample.csv",
                 "summary_json": summary_path,
                 "by_session_csv": MANUAL_V1_DIR /
                 "manual_vs_automatic_by_session.csv",
                 "by_status_csv": MANUAL_V1_DIR /
                 "manual_vs_automatic_by_status.csv",
                 "by_depth_quartile_csv": MANUAL_V1_DIR /
                 "manual_vs_automatic_by_depth_quartile.csv",
                 "scatter_png": QA_DIR / "manual_vs_automatic_scatter.png",
                 "ecdf_png": QA_DIR / "manual_vs_automatic_ecdf.png",
                 "width_height_png": QA_DIR /
                 "manual_width_height_distribution.png",
                 "outliers_png": QA_DIR / "manual_vs_automatic_outliers.png"},
        extra={"decision": "QA_ONLY"})
    return 0


STATUS_COLORS = {"PASS": "#1a9850", "REVIEW": "#fdae61", "STOP": "#d73027"}


def _plot_scatter(samples: list[dict], versions: tuple, bands: dict) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 10),
                                constrained_layout=True)
    for axis, version in zip(axes.flat, versions):
        version_samples = [s for s in samples if s["version"] == version]
        for status, color in STATUS_COLORS.items():
            selected = [s for s in version_samples
                        if s["auto_status"] == status]
            if selected:
                axis.scatter([s["delta_u_um"] for s in selected],
                             [s["delta_v_um"] for s in selected],
                             s=12, alpha=0.75, color=color, label=status,
                             edgecolors="none")
        for radius, style in ((bands["close"], "--"), (bands["moderate"], ":")):
            circle = plt.Circle((0, 0), radius, fill=False,
                                color="0.35", linestyle=style, linewidth=1.0)
            axis.add_patch(circle)
        limits = max(8.0, 1.1*max(s["center_disagreement_um"]
                                  for s in version_samples))
        axis.set_xlim(-limits, limits)
        axis.set_ylim(-limits, limits)
        axis.set_aspect("equal")
        axis.axhline(0, color="0.8", linewidth=0.5)
        axis.axvline(0, color="0.8", linewidth=0.5)
        radial = [s["center_disagreement_um"] for s in version_samples]
        axis.set_title(f"{version} | radial median "
                       f"{summarize_radial(radial)['median_um']:.2f} um | "
                       f"> {bands['moderate']} um: "
                       f"{100*band_fractions(radial, close_um=bands['close'], moderate_um=bands['moderate'])['frac_above_moderate']:.1f}%",
                       fontsize=10)
        axis.set_xlabel("delta_u (auto - manual), um", fontsize=9)
        axis.set_ylabel("delta_v (auto - manual), um", fontsize=9)
        axis.legend(fontsize=7, loc="upper right")
    axes.flat[5].axis("off")
    axes.flat[5].text(0.02, 0.95, (
        "Manual (annotator A) vs automatic centres\n"
        "consistency scatter, canonical session frame.\n"
        "Dashed circle: close band "
        f"(<= {bands['close']} um).\n"
        f"Dotted circle: moderate band (<= {bands['moderate']} um).\n"
        "Bands label consistency levels only --\n"
        "they are not error thresholds and are not\n"
        "used to select between automatic versions."),
        va="top", fontsize=10)
    figure.suptitle("Manual vs automatic registration centres "
                    "(consistency, not absolute error)", fontsize=13)
    figure.savefig(QA_DIR / "manual_vs_automatic_scatter.png", dpi=130)
    plt.close(figure)


def _plot_ecdf(samples: list[dict], versions: tuple, bands: dict) -> None:
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for version in versions:
        radial = sorted(s["center_disagreement_um"] for s in samples
                        if s["version"] == version)
        axis.step([0.0]+radial, [0.0]+[(i+1)/len(radial)
                                       for i in range(len(radial))],
                  where="post", label=version, linewidth=1.4)
    for radius, style in ((bands["close"], "--"), (bands["moderate"], ":")):
        axis.axvline(radius, color="0.4", linestyle=style, linewidth=1.0)
    axis.set_xlabel("centre disagreement (auto - manual), um", fontsize=10)
    axis.set_ylabel("empirical CDF", fontsize=10)
    axis.set_xlim(0, None)
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)
    axis.set_title("Manual vs automatic centre disagreement ECDF "
                   "(consistency, not absolute error)", fontsize=11)
    figure.savefig(QA_DIR / "manual_vs_automatic_ecdf.png", dpi=130)
    plt.close(figure)


def _plot_width_height(manual_rows: list[dict]) -> None:
    widths = np.array([float(row["annotator_a_width_um"])
                       for row in manual_rows])
    heights = np.array([float(row["annotator_a_height_um"])
                        for row in manual_rows])
    sessions = sorted({row["session_id"] for row in manual_rows})
    colors = {session: f"C{index}" for index, session in enumerate(sessions)}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                                constrained_layout=True)
    axes[0].hist(widths, bins=30, color="#4393c3")
    axes[0].axvline(180, color="red", linestyle="--", linewidth=1)
    axes[0].axvline(220, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("manual observed width, um", fontsize=9)
    axes[0].set_title("width distribution (gate bounds 180-220 um)",
                      fontsize=10)
    axes[1].hist(heights, bins=30, color="#4393c3")
    axes[1].axvline(180, color="red", linestyle="--", linewidth=1)
    axes[1].axvline(220, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("manual observed height, um", fontsize=9)
    axes[1].set_title("height distribution (gate bounds 180-220 um)",
                      fontsize=10)
    for session in sessions:
        selected_w = [float(row["annotator_a_width_um"]) for row in
                      manual_rows if row["session_id"] == session]
        selected_h = [float(row["annotator_a_height_um"]) for row in
                      manual_rows if row["session_id"] == session]
        axes[2].scatter(selected_w, selected_h, s=12, alpha=0.7,
                        color=colors[session], label=session.replace(
                            "zro2_", ""), edgecolors="none")
    axes[2].axvline(180, color="red", linestyle="--", linewidth=1)
    axes[2].axvline(220, color="red", linestyle="--", linewidth=1)
    axes[2].axhline(180, color="red", linestyle="--", linewidth=1)
    axes[2].axhline(220, color="red", linestyle="--", linewidth=1)
    axes[2].set_xlabel("manual observed width, um", fontsize=9)
    axes[2].set_ylabel("manual observed height, um", fontsize=9)
    axes[2].legend(fontsize=7)
    axes[2].set_title("width vs height by session", fontsize=10)
    figure.suptitle("Manual four-edge observed box sizes "
                    "(QA observation; nominal box stays 200 x 200 um)",
                     fontsize=12)
    figure.savefig(QA_DIR / "manual_width_height_distribution.png", dpi=130)
    plt.close(figure)


def _plot_outliers(samples: list[dict], primary: str,
                   bands: dict) -> None:
    primary_samples = sorted(
        (s for s in samples if s["version"] == primary),
        key=lambda s: s["center_disagreement_um"])
    figure, axis = plt.subplots(figsize=(9, 9), constrained_layout=True)
    values = [s["center_disagreement_um"] for s in primary_samples]
    colors = [STATUS_COLORS[s["auto_status"]] for s in primary_samples]
    axis.barh(range(len(values)), values, color=colors, height=0.8)
    axis.set_yticks(range(len(values)))
    axis.set_yticklabels(
        [f"{s['session_id'].replace('zro2_', '')}:{s['sample_id']}"
         for s in primary_samples], fontsize=4.5)
    axis.axvline(bands["close"], color="0.35", linestyle="--", linewidth=1.0)
    axis.axvline(bands["moderate"], color="0.35", linestyle=":",
                 linewidth=1.0)
    for index, sample in enumerate(primary_samples):
        if sample["center_disagreement_um"] > bands["moderate"]:
            axis.text(sample["center_disagreement_um"]+0.1, index,
                      f" {sample['auto_status']}", fontsize=5,
                      va="center", color=colors[index])
    axis.set_xlabel("centre disagreement vs manual, um", fontsize=10)
    axis.set_title(
        f"{primary} vs manual per-sample disagreement\n"
        "outlier review aid: only for spotting obvious manual entry errors -- "
        "large disagreement alone never edits a box",
        fontsize=10)
    figure.savefig(QA_DIR / "manual_vs_automatic_outliers.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
