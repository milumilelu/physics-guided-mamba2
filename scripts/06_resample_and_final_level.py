#!/usr/bin/env python3
"""WP9: export mask-aware H_reg/H_200 and perform final external leveling.

Supports versioned runs: ``--registration-metrics`` selects an alternative
frozen registration table (e.g. the manual_v1 table), ``--output-tag``
redirects every writable artefact under ``<outputs_root>/<tag>/`` (reading
the matching tagged common-FOV summary) so legacy archives are never
overwritten, and ``--npz-metadata`` injects provenance metadata (method
name, source hashes, generation time) into every exported NPZ.  Without
these arguments the behaviour is byte-compatible with the original v2
pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data_contracts import HeightMap  # noqa: E402
from src.io_cag import CagHeightReader  # noqa: E402
from src.io_npz import save_height_npz  # noqa: E402
from src.leveling import fit_outer_reference_plane  # noqa: E402
from src.manual_registration_evaluation import resolve_pipeline_paths  # noqa: E402
from src.resampling import resample_center_crop, resample_to_canonical  # noqa: E402
from src.stage_manifest import append_stage_record, sha256_of  # noqa: E402


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration-metrics", default=None,
        help="alternative frozen registration table (repo-relative path); "
             "default comes from config paths.registration_metrics")
    parser.add_argument(
        "--output-tag", default=None,
        help="read/write all pipeline artefacts under <outputs_root>/<tag>/ "
             "instead of the legacy locations (e.g. manual_v1)")
    parser.add_argument("--stage-manifest", default=None,
                        help="append a stage record to this run manifest")
    parser.add_argument(
        "--npz-metadata", default=None,
        help="JSON object merged into every exported NPZ metadata dict "
             "(e.g. registration_method, manual_annotation_sha256, "
             "config_sha256)")
    args = parser.parse_args(argv)

    config = yaml.safe_load((REPO / "config/rectangle_registration.yaml")
                            .read_text(encoding="utf-8"))
    root = REPO / config["paths"]["outputs_root"]
    paths = resolve_pipeline_paths(root, args.output_tag)
    registration_path = (REPO / args.registration_metrics
                         if args.registration_metrics
                         else REPO / config["paths"]["registration_metrics"])
    extra_metadata = (json.loads(args.npz_metadata)
                      if args.npz_metadata else None)
    canvas_summary = json.loads(
        paths.common_fov_summary_json.read_text(encoding="utf-8"))
    if canvas_summary.get("decision") != "PASS":
        print("STOP: common FOV gate is not PASS", file=sys.stderr)
        return 2
    canvas_by_session = {row["session_id"]: row for row in canvas_summary["sessions"]
                         if row["status"] == "PASS"}
    fov_diagnostics = read_csv(paths.sample_fov_diagnostics_csv)
    included_keys = {
        (row["session_id"], int(row["sample_id"]))
        for row in fov_diagnostics if row["included_in_phase_a"].lower() == "true"}
    registrations = read_csv(registration_path)
    planes = read_csv(root / "metrics/coarse_leveling_metrics.csv")
    plane_by_key = {(row["session_id"], int(row["measurement_id"])): row
                    for row in planes}
    sessions = read_csv(REPO / config["paths"]["session_manifest"])
    resampling_cfg = config["resampling"]
    final_cfg = config["final_leveling"]
    h200_pixels = math.floor(200.0/max(float(row["pixel_um"])
                                      for row in canvas_by_session.values()))
    metrics: list[dict] = []
    errors: list[dict] = []
    for session in sessions:
        sid = session["session_id"]
        if sid not in canvas_by_session:
            continue
        canvas = canvas_by_session[sid]
        session_rows = sorted(
            (row for row in registrations
             if row["session_id"] == sid
             and (sid, int(row["sample_id"])) in included_keys),
            key=lambda row: (int(row["measurement_id"]), int(row["sample_id"])))
        with CagHeightReader(REPO / session["cag_path"]) as reader:
            cached_measurement = None
            cached_map = None
            for row in session_rows:
                measurement_id = int(row["measurement_id"])
                sample_id = int(row["sample_id"])
                try:
                    if cached_measurement != measurement_id:
                        cached_map = reader.read_height_map(measurement_id)
                        cached_measurement = measurement_id
                    plane = plane_by_key[(sid, measurement_id)]
                    metadata = {
                        "session_id": sid, "measurement_id": measurement_id,
                        "sample_id": sample_id, "d4_transform": "identity",
                        "source": str(session["cag_path"]),
                    }
                    if extra_metadata:
                        metadata.update(extra_metadata)
                        metadata["generated_at_utc"] = (
                            datetime.now(timezone.utc).isoformat())
                    coarse_reg = resample_to_canonical(
                        cached_map,
                        plane=tuple(float(plane[key]) for key in ("a", "b", "c")),
                        center_x_um=float(row["center_x_um"]),
                        center_y_um=float(row["center_y_um"]),
                        theta_deg=float(row["theta_session_deg"]),
                        length_um=float(canvas["registered_fov_um"]),
                        pixels=int(canvas["grid_pixels"]),
                        minimum_mask_weight=float(
                            resampling_cfg["minimum_mask_weight"]),
                        order=1, metadata=metadata)
                    frame_width = coarse_reg.width_um/2.0-float(
                        final_cfg["exclusion_halfwidth_um"])
                    if frame_width < float(final_cfg["minimum_reference_frame_width_um"]):
                        raise ValueError(
                            f"final reference frame {frame_width:.3f} um below minimum")
                    final_fit = fit_outer_reference_plane(
                        coarse_reg, frame_width_um=frame_width,
                        sigma_low=float(config["coarse_leveling"]["sigma_low"]),
                        sigma_high=float(config["coarse_leveling"]["sigma_high"]),
                        max_iterations=int(config["coarse_leveling"]["max_iterations"]),
                        minimum_reference_valid_fraction=float(
                            final_cfg["minimum_reference_valid_fraction"]),
                        max_fit_points=int(config["coarse_leveling"]["max_fit_points"]),
                    )
                    if final_fit.status != "PASS":
                        raise ValueError(f"final leveling: {final_fit.warning}")
                    x = coarse_reg.x_um-(coarse_reg.x_um[0]+coarse_reg.x_um[-1])/2
                    y = coarse_reg.y_um-(coarse_reg.y_um[0]+coarse_reg.y_um[-1])/2
                    final_z = coarse_reg.z-(final_fit.a*x[None, :]
                                           + final_fit.b*y[:, None]+final_fit.c)
                    h_reg = HeightMap(
                        z=np.where(coarse_reg.valid_mask, final_z, np.nan),
                        valid_mask=coarse_reg.valid_mask, dx_um=coarse_reg.dx_um,
                        dy_um=coarse_reg.dy_um, x_um=coarse_reg.x_um,
                        y_um=coarse_reg.y_um,
                        metadata={**coarse_reg.metadata, "object": "H_reg",
                                  "final_plane": final_fit.to_dict()})
                    h_200 = resample_center_crop(
                        h_reg, length_um=200.0, pixels=h200_pixels,
                        minimum_mask_weight=float(
                            resampling_cfg["minimum_mask_weight"]))
                    stem = f"{sid}__sample_{sample_id:03d}"
                    hreg_path = (paths.registered_h_reg_dir
                                  / f"{stem}.npz")
                    h200_path = (paths.registered_h_200_dir
                                 / f"{stem}.npz")
                    mask_path = (paths.registered_masks_dir
                                 / f"{stem}.npz")
                    save_height_npz(hreg_path, h_reg)
                    save_height_npz(h200_path, h_200)
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    mask_metadata = {
                        **metadata,
                        "object": "valid_masks",
                        "h_reg_shape": list(h_reg.valid_mask.shape),
                        "h_200_shape": list(h_200.valid_mask.shape),
                        "h_reg_valid_fraction": h_reg.valid_fraction,
                        "h_200_valid_fraction": h_200.valid_fraction,
                    }
                    if extra_metadata:
                        mask_metadata["generated_at_utc"] = (
                            datetime.now(timezone.utc).isoformat())
                    np.savez_compressed(
                        mask_path, h_reg=h_reg.valid_mask,
                        h_200=h_200.valid_mask,
                        metadata_json=json.dumps(mask_metadata,
                                                 ensure_ascii=False))
                    metrics.append({
                        **row,
                        "coarse_plane_rmse_um": float(plane["rmse_um"]),
                        "final_plane_a": final_fit.a,
                        "final_plane_b": final_fit.b,
                        "final_plane_c": final_fit.c,
                        "final_plane_rmse_um": final_fit.rmse_um,
                        "final_reference_fraction": final_fit.reference_valid_fraction,
                        "final_reference_retained_fraction": final_fit.retained_fraction,
                        "final_reference_quadrants": final_fit.quadrant_count,
                        "registered_fov_um": h_reg.width_um,
                        "registered_pixel_um": h_reg.dx_um,
                        "registered_valid_fraction": h_reg.valid_fraction,
                        "h200_pixel_um": h_200.dx_um,
                        "h200_valid_fraction": h_200.valid_fraction,
                        "h_reg_path": str(hreg_path.relative_to(REPO)),
                        "h_200_path": str(h200_path.relative_to(REPO)),
                        "mask_path": str(mask_path.relative_to(REPO)),
                        "final_leveling_failed": False,
                    })
                except Exception as exc:
                    errors.append({"session_id": sid, "measurement_id": measurement_id,
                                   "sample_id": sample_id,
                                   "error": f"{type(exc).__name__}: {exc}"})
    write_csv(paths.registration_metrics_csv, metrics)
    decision = "PASS" if not errors and len(metrics) == len(included_keys) else "STOP"
    summary = {
        "stage": "WP9_resampling_and_final_leveling" + (
            f"_{args.output_tag}" if args.output_tag else ""),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision, "expected_samples": len(included_keys),
        "exported_h_reg": len(metrics), "exported_h_200": len(metrics),
        "output_tag": args.output_tag,
        "registration_metrics": str(registration_path.relative_to(REPO)),
        "exclusion_policy": canvas_summary.get("exclusion_policy"),
        "exclusion_policy_sha256": canvas_summary.get(
            "exclusion_policy_sha256"),
        "h200_pixels": h200_pixels, "errors": errors,
        "settings": {
            "resampling": resampling_cfg,
            "final_leveling": final_cfg,
            "coarse_leveling": {
                key: config["coarse_leveling"][key]
                for key in ("sigma_low", "sigma_high", "max_iterations",
                            "max_fit_points")},
            "h200_pixels": h200_pixels,
            "npz_metadata": extra_metadata,
        },
        "provenance_inputs": {
            "rectangle_registration_yaml": {
                "path": "config/rectangle_registration.yaml",
                "sha256": sha256_of(
                    REPO / "config/rectangle_registration.yaml")},
            "session_manifest": {
                "path": config["paths"]["session_manifest"],
                "sha256": sha256_of(
                    REPO / config["paths"]["session_manifest"])},
        },
    }
    paths.resampling_summary_json.parent.mkdir(parents=True, exist_ok=True)
    paths.resampling_summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.stage_manifest:
        command = [sys.executable,
                   str(Path(__file__).relative_to(REPO)),
                   "--registration-metrics",
                   str(registration_path.relative_to(REPO)),
                   "--output-tag", str(args.output_tag)]
        if args.npz_metadata:
            command += ["--npz-metadata", args.npz_metadata]
        append_stage_record(
            Path(args.stage_manifest), stage=summary["stage"],
            command=command,
            exit_code=0 if decision == "PASS" else 2,
            inputs={"registration_metrics": registration_path,
                    "common_fov_summary": paths.common_fov_summary_json,
                    "coarse_leveling_metrics": root /
                    "metrics/coarse_leveling_metrics.csv",
                    "rectangle_registration_yaml": REPO /
                    "config/rectangle_registration.yaml",
                    "session_manifest": REPO /
                    config["paths"]["session_manifest"]},
            outputs={"registered_h_reg": paths.registered_h_reg_dir,
                     "registered_h_200": paths.registered_h_200_dir,
                     "registered_masks": paths.registered_masks_dir,
                     "registration_metrics": paths.registration_metrics_csv,
                     "resampling_summary": paths.resampling_summary_json},
            extra={"decision": decision,
                   "expected_samples": len(included_keys),
                   "exported_h_reg": len(metrics)})
    return 0 if decision == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
