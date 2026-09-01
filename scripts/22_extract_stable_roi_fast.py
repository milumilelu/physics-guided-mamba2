#!/usr/bin/env python3
"""Fast full extraction of 200 fixed-size manual-centred stable ROIs.

The script reuses the already-passed measurement outer-plane fits, never
depends on the blocked 260 um H_reg route, and writes raw plus conservatively
repaired height maps with explicit masks and provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.conical_dropout import (  # noqa: E402
    ConicalDropoutConfig, repair_compact_dropouts)
from src.data_contracts import HeightMap  # noqa: E402
from src.io_cag import CagHeightReader  # noqa: E402
from src.io_npz import save_height_npz  # noqa: E402
from src.resampling import resample_to_canonical  # noqa: E402
from src.stage_manifest import sha256_of  # noqa: E402


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


def finite_metrics(z: np.ndarray, valid: np.ndarray, pixel_um: float,
                   core_halfwidth_um: float,
                   edge_ring_width_um: float) -> dict:
    values = z[valid]
    rows, cols = z.shape
    yy = (np.arange(rows)+0.5)*pixel_um-rows*pixel_um/2
    xx = (np.arange(cols)+0.5)*pixel_um-cols*pixel_um/2
    core = ((np.abs(xx)[None, :] <= core_halfwidth_um)
            & (np.abs(yy)[:, None] <= core_halfwidth_um) & valid)
    ring = ((np.abs(xx)[None, :] >= cols*pixel_um/2-edge_ring_width_um)
            | (np.abs(yy)[:, None] >= rows*pixel_um/2-edge_ring_width_um)) & valid
    filled = np.where(valid, z, np.nanmedian(values))
    gy, gx = np.gradient(filled, pixel_um, pixel_um)
    grad = np.hypot(gx, gy)
    centre = float(np.median(values))
    return {
        "z_median_um": centre,
        "z_p05_um": float(np.quantile(values, .05)),
        "z_p95_um": float(np.quantile(values, .95)),
        "Sa_um": float(np.mean(np.abs(values-centre))),
        "Sq_um": float(np.sqrt(np.mean((values-centre)**2))),
        "core_gradient_median": float(np.median(grad[core])),
        "edge_ring_gradient_median": float(np.median(grad[ring])),
        "edge_to_core_gradient_ratio": float(
            np.median(grad[ring])/max(np.median(grad[core]), 1e-12)),
    }


def profile_rows(z: np.ndarray, valid: np.ndarray, x_um: np.ndarray,
                 y_um: np.ndarray, base: dict) -> list[dict]:
    # Robust profiles use the central 60% tangential band.  They are QA only;
    # the extraction size is fixed before inspecting them.
    x_band = np.abs(x_um) <= 0.30*(x_um[-1]-x_um[0]+(x_um[1]-x_um[0]))
    y_band = np.abs(y_um) <= 0.30*(y_um[-1]-y_um[0]+(y_um[1]-y_um[0]))
    out: list[dict] = []
    for index, position in enumerate(x_um):
        mask = valid[y_band, index]
        vals = z[y_band, index][mask]
        out.append({**base, "profile_axis": "u", "position_um": position,
                    "height_median_um": (float(np.median(vals))
                                           if vals.size else ""),
                    "valid_count": int(vals.size)})
    for index, position in enumerate(y_um):
        mask = valid[index, x_band]
        vals = z[index, x_band][mask]
        out.append({**base, "profile_axis": "v", "position_um": position,
                    "height_median_um": (float(np.median(vals))
                                           if vals.size else ""),
                    "valid_count": int(vals.size)})
    return out


def save_montage(items: list[dict], path: Path) -> None:
    if not items:
        return
    columns = 3
    fig, axes = plt.subplots(len(items), columns,
                             figsize=(10, 2.7*len(items)), squeeze=False)
    for row, item in enumerate(items):
        raw, repaired, correction = item["raw"], item["repaired"], item["correction"]
        finite = raw[np.isfinite(raw)]
        lo, hi = np.quantile(finite, [.02, .98])
        axes[row, 0].imshow(raw, cmap="viridis", vmin=lo, vmax=hi)
        axes[row, 1].imshow(repaired, cmap="viridis", vmin=lo, vmax=hi)
        vmax = max(float(np.nanmax(correction)), .01)
        axes[row, 2].imshow(correction, cmap="magma", vmin=0, vmax=vmax)
        axes[row, 0].set_ylabel(item["label"], fontsize=8)
        for col in range(columns):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    axes[0, 0].set_title("raw levelled")
    axes[0, 1].set_title("conservative repaired")
    axes[0, 2].set_title("repair delta")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("config/manual_internal_roi_v1.yaml"))
    args = parser.parse_args(argv)
    config_path = (REPO/args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = REPO/cfg["output_root"]
    roi = cfg["stable_roi"]
    width_um, height_um = float(roi["width_um"]), float(roi["height_um"])
    if width_um != height_um:
        raise ValueError("fast extractor currently requires a square ROI")
    pixel_um = float(roi["pixel_um"])
    pixels = int(round(width_um/pixel_um))
    if not math.isclose(pixels*pixel_um, width_um, abs_tol=1e-9):
        raise ValueError("ROI size must be an integer number of pixels")

    registrations = read_csv(REPO/cfg["registration_csv"])
    planes = {(r["session_id"], int(r["measurement_id"])): r
              for r in read_csv(REPO/cfg["plane_csv"])}
    sessions = read_csv(REPO/cfg["session_manifest"])
    sources = {(r["session_id"], int(r["measurement_id"])): r
               for r in read_csv(REPO/cfg["height_source_manifest"])}
    if len(registrations) != 200 or len(planes) != 160:
        raise RuntimeError(f"input cardinality mismatch: registrations={len(registrations)}, planes={len(planes)}")
    if any(r["status"] != "PASS" for r in registrations):
        raise RuntimeError("not all frozen manual registrations are PASS")
    if any(r["status"] != "PASS" for r in planes.values()):
        raise RuntimeError("not all measurement outer-plane fits are PASS")
    min_half_width = min(float(r["manual_width_um"])/2 for r in registrations)
    min_half_height = min(float(r["manual_height_um"])/2 for r in registrations)
    if width_um/2 > min_half_width or height_um/2 > min_half_height:
        raise RuntimeError("fixed stable ROI is not inside every manual box")

    repair_cfg = ConicalDropoutConfig(**cfg["conical_dropout"])
    config_sha = sha256_of(config_path)
    annotation_path = REPO/cfg["annotation_csv"]
    annotation_sha = sha256_of(annotation_path)
    metrics: list[dict] = []
    components: list[dict] = []
    profiles: list[dict] = []
    errors: list[dict] = []
    montage: list[dict] = []
    montage_n = int(cfg["qa"]["montage_samples_per_session"])

    for session in sessions:
        sid = session["session_id"]
        rows = sorted((r for r in registrations if r["session_id"] == sid),
                      key=lambda r: (int(r["measurement_id"]), int(r["sample_id"])))
        selected_ids = {int(r["sample_id"]) for r in rows[:montage_n]}
        with CagHeightReader(REPO/session["cag_path"]) as reader:
            cached_mid = None
            hm = None
            for row in rows:
                mid, sample_id = int(row["measurement_id"]), int(row["sample_id"])
                stem = f"{sid}__sample_{sample_id:03d}"
                try:
                    if cached_mid != mid:
                        hm = reader.read_height_map(mid)
                        cached_mid = mid
                    assert hm is not None
                    plane_row = planes[(sid, mid)]
                    plane = tuple(float(plane_row[k]) for k in ("a", "b", "c"))
                    source = sources[(sid, mid)]
                    metadata = {
                        "object": "H_stable_raw",
                        "method": cfg["method"], "evidence_level": 3,
                        "session_id": sid, "measurement_id": mid,
                        "sample_id": sample_id,
                        "center_x_um": float(row["center_x_um"]),
                        "center_y_um": float(row["center_y_um"]),
                        "theta_session_deg": float(row["theta_session_deg"]),
                        "roi_width_um": width_um, "roi_height_um": height_um,
                        "manual_width_um": float(row["manual_width_um"]),
                        "manual_height_um": float(row["manual_height_um"]),
                        "manual_boundary_clearance_x_um": float(row["manual_width_um"])/2-width_um/2,
                        "manual_boundary_clearance_y_um": float(row["manual_height_um"])/2-height_um/2,
                        "config_sha256": config_sha,
                        "manual_annotation_sha256": annotation_sha,
                        "source_csv_sha256": source["csv_sha256"],
                        "plane": {k: plane_row[k] for k in ("a", "b", "c", "rmse_um", "status")},
                    }
                    raw = resample_to_canonical(
                        hm, plane=plane,
                        center_x_um=float(row["center_x_um"]),
                        center_y_um=float(row["center_y_um"]),
                        theta_deg=float(row["theta_session_deg"]),
                        length_um=width_um, pixels=pixels,
                        minimum_mask_weight=.99, order=1,
                        metadata=metadata)
                    repaired_z, repair_mask, records, repair_metrics = repair_compact_dropouts(
                        raw.z, raw.valid_mask, dx_um=raw.dx_um, dy_um=raw.dy_um,
                        config=repair_cfg)
                    repaired = HeightMap(
                        z=repaired_z, valid_mask=raw.valid_mask.copy(),
                        dx_um=raw.dx_um, dy_um=raw.dy_um,
                        x_um=raw.x_um.copy(), y_um=raw.y_um.copy(),
                        metadata={**metadata, "object": "H_stable_repaired",
                                  "repair_metrics": repair_metrics,
                                  "repair_config": asdict(repair_cfg)})
                    raw_path = output/"registered/H_stable_raw"/f"{stem}.npz"
                    repaired_path = output/"registered/H_stable_repaired"/f"{stem}.npz"
                    mask_path = output/"registered/repair_masks"/f"{stem}.npz"
                    save_height_npz(raw_path, raw)
                    save_height_npz(repaired_path, repaired)
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        mask_path, valid_mask=raw.valid_mask,
                        repair_mask=repair_mask,
                        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
                    raw_stats = finite_metrics(
                        raw.z, raw.valid_mask, pixel_um,
                        float(cfg["qa"]["core_halfwidth_um"]),
                        float(cfg["qa"]["edge_ring_width_um"]))
                    metric = {
                        "session_id": sid, "measurement_id": mid,
                        "sample_id": sample_id, "status": (
                            "PASS" if raw.valid_fraction >= float(roi["minimum_valid_fraction"])
                            else "REVIEW"),
                        "roi_width_um": width_um, "roi_height_um": height_um,
                        "pixels_x": pixels, "pixels_y": pixels,
                        "valid_fraction": raw.valid_fraction,
                        "manual_boundary_clearance_x_um": metadata["manual_boundary_clearance_x_um"],
                        "manual_boundary_clearance_y_um": metadata["manual_boundary_clearance_y_um"],
                        **raw_stats, **{f"repair_{k}": v for k, v in repair_metrics.items()
                                      if k != "config"},
                        "raw_path": str(raw_path.relative_to(REPO)),
                        "repaired_path": str(repaired_path.relative_to(REPO)),
                        "mask_path": str(mask_path.relative_to(REPO)),
                    }
                    metrics.append(metric)
                    base = {"session_id": sid, "measurement_id": mid,
                            "sample_id": sample_id}
                    profiles.extend(profile_rows(raw.z, raw.valid_mask,
                                                 raw.x_um, raw.y_um, base))
                    for record in records:
                        components.append({**base, **record})
                    if sample_id in selected_ids:
                        montage.append({"label": f"{sid} #{sample_id}",
                                        "raw": raw.z,
                                        "repaired": repaired.z,
                                        "correction": repaired.z-raw.z})
                    print(f"{sid} sample {sample_id}: valid={raw.valid_fraction:.4f} "
                          f"cones={repair_metrics['accepted_components']} "
                          f"pixels={repair_metrics['repaired_pixels']}")
                except Exception as exc:
                    errors.append({"session_id": sid, "measurement_id": mid,
                                   "sample_id": sample_id,
                                   "error": f"{type(exc).__name__}: {exc}"})
                    print(f"ERROR {sid} sample {sample_id}: {exc}", file=sys.stderr)

    write_csv(output/"metrics/stable_roi_metrics.csv", metrics)
    write_csv(output/"metrics/conical_components.csv", components)
    write_csv(output/"metrics/stable_profiles.csv", profiles)
    save_montage(montage, output/"qa/stable_roi_raw_repaired_montage.png")
    status_counts = {s: sum(r["status"] == s for r in metrics)
                     for s in sorted({r["status"] for r in metrics})}
    summary = {
        "stage": "fast_manual_internal_roi_extraction",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if len(metrics) == 200 and not errors else "STOP",
        "method": cfg["method"], "evidence_level": 3,
        "expected_samples": 200, "exported_samples": len(metrics),
        "status_counts": status_counts, "errors": errors,
        "stable_roi": roi,
        "minimum_manual_boundary_clearance_x_um": min_half_width-width_um/2,
        "minimum_manual_boundary_clearance_y_um": min_half_height-height_um/2,
        "accepted_conical_components": len(components),
        "total_repaired_pixels": sum(int(r["repair_repaired_pixels"]) for r in metrics),
        "config": {"path": str(config_path.relative_to(REPO)), "sha256": config_sha},
        "annotation": {"path": str(annotation_path.relative_to(REPO)), "sha256": annotation_sha},
        "notes": [
            "Stable ROI is fixed before profile inspection and shared by all samples.",
            "Raw height is authoritative and is never overwritten.",
            "Repaired height is a conservative Level-3 derived product with an explicit mask.",
            "This fast route does not depend on the blocked 260 um H_reg export."],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output/"run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    approval = (
        "# Stable ROI extraction\n\n"
        f"Status: {'PENDING' if summary['decision']=='PASS' else 'BLOCKED'}\n\n"
        f"Exported: {len(metrics)}/200\n\n"
        "Automatic extraction completed. Review the montage and metrics before downstream modelling.\n")
    (output/"STABLE_ROI_APPROVAL.md").write_text(approval, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
