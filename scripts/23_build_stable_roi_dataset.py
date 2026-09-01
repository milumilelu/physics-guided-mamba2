#!/usr/bin/env python3
"""Bundle the 200 stable ROIs into one directly consumable NPZ dataset."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.io_npz import load_height_npz  # noqa: E402

ROOT = REPO/"outputs/rectangle_registration/manual_internal_roi_v1"


def main() -> int:
    metrics_path = ROOT/"metrics/stable_roi_metrics.csv"
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda r: (r["session_id"], int(r["sample_id"])))
    if len(rows) != 200 or any(r["status"] != "PASS" for r in rows):
        raise RuntimeError("stable ROI metrics are not 200/200 PASS")

    raw_maps, repaired_maps = [], []
    valid_masks, repair_masks = [], []
    index_rows = []
    x_um = y_um = None
    for index, row in enumerate(rows):
        raw = load_height_npz(REPO/row["raw_path"])
        repaired = load_height_npz(REPO/row["repaired_path"])
        with np.load(REPO/row["mask_path"], allow_pickle=False) as masks:
            valid = masks["valid_mask"].astype(bool)
            repair = masks["repair_mask"].astype(bool)
        if raw.shape != (160, 160) or repaired.shape != raw.shape:
            raise RuntimeError(f"unexpected shape for {row['raw_path']}")
        if not np.array_equal(raw.valid_mask, valid):
            raise RuntimeError(f"valid mask mismatch for {row['raw_path']}")
        if not np.allclose(repaired.z[~repair], raw.z[~repair], equal_nan=True):
            raise RuntimeError(f"repaired values changed outside mask for {row['raw_path']}")
        raw_maps.append(raw.z.astype(np.float32))
        repaired_maps.append(repaired.z.astype(np.float32))
        valid_masks.append(valid)
        repair_masks.append(repair)
        if x_um is None:
            x_um, y_um = raw.x_um, raw.y_um
        index_rows.append({
            "dataset_index": index,
            "session_id": row["session_id"],
            "measurement_id": int(row["measurement_id"]),
            "sample_id": int(row["sample_id"]),
            "status": row["status"],
            "valid_fraction": row["valid_fraction"],
            "repair_fraction": row["repair_repaired_fraction"],
            "raw_path": row["raw_path"],
            "repaired_path": row["repaired_path"],
            "mask_path": row["mask_path"],
        })

    bundle = ROOT/"dataset/stable_roi_80um_dataset.npz"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    session_id = np.array([r["session_id"] for r in index_rows], dtype="U24")
    measurement_id = np.array([r["measurement_id"] for r in index_rows], dtype=np.int32)
    sample_id = np.array([r["sample_id"] for r in index_rows], dtype=np.int32)
    metadata = {
        "method": "manual_internal_roi_v1_fast_80um",
        "evidence_level": 3,
        "shape": [200, 160, 160],
        "pixel_um": 0.5,
        "roi_um": [80.0, 80.0],
        "primary_array": "height_raw",
        "optional_array": "height_repaired",
        "warning": "repair is Level-3 model-derived; retain repair_mask and use raw as authority",
        "metrics": str(metrics_path.relative_to(REPO)),
    }
    np.savez_compressed(
        bundle,
        height_raw=np.stack(raw_maps),
        height_repaired=np.stack(repaired_maps),
        valid_mask=np.stack(valid_masks),
        repair_mask=np.stack(repair_masks),
        session_id=session_id,
        measurement_id=measurement_id,
        sample_id=sample_id,
        x_um=x_um.astype(np.float32),
        y_um=y_um.astype(np.float32),
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    index_path = ROOT/"dataset/stable_roi_80um_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
        writer.writeheader(); writer.writerows(index_rows)
    summary = {
        "decision": "PASS", "samples": 200,
        "shape": [200, 160, 160], "dtype": "float32",
        "bundle": str(bundle.relative_to(REPO)),
        "index": str(index_path.relative_to(REPO)),
        "primary": "height_raw", "optional": "height_repaired",
    }
    (ROOT/"dataset/dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
