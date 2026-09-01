#!/usr/bin/env python3
"""Freeze a blinded validation list: calibration union v6 non-PASS."""

from pathlib import Path
import argparse
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.manual_four_edge_annotation import ANNOTATION_FIELDS

ROOT = REPO / "outputs/rectangle_registration"

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--force-empty",action="store_true")
    parser.add_argument("--all-samples-preserve", action="store_true",
                        help="Expand to all v6 samples while preserving existing annotations by key.")
    args=parser.parse_args()
    calibration = pd.read_csv(ROOT / "calibration/calibration_sample_ids.csv")
    v6 = pd.read_csv(ROOT / "registration/translation_metrics_v6.csv")
    if args.all_samples_preserve:
        keys = v6[["session_id", "sample_id"]].copy()
    else:
        keys = pd.concat([
            calibration[["session_id", "sample_id"]],
            v6.loc[~v6.status.eq("PASS"), ["session_id", "sample_id"]],
        ]).drop_duplicates()
    keys = keys.sort_values(["session_id", "sample_id"])
    lookup = v6[["session_id", "sample_id", "measurement_id", "roi_within_measurement"]]
    output = keys.merge(lookup, on=["session_id", "sample_id"], validate="one_to_one")
    for annotator in ("a", "b"):
        for field in ANNOTATION_FIELDS:
            output[f"annotator_{annotator}_{field}"] = ""
    path = ROOT / "registration/manual_four_edge_validation.csv"
    if args.all_samples_preserve:
        if not path.exists():
            raise SystemExit("Cannot preserve annotations: the existing table is missing.")
        existing = pd.read_csv(path, keep_default_na=False)
        annotation_columns = [c for c in existing.columns if c.startswith("annotator_")]
        preserved = existing[["session_id", "sample_id", *annotation_columns]]
        output = output.drop(columns=[c for c in annotation_columns if c in output], errors="ignore")
        output = output.merge(
            preserved, on=["session_id", "sample_id"], how="left",
            validate="one_to_one"
        )
        for column in annotation_columns:
            output[column] = output[column].fillna("")
        output.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{path}: expanded to {len(output)} rows; existing annotations preserved")
        return
    if path.exists() and not args.force_empty:
        existing=pd.read_csv(path,keep_default_na=False)
        state_columns=[c for c in existing if c.endswith("_state")]
        if state_columns and existing[state_columns].isin(["complete","unusable"]).any().any():
            raise SystemExit("Refusing to overwrite existing annotations; use --force-empty only intentionally.")
    output.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{path}: {len(output)} frozen rows")


if __name__ == "__main__":
    main()
