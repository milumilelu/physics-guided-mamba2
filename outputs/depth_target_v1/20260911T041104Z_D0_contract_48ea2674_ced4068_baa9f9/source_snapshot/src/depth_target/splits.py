"""Frozen-partition loading and leakage validation for depth_target_v1.

Reuses the E00 audit split manifests (already outcome-blind and validated).
This module re-validates every outer/inner partition before any fit and
exposes the MAIN180 frame joined with component/fold identity.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.task_state_learning.grouping import require, validate_partition

MAIN_ROLES = ("formal", "pass_main")


def load_frozen_partition(audit_dir):
    """Return (frame, inner_table) for MAIN180 with frozen folds, fully re-validated."""
    audit = Path(audit_dir)
    targets = pd.read_csv(audit / "frozen_targets.csv")
    split = pd.read_csv(audit / "split_manifest.csv")
    inner = pd.read_csv(audit / "inner_split_manifest.csv")
    frame = targets.merge(split, on=["dataset_index", "session_id", "sample_id",
                                     "shared_height_source_id", "session_role"],
                          validate="one_to_one")
    require(set(frame.session_role) <= set(MAIN_ROLES), "MAIN180 role contamination")
    require(len(frame) == 180, "MAIN180 coverage mismatch")
    for fold in sorted(frame.outer_fold.unique()):
        train = frame[frame.outer_fold != fold]
        test = frame[frame.outer_fold == fold]
        validate_partition(frame, train.dataset_index, test.dataset_index)
        inner_rows = inner[inner.outer_fold == fold]
        for k in sorted(inner_rows.inner_fold.unique()):
            val = inner_rows[inner_rows.inner_fold == k]
            fit = inner_rows[inner_rows.inner_fold != k]
            require(set(val.dataset_index) | set(fit.dataset_index) == set(train.dataset_index),
                    "Inner partition coverage mismatch")
            validate_partition(train, fit.dataset_index, val.dataset_index)
    return frame, inner
