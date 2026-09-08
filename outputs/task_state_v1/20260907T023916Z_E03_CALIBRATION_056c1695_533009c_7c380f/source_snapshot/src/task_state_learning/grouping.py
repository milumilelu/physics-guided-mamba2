"""Identity-stable connected components and outcome-blind nested splits."""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import numpy as np
import pandas as pd

KEYS = ["session_id", "sample_id"]
FAMILY = ["pulse_duration_fs", "frequency_kHz", "velocity_mm_s", "hatch_spacing_um"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def decimal_key(value):
    number = Decimal(str(value))
    require(number.is_finite(), "Nonfinite family value")
    return format(number.normalize(), "f")


def components(frame):
    required = KEYS + FAMILY + ["dataset_index", "shared_height_source_id", "session_role"]
    require(set(required) <= set(frame), "Missing grouping columns")
    require(not frame[required].isna().any().any(), "Null grouping identity")
    require(not frame.duplicated(KEYS).any(), "Duplicate specimen identity")
    require(not frame.dataset_index.duplicated().any(), "Duplicate dataset_index")
    require(np.equal(frame.sample_id, np.floor(frame.sample_id)).all(), "Noninteger sample_id")
    out = frame.sort_values(KEYS).reset_index(drop=True).copy()
    parent = list(range(len(out)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen_source, seen_family = {}, {}
    family_keys = []
    for i, row in enumerate(out.itertuples(index=False)):
        family = tuple(decimal_key(getattr(row, c)) for c in FAMILY)
        family_keys.append("|".join(family))
        for key, seen in ((str(row.shared_height_source_id), seen_source), (family, seen_family)):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    members = {}
    for i in range(len(out)):
        members.setdefault(find(i), []).append(i)
    identifiers = {}
    for root, indices in members.items():
        keys = sorted((str(out.iloc[i].session_id), int(out.iloc[i].sample_id)) for i in indices)
        payload = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        identifiers[root] = hashlib.sha256(payload).hexdigest()
    out["component_id"] = [identifiers[find(i)] for i in range(len(out))]
    out["base_family_key"] = family_keys
    return out.sort_values("dataset_index").reset_index(drop=True)


def assign_folds(frame, n_splits):
    require(n_splits >= 2, "At least two folds required")
    groups = list(frame.groupby("component_id", sort=True))
    require(len(groups) >= n_splits, "Insufficient components")
    counts = [[0, 0, 0] for _ in range(n_splits)]
    allocation = {}
    for key, group in sorted(groups, key=lambda item: (-len(item[1]), item[0])):
        fold = min(range(n_splits), key=lambda j: (*counts[j], j))
        allocation[key] = fold
        counts[fold][0] += len(group)
        counts[fold][1] += int(group.session_role.eq("formal").sum())
        counts[fold][2] += int(group.session_role.eq("pass_main").sum())
    return frame.component_id.map(allocation).astype(int)


def validate_partition(frame, train_ids, test_ids):
    train_ids, test_ids = set(train_ids), set(test_ids)
    require(train_ids and test_ids, "Empty train/test partition")
    require(not train_ids & test_ids, "Specimen leakage")
    require(train_ids | test_ids == set(frame.dataset_index), "Partition coverage mismatch")
    a = frame[frame.dataset_index.isin(train_ids)]
    b = frame[frame.dataset_index.isin(test_ids)]
    for column in ["component_id", "shared_height_source_id", "base_family_key"]:
        require(not set(a[column]) & set(b[column]), f"Leakage in {column}")


def nested_splits(frame, outer_splits=5, inner_splits=3):
    out = frame.copy()
    out["outer_fold"] = assign_folds(out, outer_splits)
    inner_rows = []
    for fold in range(outer_splits):
        train = out[out.outer_fold != fold].copy()
        test = out[out.outer_fold == fold]
        validate_partition(out, train.dataset_index, test.dataset_index)
        train["inner_fold"] = assign_folds(train, inner_splits)
        for inner in range(inner_splits):
            validate_partition(train, train[train.inner_fold != inner].dataset_index,
                               train[train.inner_fold == inner].dataset_index)
        train["outer_fold"] = fold  # identifies the parent training set
        inner_rows.append(train[["dataset_index", "component_id", "outer_fold", "inner_fold"]])
    return out, pd.concat(inner_rows, ignore_index=True)


def pair_coverage(frame, fold_column):
    rows = []
    for (fold, session), group in frame.groupby([fold_column, "session_id"]):
        sizes = group.groupby("component_id").size()
        rows.append({"fold": int(fold), "session_id": session, "n_rows": len(group),
                     "n_components": len(sizes),
                     "n_component_pairs": len(sizes) * (len(sizes) - 1) // 2,
                     "n_roi_pairs": int((len(group)**2 - (sizes**2).sum()) // 2),
                     "metric_status": "VALID" if len(sizes) >= 2 else "NOT_ESTIMABLE"})
    return pd.DataFrame(rows)
