"""Depth-primary model zoo (D1): B0/BQ/BU/BA/BP/BPE/BG/BT/BM/BUM.

Target is scalar canonical D (um). Estimators reuse the parity-tested
task_state_v1 Regressor (Ridge / GAM = SplineTransformer + Ridge) with
component-balanced weights. Alpha is selected training-side only via the
frozen inner partitions, using component-balanced MSE on D.
"""
from __future__ import annotations

import numpy as np

from src.task_state_learning.diagnostics import Regressor, group_weights
from src.task_state_learning.grouping import require
from .features import ALL_MORPHOLOGY, BLOCKS, U_COLUMNS

DEPTH_MODELS = ["B0", "BQ", "BU", "BA", "BP", "BPE", "BG", "BT", "BM", "BUM"]


def design_matrix(frame, features, model_id):
    """Input matrix for a depth model. Morphology comes from ``features`` (aligned to frame)."""
    if model_id == "B0":
        return np.zeros((len(frame), 1))
    if model_id == "BQ":
        dose = frame["areal_dose_proxy_J_per_mm2"].to_numpy(float)
        require((dose > 0).all(), "Nonpositive dose proxy")
        return np.log(dose)[:, None]
    u = frame[U_COLUMNS].to_numpy(float)
    require((u > 0).all(), "Nonpositive process input")
    logu = np.log(u)
    if model_id == "BU":
        return logu
    blocks = {"BA": BLOCKS["A"], "BP": BLOCKS["P"], "BPE": BLOCKS["PE"],
              "BG": BLOCKS["G"], "BT": BLOCKS["T"], "BM": ALL_MORPHOLOGY}
    if model_id in blocks:
        return features[blocks[model_id]].to_numpy(float)
    if model_id == "BUM":
        return np.column_stack([logu, features[ALL_MORPHOLOGY].to_numpy(float)])
    raise ValueError(f"Unknown depth model {model_id}")


def component_balanced_mse(y, pred, component_id):
    """Mean of per-component MSE (each component contributes equally)."""
    import pandas as pd
    err = pd.Series((np.asarray(y, float) - np.asarray(pred, float)) ** 2)
    return float(err.groupby(np.asarray(component_id)).mean().mean())


def select_alpha(train, inner_table, train_features, model_id, family, alpha_grid):
    """Training-side alpha selection over frozen inner folds; tie-break to smaller alpha."""
    y = train.D.to_numpy(float)
    choices = []
    for alpha in alpha_grid:
        losses = []
        for k in sorted(inner_table.inner_fold.unique()):
            val_ids = inner_table[inner_table.inner_fold == k].dataset_index
            is_val = train.dataset_index.isin(val_ids).to_numpy()
            a, b = train[~is_val], train[is_val]
            xa = design_matrix(a, train_features[~is_val], model_id)
            xb = design_matrix(b, train_features[is_val], model_id)
            if model_id == "B0":
                pred = np.full(len(b), float(group_weights(a) @ a.D.to_numpy(float)))
            else:
                est = Regressor(family, float(alpha)).fit(xa, a.D.to_numpy(float), group_weights(a))
                pred = est.predict(xb)
            losses.append(component_balanced_mse(b.D.to_numpy(float), pred, b.component_id))
        choices.append((float(np.mean(losses)), float(alpha)))
    return min(choices)


def fit_depth_model(train, test, train_features, test_features, model_id, family, alpha):
    """Fit on outer train, predict outer test. Returns (pred, train_null)."""
    w = group_weights(train)
    null = float(w @ train.D.to_numpy(float))
    if model_id == "B0":
        return np.full(len(test), null), null
    xa = design_matrix(train, train_features, model_id)
    xb = design_matrix(test, test_features, model_id)
    est = Regressor(family, float(alpha)).fit(xa, train.D.to_numpy(float), w)
    pred = est.predict(xb)
    require(np.isfinite(pred).all(), "Nonfinite depth prediction")
    return pred, null
