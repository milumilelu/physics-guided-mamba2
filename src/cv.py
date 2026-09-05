"""Canonical grouped-CV contracts and Ridge alpha selection.

Migrated verbatim (WP1 canonical migration; parity-tested in
``tests/test_src_cv.py``) from the frozen per-phase libraries:

* ``gkf_splits`` / ``gss_splits`` / ``check_gkf_contract`` /
  ``check_gss_contract``  <- phase2 ``_lib`` (frozen Phase 2 semantics);
* ``make_ridge_alpha_grid`` / ``make_ridge`` / ``ridge_alpha_inner_gkf``
  <- phase2_6 ``_lib`` (frozen inner GKF(5) mean-MSE protocol).

Frozen-split semantics (Phase 2.8 v2.1 F1 -- identical to Phase 2/2.5/2.7):

* ``src_gkf``  = GroupKFold(5), groups = ``shared_height_source_id`` (160);
* ``proc_gkf`` = GroupKFold(5), groups = ``cv_process_group``       (134).

Both are deterministic GroupKFold splits (no shuffle, no seed).  A shuffle
variant must be named ``proc_gss_sensitivity`` and built from ``gss_splits``
-- never called ``proc_gkf``.

New in Phase 2.8 (does NOT rewrite Phase 2.7): ``select_alpha_inner``
selects Ridge alpha by a **target-native inner scorer** so the inner
objective and the outer skill belong to the same loss family:

* ``"mse"``              scalar targets          -> inner mean MSE;
* ``"multi_mse"``        multivariate targets    -> inner mean Euclidean MSE;
* ``"aitchison_ilr_q2"`` ILR composition targets -> inner mean Aitchison Q2
  (maximized; equivalent to minimizing ILR-space MSE).

Versioning discipline (v2.1 §4.2): frozen semantics are immutable -- the
Phase 2.7 alpha protocol is preserved under the explicit ``_v1`` name;
new semantics only ever appear under new names.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

__all__ = [
    "gkf_splits", "gss_splits", "check_gkf_contract", "check_gss_contract",
    "make_ridge_alpha_grid", "make_ridge", "ridge_alpha_inner_gkf_v1",
    "select_alpha_inner", "q2_aitchison_ilr_v1",
]

_RIDGE_ALPHA_GRID: np.ndarray | None = None


# --------------------------------------------------------------------------- #
# grouped CV splits + contracts (verbatim phase2 semantics)
# --------------------------------------------------------------------------- #

def gkf_splits(groups, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold
    groups = np.asarray(groups)
    return [(np.asarray(tr), np.asarray(te)) for tr, te in
            GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)),
                                                groups=groups)]


def gss_splits(groups, n_splits: int, test_size: float,
               seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupShuffleSplit
    groups = np.asarray(groups)
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size,
                                 random_state=seed)
    return [(np.asarray(tr), np.asarray(te)) for tr, te in
            splitter.split(np.zeros(len(groups)), groups=groups)]


def _group_sets(groups: np.ndarray, splits) -> list[tuple[set, set]]:
    return [(set(groups[tr].tolist()), set(groups[te].tolist()))
            for tr, te in splits]


def check_gkf_contract(groups: np.ndarray, splits) -> None:
    """GroupKFold: per-split train/test disjoint; test groups pairwise
    disjoint across folds; union of test groups == all groups."""
    groups = np.asarray(groups)
    gs = _group_sets(groups, splits)
    test_sets = [te for _, te in gs]
    for a, b in itertools.combinations(range(len(test_sets)), 2):
        require_g(test_sets[a].isdisjoint(test_sets[b]),
                  "gkf: test groups overlap across folds")
    require_g(set().union(*test_sets) == set(groups.tolist()),
              "gkf: test union != all groups")
    for tr_g, te_g in gs:
        require_g(tr_g.isdisjoint(te_g), "gkf: train/test group overlap")


def check_gss_contract(groups: np.ndarray, splits) -> pd.Series:
    """GroupShuffleSplit: only per-split train/test disjointness is required.
    Test groups MAY repeat across splits and need not cover all groups; the
    per-group test membership count is reported instead."""
    groups = np.asarray(groups)
    counts = pd.Series(0, index=sorted(set(groups.tolist())), dtype=int)
    for tr, te in splits:
        tr_g = set(groups[tr].tolist())
        te_g = set(groups[te].tolist())
        require_g(not (tr_g & te_g), "gss: train/test group overlap in a split")
        counts.loc[sorted(te_g)] += 1
    return counts


def require_g(condition: bool, message: str) -> None:
    from src.provenance import require
    require(condition, message)


# --------------------------------------------------------------------------- #
# Ridge family + frozen Phase 2.7 alpha protocol (v1 semantics)
# --------------------------------------------------------------------------- #

def make_ridge_alpha_grid() -> np.ndarray:
    global _RIDGE_ALPHA_GRID
    if _RIDGE_ALPHA_GRID is None:
        _RIDGE_ALPHA_GRID = np.logspace(-3, 3, 13)
    return _RIDGE_ALPHA_GRID


def make_ridge(alpha: float):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scale", StandardScaler()),
                     ("ridge", Ridge(alpha=float(alpha)))])


def ridge_alpha_inner_gkf_v1(X_train, y_train, groups_train, *,
                             n_splits: int = 5) -> float:
    """Frozen Phase 2.6/2.7 protocol: fold-internal alpha selection by inner
    GKF(5) mean MSE.  Kept under the explicit ``_v1`` name; Phase 2.7 and
    earlier keep calling this via their ``_lib`` re-exports."""
    grid = make_ridge_alpha_grid()
    inner = gkf_splits(pd.Series(groups_train), n_splits)
    check_gkf_contract(pd.Series(groups_train), inner)
    scores = []
    for alpha in grid:
        fold_mse = []
        for tr, te in inner:
            model = make_ridge(alpha).fit(np.asarray(X_train)[tr],
                                          np.asarray(y_train)[tr])
            pred = model.predict(np.asarray(X_train)[te])
            fold_mse.append(float(np.mean(
                (np.asarray(y_train)[te] - pred) ** 2)))
        scores.append(float(np.mean(fold_mse)))
    return float(grid[int(np.argmin(scores))])


def q2_aitchison_ilr_v1(z_test: np.ndarray, z_pred: np.ndarray,
                        z_train: np.ndarray) -> float:
    """ILR-coordinate-space Q2 with train-mean null (frozen definition,
    provenance Phase 2.5 `12_` script via Phase 2.7 ``_lib.q2_aitchison_ilr``)."""
    z_test = np.asarray(z_test, dtype=float)
    z_pred = np.asarray(z_pred, dtype=float)
    z_train = np.asarray(z_train, dtype=float)
    denom = float(((z_test - z_train.mean(axis=0)) ** 2).sum())
    if denom <= 0:
        return np.nan
    return float(1.0 - ((z_test - z_pred) ** 2).sum() / denom)


# --------------------------------------------------------------------------- #
# NEW (Phase 2.8): target-native inner alpha selection
# --------------------------------------------------------------------------- #

_INNER_SCORERS = ("mse", "multi_mse", "aitchison_ilr_q2")


def _mean_inner_score(X: np.ndarray, y: np.ndarray, inner, alpha: float,
                      scorer: str) -> float:
    fold_scores = []
    for tr, te in inner:
        model = make_ridge(alpha).fit(X[tr], y[tr])
        pred = model.predict(X[te])
        if scorer == "mse":
            fold_scores.append(float(np.mean((y[te] - pred) ** 2)))
        elif scorer == "multi_mse":
            fold_scores.append(float(np.mean((y[te] - pred) ** 2)))
        else:  # aitchison_ilr_q2 (maximized)
            fold_scores.append(q2_aitchison_ilr_v1(y[te], pred, y[tr]))
    arr = np.asarray(fold_scores, dtype=float)
    if np.isnan(arr).any():
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return float("-inf") if scorer == "aitchison_ilr_q2" else float("inf")
    return float(arr.mean())


def select_alpha_inner(X, y, groups, *, scorer: str, n_splits: int = 5,
                       grid=None, return_scores: bool = False):
    """Fold-internal alpha selection with a target-native scorer.

    scorer: "mse" (scalar, minimize), "multi_mse" (Euclidean, minimize),
    "aitchison_ilr_q2" (ILR-space Q2, maximize).  The inner objective belongs
    to the same loss family as the Phase 2.8 outer skill -- this replaces the
    Phase 2.7 practice of selecting alpha on the first ILR coordinate
    (``ridge_alpha_inner_gkf_v1``), which is preserved unchanged for the
    frozen phases.
    """
    if scorer not in _INNER_SCORERS:
        raise ValueError(f"scorer must be one of {_INNER_SCORERS}, got {scorer!r}")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = make_ridge_alpha_grid() if grid is None else np.asarray(grid, float)
    inner = gkf_splits(pd.Series(groups), n_splits)
    check_gkf_contract(pd.Series(groups), inner)
    scores = np.array([_mean_inner_score(X, y, inner, float(a), scorer)
                       for a in grid])
    best = int(np.argmax(scores)) if scorer == "aitchison_ilr_q2" \
        else int(np.argmin(scores))
    alpha = float(grid[best])
    return (alpha, scores) if return_scores else alpha
