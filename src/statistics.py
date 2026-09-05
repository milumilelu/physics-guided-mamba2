"""Canonical statistics: exact sign-flip test, Moran's I, cluster
bootstrap resampling, five-class TV and permutation p, logistic slope.

Migrated verbatim (WP1 canonical migration; parity-tested in
``tests/test_src_statistics.py``) from the frozen libraries:

* ``sign_matrix`` / ``exact_signflip_test`` / ``require_no_n4_to_5`` /
  ``knn_row_standardized_graph`` / ``moran_i`` / ``moran_permutation_p``
  <- phase2_5 ``_lib``;
* ``cluster_lists`` / ``boot_draw`` / ``build_resample_bank``  <- phase1_5
  ``_lib`` (the PCA-angle bootstrap family stays in phase1_5: it depends on
  l15 gram-PCA machinery and has no Phase 2.8 consumer -- 细则 §3.1 执行修订);
* ``tv`` / ``tv_perm_p`` / ``logistic_slope``  <- phase2_7 ``_lib``.

Binding spec: Phase 2.8 v2.1 section 4.1 (`src/statistics.py` row).
"""

from __future__ import annotations

import numpy as np

from src.provenance import require

__all__ = [
    "sign_matrix", "exact_signflip_test", "require_no_n4_to_5",
    "knn_row_standardized_graph", "moran_i", "moran_permutation_p",
    "cluster_lists", "boot_draw", "build_resample_bank",
    "tv", "tv_perm_p", "logistic_slope",
]


def cluster_lists(man: pd.DataFrame, key: str = "shared_height_source_id"):
    return [g.to_numpy() for _, g in man.groupby(key)["dataset_index"]]


def boot_draw(clusters: list, rng: np.random.Generator) -> np.ndarray:
    pick = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[p] for p in pick])


def build_resample_bank(clusters: list, B: int,
                        seed: int) -> list[np.ndarray]:
    """Pre-generate B cluster resamples; reuse the same bank across fields."""
    rng = np.random.default_rng(seed)
    return [boot_draw(clusters, rng) for _ in range(B)]


def sign_matrix(m: int) -> np.ndarray:
    """All 2^m sign configurations as an (2^m, m) float matrix (+1/-1)."""
    bits = np.arange(m)
    return 1.0 - 2.0 * ((np.arange(2 ** m)[:, None] >> bits) & 1)


def exact_signflip_test(dz: np.ndarray) -> dict:
    """Global mean-norm statistic with EXACT enumeration.

    p_exact = #{T_null >= T_obs} / 2^m — no Monte-Carlo +1 correction; the
    observed all-+1 configuration is part of the enumeration space, so
    p >= 1/2^m automatically (细则 §0.16). Also returns coordinate-wise
    two-sided exact p for each column mean.
    """
    dz = np.asarray(dz, dtype=float)
    m = dz.shape[0]
    require(m <= 20, f"exact enumeration with m={m} is infeasible")
    S = sign_matrix(m).astype(float)
    means = (S @ dz) / m
    T = np.linalg.norm(means, axis=1)
    T_obs = float(np.linalg.norm(dz.mean(axis=0)))
    p_global = float(np.count_nonzero(T >= T_obs)) / 2 ** m
    coord = []
    for j in range(dz.shape[1]):
        obs = abs(float(dz[:, j].mean()))
        p_j = float(np.count_nonzero(np.abs(means[:, j]) >= obs)) / 2 ** m
        coord.append({"coordinate": j, "mean_dz": float(dz[:, j].mean()),
                      "p_exact_two_sided": p_j})
    return {"T_obs": T_obs, "p_exact_global": p_global,
            "n_configurations": 2 ** m, "coordinates": coord}


def require_no_n4_to_5(steps) -> None:
    """N4->5 is session-confounded (v2 §10.2): analysing that STEP is
    forbidden. steps = iterable of (from_pass, to_pass) tuples."""
    for s_from, s_to in steps:
        require((int(s_from), int(s_to)) != (4, 5),
                "N4->5 step is session-confounded and must not be analysed")


# --------------------------------------------------------------------------- #
# Moran I with kNN binary graph (Task 14B)
# --------------------------------------------------------------------------- #

def knn_row_standardized_graph(X: np.ndarray, k: int) -> np.ndarray:
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    W = np.zeros((len(X), len(X)))
    rows = np.repeat(np.arange(len(X)), k)
    W[rows, idx.ravel()] = 1.0
    W = np.maximum(W, W.T)                      # symmetric kNN graph
    return W / np.maximum(W.sum(axis=1, keepdims=True), 1e-300)


def moran_i(z: np.ndarray, W: np.ndarray) -> float:
    zc = np.asarray(z, dtype=float) - np.mean(z)
    return float((zc @ W @ zc) / max(zc @ zc, 1e-300))


def moran_permutation_p(z: np.ndarray, W: np.ndarray, n_perm: int,
                        seed: int) -> tuple[float, float]:
    """Monte-Carlo permutation (NOT exact enumeration): keep the
    (1+b)/(1+n_perm) correction here, unlike Task 13's exact test."""
    rng = np.random.default_rng(seed)
    i_obs = moran_i(z, W)
    b = 0
    for _ in range(n_perm):
        if moran_i(rng.permutation(z), W) >= i_obs:
            b += 1
    return i_obs, (1 + b) / (1 + n_perm)


def tv(q_a: np.ndarray, q_b: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(q_a, dtype=float)
                              - np.asarray(q_b, dtype=float)).sum())


def tv_perm_p(q_obs_h: dict, q_null_h: dict, weights: dict, *,
              n_perm: int) -> dict:
    """v2.1 frozen permutation p: pooled null center per h, weighted TV.

    q_obs_h / q_null_h: {h: five-class vector}; q_null_h holds per-perm
    distributions keyed 0..n_perm-1.  Weights w_h = n_h/N (observed).
    """
    h_levels = sorted(q_obs_h)
    q_bar = {h: np.mean([q_null_h[h][b] for b in range(n_perm)], axis=0)
             for h in h_levels}
    t_obs = sum(weights[h] * tv(q_obs_h[h], q_bar[h]) for h in h_levels)
    t_b = np.empty(n_perm)
    for b in range(n_perm):
        t_b[b] = sum(weights[h] * tv(q_null_h[h][b], q_bar[h])
                     for h in h_levels)
    p_value = float((1 + int((t_b >= t_obs).sum())) / (1 + n_perm))
    return {"t_obs": float(t_obs), "p_value": p_value,
            "t_null_median": float(np.median(t_b)),
            "q_bar": q_bar}


def logistic_slope(h: np.ndarray, is_m2: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty=None, max_iter=1000)
    model.fit(np.asarray(h, dtype=float).reshape(-1, 1),
              np.asarray(is_m2, dtype=int))
    return float(model.coef_[0][0])
