"""Shared frozen-input loaders and fast cluster-bootstrap PCA for Phase 1.5.

Low-model-assumption diagnostics only. Residual definition matches Phase 1
exactly: R = H - per-sample valid-median, computed from height_raw.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[2]


def log(message: str = "") -> None:
    print(message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"HARD ASSERTION FAILED: {message}")


def load_config(description: str) -> tuple[dict, bool]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(Path(__file__).with_name(
        "phase1_5_config.yaml")))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    with open(REPO / args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg, args.quick


def n_boot(cfg: dict, quick: bool) -> int:
    b = cfg["bootstrap"]
    return int(b["n_replicates_quick"]) if quick else int(b["n_replicates"])


def output_dir(cfg: dict) -> Path:
    out = REPO / cfg["paths"]["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    return out


def to_2d(mat: np.ndarray) -> np.ndarray:
    return mat.reshape(mat.shape[0], -1)


def load_frozen(cfg: dict) -> dict:
    """Load Phase 1 manifest + NPZ and rebuild residuals exactly as Phase 1."""
    man = pd.read_csv(REPO / cfg["paths"]["exploration_manifest"])
    require(len(man) == 200, f"manifest rows {len(man)} != 200")
    require(not man.duplicated(["session_id", "sample_id"]).any(),
            "(session_id, sample_id) not unique")
    require(man["shared_height_source_id"].nunique() == 160,
            "unique shared sources != 160")
    require({"median_depth_um", "residual_Sq_um", "session_role",
             "design_group"} <= set(man.columns),
            "manifest lacks Phase 1 columns; run Phase 1 first")

    data = np.load(REPO / cfg["paths"]["dataset_npz"])
    H = data["height_raw"].astype(np.float64)
    V = data["valid_mask"].astype(bool)
    require(H.shape == (200, 160, 160), "NPZ shape mismatch")
    require((man["session_id"].to_numpy() == data["session_id"].astype(str)).all()
            and (man["sample_id"].to_numpy(np.int64)
                 == data["sample_id"].astype(np.int64)).all(),
            "NPZ row order != manifest order")
    bad = int(np.count_nonzero(~np.isfinite(H[V])))
    require(bad == 0, f"{bad} non-finite valid pixels")
    Hnan = np.where(V, H, np.nan)
    med = np.nanmedian(Hnan, axis=(1, 2))
    R = Hnan - med[:, None, None]
    log(f"  frozen inputs OK: 200 ROIs, 160 clusters, "
        f"valid_fraction min = {V.reshape(200, -1).mean(1).min():.4f}")
    return {"man": man, "H": H, "V": V, "R": R, "Hnan": Hnan}


# --------------------------------------------------------------------------- #
# scale decomposition
# --------------------------------------------------------------------------- #

def gaussian_smooth(R3: np.ndarray, sigma_px: float) -> np.ndarray:
    out = np.empty_like(R3)
    for i in range(R3.shape[0]):
        out[i] = gaussian_filter(R3[i], float(sigma_px), mode="reflect")
    return out


def make_bands(R3: np.ndarray, sigma_low_px: float,
               sigma_high_px: float) -> dict[str, np.ndarray]:
    """Disjoint band split (spec formulas, sigma_low > sigma_high):

    R_low  = G_{sigma_low} * R           (coarser than ~sigma_low px)
    R_high = R - G_{sigma_high} * R      (finer than ~sigma_high px)
    R_mid  = R - R_low - R_high          (band between the two cutoffs)
    """
    require(sigma_low_px > sigma_high_px, "need sigma_low > sigma_high")
    n_nan = int(np.count_nonzero(~np.isfinite(R3)))
    if n_nan:
        log(f"  WARNING: {n_nan} NaN pixels in R; filling with per-sample "
            "median before filtering")
        fill = np.nanmedian(R3, axis=(1, 2))
        R3 = np.where(np.isfinite(R3), R3, fill[:, None, None])
    R_low = gaussian_smooth(R3, sigma_low_px)
    R_high = R3 - gaussian_smooth(R3, sigma_high_px)
    R_mid = R3 - R_low - R_high
    return {"low": R_low, "mid": R_mid, "high": R_high}


# --------------------------------------------------------------------------- #
# PCA + cluster bootstrap
# --------------------------------------------------------------------------- #

def gram_pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact PCA via n x n Gram matrix. Returns (components (k, F), evr (k,))."""
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    G = (Xc @ Xc.T) / n
    return pca_from_gram(G, Xc, k)


def pca_from_gram(G: np.ndarray, Xc: np.ndarray,
                  k: int) -> tuple[np.ndarray, np.ndarray]:
    w, U = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:k]
    top = w[order]
    s = np.sqrt(np.maximum(top, 0.0))
    safe = np.where(s > 0, s, 1.0)
    Vt = (Xc.T @ U[:, order]) / safe[None, :]
    # explicit unit normalisation: correct regardless of how G was scaled
    # (raw Xc Xc^T vs /n), which changes lambda by a constant factor
    norms = np.linalg.norm(Vt, axis=0)
    Vt = Vt / np.where(norms > 0, norms, 1.0)[None, :]
    evr = top / max(np.trace(G), 1e-300)
    return Vt.T, evr


def principal_angles(U_a: np.ndarray, U_b: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(U_a.T @ U_b, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def cluster_lists(man: pd.DataFrame, key: str = "shared_height_source_id"):
    return [g.to_numpy() for _, g in man.groupby(key)["dataset_index"]]


def boot_draw(clusters: list, rng: np.random.Generator) -> np.ndarray:
    pick = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[p] for p in pick])


def boot_angles(G: np.ndarray, X: np.ndarray, clusters: list,
                ref_comps: np.ndarray, k_max: int, B: int,
                seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Max principal angle of bootstrap top-k subspace vs reference, per rep.

    Uses the precomputed Gram G = X X^T: the bootstrap Gram of centred rows is
    G_b - (G_b 1 1^T + 1 1^T G_b)/m + (1^T G_b 1 / m^2) 1 1^T, all m x m ops.
    Returns (angles (B, k_max), evr (B, 3)).
    """
    rng = np.random.default_rng(seed)
    angles = np.zeros((B, k_max))
    evr = np.zeros((B, min(3, k_max)))
    for b in range(B):
        idx = boot_draw(clusters, rng)
        m = len(idx)
        Gb = G[np.ix_(idx, idx)]
        one = np.ones(m)
        Gb1 = Gb @ one
        c11 = float(one @ Gb1)
        Gc = Gb - (np.outer(Gb1, one) + np.outer(one, Gb1)) / m \
            + (c11 / (m * m)) * np.outer(one, one)
        w, U = np.linalg.eigh(Gc)
        order = np.argsort(w)[::-1][:k_max]
        lam = w[order]
        evr[b] = lam[:evr.shape[1]] / max(np.trace(Gc), 1e-300)
        Xb = X[idx]
        Xc = Xb - Xb.mean(axis=0, keepdims=True)
        s = np.sqrt(np.maximum(lam, 0.0))
        comps = (Xc.T @ U[:, order]) / np.where(s > 0, s, 1.0)[None, :]
        for k in range(1, k_max + 1):
            angles[b, k - 1] = principal_angles(ref_comps[:k].T, comps[:, :k])[-1]
    return angles, evr


def _self_test() -> None:
    """Verify fast bootstrap Gram/components against direct computation."""
    rng = np.random.default_rng(0)
    n, f = 30, 400
    X = rng.normal(size=(n, f)) @ rng.normal(size=(f, 12))
    clusters = [np.array([i]) for i in range(n)]
    ref, _ = gram_pca(X, 4)
    require(np.allclose(ref @ ref.T, np.eye(4), atol=1e-8),
            "self-test: reference components not orthonormal")
    G = X @ X.T
    angles_fast, evr_fast = boot_angles(G, X, clusters, ref, 3, B=5, seed=7)
    require(np.all((angles_fast >= 0) & (angles_fast <= 90)),
            "self-test: angles outside [0, 90]")
    rng2 = np.random.default_rng(7)
    for b in range(5):
        pick = rng2.integers(0, n, size=n)
        idx = np.concatenate([clusters[p] for p in pick])
        comps_dir, evr_dir = gram_pca(X[idx], 3)
        require(np.allclose(comps_dir @ comps_dir.T, np.eye(3), atol=1e-8),
                f"self-test: direct components not orthonormal (rep {b})")
        for k in range(1, 4):
            ang_dir = principal_angles(ref[:k].T, comps_dir[:k].T)[-1]
            require(abs(ang_dir - angles_fast[b, k - 1]) < 1e-7,
                    f"self-test angle mismatch at rep {b} k {k}")
        require(np.allclose(evr_dir[:3], evr_fast[b], atol=1e-10),
                f"self-test EVR mismatch at rep {b}")


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #

def pairwise_rmse_from_gram(G: np.ndarray, n_features: int) -> np.ndarray:
    sq = np.diag(G).copy()
    d2 = np.clip(sq[:, None] + sq[None, :] - 2.0 * G, 0.0, None)
    return np.sqrt(d2 / n_features)


def sentinel_rows(man: pd.DataFrame, cfg: dict) -> tuple[int, int]:
    sent = cfg["sentinel"]
    sel = man[(man["session_id"] == sent["session"])
              & (man["processing_order"].isin(sent["processing_orders"]))]
    require(len(sel) == 2, "sentinel rows != 2")
    return tuple(int(x) for x in sel["dataset_index"].to_numpy())


def ordinary_pair_mask(man: pd.DataFrame, i_a: int, i_b: int) -> tuple:
    iu = np.triu_indices(len(man), k=1)
    same = (man["shared_height_source_id"].to_numpy()[iu[0]]
            == man["shared_height_source_id"].to_numpy()[iu[1]])
    is_sent = ((iu[0] == i_a) & (iu[1] == i_b)) | \
              ((iu[0] == i_b) & (iu[1] == i_a))
    keep = ~same & ~is_sent
    return iu[0][keep], iu[1][keep]


def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"
