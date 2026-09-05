"""Shared frozen-input loaders and fast cluster-bootstrap PCA for Phase 1.5R.

Low-model-assumption diagnostics only. Residual definition matches Phase 1
exactly: R = H - per-sample valid-median, computed from height_raw.

1.5R revisions:
- scales are named by their filter (Gsigma2/4/8/16 px) and by physical
  wavelength DCT bands; Gaussian -3dB wavelengths are reported explicitly;
- bootstrap uses a pre-generated cluster resample bank (shared across fields)
  and reports Q25/Q50/Q75/Q90/Q95 (the angle distribution is asymmetric);
- conditional baselines draw from the same session with matched ROI count and
  matched within-subset cluster-size pattern;
- leave-one-cluster-out influence and eigengap diagnostics are provided.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# WP1 canonical migration (parity-tested, tests/test_src_data.py): the
# frozen-input loader now lives in src/data.py; the frozen name is kept as a
# thin re-export (Phase 2.8 v2.1 §4.2 migration protocol step 5).
from src.data import load_frozen  # noqa: E402,F401


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


# --------------------------------------------------------------------------- #
# scale fields: Gaussian low-pass (Gsigma) and physical-wavelength DCT bands
# --------------------------------------------------------------------------- #

def gaussian_smooth(R3: np.ndarray, sigma_px: float) -> np.ndarray:
    out = np.empty_like(R3)
    for i in range(R3.shape[0]):
        out[i] = gaussian_filter(R3[i], float(sigma_px), mode="reflect")
    return out


def lambda_3db_px(sigma_px: float) -> float:
    """-3 dB wavelength of the continuous Gaussian transfer function.

    |H(f)| = exp(-2 pi^2 sigma^2 f^2) = 1/sqrt(2) at
    f = sqrt(ln 2) / (2 pi sigma)  ->  lambda_3dB = 2 pi sigma / sqrt(ln 2).
    """
    return 2.0 * np.pi * float(sigma_px) / np.sqrt(np.log(2.0))


def numeric_lambda_3db_px(sigma_px: float, n: int = 8192) -> float:
    """Empirical -3 dB wavelength from the DFT of the discrete kernel."""
    radius = max(int(8.0 * float(sigma_px)), 4)
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-x ** 2 / (2.0 * float(sigma_px) ** 2))
    kernel /= kernel.sum()
    Hf = np.abs(np.fft.rfft(kernel, n=n))
    target = 1.0 / np.sqrt(2.0)
    below = np.flatnonzero(Hf < target)
    j = int(below[0])
    f_cross = (j - 1 + (Hf[j - 1] - target) / (Hf[j - 1] - Hf[j])) / n
    return float(1.0 / f_cross)


def dct_lambda_grid(shape: tuple[int, int], pixel_um: float) -> np.ndarray:
    """Wavelength [um] of each DCT-II coefficient (isotropic |f|).

    DCT-II mode k over N samples of spacing d has spatial frequency
    k / (2 N d) cycles/um; lam = 1/|f| with f = hypot(fx, fy); DC -> inf.
    """
    nx, ny = shape
    fx = np.arange(nx) / (2.0 * nx * pixel_um)
    fy = np.arange(ny) / (2.0 * ny * pixel_um)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    f = np.hypot(FX, FY)
    lam = np.full(shape, np.inf)
    np.divide(1.0, f, out=lam, where=f > 0)
    return lam


def dct_band_fields(R3: np.ndarray, pixel_um: float,
                    bands_um: list) -> tuple[dict[str, np.ndarray], float]:
    """Physical-wavelength band-pass fields via DCT coefficient masking.

    Returns (dict band-name -> field, coverage fraction of the pixel grid by
    the union of the bands). Band names: DCT_<lo>_<hi> (um).
    """
    lam = dct_lambda_grid(R3.shape[1:], pixel_um)
    out: dict[str, np.ndarray] = {}
    covered = np.zeros(R3.shape[1:], dtype=bool)
    for lo_raw, hi_raw in bands_um:
        lo, hi = float(lo_raw), float(hi_raw)
        if np.isfinite(hi) and hi < 1e8:
            mask = (lam >= lo) & (lam < hi)
            name = f"DCT_{lo:g}_{hi:g}"
        else:
            mask = lam >= lo
            name = f"DCT_{lo:g}_inf"
        fields = np.empty_like(R3)
        for i in range(R3.shape[0]):
            C = dctn(R3[i], norm="ortho")
            fields[i] = idctn(C * mask, norm="ortho")
        out[name] = fields
        covered |= mask
    return out, float(covered.mean())


def multiscale_fields(R3: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """total + Gaussian low-pass fields + DCT band fields, 2-D flattened."""
    pixel_um = float(cfg["scales"]["pixel_um"])
    fields = {"total": to_2d(R3)}
    for sigma in cfg["scales"]["sigmas_px"]:
        fields[f"G{int(sigma)}"] = to_2d(gaussian_smooth(R3, float(sigma)))
    dct_fields, coverage = dct_band_fields(R3, pixel_um,
                                           cfg["scales"]["dct_bands_um"])
    fields.update({k: to_2d(v) for k, v in dct_fields.items()})
    return fields


# --------------------------------------------------------------------------- #
# PCA + cluster bootstrap (bank based)
# --------------------------------------------------------------------------- #

def gram_eigenvalues(X: np.ndarray, k: int = 2) -> np.ndarray:
    """Top-k raw eigenvalues of the centred Gram matrix (descending)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    w = np.linalg.eigvalsh(Xc @ Xc.T)
    return w[::-1][:k]


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


# cluster resampling trio: canonical implementation in src/statistics.py
# (WP1 migration, parity-tested); frozen names kept as thin re-exports.
from src.statistics import (  # noqa: E402,F401
    boot_draw,
    build_resample_bank,
    cluster_lists,
)

def boot_angles_bank(G: np.ndarray, X: np.ndarray, bank: list,
                     ref_comps: np.ndarray,
                     k_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Max principal angle of each bank resample's top-k subspace vs reference.

    Uses the precomputed Gram G = X X^T: the resample Gram of centred rows is
    G_b - (G_b 1 1^T + 1 1^T G_b)/m + (1^T G_b 1 / m^2) 1 1^T, all m x m ops.
    Returns (angles (B, k_max), evr (B, 3)).
    """
    B = len(bank)
    angles = np.zeros((B, k_max))
    evr = np.zeros((B, min(3, k_max)))
    for b, idx in enumerate(bank):
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


def boot_angles(G: np.ndarray, X: np.ndarray, clusters: list,
                ref_comps: np.ndarray, k_max: int, B: int,
                seed: int) -> tuple[np.ndarray, np.ndarray]:
    bank = build_resample_bank(clusters, B, seed)
    return boot_angles_bank(G, X, bank, ref_comps, k_max)


def angle_quantiles(angles: np.ndarray) -> dict[str, np.ndarray]:
    """Q25/Q50/Q75/Q90/Q95 per column (the distribution is asymmetric)."""
    qs = np.percentile(angles, [25, 50, 75, 90, 95], axis=0)
    return {"q25": qs[0], "q50": qs[1], "q75": qs[2], "q90": qs[3],
            "q95": qs[4]}


def loco_angles(X: np.ndarray, clusters: list, k: int = 1) -> np.ndarray:
    """Leave-one-cluster-out influence: angle between the full-subset top-k
    subspace and the subspace without each cluster."""
    ref, _ = gram_pca(X, k)
    all_rows = np.arange(len(X))
    out = np.zeros(len(clusters))
    for i, c in enumerate(clusters):
        keep = np.setdiff1d(all_rows, c)
        ref_k, _ = gram_pca(X[keep], k)
        out[i] = principal_angles(ref[:k].T, ref_k[:k].T)[-1]
    return out


def occupancy_signature(man_sub: pd.DataFrame,
                        key: str = "shared_height_source_id") -> tuple:
    """Sorted within-subset cluster sizes, e.g. (1,1,...,1) or (2,2,1,...)."""
    sizes = man_sub.groupby(key).size().to_numpy()
    return tuple(sorted((int(s) for s in sizes), reverse=True))


def draw_matched_subset(pool: list, sig_sizes: tuple,
                        rng: np.random.Generator) -> np.ndarray:
    """Draw a random ROI subset with the same cluster count, ROI count and
    within-subset cluster-size pattern as a conditional subset.

    pool: list of global cluster member-index arrays. For each required
    within-subset size s (largest first) one cluster is drawn without
    replacement and min(s, len(cluster)) members are taken (random slots).
    """
    remaining = list(range(len(pool)))
    picked = []
    for s in sorted(sig_sizes, reverse=True):
        ok = [j for j in remaining if len(pool[j]) >= s]
        if not ok:
            ok = remaining
        j = ok[int(rng.integers(0, len(ok)))]
        remaining.remove(j)
        members = pool[j]
        take = rng.permutation(members)[:s]
        picked.append(np.sort(take))
    return np.sort(np.concatenate(picked))


def session_cluster_pools(man: pd.DataFrame,
                          key: str = "shared_height_source_id") -> dict:
    """session_id -> list of cluster member-index arrays (global dataset)."""
    out = {}
    for s, grp in man.groupby("session_id"):
        out[s] = [g.to_numpy() for _, g in grp.groupby(key)["dataset_index"]]
    return out


# --------------------------------------------------------------------------- #
# pairwise helpers
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
    bank = build_resample_bank(clusters, 5, 7)
    angles_fast, evr_fast = boot_angles_bank(G, X, bank, ref, 3)
    require(np.all((angles_fast >= 0) & (angles_fast <= 90)),
            "self-test: angles outside [0, 90]")
    rng2 = np.random.default_rng(7)
    for b in range(5):
        pick = rng2.integers(0, n, size=n)
        idx = np.concatenate([clusters[p] for p in pick])
        require(np.array_equal(idx, bank[b]), "self-test: bank mismatch")
        comps_dir, evr_dir = gram_pca(X[idx], 3)
        require(np.allclose(comps_dir @ comps_dir.T, np.eye(3), atol=1e-8),
                f"self-test: direct components not orthonormal (rep {b})")
        for k in range(1, 4):
            ang_dir = principal_angles(ref[:k].T, comps_dir[:k].T)[-1]
            require(abs(ang_dir - angles_fast[b, k - 1]) < 1e-7,
                    f"self-test angle mismatch at rep {b} k {k}")
        require(np.allclose(evr_dir[:3], evr_fast[b], atol=1e-10),
                f"self-test EVR mismatch at rep {b}")
    # -3 dB wavelength: analytic vs numeric DFT crossing
    for sigma in (2.0, 7.3):
        ana = lambda_3db_px(sigma)
        num = numeric_lambda_3db_px(sigma)
        require(abs(ana - num) / ana < 0.02,
                f"self-test: lambda_3dB analytic {ana:.3f} vs numeric "
                f"{num:.3f}")
