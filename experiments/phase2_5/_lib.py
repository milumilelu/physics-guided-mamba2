"""Phase 2.5 shared library: five-part spectral composition, ILR geometry,
radial spectrum, directional FFT metrics, exact sign-flip enumeration, Moran I.

Loads the frozen Phase 2 library by explicit file location (module name
`phase2_lib_p25`; it in turn loads Phase 1.5 as `l15`). No implementation is
copied. Binding spec: Phase2.5_落地执行细则.md.

Conventions (细则 §0.1, rev2):
  - five-part composition uses NON-DC DCT coefficient energies:
      p_b = sum_{k in band, k != DC} C_k^2 / sum_{k != DC} C_k^2
  - `dc_offset_frac = mean(R)^2 / mean(R^2)`  (= C_DC^2 / sum_all C^2), kept
    OUT of the composition as a separate DC/mean-offset descriptor;
  - frozen fractions (E_b = mean(R_b^2)/mean(R^2), with DC inside the >=64 um
    band) reconcile as: E_b^frozen = (1-r_DC) p_b  (DC-free bands)
                      and   E_64^frozen = r_DC + (1-r_DC) p_64.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.fft import dctn

REPO = Path(__file__).resolve().parents[2]

_spec2 = importlib.util.spec_from_file_location(
    "phase2_lib_p25",
    Path(__file__).resolve().parents[1] / "phase2" / "_lib.py")
p2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(p2)
l15 = p2.l15

ILR_BANDS = ["lt8", "8_16", "16_32", "32_64", "64_inf"]


def log(message: str = "") -> None:
    print(message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"HARD ASSERTION FAILED: {message}")


def load_config(description: str) -> tuple[dict, bool]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(Path(__file__).with_name(
        "phase2_5_config.yaml")))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    with open(REPO / args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if args.quick:
        # quick outputs fully isolated from the formal root (细则 §0.9)
        import copy
        cfg = copy.deepcopy(cfg)
        cfg["paths"]["output_root"] = cfg["paths"]["output_root"] + "_quick"
    return cfg, args.quick


def output_dir(cfg: dict, sub: str = "") -> Path:
    out = REPO / cfg["paths"]["output_root"]
    if sub:
        out = out / sub
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# five-part composition (non-DC coefficient energies) + DC offset descriptor
# --------------------------------------------------------------------------- #

def five_part_composition(R: np.ndarray, pixel_um: float) -> tuple[dict, np.ndarray]:
    """Clean five-part composition + dc_offset_frac per sample.

    p_b uses non-DC coefficient energies only; dc_offset_frac = mu^2/M_2 with
    mu = mean(R), M_2 = mean(R^2) (Parseval: = C_DC^2 / sum_all C^2).
    """
    C = dctn(R, axes=(1, 2), norm="ortho")
    lam = l15.dct_lambda_grid(R.shape[1:], pixel_um)
    nonDC = np.isfinite(lam)
    sq = C ** 2
    denom = (sq * nonDC).sum(axis=(1, 2))
    require(np.all(denom > 0), "zero non-DC DCT energy")
    p = {"lt8": (sq * (nonDC & (lam < 8))).sum(axis=(1, 2)) / denom}
    for lo, hi, name in ((8, 16, "8_16"), (16, 32, "16_32"), (32, 64, "32_64")):
        p[name] = (sq * (nonDC & (lam >= lo) & (lam < hi))).sum(axis=(1, 2)) / denom
    p["64_inf"] = (sq * (nonDC & (lam >= 64))).sum(axis=(1, 2)) / denom
    dc_offset_frac = C[:, 0, 0] ** 2 / (C ** 2).sum(axis=(1, 2))
    return p, dc_offset_frac


def frozen_band_fractions(R: np.ndarray, bands_um: list,
                          pixel_um: float) -> tuple[dict, float]:
    """Replicates the Phase 1.5-05 convention exactly:
    E_b = mean(R_b^2) / mean(R^2)  (var_R there is the SECOND moment)."""
    fields, coverage = l15.dct_band_fields(R, pixel_um, bands_um)
    M2 = np.mean(R ** 2, axis=(1, 2))
    E = {name: np.mean(f ** 2, axis=(1, 2)) / M2 for name, f in fields.items()}
    return E, coverage


def apply_zero_replacement(p: np.ndarray, zero_threshold: float,
                           delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicative replacement for confirmed numerical zeros; returns
    (composition, replaced_mask). Caller must STOP if too many rows affected
    (细则 §0.2: this dataset never triggers it)."""
    p = np.asarray(p, dtype=float).copy()
    replaced = p < zero_threshold
    if not replaced.any():
        return p, replaced
    tiny = np.where(replaced, delta, p)
    p = tiny / tiny.sum(axis=1, keepdims=True)
    return p, replaced


# --------------------------------------------------------------------------- #
# ILR geometry (Z1..Z4 sequential balances on [lt8, 8_16, 16_32, 32_64, 64_inf])
# --------------------------------------------------------------------------- #

def ilr_matrix() -> np.ndarray:
    """4x5 orthonormal-row contrast matrix (AA^T = I4; A 1 = 0)."""
    z1 = np.sqrt(6.0 / 5.0) * np.array([0.5, 0.5, -1 / 3, -1 / 3, -1 / 3])
    z2 = np.sqrt(0.5) * np.array([1.0, -1.0, 0.0, 0.0, 0.0])
    z3 = np.sqrt(2.0 / 3.0) * np.array([0.0, 0.0, 1.0, -0.5, -0.5])
    z4 = np.sqrt(0.5) * np.array([0.0, 0.0, 0.0, 1.0, -1.0])
    return np.vstack([z1, z2, z3, z4])


ILR_A = ilr_matrix()


def ilr_transform(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    require(np.all(p > 0), "ILR needs strictly positive compositions")
    return np.log(p) @ ILR_A.T


def ilr_inverse(z: np.ndarray) -> np.ndarray:
    """Closure of exp(A^T z); A 1 = 0 makes the centering constant vanish."""
    z = np.asarray(z, dtype=float)
    logp = z @ ILR_A
    logp = logp - logp.mean(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def aitchison_distance(p_hat: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.linalg.norm(ilr_transform(p_hat) - ilr_transform(p), axis=1)


# --------------------------------------------------------------------------- #
# radial spectrum on the frozen DCT wavelength grid
# --------------------------------------------------------------------------- #

def radial_spectrum(R: np.ndarray, pixel_um: float, n_bins: int,
                    lam_lo: float, lam_hi: float) -> tuple[dict, np.ndarray]:
    """Log-binned non-DC DCT energy spectrum. Returns (per-bin dict, edges).

    Per-bin arrays are (n_samples, n_bins): energy, energy_fraction,
    n_modes, low_mode_count; plus the uncovered non-DC energy fraction.
    """
    C = dctn(R, axes=(1, 2), norm="ortho")
    lam = l15.dct_lambda_grid(R.shape[1:], pixel_um)
    nonDC = np.isfinite(lam)
    sq = C ** 2
    denom = (sq * nonDC).sum(axis=(1, 2))
    edges = np.geomspace(lam_lo, lam_hi, n_bins + 1)
    energy = np.empty((R.shape[0], n_bins))
    n_modes = np.empty((n_bins,), dtype=int)
    covered = np.zeros_like(nonDC)
    for b in range(n_bins):
        msk = nonDC & (lam >= edges[b]) & \
            ((lam < edges[b + 1]) if b < n_bins - 1 else (lam <= edges[b + 1]))
        energy[:, b] = (sq * msk).sum(axis=(1, 2))
        n_modes[b] = int(msk.sum())
        covered |= msk
    uncovered_frac = ((sq * (nonDC & ~covered)).sum(axis=(1, 2)) / denom)
    frac = energy / denom[:, None]
    out = {"energy": energy, "energy_fraction": frac,
           "n_modes": np.broadcast_to(n_modes, frac.shape).copy(),
           "low_mode_count": np.broadcast_to(n_modes < 20, frac.shape).copy(),
           "lambda_geo_um": np.sqrt(edges[:-1] * edges[1:]),
           "lambda_lo_um": edges[:-1], "lambda_hi_um": edges[1:],
           "uncovered_frac": uncovered_frac}
    return out, edges


def spectrum_descriptors(frac: np.ndarray, lam_geo: np.ndarray) -> dict:
    """Centroid / entropy / effective band number / peak (descriptive)."""
    q = np.clip(frac, 0.0, None)
    qsum = q.sum(axis=1, keepdims=True)
    q = q / np.maximum(qsum, 1e-300)
    logl = np.log(lam_geo)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        qlogq = np.where(q > 0, q * np.log(np.maximum(q, 1e-300)), 0.0)
    mu = (q * logl).sum(axis=1)
    return {"spectral_centroid_log_um": mu,
            "spectral_centroid_um": np.exp(mu),
            "spectral_entropy": -qlogq.sum(axis=1) / np.log(len(lam_geo)),
            "effective_band_number": np.exp(-qlogq.sum(axis=1)),
            "lambda_peak_um": lam_geo[np.argmax(q, axis=1)]}


# --------------------------------------------------------------------------- #
# directional spectrum (2D FFT, separable Hann, window-energy normalized)
# --------------------------------------------------------------------------- #

def directional_band_metrics(R: np.ndarray, pixel_um: float,
                             bands: list, theta_bins: int) -> pd.DataFrame:
    """Per-sample x per-band: A2 (second angular moment), wave-vector angle,
    real-space stripe angle (= wave-vector + 90 deg mod 180), angular entropy.

    Preprocessing per 细则 §4: subtract DC only, apply separable Hann window,
    window-energy normalization. Orientation is IMAGE-FRAME relative.
    """
    nx, ny = R.shape[1:]
    wx = np.hanning(nx)
    wy = np.hanning(ny)
    win = wx[:, None] * wy[None, :]
    win_energy = float((win ** 2).sum())
    # array convention (frozen dataset): axis0 = y, axis1 = x.
    # Explicit broadcasts (square 160x160): KX[i, j] = kx[j] along axis1,
    # KY[i, j] = ky[i] along axis0.
    kx = np.fft.fftfreq(nx, d=pixel_um)
    ky = np.fft.fftfreq(ny, d=pixel_um)
    KX, KY = np.meshgrid(kx, ky)                # 'xy': KX varies along axis1
    lam = np.full((nx, ny), np.inf)
    f = np.hypot(KX, KY)
    np.divide(1.0, f, out=lam, where=f > 0)
    theta = np.arctan2(KY, KX)
    edges = np.linspace(0.0, np.pi, theta_bins + 1)
    rows = []
    for i in range(R.shape[0]):
        x = R[i] - R[i].mean()
        P = np.abs(np.fft.fft2(x * win)) ** 2 / win_energy
        for lo, hi, name in bands:
            msk = np.isfinite(lam) & (lam >= lo) & \
                ((lam < hi) if np.isfinite(hi) else True)
            if name == "64_inf":
                msk = np.isfinite(lam) & (lam >= lo)
            Pm = P[msk]
            th = theta[msk]
            s2 = Pm @ np.exp(2j * th)
            a2 = float(np.abs(s2) / Pm.sum())
            theta_k = float(0.5 * np.degrees(np.angle(s2)))
            stripe = (theta_k + 90.0) % 180.0
            th_fold = np.mod(th, np.pi)
            hist, _ = np.histogram(th_fold, bins=edges)
            q = hist / max(hist.sum(), 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                qlogq = np.where(q > 0, q * np.log(np.maximum(q, 1e-300)), 0.0)
            ent = float(-qlogq.sum() / np.log(theta_bins))
            rows.append((i, name, a2, theta_k, stripe, ent))
    return pd.DataFrame(rows, columns=["dataset_index", "band", "A2",
                                       "theta_k_deg", "theta_stripe_deg",
                                       "angular_entropy"])


# --------------------------------------------------------------------------- #
# exact sign-flip enumeration (Task 13)
# --------------------------------------------------------------------------- #

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


def require_no_n4_to_5(groups: np.ndarray, pass_counts: np.ndarray) -> None:
    """N4->5 is session-confounded (v2 §10.2) and must never be analysed."""
    pairs = set(zip(groups.tolist(), pass_counts.tolist()))
    bases = {g for g, c in pairs if c == 4}
    bad = [g for g, c in pairs if c == 5 and g in bases]
    require(not bad, f"N4->5 analysis attempted for confounded bases {bad}")


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


# --------------------------------------------------------------------------- #
# grouped-CV machinery re-exported from the frozen Phase 2 library
# --------------------------------------------------------------------------- #

gkf_splits = p2.gkf_splits
gss_splits = p2.gss_splits
check_gkf_contract = p2.check_gkf_contract
check_gss_contract = p2.check_gss_contract


def read_phase2_manifest(cfg: dict, require_loco: bool = True) -> pd.DataFrame:
    """Read the FORMAL Phase 2 manifest (paths.phase2_manifest). Quick mode
    never redirects this: the manifest is a deterministic frozen input."""
    path = REPO / cfg["paths"]["phase2_manifest"]
    require(path.exists(), "phase2_manifest.csv missing; run Phase 2 01 first")
    man = pd.read_csv(path)
    if require_loco:
        require(set(p2.LOCO_COLS) <= set(man.columns),
                "manifest lacks LOCO backfill; run Phase 2 01 first")
    return man
