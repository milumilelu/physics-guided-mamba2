"""Canonical spectral composition + ILR geometry (Aitchison).

Migrated verbatim (WP1 canonical migration; parity-tested in
``tests/test_src_composition.py``) from the frozen phase2_5 ``_lib``:

* ``five_part_composition`` / ``apply_zero_replacement``;
* ``ilr_matrix`` / ``ilr_transform`` / ``ilr_inverse`` /
  ``aitchison_distance`` (sequential balances Z1..Z4 on
  [lt8, 8_16, 16_32, 32_64, 64_inf]).

``frozen_band_fractions`` intentionally stays in the phase2_5 ``_lib``: it
replicates the Phase 1.5-05 second-moment convention through
``l15.dct_band_fields`` and has no Phase 2.8 consumer (细则 §3.1 执行修订).
"""

from __future__ import annotations

import numpy as np

from src.provenance import require
from src.spectrum import dct_lambda_grid

__all__ = [
    "five_part_composition", "apply_zero_replacement",
    "ilr_matrix", "ILR_A", "ilr_transform", "ilr_inverse",
    "aitchison_distance",
]


def five_part_composition(R: np.ndarray, pixel_um: float) -> tuple[dict, np.ndarray]:
    """Clean five-part composition + dc_offset_frac per sample.

    p_b uses non-DC coefficient energies only; dc_offset_frac = mu^2/M_2 with
    mu = mean(R), M_2 = mean(R^2) (Parseval: = C_DC^2 / sum_all C^2).
    """
    from scipy.fft import dctn
    C = dctn(R, axes=(1, 2), norm="ortho")
    lam = dct_lambda_grid(R.shape[1:], pixel_um)
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


def apply_zero_replacement(p: np.ndarray, zero_threshold: float,
                           delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicative replacement for confirmed numerical zeros; returns
    (composition, replaced_mask). Caller must STOP if too many rows affected
    (Phase 2.5 细则 §0.2: this dataset never triggers it)."""
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
    # log-floor keeps extreme inverse-ILR predictions numerically safe
    return np.log(np.maximum(p, 1e-300)) @ ILR_A.T


def ilr_inverse(z: np.ndarray) -> np.ndarray:
    """Closure of exp(A^T z); A 1 = 0 makes the centering constant vanish."""
    z = np.asarray(z, dtype=float)
    logp = z @ ILR_A
    logp = logp - logp.mean(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def aitchison_distance(p_hat: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.linalg.norm(ilr_transform(p_hat) - ilr_transform(p), axis=1)
