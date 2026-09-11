"""Morphology features for the depth-primary task (D1).

All morphology features are computed from the median-centered residual
r = H - median(H) only. Forbidden by protocol: canonical depth, absolute
height mean/minimum, raw DC, target-normalized features, target-equivalent
volume. Translation invariance is enforced by ``translation_invariance_error``
and re-checked in experiments/depth_target_v1/00_contract.py.

Canonical columns (D, A_med, ilr_z1..4, A2_8_16, entropy_8_16) are reused
verbatim from the E00 frozen targets (parity-tested in E01); only the new
features below are computed here from ``height_raw``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.fft import dctn

from src.spectrum import dct_lambda_grid
from src.task_state_learning.grouping import require

PIXEL_UM = 0.5
BAND_EDGES = [(0.0, 8.0, "lt8"), (8.0, 16.0, "8_16"), (16.0, 32.0, "16_32"),
              (32.0, 64.0, "32_64"), (64.0, np.inf, "64_inf")]

BLOCKS = {
    "A": ["A_med", "sq_mean_um", "iqr_um", "tail_width_um"],
    "P": ["ilr_z1", "ilr_z2", "ilr_z3", "ilr_z4"],
    "PE": ["log10_E_lt8", "log10_E_8_16", "log10_E_16_32", "log10_E_32_64", "log10_E_64_inf"],
    "G": ["rms_slope", "corr_length_um"],
    "T": ["A2_8_16", "entropy_8_16"],
}
REUSED_CANONICAL = ["A_med", "ilr_z1", "ilr_z2", "ilr_z3", "ilr_z4", "A2_8_16", "entropy_8_16"]
NEW_FEATURES = ["sq_mean_um", "iqr_um", "tail_width_um"] + BLOCKS["PE"] + BLOCKS["G"]
ALL_MORPHOLOGY = BLOCKS["A"] + BLOCKS["P"] + BLOCKS["PE"] + BLOCKS["G"] + BLOCKS["T"]
U_COLUMNS = ["pulse_duration_fs", "frequency_kHz", "velocity_mm_s", "hatch_spacing_um", "pass_count"]

_ENERGY_FLOOR = 1e-12


def residual_median_centered(height, valid_mask):
    """r = H - median_valid(H). Returns float64 residual (full grid required)."""
    h = np.asarray(height, dtype=np.float64)
    v = np.asarray(valid_mask, dtype=bool)
    require(h.ndim == 3 and h.shape == v.shape, "Expected N x H x W height and mask")
    require(v.reshape(len(v), -1).all(axis=1).all(), "Feature computation requires full valid grid")
    med = np.median(h.reshape(len(h), -1), axis=1)
    return h - med[:, None, None]


def band_energies_um2(r, pixel_um=PIXEL_UM):
    """Absolute non-DC DCT band energies per pixel, unit um^2.

    Uses the same orthonormal DCT coefficients as the canonical composition,
    divided by pixel count (plan section 7.1: do NOT approximate via p_b*A_med^2).
    Returns array N x 5 ordered by BAND_EDGES.
    """
    n, hh, ww = r.shape
    c = dctn(r, axes=(1, 2), norm="ortho")
    lam = dct_lambda_grid((hh, ww), pixel_um)
    nondc = np.isfinite(lam)
    sq = c ** 2
    out = np.zeros((n, len(BAND_EDGES)))
    for j, (lo, hi, _) in enumerate(BAND_EDGES):
        band = nondc & (lam >= lo) & (lam < hi)
        out[:, j] = (sq * band[None]).sum(axis=(1, 2)) / (hh * ww)
    return out


def rms_slope(r, pixel_um=PIXEL_UM):
    """RMS of the gradient magnitude of r (central differences, spacing pixel_um)."""
    gy, gx = np.gradient(r, pixel_um, axis=(1, 2))
    return np.sqrt(np.mean(gx ** 2 + gy ** 2, axis=(1, 2)))


def correlation_length_um(r, pixel_um=PIXEL_UM):
    """First radial lag where the normalized autocorrelation of r crosses 1/e.

    ACF via FFT (periodic, mean-centered input), radially binned at integer
    pixel radii, linear interpolation of the crossing. If the ACF never drops
    below 1/e within half the window, returns the half-window cap.
    """
    n, hh, ww = r.shape
    r0 = r - r.mean(axis=(1, 2), keepdims=True)
    power = np.abs(np.fft.fft2(r0)) ** 2
    acf = np.fft.fftshift(np.fft.ifft2(power).real, axes=(1, 2))
    acf /= acf[:, hh // 2, ww // 2][:, None, None]
    iy, ix = np.indices((hh, ww))
    rad = np.hypot(iy - hh // 2, ix - ww // 2)
    max_bin = min(hh, ww) // 2
    out = np.full(n, max_bin * pixel_um)
    threshold = np.exp(-1.0)
    for i in range(n):
        profile = np.array([acf[i][(rad >= b - 0.5) & (rad < b + 0.5)].mean() for b in range(max_bin + 1)])
        below = np.flatnonzero(profile < threshold)
        if len(below) == 0:
            continue
        b = int(below[0])
        if b == 0:
            out[i] = 0.0
            continue
        y1, y0 = profile[b], profile[b - 1]
        frac = (y0 - threshold) / max(y0 - y1, 1e-12)
        out[i] = (b - 1 + min(max(frac, 0.0), 1.0)) * pixel_um
    return out


def compute_new_features(height, valid_mask, pixel_um=PIXEL_UM):
    """Compute the non-canonical morphology features from the median residual."""
    r = residual_median_centered(height, valid_mask)
    n = len(r)
    flat = r.reshape(n, -1)
    centered = flat - flat.mean(axis=1, keepdims=True)
    feats = {
        "sq_mean_um": np.sqrt(np.mean(centered ** 2, axis=1)),
        "iqr_um": np.quantile(flat, 0.75, axis=1) - np.quantile(flat, 0.25, axis=1),
        "tail_width_um": np.quantile(flat, 0.95, axis=1) - np.quantile(flat, 0.05, axis=1),
        "rms_slope": rms_slope(r, pixel_um),
        "corr_length_um": correlation_length_um(r, pixel_um),
    }
    energies = band_energies_um2(r, pixel_um)
    for j, (_, _, name) in enumerate(BAND_EDGES):
        feats[f"log10_E_{name}"] = np.log10(np.maximum(energies[:, j], _ENERGY_FLOOR))
    frame = pd.DataFrame(feats)[NEW_FEATURES]
    require(np.isfinite(frame.to_numpy()).all(), "Nonfinite morphology feature")
    return frame


def translation_invariance_error(height, valid_mask, shift_um=3.7, pixel_um=PIXEL_UM):
    """Max abs change of every new feature when the whole ROI is shifted by a constant.

    Protocol gate: must be numerically zero (features read the median residual only).
    """
    base = compute_new_features(height, valid_mask, pixel_um)
    moved = compute_new_features(np.asarray(height, dtype=np.float64) + shift_um, valid_mask, pixel_um)
    return (moved - base).abs().max(axis=0)


def align_height_rows(frozen_targets, pack):
    """Map each frozen_targets row to its height-package row via (session_id, sample_id).

    Verifies one-to-one matching and canonical D/A_med parity against height_raw.
    """
    keys = {}
    for i, (s, smp) in enumerate(zip(pack["session_id"].tolist(), pack["sample_id"].tolist())):
        key = (str(s), int(smp))
        require(key not in keys, f"Duplicate height-package key {key}")
        keys[key] = i
    idx = np.array([keys.get((r.session_id, int(r.sample_id)), -1)
                    for r in frozen_targets.itertuples()])
    require((idx >= 0).all(), "Unmatched frozen target row in height package")
    require(len(set(idx.tolist())) == len(idx), "Height-package rows not used one-to-one")
    h = pack["height_raw"][idx].astype(np.float64)
    v = pack["valid_mask"][idx]
    med = np.array([np.median(h[i][v[i]]) for i in range(len(h))])
    res = h - med[:, None, None]
    amed = np.sqrt(np.mean(res.reshape(len(res), -1) ** 2, axis=1))
    require(np.abs(-med - frozen_targets.D.to_numpy()).max() < 1e-4, "Canonical D parity failure")
    require(np.abs(amed - frozen_targets.A_med.to_numpy()).max() < 1e-4, "Canonical A_med parity failure")
    return idx


def build_feature_frame(frozen_targets, height, valid_mask, pixel_um=PIXEL_UM):
    """Full morphology feature table: reused canonical columns + new features."""
    new = compute_new_features(height, valid_mask, pixel_um)
    canonical = frozen_targets[REUSED_CANONICAL].reset_index(drop=True)
    frame = pd.concat([canonical, new.reset_index(drop=True)], axis=1)
    require(list(frame.columns) == REUSED_CANONICAL + NEW_FEATURES, "Feature column order drift")
    require(set(ALL_MORPHOLOGY) <= set(frame.columns), "Missing morphology block column")
    return frame
