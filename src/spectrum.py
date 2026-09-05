"""Canonical spectral primitives: DCT wavelength grid, radial spectrum,
spectral descriptors, directional band metrics, and the Fourier-phase
(phase-only) realization-diagnostic helpers.

Migrated verbatim (WP1 canonical migration; parity-tested in
``tests/test_src_spectrum.py``) from the frozen libraries:

* ``dct_lambda_grid``            <- phase1_5 ``_lib``;
* ``radial_spectrum`` /
  ``spectrum_descriptors`` /
  ``directional_band_metrics``   <- phase2_5 ``_lib``.

New in Phase 2.8 (realization diagnostic, v2.1 §2.8 -- NOT a regression
target and excluded from the Predictability Spectrum):

* ``phase_only_field``:  q = Re F⁻¹[ R̂ / (|R̂| + ε) ]  keeps the Fourier
  phase, flattens amplitude (DC removed);
* ``shift_invariant_phase_distance``: d_phi(i, j) = 1 - max_{|Δ|<=w}
  corr(q_i, q_j(·+Δ)) -- invariant to small global phase ramps from ROI
  translations.  Moran's I was rejected as a phi proxy: I ~ RᵀWR is a
  second-order spatial statistic dominated by the power distribution
  |R̂(k)|², i.e. it re-mixes P(λ)/O_θ information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.provenance import require

__all__ = [
    "dct_lambda_grid", "radial_spectrum", "spectrum_descriptors",
    "directional_band_metrics",
    "phase_only_field", "shift_invariant_phase_distance",
]


# --------------------------------------------------------------------------- #
# DCT wavelength grid (verbatim phase1_5 semantics)
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# radial spectrum on the frozen DCT wavelength grid (verbatim phase2_5)
# --------------------------------------------------------------------------- #

def radial_spectrum(R: np.ndarray, pixel_um: float, n_bins: int,
                    lam_lo: float, lam_hi: float) -> tuple[dict, np.ndarray]:
    """Log-binned non-DC DCT energy spectrum. Returns (per-bin dict, edges).

    Per-bin arrays are (n_samples, n_bins): energy, energy_fraction,
    n_modes, low_mode_count; plus the uncovered non-DC energy fraction.
    """
    from scipy.fft import dctn
    C = dctn(R, axes=(1, 2), norm="ortho")
    lam = dct_lambda_grid(R.shape[1:], pixel_um)
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
# verbatim phase2_5 semantics
# --------------------------------------------------------------------------- #

def directional_band_metrics(R: np.ndarray, pixel_um: float,
                             bands: list, theta_bins: int) -> pd.DataFrame:
    """Per-sample x per-band: A2 (second angular moment), wave-vector angle,
    real-space stripe angle (= wave-vector + 90 deg mod 180), angular entropy.

    Preprocessing per Phase 2.5 细则 §4: subtract DC only, apply separable
    Hann window, window-energy normalization.  Orientation is IMAGE-FRAME
    relative.
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
            # weighted by PSD power (Phase 2.5 rev2 fix): entropy describes
            # the angular POWER distribution, not the grid-point count
            hist, _ = np.histogram(th_fold, bins=edges, weights=Pm)
            q = hist / max(hist.sum(), 1e-300)
            with np.errstate(divide="ignore", invalid="ignore"):
                qlogq = np.where(q > 0, q * np.log(np.maximum(q, 1e-300)), 0.0)
            ent = float(-qlogq.sum() / np.log(theta_bins))
            rows.append((i, name, a2, theta_k, stripe, ent))
    return pd.DataFrame(rows, columns=["dataset_index", "band", "A2",
                                       "theta_k_deg", "theta_stripe_deg",
                                       "angular_entropy"])


# --------------------------------------------------------------------------- #
# NEW (Phase 2.8): Fourier-phase realization diagnostic helpers
# --------------------------------------------------------------------------- #

def phase_only_field(field: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """q = Re F⁻¹[ Q ], Q = F / (|F| + eps) with the DC bin zeroed (the DC
    carries offset, not realization information).  Keeps the Fourier phase,
    flattens the amplitude.  ``field`` is a single 2D height/residual map."""
    F = np.fft.fft2(np.asarray(field, dtype=float))
    Q = F / (np.abs(F) + eps)
    Q[0, 0] = 0.0
    q = np.fft.ifft2(Q)
    # conjugate-symmetric spectrum: imaginary part is numerical noise
    require(float(np.abs(q.imag).max()) < 1e-8, "phase-only field not real")
    return q.real


def shift_invariant_phase_distance(q_i: np.ndarray, q_j: np.ndarray,
                                   max_shift_px: int = 4) -> float:
    """d_phi = 1 - max corr(q_i, q_j shifted by integer (dy, dx)) over
    |dy|, |dx| <= max_shift_px (removes global phase ramps from small ROI
    translations).  Returns in [0, 2]."""
    a = np.asarray(q_i, dtype=float)
    b = np.asarray(q_j, dtype=float)
    require(a.shape == b.shape, "phase-only fields must share shape")
    best = -np.inf
    for dy in range(-max_shift_px, max_shift_px + 1):
        for dx in range(-max_shift_px, max_shift_px + 1):
            bs = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
            a_ = a[max_shift_px:-max_shift_px, max_shift_px:-max_shift_px] \
                if max_shift_px > 0 else a
            b_ = bs[max_shift_px:-max_shift_px, max_shift_px:-max_shift_px] \
                if max_shift_px > 0 else bs
            af = a_.ravel() - a_.mean()
            bf = b_.ravel() - b_.mean()
            denom = np.sqrt((af * af).sum() * (bf * bf).sum())
            if denom <= 0:
                continue
            best = max(best, float((af * bf).sum() / denom))
    if best == -np.inf:
        return float("nan")
    return 1.0 - best
