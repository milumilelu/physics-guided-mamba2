"""Canonical forward models: Hann Fourier projection, cycle measurability,
finite-array 2D synthesis, and the same-pipeline peak-class observation
operator; plus the Phase 2.8 model-family extensions (frozen v2.1 S3.2).

Migrated verbatim (WP1 canonical migration; parity-tested in
``tests/test_src_forward_models.py``) from the frozen phase2_7 ``_lib``:
``hann_projection`` / ``cycles_level`` / ``synth_field`` / ``field_class``.

New in Phase 2.8 (G28-B model family; v2.1 F6/F7 fixes):

* ``saturate(s, D_sat)``         pointwise monotone saturation family
                                 F_beta(s) = D_sat (1 - e^(-s/D_sat));
* ``pairwise_interaction_field`` L3b: z = sum g_n + gamma_per_um * sum
                                 g_n * g_(n+1)  ([gamma] = um^-1);
* ``array_transfer``             |A_array(k)|^2 for a finite line array;
* ``overlap_descriptor``         normalized O(h) over the common support with
                                 edge guard (v2.1 normalized definition);
* ``physical_validity_field``    candidate screening: simulated field must
                                 never dip below -tol removal depth
                                 (physical-invalid; never post-hoc clipped).

Sign convention: removal-depth positive (g >= 0 = depth into the material).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.geometry import (  # noqa: F401
    CODE_INVALID,
    assign_class,
    lambda_peak_4_32,
)
from src.provenance import require
from src.spectrum import radial_spectrum

__all__ = [
    "hann_projection", "cycles_level", "synth_field", "field_class",
    "saturate", "pairwise_interaction_field", "array_transfer",
    "overlap_descriptor", "physical_validity_field",
]


def hann_projection(profile: np.ndarray, x: np.ndarray, k: float) -> float:
    """Hann-windowed continuous Fourier projection |Σ w g e^{-i2πkx}|².

    Evaluated at the exact requested k -- no nearest-bin reading.
    """
    g = np.asarray(profile, dtype=float)
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(g)
    g, x = g[finite], x[finite]
    w = np.hanning(g.size + 2)[1:-1]
    return float(abs(np.sum(w * g * np.exp(-1j * 2 * np.pi * k * x))) ** 2)


def cycles_level(lam: float, fov_um: float = 17.834048) -> str:
    """Frozen three-level measurability: HIGH >=2, LOW [1.2,2), else UNMEASURABLE."""
    n_cycles = fov_um / float(lam)
    if n_cycles >= 2.0:
        return "HIGH"
    if n_cycles >= 1.2:
        return "LOW"
    return "UNMEASURABLE"


def synth_field(profile: np.ndarray, x_profile: np.ndarray, h: float,
                phi: float, c: float, *, pixel_um: float = 0.5,
                roi_um: float = 80.0) -> np.ndarray:
    """Finite-array 2D field: z(x,y) = Σ_n a_n g(x - n h - φ), replicated
    along y on the 80 µm ROI (160 px @ 0.5 µm).  a_n = 1 + c(-1)^n."""
    n_grid = int(round(roi_um / pixel_um))
    x = (np.arange(n_grid) + 0.5) * pixel_um
    field = np.zeros((n_grid, n_grid), dtype=float)
    n_lines = int(np.floor((roi_um - phi) / h)) + 1
    amp = 1.0
    for n in range(n_lines):
        center = phi + n * h
        a_n = 1.0 + c * ((-1.0) ** n)
        lo = np.searchsorted(x, center - x_profile.max())
        hi = np.searchsorted(x, center - x_profile.min())
        field[:, lo:hi] += a_n * np.interp(
            x[lo:hi] - center, x_profile, profile, left=0.0, right=0.0)
        amp = a_n
    return field


def field_class(field: np.ndarray, *, h: float, pixel_um: float = 0.5,
                window_um: tuple[float, float] = (4.0, 32.0)) -> tuple[int, float]:
    """Same-pipeline peak extraction: 2D residual -> radial_spectrum (src.spectrum) ->
    the frozen 4–32 µm peak validity → r = λ_peak/h → interval assignment."""
    r = np.asarray(field, dtype=float)[None, :, :]
    r = r - np.median(r)
    out, _ = radial_spectrum(r, pixel_um, 24, 0.7, 160.0)
    long_rows = [{"dataset_index": 0, "bin": b,
                  "lambda_geo_um": float(out["lambda_geo_um"][b]),
                  "energy": float(out["energy"][0, b]),
                  "n_modes": int(out["n_modes"][0, b])}
                 for b in range(24)]
    peak = lambda_peak_4_32(pd.DataFrame(long_rows),
                                window_um=window_um, n_modes_min=20,
                                share_min=0.20)
    valid = bool(peak.loc[0, "lambda_peak_valid"])
    lam = float(peak.loc[0, "lambda_peak_4_32_um"]) if valid else np.nan
    ratio = lam / h if valid else np.nan
    cls = int(assign_class(np.array([ratio]), np.array([valid]))[0]) if valid \
        else CODE_INVALID
    return cls, lam


# --------------------------------------------------------------------------- #
# NEW (Phase 2.8): L2 saturation, L3b pairwise interaction, descriptors
# --------------------------------------------------------------------------- #
def saturate(s: np.ndarray, d_sat_um: float) -> np.ndarray:
    """F_beta(s) = D_sat * (1 - exp(-s / D_sat)): monotone on s >= 0,
    saturating at D_sat.  Frozen L2 family -- no alternative branch."""
    s = np.asarray(s, dtype=float)
    d = float(d_sat_um)
    require(d > 0, "D_sat must be positive")
    return d * (1.0 - np.exp(-s / d))


def _array_accumulate(profile: np.ndarray, x_profile: np.ndarray, h: float,
                      phi: float, *, pixel_um: float = 0.5,
                      roi_um: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
    """(sum_n g_n, sum_n g_n g_{n+1}) on the ROI grid.  Mirrors the frozen
    ``synth_field`` accumulation so that L1 = synth_field(...) equals
    pairwise_interaction_field(..., gamma_per_um=0.0) bit-for-bit."""
    n_grid = int(round(roi_um / pixel_um))
    x = (np.arange(n_grid) + 0.5) * pixel_um
    total = np.zeros((n_grid, n_grid), dtype=float)
    cross = np.zeros((n_grid, n_grid), dtype=float)
    n_lines = int(np.floor((roi_um - phi) / h)) + 1
    prev = None
    for n in range(n_lines):
        center = phi + n * h
        lo = np.searchsorted(x, center - x_profile.max())
        hi = np.searchsorted(x, center - x_profile.min())
        g_n = np.interp(x[lo:hi] - center, x_profile, profile,
                        left=0.0, right=0.0)
        tile = np.zeros_like(total)
        tile[:, lo:hi] = g_n[None, :]
        total += tile
        if prev is not None:
            cross += prev * tile
        prev = tile
    return total, cross


def pairwise_interaction_field(profile: np.ndarray, x_profile: np.ndarray,
                               h: float, phi: float, gamma_per_um: float,
                               *, pixel_um: float = 0.5,
                               roi_um: float = 80.0) -> np.ndarray:
    """L3b: z = sum_n g_n + gamma_per_um * sum_n g_n * g_{n+1}.

    Lowest-order nonlinear neighbour-interaction cross term; gamma carries
    um^-1 (the cross term is um^2).  gamma = 0 reproduces ``synth_field``
    exactly.  Candidates are screened by ``physical_validity_field`` BEFORE
    selection -- never post-hoc clipped."""
    total, cross = _array_accumulate(profile, x_profile, h, phi,
                                     pixel_um=pixel_um, roi_um=roi_um)
    return total + float(gamma_per_um) * cross


def array_transfer(k: np.ndarray, h: float, n_lines: int) -> np.ndarray:
    """|A_array(k)|^2, A_array(k) = sum_{n=0}^{N-1} exp(-i 2 pi k n h)
    (finite Dirichlet kernel).  Spectral descriptor only:
    S_z(k) = S_g(k) |A_array(k)|^2 -- NOT S_g^2 (v2.1 F7/B7)."""
    k = np.asarray(k, dtype=float)
    phase = 2.0 * np.pi * k * float(h)
    amp = np.sin(n_lines * phase / 2.0) / np.sin(phase / 2.0 + 1e-300)
    return np.abs(amp) ** 2


def overlap_descriptor(profile: np.ndarray, dx_um: float, h_um: float,
                       *, edge_px: int = 3) -> float:
    """Normalized overlap O(h) over the common support (v2.1 definition):

        O(h) = <g, g(. - h)> / sqrt(<g, g> <g(. - h), g(. - h)>)

    with g the baseline-corrected (removal-positive) profile and the common
    support trimmed by ``edge_px`` guard samples at each open end.  The
    shift is applied by linear interpolation (h need not be an integer
    multiple of dx).  Returns in [-1, 1]."""
    g = np.asarray(profile, dtype=float)
    finite = np.isfinite(g)
    require(finite.any(), "overlap descriptor: no finite profile samples")
    idx = np.flatnonzero(finite)
    i0, i1 = int(idx[0]), int(idx[-1])
    x = np.arange(i0, i1 + 1, dtype=float) * float(dx_um)
    base = g[i0:i1 + 1]
    base = base - base.mean()
    x_shift = x - float(h_um)
    shifted = np.interp(x_shift, x, base, left=np.nan, right=np.nan)
    lo, hi = edge_px, len(base) - edge_px
    if hi - lo < 8:
        raise ValueError("common support too small for overlap descriptor")
    a = base[lo:hi]
    b = shifted[lo:hi]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)


def physical_validity_field(field: np.ndarray, tol_um: float = 1e-9) -> bool:
    """True when the simulated field never dips below -tol removal depth
    (removal-positive convention).  Invalidation is a candidate-screening
    criterion, not a clipping licence."""
    return bool(np.nanmin(field) >= -float(tol_um))


# Versioned corrections: historical functions above retain frozen semantics.
__all__ += ["array_transfer_v2", "physical_validity_relative_v2", "phase_grid_v2"]


def array_transfer_v2(k: np.ndarray, h: float, n_lines: int) -> np.ndarray:
    """Stable finite-array power, including DC and reciprocal-lattice peaks."""
    k = np.asarray(k, dtype=float)
    if not np.isfinite(k).all() or not np.isfinite(h) or h <= 0:
        raise ValueError("finite frequencies and positive finite hatch required")
    if isinstance(n_lines, bool) or int(n_lines) != n_lines or n_lines < 1:
        raise ValueError("n_lines must be a positive integer")
    cycles = np.remainder(k * float(h), 1.0)
    amp = np.exp(-2j * np.pi * cycles[..., None] * np.arange(int(n_lines)))
    return np.abs(amp.sum(axis=-1)) ** 2


def physical_validity_relative_v2(field: np.ndarray, base: np.ndarray,
                                  tol_um: float = 0.01) -> dict:
    """Noise-aware admissibility: z >= min(base, 0) - tol on EVERY pixel.

    Preserve measured negative baseline noise; do not create negative removal
    where base is positive. Never clip. Apply also to held-out fields after
    selection, without consulting their observed classes.
    """
    z, s = np.asarray(field, float), np.asarray(base, float)
    if z.shape != s.shape or z.size == 0:
        raise ValueError("field and base must have matching nonempty shapes")
    if not np.isfinite(tol_um) or tol_um < 0:
        raise ValueError("tol_um must be finite and nonnegative")
    finite = bool(np.isfinite(z).all() and np.isfinite(s).all())
    margin = z - np.minimum(s, 0.0)
    return {"valid": bool(finite and np.all(margin >= -tol_um)),
            "finite": finite,
            "min_field_um": float(z.min()) if finite else None,
            "min_margin_um": float(margin.min()) if finite else None,
            "n_violating_pixels": int((margin < -tol_um).sum()) if finite
                                  else int(z.size)}


def phase_grid_v2(h: float, n_phases: int, level: str,
                  param: float | None = None) -> np.ndarray:
    """Complete phase period, retaining Phase 2.7 finite-array convention."""
    if not np.isfinite(h) or h <= 0 or int(n_phases) != n_phases or n_phases < 1:
        raise ValueError("positive hatch and integer phase count required")
    period = 2.0 * h if level == "L3a" and param not in (None, 0.0) else h
    return np.arange(int(n_phases), dtype=float) * (period / n_phases)
