"""Phase 2.7 shared library: m/OUT interval assignment, two-layer class
distributions, four/five-class TV, block-shuffle permutation p, Hann Fourier
projection, finite-array 2D synthesis (same Phase 2.5 spectrum pipeline),
LOHO period-2 selection, and the frozen G27-3 verdict order.

Loads the frozen Phase 2.6 library by explicit file location (module name
`phase2_6_lib_p27`; it chains p25/p2/l15).  No frozen implementation is
copied.  Binding spec: Phase2.7_落地执行细则.md (FROZEN) + 任务说明 v2.1 (FROZEN).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

_spec26 = importlib.util.spec_from_file_location(
    "phase2_6_lib_p27",
    Path(__file__).resolve().parents[1] / "phase2_6" / "_lib.py")
p26 = importlib.util.module_from_spec(_spec26)
_spec26.loader.exec_module(p26)
p25 = p26.p25
p2 = p26.p2
l15 = p26.l15

log = p26.log
require = p26.require

CLASS_NAMES = ["INVALID", "OUT", "m1", "m2", "m3"]
CODE_INVALID, CODE_OUT, CODE_M1, CODE_M2, CODE_M3 = 0, 1, 2, 3, 4
# frozen mutually exclusive intervals (任务说明 v2.1 blocker①)
INTERVALS = {1: (0.75, 1.25), 2: (1.75, 2.25), 3: (2.75, 3.25)}


def assign_class(r: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Five-class codes: 0=INVALID (not peak-valid or non-finite r), 1=OUT,
    2..4 = m1..m3.  Mutually exclusive closed intervals -- no tie logic."""
    r = np.asarray(r, dtype=float)
    out = np.full(r.shape, CODE_OUT, dtype=int)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(r)
    out[~valid] = CODE_INVALID
    for code, m in ((CODE_M1, 1), (CODE_M2, 2), (CODE_M3, 3)):
        lo, hi = INTERVALS[m]
        out[valid & (r >= lo) & (r <= hi)] = code
    return out


def q_distribution(classes: np.ndarray, codes=CLASS_NAMES) -> np.ndarray:
    """Five-class probability vector over `codes` order."""
    classes = np.asarray(classes, dtype=int)
    return np.array([(classes == code).mean() for code in range(5)])


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


def profile_suitable(profile: np.ndarray, *, edge_frac_max: float = 0.15
                     ) -> bool:
    """Profile edges must return to background (≤ edge_frac_max × D_max)."""
    g = np.asarray(profile, dtype=float)
    finite = np.isfinite(g)
    if not finite.any():
        return False
    d_max = float(np.nanmax(g))
    if d_max <= 0:
        return False
    edge = float(np.nanmax(np.abs(np.r_[g[:3], g[-3:]])))
    return edge <= edge_frac_max * d_max


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


def field_class(field: np.ndarray, *, pixel_um: float = 0.5,
                window_um: tuple[float, float] = (4.0, 32.0)) -> tuple[int, float]:
    """Same-pipeline peak extraction: 2D residual → p25.radial_spectrum →
    the frozen 4–32 µm peak validity → interval assignment."""
    r = np.asarray(field, dtype=float)[None, :, :]
    r = r - np.median(r)
    out, _ = p25.radial_spectrum(r, pixel_um, 24, 0.7, 160.0)
    long_rows = [{"bin": b, "lambda_geo_um": float(out["lambda_geo_um"][b]),
                  "energy": float(out["energy"][0, b]),
                  "n_modes": int(out["n_modes"][0, b])}
                 for b in range(24)]
    peak = p26.lambda_peak_4_32(pd.DataFrame(long_rows),
                                window_um=window_um, n_modes_min=20,
                                share_min=0.20)
    valid = bool(peak.loc[0, "lambda_peak_valid"])
    lam = float(peak.loc[0, "lambda_peak_4_32_um"]) if valid else np.nan
    cls = int(assign_class(np.array([lam]), np.array([valid]))[0]) if valid \
        else CODE_INVALID
    return cls, lam


def q2_aitchison_ilr(z_test: np.ndarray, z_pred: np.ndarray,
                     z_train: np.ndarray) -> float:
    """Task 12 ILR-coordinate-space Q2 (same definition as
    18_scale_bridge_model_compare.py; provenance: Phase 2.5 `12_` script)."""
    denom = float(((z_test - z_train.mean(axis=0)) ** 2).sum())
    if denom <= 0:
        return np.nan
    return float(1.0 - ((z_test - z_pred) ** 2).sum() / denom)


def logistic_slope(h: np.ndarray, is_m2: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty=None, max_iter=1000)
    model.fit(np.asarray(h, dtype=float).reshape(-1, 1),
              np.asarray(is_m2, dtype=int))
    return float(model.coef_[0][0])


def verdict_g27_3(tv_w_const: float, tv_w_p2: float, delta_tv: float,
                  ci_low: float, n_h_win: int, n_h_evaluable: int,
                  d_i_values: list[float], n_eval: int, *, thresholds: dict
                  ) -> dict:
    """Frozen v2.1 verdict order (mutually exclusive):
    MODEL_INADEQUATE → NOT_SUPPORTED → SUPPORTED/PARTIAL → d_i guard cap."""
    tv = thresholds["tv"]
    inadequate = (tv_w_const > tv["inadequate"]
                  and tv_w_p2 > tv["inadequate"])
    if inadequate:
        return {"G_SL3": "MODEL_INADEQUATE",
                "note": "linear array model family insufficient; material "
                        "nonlinearity is one candidate, not established"}
    if delta_tv <= 0 and (tv_w_const <= tv["inadequate"]
                          or tv_w_p2 <= tv["inadequate"]):
        verdict = "NOT_SUPPORTED"
    else:
        cond = (delta_tv >= tv["delta_min"]
                and tv_w_p2 <= tv["period2_max"]
                and ci_low > 0
                and n_h_win >= thresholds["h_consistency"]["min_wins"]
                and n_h_evaluable >= thresholds["h_consistency"]["min_evaluable"])
        verdict = "SUPPORTED" if cond else "PARTIAL"
    contradictions = int(sum(1 for d in d_i_values if d < 0))
    if (n_eval >= thresholds["d_guard"]["n_eval_min"]
            and contradictions / n_eval
            > thresholds["d_guard"]["contradiction_frac"]
            and verdict == "SUPPORTED"):
        verdict = "PARTIAL"
    return {"G_SL3": verdict,
            "n_hard_contradictions": contradictions, "n_eval": n_eval}
