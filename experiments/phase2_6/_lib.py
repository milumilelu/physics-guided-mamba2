"""Phase 2.6 shared library: single-line axis sampling, threshold widths,
lambda* descriptors, box membership, block-structured hatch shuffling.

Loads the frozen Phase 2.5 library by explicit file location (module name
`phase2_5_lib_p26`; it in turn loads Phase 2 and Phase 1.5).  No frozen
implementation is copied: composition/ILR/CV primitives are reached through
`p25.*` / `p2.*`.  Binding spec: Phase2.6_落地执行细则.md (FROZEN_EXECUTED).

Geometry convention (细则 §0.20): profiles are sampled directly along the
frozen line axis in ORIGINAL map coordinates -- no whole-map rotation.
  raw_centered = anchor + s*t_hat + v*n_hat
  t_hat = (cos theta, sin theta); n_hat = (-sin theta, cos theta)
with theta = theta_line_deg (identical convention to
`src/resampling.resample_to_canonical`: canonical u runs along t_hat,
canonical +v along n_hat) and anchor = (orientation_center_x_um,
orientation_center_y_um) from the frozen view manifest.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]

_spec25 = importlib.util.spec_from_file_location(
    "phase2_5_lib_p26",
    Path(__file__).resolve().parents[1] / "phase2_5" / "_lib.py")
p25 = importlib.util.module_from_spec(_spec25)
_spec25.loader.exec_module(p25)
p2 = p25.p2
l15 = p25.l15

log = p25.log
require = p25.require

WIDTH_Q_KEYS = ("W20", "W50", "W80")
Q_BY_KEY = {"W20": 0.2, "W50": 0.5, "W80": 0.8}


def load_config(description: str) -> tuple[dict, bool]:
    """Read `phase2_6_config.yaml` next to this file; honor `--quick`."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke option: only the first N groups")
    args, _ = parser.parse_known_args()
    cfg = yaml.safe_load((Path(__file__).resolve().parent
                          / "phase2_6_config.yaml").read_text(encoding="utf-8"))
    cfg["_output_root"] = (str(cfg["meta"]["quick_output_root"]) if args.quick
                           else "outputs/phase2_6")
    cfg["_quick"] = bool(args.quick)
    cfg["_limit"] = args.limit
    return cfg, quick


def output_dir(cfg: dict, sub: str = "") -> Path:
    path = REPO / cfg["_output_root"]
    if sub:
        path = path / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Line-axis frame and profile sampling (§0.20: direct axis sampling)
# --------------------------------------------------------------------------- #
def axis_frame(theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangent (along the line) and unit normal (lateral) in centered coords."""
    theta = np.deg2rad(float(theta_deg))
    t_hat = np.array([np.cos(theta), np.sin(theta)])
    n_hat = np.array([-np.sin(theta), np.cos(theta)])
    return t_hat, n_hat


def _pixel_indices(hm, centered_x: np.ndarray, centered_y: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Centered physical coords -> fractional pixel indices (same frame as
    `src.resampling.resample_to_canonical`)."""
    x_absolute = centered_x + hm.width_um / 2.0
    y_absolute = centered_y + hm.height_um / 2.0
    columns = (x_absolute - hm.x_um[0]) / hm.dx_um
    rows = (y_absolute - hm.y_um[0]) / hm.dy_um
    return rows, columns


def sample_profiles(depth: np.ndarray, valid: np.ndarray, hm,
                    theta_deg: float, anchor: tuple[float, float],
                    s_positions: np.ndarray, v_positions: np.ndarray,
                    *, order: int = 1, mask_weight_min: float = 0.99
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Sample perpendicular profiles along the frozen axis, mask-aware.

    Returns `(profiles, weights)` with shape `(n_s, n_v)`; out-of-FOV samples
    are NaN (they count as FOV censoring, never as real background).
    """
    t_hat, n_hat = axis_frame(theta_deg)
    s_grid, v_grid = np.meshgrid(np.asarray(s_positions, dtype=float),
                                 np.asarray(v_positions, dtype=float),
                                 indexing="ij")
    centered_x = anchor[0] + s_grid * t_hat[0] + v_grid * n_hat[0]
    centered_y = anchor[1] + s_grid * t_hat[1] + v_grid * n_hat[1]
    rows, columns = _pixel_indices(hm, centered_x, centered_y)
    numerator = ndimage.map_coordinates(
        np.where(valid, np.nan_to_num(depth, nan=0.0), 0.0),
        [rows, columns], order=order, mode="constant", cval=0.0,
        prefilter=order > 1)
    weight = ndimage.map_coordinates(
        valid.astype(float), [rows, columns], order=order,
        mode="constant", cval=0.0, prefilter=order > 1)
    profiles = np.full(weight.shape, np.nan, dtype=float)
    hit = weight >= mask_weight_min
    profiles[hit] = numerator[hit] / weight[hit]
    return profiles, weight


def lateral_positions(n_v: int, dy_um: float) -> np.ndarray:
    """Centered lateral sample positions matching the raw row centers."""
    return (np.arange(n_v, dtype=float) - (n_v - 1) / 2.0) * dy_um


# --------------------------------------------------------------------------- #
# Line detection / extent / stable region (§4.1)
# --------------------------------------------------------------------------- #
def detect_online_flags(profiles: np.ndarray, threshold_um: float,
                        min_profile_points: int) -> np.ndarray:
    """A position is on-line when >= min_profile_points valid pixels exceed
    the groove threshold (pilot rule D > k * sigma_ref)."""
    above = np.isfinite(profiles) & (profiles > threshold_um)
    return above.sum(axis=1) >= min_profile_points


def line_extent(s_scan: np.ndarray, online: np.ndarray, *,
                min_run_um: float, merge_gap_um: float) -> tuple[float, float]:
    """Longest merged on-line run; isolated detections are discarded."""
    step = float(np.min(np.diff(s_scan))) if s_scan.size > 1 else 1.0
    min_run = max(1, int(round(min_run_um / step)))
    runs: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(online):
        if flag and start is None:
            start = index
        if (not flag or index == online.size - 1) and start is not None:
            end = index if flag else index - 1
            runs.append((start, end))
            start = None
    if not runs:
        raise ValueError("no on-line positions detected")
    merged: list[list[int]] = []
    for run in runs:
        if merged and (s_scan[run[0]] - s_scan[merged[-1][1]]) <= merge_gap_um:
            merged[-1][1] = run[1]
        else:
            merged.append(list(run))
    longest = max(merged, key=lambda r: s_scan[r[1]] - s_scan[r[0]])
    if s_scan[longest[1]] - s_scan[longest[0]] < min_run_um:
        raise ValueError("longest on-line run shorter than min_run_um")
    return float(s_scan[longest[0]]), float(s_scan[longest[1]])


def stable_region(s_start: float, s_end: float, *, pad_low: float,
                  pad_high: float) -> tuple[float, float]:
    """Central (1 - pad_low - pad_high) fraction of the detected extent."""
    length = s_end - s_start
    return s_start + pad_low * length, s_end - pad_high * length


def section_positions(stable: tuple[float, float], step_um: float
                      ) -> np.ndarray:
    """2-um-spaced section centers inside the stable region."""
    lo, hi = stable
    count = int(np.floor((hi - lo) / step_um)) + 1
    positions = lo + step_um * np.arange(count, dtype=float)
    return positions[(positions >= lo) & (positions <= hi)]


# --------------------------------------------------------------------------- #
# Threshold widths / equivalent-area / descriptors (§4.2, frozen)
# --------------------------------------------------------------------------- #
def _run_boundaries(values: np.ndarray, positions: np.ndarray, run: tuple[int, int],
                    q: float, dv: float) -> tuple[float, float, bool, bool]:
    """Sub-pixel boundaries of one above-q run; censored = touches a profile
    end (index 0 / n-1) or is bounded by an out-of-FOV sample."""
    i, j = run
    n = values.size
    left_censored = (i == 0) or not np.isfinite(values[i - 1])
    right_censored = (j == n - 1) or not np.isfinite(values[j + 1])
    if left_censored:
        v_left = positions[i] - dv / 2.0
    else:
        v_left = positions[i] - (values[i] - q) * dv / (values[i] - values[i - 1])
    if right_censored:
        v_right = positions[j] + dv / 2.0
    else:
        v_right = positions[j] + (q - values[j]) * dv / (values[j + 1] - values[j])
    return v_left, v_right, left_censored or (i == 0), right_censored or (j == n - 1)


def section_features(profile: np.ndarray, v_positions: np.ndarray,
                     thresholds_q: tuple[float, ...], *,
                     affected_delta_um: float) -> dict:
    """All frozen §4.2 features for one perpendicular profile.

    `profile` may contain NaN (out-of-FOV).  d_n = D / D_max with z_bg = 0.
    """
    dv = float(np.mean(np.diff(v_positions)))
    positions = v_positions
    finite = np.isfinite(profile)
    out = {"n_valid_samples": int(finite.sum()), "D_max_um": np.nan}
    if not finite.any():
        out.update({key: np.nan for key in (
            "D_max_um", "W20_um", "W50_um", "W80_um", "n_runs_W20", "n_runs_W50",
            "n_runs_W80", "total_width_W20_um", "total_width_W50_um",
            "total_width_W80_um", "censored_W20", "censored_W50", "censored_W80",
            "A_remove_um2", "W_eq_um", "W_affected_um", "left_slope",
            "right_slope", "edge_asymmetry", "ridge_left_um", "ridge_right_um",
            "ridge_separation_um", "profile_skewness", "n_above_threshold")})
        return out
    values = np.where(finite, profile, np.nan)
    d_max = float(np.nanmax(values))
    out["D_max_um"] = d_max
    out["n_above_threshold"] = int(np.nansum(values > 0.0))
    positive = np.where(finite, np.maximum(profile, 0.0), 0.0)
    out["A_remove_um2"] = float(np.nansum(positive) * dv)
    out["W_eq_um"] = (out["A_remove_um2"] / d_max) if d_max > 0 else np.nan
    if d_max <= 0:
        for key in ("W20_um", "W50_um", "W80_um", "n_runs_W20", "n_runs_W50",
                    "n_runs_W80", "total_width_W20_um", "total_width_W50_um",
                    "total_width_W80_um", "censored_W20", "censored_W50",
                    "censored_W80", "W_affected_um", "left_slope", "right_slope",
                    "edge_asymmetry", "ridge_left_um", "ridge_right_um",
                    "ridge_separation_um", "profile_skewness"):
            out[key] = np.nan
        return out
    d_n = values / d_max
    for key, q in Q_BY_KEY.items():
        above = np.isfinite(d_n) & (d_n >= q)
        if not above.any():
            out[f"{key}_um"] = np.nan
            out[f"n_runs_{key}"] = 0
            out[f"total_width_{key}_um"] = 0.0
            out[f"censored_{key}"] = False
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([0], above.view(np.int8), [0]))))
        starts, stops = edges[0::2], edges[1::2]
        runs = [(int(a), int(b - 1)) for a, b in zip(starts, stops)]
        widths, censor_flags = [], []
        for run in runs:
            v_left, v_right, cen_l, cen_r = _run_boundaries(
                d_n, v_positions, run, q, dv)
            widths.append(v_right - v_left)
            censor_flags.append(bool(cen_l or cen_r))
        best = int(np.argmax(widths))
        out[f"{key}_um"] = float(widths[best])
        out[f"n_runs_{key}"] = len(runs)
        out[f"total_width_{key}_um"] = float(np.sum(widths))
        out[f"censored_{key}"] = bool(censor_flags[best])
    # affected width (secondary): longest run of |D| > delta beyond the groove
    delta = affected_delta_um
    signed = np.isfinite(values) & (np.abs(values) > delta)
    if signed.any():
        edges = np.flatnonzero(np.diff(np.concatenate(([0], signed.view(np.int8), [0]))))
        starts, stops = edges[0::2], edges[1::2]
        spans = [(positions[b - 1] - positions[a]) for a, b in zip(starts, stops)]
        out["W_affected_um"] = float(np.max(spans)) + dv
    else:
        out["W_affected_um"] = 0.0
    # groove-wall slopes around D_max, edge asymmetry, ridges
    k = int(np.nanargmax(values))
    best = None
    above50 = np.isfinite(d_n) & (d_n >= 0.5)
    if above50.any():
        edges = np.flatnonzero(np.diff(np.concatenate(([0], above50.view(np.int8), [0]))))
        starts, stops = edges[0::2], edges[1::2]
        runs = [(int(a), int(b - 1)) for a, b in zip(starts, stops)]
        best = max(runs, key=lambda r: r[1] - r[0])
    if best is not None and best[0] <= k <= best[1]:
        left_idx, right_idx = best
        out["edge_asymmetry"] = float(
            ((positions[right_idx] - positions[k]) - (positions[k] - positions[left_idx]))
            / max(positions[right_idx] - positions[left_idx], 1e-12))
        left_seg = slice(left_idx, k + 1)
        right_seg = slice(k, right_idx + 1)
        out["left_slope"] = _slope(positions[left_seg], values[left_seg])
        out["right_slope"] = _slope(positions[right_seg], values[right_seg])
        out["ridge_left_um"] = _ridge(values, positions, slice(0, left_idx))
        out["ridge_right_um"] = _ridge(values, positions, slice(right_idx + 1, values.size))
        ridge_sep = np.nan
        left_part = values[:left_idx]
        right_part = values[right_idx + 1:]
        left_trough = (float(-np.nanmin(left_part))
                       if left_part.size and np.isfinite(left_part).any() else 0.0)
        right_trough = (float(-np.nanmin(right_part))
                        if right_part.size and np.isfinite(right_part).any() else 0.0)
        if left_trough > 0 and right_trough > 0:
            l_arg = int(np.nanargmin(left_part))
            r_arg = right_idx + 1 + int(np.nanargmin(right_part))
            ridge_sep = abs(float(positions[r_arg]) - float(positions[l_arg]))
        out["ridge_separation_um"] = ridge_sep
    else:
        out["edge_asymmetry"] = np.nan
        out["left_slope"] = np.nan
        out["right_slope"] = np.nan
        out["ridge_left_um"] = _ridge(values, positions, slice(0, k))
        out["ridge_right_um"] = _ridge(values, positions, slice(k + 1, values.size))
        out["ridge_separation_um"] = np.nan
    finite_vals = values[finite]
    centered = finite_vals - finite_vals.mean()
    std = float(centered.std())
    out["profile_skewness"] = float(np.mean(centered**3) / std**3) if std > 0 else 0.0
    return out


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x, y = x[mask], y[mask]
    if np.ptp(x) <= 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def _ridge(values: np.ndarray, positions: np.ndarray, seg: slice) -> float:
    part = values[seg]
    if part.size == 0 or not np.isfinite(part).any():
        return 0.0
    return float(max(0.0, -np.nanmin(part)))


def aggregate_line(sections: pd.DataFrame, *, min_sections: int,
                   censored_frac_limit: float) -> dict:
    """Frozen line-level aggregates + width_identifiability (§4.3)."""
    used = sections[sections["n_above_threshold"] > 0].reset_index(drop=True)
    row: dict[str, float | str] = {
        "n_sections_total": int(len(sections)),
        "n_sections_used": int(len(used)),
    }
    for key in WIDTH_Q_KEYS:
        column = f"{key}_um"
        uncensored = used[~used[f"censored_{key}"].astype(bool)][column].dropna()
        all_values = used[column].dropna()
        row[f"median_{key}_um"] = float(uncensored.median()) if len(uncensored) else np.nan
        row[f"iqr_{key}_um"] = (float(uncensored.quantile(0.75) - uncensored.quantile(0.25))
                                if len(uncensored) else np.nan)
        row[f"p10_{key}_um"] = float(uncensored.quantile(0.10)) if len(uncensored) else np.nan
        row[f"p90_{key}_um"] = float(uncensored.quantile(0.90)) if len(uncensored) else np.nan
        row[f"censored_frac_{key}"] = (float(used[f"censored_{key}"].astype(bool).mean())
                                       if len(used) else np.nan)
        row[f"n_uncensored_sections_{key}"] = int(len(uncensored))
    median_w50 = row["median_W50_um"]
    iqr_w50 = row["iqr_W50_um"]
    row["CV_W50"] = (iqr_w50 / median_w50) if (np.isfinite(median_w50)
                                               and median_w50 > 0 and np.isfinite(iqr_w50)) else np.nan
    row["median_W_eq_um"] = float(used["W_eq_um"].dropna().median()) if len(used) else np.nan
    row["median_max_depth_um"] = float(used["D_max_um"].dropna().median()) if len(used) else np.nan
    if row["n_sections_used"] < min_sections:
        row["width_identifiability"] = "insufficient_sections"
    elif (np.isfinite(row["censored_frac_W50"])
          and row["censored_frac_W50"] > censored_frac_limit):
        row["width_identifiability"] = "right_censored"
    else:
        row["width_identifiability"] = "estimable"
    return row


# --------------------------------------------------------------------------- #
# Phase 2.5-side new descriptors (§0.2 / §0.18)
# --------------------------------------------------------------------------- #
def lambda_star_4_32(radial_long: pd.DataFrame, *, window_um: tuple[float, float],
                     guard: float) -> pd.DataFrame:
    """Energy-weighted geometric-mean wavelength inside [window), with NA guard.

    Input: `radial_spectrum_long.csv` (dataset_index, lambda_geo_um, energy,...).
    Output columns: lambda_star_4_32_um, lambda_star_valid, band_energy_fraction.
    """
    rows = []
    for index, frame in radial_long.groupby("dataset_index"):
        energy = frame["energy"].to_numpy(dtype=float)
        lam = frame["lambda_geo_um"].to_numpy(dtype=float)
        inside = (lam >= window_um[0]) & (lam < window_um[1])
        total = float(energy.sum())
        band = float(energy[inside].sum())
        fraction = band / total if total > 0 else 0.0
        valid = bool(inside.any() and total > 0 and fraction >= guard)
        value = (float(np.exp(np.sum(energy[inside] * np.log(lam[inside])) / band))
                 if valid else np.nan)
        rows.append({"dataset_index": int(index),
                     "band_energy_fraction_4_32": fraction,
                     "lambda_star_valid": valid,
                     "lambda_star_4_32_um": value})
    return pd.DataFrame(rows)


def lambda_peak_4_32(radial_long: pd.DataFrame, *, window_um: tuple[float, float],
                     n_modes_min: int, share_min: float) -> pd.DataFrame:
    """Restricted spectral peak with validity gates (H2 primary, §0.18).

    Bins with n_modes < n_modes_min are dropped from the peak search; the peak
    bin must additionally hold >= share_min of the window energy.
    """
    rows = []
    for index, frame in radial_long.groupby("dataset_index"):
        frame = frame.assign(
            lam=frame["lambda_geo_um"].to_numpy(dtype=float),
            e=frame["energy"].to_numpy(dtype=float),
            modes=frame["n_modes"].to_numpy(dtype=float))
        inside = (frame["lam"] >= window_um[0]) & (frame["lam"] < window_um[1])
        candidate = frame[inside & (frame["modes"] >= n_modes_min)]
        window_energy = float(frame.loc[inside, "e"].sum())
        if not len(candidate) or window_energy <= 0:
            rows.append({"dataset_index": int(index), "lambda_peak_valid": False,
                         "lambda_peak_4_32_um": np.nan,
                         "peak_energy_share_in_window": 0.0})
            continue
        top = candidate.loc[candidate["e"].idxmax()]
        share = float(top["e"]) / window_energy
        valid = bool(share >= share_min)
        rows.append({"dataset_index": int(index), "lambda_peak_valid": valid,
                     "lambda_peak_4_32_um": float(top["lam"]) if valid else np.nan,
                     "peak_energy_share_in_window": share})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Bridge helpers (§0.1 / §0.17 / §0.19)
# --------------------------------------------------------------------------- #
def in_box_mask(manifest: pd.DataFrame, box: dict) -> pd.Series:
    """Closed-interval box membership on (tau, f, v, N) (frozen 101/200)."""
    return (manifest["pulse_duration_fs"].between(*box["tau_fs"])
            & manifest["frequency_kHz"].between(*box["f_khz"])
            & manifest["velocity_mm_s"].between(*box["v_mm_s"])
            & manifest["pass_count"].between(*box["pass"]))


def condition_key(manifest: pd.DataFrame) -> pd.Series:
    return (manifest["pulse_duration_fs"].astype(str) + ":"
            + manifest["frequency_kHz"].astype(str) + ":"
            + manifest["pass_count"].astype(str) + ":"
            + manifest["velocity_mm_s"].astype(str))


def shuffle_h_by_block(manifest: pd.DataFrame, *, unit_columns: tuple[str, ...],
                       seed: int) -> pd.Series:
    """Permute the hatch assignment at the DOE assignment unit level (§0.19).

    Units are unique `unit_columns` tuples; hatch is constant within a unit
    (verified on the frozen manifest); permutation happens independently inside
    each session block so block sizes and within-unit sharing are preserved.
    """
    require((manifest.groupby(list(unit_columns))["hatch_spacing_um"]
             .nunique() == 1).all(),
            "HARD ASSERTION FAILED: hatch must be constant within a shuffle unit")
    rng = np.random.default_rng(seed)
    shuffled = manifest["hatch_spacing_um"].astype(float).copy()
    for _, block in manifest.groupby("session_id", sort=False):
        grouped = block.groupby(list(unit_columns), sort=False)
        unit_labels = list(grouped.groups.values())
        values = np.array([block.loc[labels[0], "hatch_spacing_um"]
                           for labels in unit_labels], dtype=float)
        order = rng.permutation(values.size)
        for labels, position in zip(unit_labels, order):
            shuffled.loc[labels] = values[position]
    return shuffled


# --------------------------------------------------------------------------- #
# Pilot reconciliation (§0.15)
# --------------------------------------------------------------------------- #
def reconcile_stable_region(my_frame: pd.DataFrame, pilot: pd.DataFrame, *,
                            groups: tuple[int, ...], tol_um: float
                            ) -> pd.DataFrame:
    """Section-level agreement between my stable flags and pilot flags.

    `my_frame` needs (加工顺序, s_center_um, stable_flag) rows at 1-um scan
    step; pilot rows are matched within `tol_um` on center-relative s.
    """
    records = []
    for group in groups:
        mine = my_frame[my_frame["加工顺序"] == group]
        theirs = pilot[pilot["加工顺序"] == group]
        if not len(mine) or not len(theirs):
            continue
        merged = pd.merge_asof(
            theirs.sort_values("s_um").rename(columns={"s_um": "s_pilot"}),
            mine.sort_values("s_center_um"),
            left_on="s_pilot", right_on="s_center_um",
            tolerance=tol_um, direction="nearest")
        merged = merged.dropna(subset=["stable_flag"])
        if not len(merged):
            continue
        agreement = float((merged["stable_flag"].astype(bool)
                           == merged["included_in_stable_region"].astype(bool)).mean())
        records.append({"加工顺序": group, "n_matched": int(len(merged)),
                        "n_unmatched_pilot": int(len(theirs) - len(merged)),
                        "agreement": agreement})
    return pd.DataFrame(records)
