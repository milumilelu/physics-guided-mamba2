#!/usr/bin/env python3
"""Task SL-04 (细则 §7) -- lambda ratio test + G-SL2 (hatch-related periodicity).

口径分工（§0.18，冻结）:
  - H2 主证据 = **lambda_peak_4_32**（有效性双条件: bin n_modes >= 20 AND 峰 bin
    持窗内能量 >= 0.20）。centroid（lambda_star_4_32）是 H1/H3 的主口径，在这里
    只作为并排的 sensitivity 臂，**不替代** G-SL2 主判定。
  - 语言红线（§14）: 整数倍关系一律称 "hatch-related periodic / integer-multiple
    scale"，全文不得出现 harmonic 一词。

统计口径（§0.19，冻结）:
  - 置换单位 = unique (session_id, base_condition_group)（formal 120 单行单位、
    pass 15 base、supplement 10 base），置换在 session 块内部进行，单位全体行
    一起带走 —— 逐行 shuffle 会把同 base 的 N1..N4 当成独立 h 指派，人为扩大
    有效样本量。
  - p = (1 + #{A_null >= A_obs}) / (1 + n_perm)；peak 与 centroid 各做各的 null。

G-SL2: A_obs >= 0.40 AND p <= 0.05 -> SUPPORTED；仅其一 -> PARTIAL；均否 ->
NOT_SUPPORTED。若 lambda_peak_valid 比例 < 0.5 -> 最高 PARTIAL（峰证据覆盖不足；
"宽谱无峰"不得冒充周期证据）。

EXPECTED outputs:
    outputs/phase2_6/scale_bridge/lambda_over_hatch.csv
    outputs/phase2_6/scale_bridge/lambda_over_width.csv
    outputs/phase2_6/scale_bridge/overlap_metrics.csv
    outputs/phase2_6/scale_bridge/shuffled_h_null.csv
    outputs/phase2_6/summary/gsl2_evaluation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import _lib as p26  # noqa: E402

EXPECTED = [
    "outputs/phase2_6/scale_bridge/lambda_over_hatch.csv",
    "outputs/phase2_6/scale_bridge/lambda_over_width.csv",
    "outputs/phase2_6/scale_bridge/overlap_metrics.csv",
    "outputs/phase2_6/scale_bridge/shuffled_h_null.csv",
    "outputs/phase2_6/summary/gsl2_evaluation.json",
]

TARGETS_OVERLAP = ["p_8_16", "A2_8_16", "angular_entropy_8_16", "ilr_z2"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def integer_distance(ratios: np.ndarray) -> np.ndarray:
    """d_int = min_{m in {1,2,3}} |r - m| (上位规划 §13; NaN-preserving)."""
    out = np.full(np.shape(ratios), np.nan, dtype=float)
    finite = np.isfinite(ratios)
    if finite.any():
        values = np.asarray(ratios, dtype=float)[finite]
        stacked = np.abs(values[:, None] - np.array([1.0, 2.0, 3.0])[None, :])
        out[finite] = stacked.min(axis=1)
    return out


def alignment_fraction(d_int: np.ndarray, valid: np.ndarray,
                       tolerance: float) -> float:
    """A = #{d_int <= tol} / n_valid (observed or null)."""
    usable = np.asarray(valid, dtype=bool) & np.isfinite(d_int)
    if not usable.any():
        return float("nan")
    return float(np.mean(np.asarray(d_int, dtype=float)[usable] <= tolerance))


def shuffled_h_null(frame: pd.DataFrame, lambda_values: np.ndarray,
                    valid: np.ndarray, *, unit_columns: tuple[str, ...],
                    base_seed: int, n_perm: int, tolerance: float
                    ) -> tuple[np.ndarray, float, float]:
    """Block-structured shuffled-h null (§0.19). Returns (A_null, A_obs, p).

    The lambda values are held fixed and only the hatch assignment moves: the
    null asks whether the observed lambda/h ratios sit closer to 1/2/3 than a
    DOE-structure-preserving reassignment of hatch would produce.
    """
    hatch = frame["hatch_spacing_um"].to_numpy(dtype=float)
    p26.require(bool(np.isfinite(hatch).all()),
                "HARD ASSERTION FAILED: rectangle hatch must never be NA")
    d_obs = integer_distance(lambda_values / hatch)
    a_obs = alignment_fraction(d_obs, valid, tolerance)

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shuffled = p26.shuffle_h_by_block(
            frame, unit_columns=unit_columns,
            seed=base_seed + i).to_numpy(dtype=float)
        d_null = integer_distance(lambda_values / shuffled)
        null[i] = alignment_fraction(d_null, valid, tolerance)

    p_value = float((1.0 + int(np.sum(null >= a_obs))) / (1.0 + n_perm))
    return null, a_obs, p_value


# --------------------------------------------------------------------------- #
def main() -> int:
    cfg, quick = p26.load_config(__doc__)
    seed = int(cfg["meta"]["random_seed"])
    ratio = cfg["ratio_test"]
    tolerance = float(ratio["d_int_tolerance"])
    a_obs_min = float(ratio["a_obs_min"])
    p_max = float(ratio["p_max"])
    n_perm = int(ratio["n_perm"]) if not quick else 200
    perm_seed = seed + int(cfg["seeds"]["permutation_offset"])
    units = tuple(ratio["shuffle_unit"])

    scale = p26.output_dir(cfg, "scale_bridge")
    summary_dir = p26.output_dir(cfg, "summary")

    # ---- inputs ----------------------------------------------------------- #
    match_path = scale / "morphology_scale_match.csv"
    p26.require(match_path.exists(),
                "HARD ASSERTION FAILED: run Task 18 before Task 19 "
                f"(missing {match_path})")
    match = pd.read_csv(match_path, encoding="utf-8-sig")

    manifest = pd.read_csv(REPO / cfg["paths"]["phase2_manifest"],
                           encoding="utf-8-sig")
    radial = pd.read_csv(REPO / cfg["paths"]["p25_radial_long_csv"],
                         encoding="utf-8-sig")

    # ---- T13: lambda recompute must reproduce Task 18 ---------------------- #
    lam_cfg = cfg["lambda_star"]
    re_star = p26.lambda_star_4_32(
        radial, window_um=tuple(lam_cfg["window_um"]),
        guard=float(lam_cfg["min_band_energy_fraction"]))
    re_peak = p26.lambda_peak_4_32(
        radial, window_um=tuple(lam_cfg["window_um"]),
        n_modes_min=int(lam_cfg["peak_n_modes_min"]),
        share_min=float(lam_cfg["peak_min_energy_share_in_window"]))
    check = (match[["dataset_index", "lambda_star_4_32_um",
                    "lambda_star_valid", "lambda_peak_4_32_um",
                    "lambda_peak_valid"]]
             .merge(re_star, on="dataset_index", suffixes=("_t18", "_re"))
             .merge(re_peak, on="dataset_index",
                    suffixes=("_t18", "_re")))
    both_star = check["lambda_star_valid_t18"].astype(bool) & check[
        "lambda_star_valid_re"].astype(bool)
    delta_star = (check.loc[both_star, "lambda_star_4_32_um_t18"]
                  - check.loc[both_star, "lambda_star_4_32_um_re"]).abs()
    p26.require(bool((delta_star <= 1e-9).all()),
                "HARD ASSERTION FAILED: Task 19 lambda_star recompute differs "
                f"from Task 18 (max |delta| = {float(delta_star.max()):.3e})")
    both_peak = check["lambda_peak_valid_t18"].astype(bool) & check[
        "lambda_peak_valid_re"].astype(bool)
    delta_peak = (check.loc[both_peak, "lambda_peak_4_32_um_t18"]
                  - check.loc[both_peak, "lambda_peak_4_32_um_re"]).abs()
    p26.require(bool((delta_peak <= 1e-9).all()),
                "HARD ASSERTION FAILED: Task 19 lambda_peak recompute differs "
                "from Task 18")
    p26.log(f"T13 recompute: lambda* max|delta|={float(delta_star.max()):.2e} | "
            f"lambda_peak max|delta|="
            f"{float(delta_peak.max()) if both_peak.any() else 0.0:.2e}")

    # ---- T14: hatch is never NA on the rectangle side ---------------------- #
    # Task 18 already carries the rectangle manifest columns through; only pull
    # in the ones it does not have (avoids duplicate *_mf columns).
    needed = ["session_id", "base_condition_group", "hatch_spacing_um"]
    missing = [column for column in needed if column not in match.columns]
    if missing:
        frame = match.merge(
            manifest[["dataset_index"] + missing], on="dataset_index")
    else:
        frame = match
    # Units must exist for the DOE-block shuffle (§0.19).
    p26.require(all(column in frame.columns for column in needed),
                "HARD ASSERTION FAILED: shuffle units (session_id, "
                "base_condition_group) or hatch_spacing_um missing from the "
                "scale-match table")
    p26.require(len(frame) == 200,
                f"HARD ASSERTION FAILED: scale-match rows {len(frame)} != 200")
    p26.require(bool(frame["hatch_spacing_um"].notna().all()),
                "HARD ASSERTION FAILED: hatch must be present for every "
                "rectangle sample (single-line rows carry h=NA and must never "
                "reach the r_h table)")

    # ---- r_h / r_h_peak table (all 200; h always defined) ------------------ #
    lam_star = frame["lambda_star_4_32_um"].to_numpy(dtype=float)
    lam_peak = frame["lambda_peak_4_32_um"].to_numpy(dtype=float)
    hatch = frame["hatch_spacing_um"].to_numpy(dtype=float)
    valid_star = frame["lambda_star_valid"].to_numpy(dtype=bool)
    valid_peak = frame["lambda_peak_valid"].to_numpy(dtype=bool)

    over_hatch = pd.DataFrame({
        "dataset_index": frame["dataset_index"].to_numpy(),
        "hatch_spacing_um": hatch,
        "lambda_star_4_32_um": lam_star,
        "lambda_star_valid": valid_star,
        "lambda_peak_4_32_um": lam_peak,
        "lambda_peak_valid": valid_peak,
        "r_h": lam_star / hatch,
        "r_h_peak": lam_peak / hatch,
        "d_int": integer_distance(lam_star / hatch),
        "d_int_peak": integer_distance(lam_peak / hatch),
    })
    over_hatch.to_csv(scale / "lambda_over_hatch.csv", index=False,
                      encoding="utf-8-sig")
    p26.log(f"r_h table: {len(over_hatch)} rows | lambda_star_valid="
            f"{int(valid_star.sum())} | lambda_peak_valid={int(valid_peak.sum())}")

    # ---- G-SL2 primary: peak statistic ------------------------------------- #
    a_obs_peak = alignment_fraction(over_hatch["d_int_peak"].to_numpy(float),
                                    valid_peak, tolerance)
    valid_peak_fraction = float(valid_peak.mean())
    p26.log(f"G-SL2 observed(peak): A_obs={a_obs_peak:.4f} on n_valid="
            f"{int(valid_peak.sum())} (valid fraction {valid_peak_fraction:.3f})")

    null_peak, a_obs_peak, p_peak = shuffled_h_null(
        frame, lam_peak, valid_peak, unit_columns=units, base_seed=perm_seed,
        n_perm=n_perm, tolerance=tolerance)
    p26.log(f"G-SL2 null(peak): A_null median="
            f"{float(np.median(null_peak)):.4f} | p={p_peak:.4f} "
            f"(n_perm={n_perm}, seed={perm_seed})")

    # ---- sensitivity arm: centroid statistic ------------------------------- #
    null_centroid, a_obs_centroid, p_centroid = shuffled_h_null(
        frame, lam_star, valid_star, unit_columns=units, base_seed=perm_seed,
        n_perm=n_perm, tolerance=tolerance)
    p26.log(f"G-SL2 sensitivity(centroid): A_obs={a_obs_centroid:.4f} | "
            f"p={p_centroid:.4f} -- parallel arm, not the primary verdict")

    pd.DataFrame({
        "statistic": (["lambda_peak_4_32"] * n_perm
                      + ["lambda_star_4_32"] * n_perm),
        "permutation": list(range(n_perm)) * 2,
        "A_null": np.concatenate([null_peak, null_centroid]),
    }).to_csv(scale / "shuffled_h_null.csv", index=False, encoding="utf-8-sig")

    # ---- r_W table (in-box 101 primary; out-of-box excluded, T14) ---------- #
    in_box = frame["in_box"].astype(bool).to_numpy()
    w_hat = frame["W_hat_um"].to_numpy(dtype=float)
    over_width = pd.DataFrame({
        "dataset_index": frame["dataset_index"].to_numpy(),
        "arm": np.where(in_box, "primary_in_box", "extrapolated_out_of_box"),
        "bridge_coverage": frame["bridge_coverage"].to_numpy(dtype=object),
        "W_hat_um": w_hat,
        "lambda_star_4_32_um": lam_star,
        "lambda_star_valid": valid_star,
        "r_W": lam_star / w_hat,
        "abs_r_W_minus_1": np.abs(lam_star / w_hat - 1.0),
    })
    over_width.to_csv(scale / "lambda_over_width.csv", index=False,
                      encoding="utf-8-sig")
    p26.require(
        not (over_width.loc[over_width["arm"] == "primary_in_box",
                            "bridge_coverage"] == "out_of_box").any(),
        "HARD ASSERTION FAILED: out_of_box samples leaked into the primary r_W arm")
    p26.log(f"r_W table: primary(in-box)={int(in_box.sum())} | "
            f"extrapolated={int((~in_box).sum())}")

    # ---- H1-side informational statistics (§7; NOT a gate) ----------------- #
    r_w = over_width["r_W"].to_numpy(dtype=float)
    usable = in_box & valid_star & np.isfinite(r_w)
    h1_side = {
        "note": ("H1-side r_W statistics on the in-box 101 arm. Descriptive "
                 "only: per 细则 §0.17 the H1 evidence order is exact-match "
                 "direct > in-box predicted > out-of-box, so these model-proxy "
                 "numbers carry the LOWEST priority and must not be reported "
                 "as a mechanism result."),
        "n_usable": int(usable.sum()),
        "median_r_W": float(np.median(r_w[usable])) if usable.any() else np.nan,
        "iqr_r_W": ([float(np.percentile(r_w[usable], 25)),
                     float(np.percentile(r_w[usable], 75))]
                    if usable.any() else [np.nan, np.nan]),
        "fraction_abs_r_W_minus_1_le_0.25": (
            float(np.mean(np.abs(r_w[usable] - 1.0) <= 0.25))
            if usable.any() else np.nan),
        "spearman_lambda_star_vs_W_hat": (
            float(spearmanr(lam_star[usable], w_hat[usable]).statistic)
            if int(usable.sum()) >= 3 else np.nan),
    }

    # ---- overlap metrics (核心图 6: eta_h vs Route P/T targets) ------------- #
    spectral = pd.read_csv(REPO / cfg["paths"]["p25_spectral_csv"],
                           encoding="utf-8-sig")
    directional = pd.read_csv(REPO / cfg["paths"]["p25_directional_csv"],
                              encoding="utf-8-sig")
    band = directional[directional["band"].astype(str) == "8_16"]
    overlap = frame[["dataset_index", "W_hat_um", "hatch_spacing_um", "eta_h",
                     "bridge_coverage", "in_box"]].merge(
        spectral[["dataset_index", "p_8_16"]], on="dataset_index").merge(
        band[["dataset_index", "A2", "angular_entropy"]].rename(
            columns={"A2": "A2_8_16",
                     "angular_entropy": "angular_entropy_8_16"}),
        on="dataset_index")
    if "ilr_z2" not in overlap.columns:
        ilr = pd.read_csv(REPO / cfg["paths"]["p25_ilr_csv"],
                          encoding="utf-8-sig")
        overlap = overlap.merge(ilr[["dataset_index", "ilr_z2"]],
                                on="dataset_index")
    overlap.to_csv(scale / "overlap_metrics.csv", index=False,
                   encoding="utf-8-sig")

    inbox_overlap = overlap[overlap["in_box"].astype(bool)]
    correlations = {}
    for target in TARGETS_OVERLAP:
        if target not in inbox_overlap.columns:
            continue
        pair = inbox_overlap[["eta_h", target]].dropna()
        if len(pair) >= 3:
            correlations[f"spearman_eta_h_vs_{target}"] = float(
                spearmanr(pair["eta_h"], pair[target]).statistic)
    p26.log("overlap (in-box 101) Spearman: "
            + (", ".join(f"{k.replace('spearman_eta_h_vs_', '')}={v:+.3f}"
                         for k, v in correlations.items()) or "n/a"))

    # ---- G-SL2 verdict ----------------------------------------------------- #
    coverage_shortfall = valid_peak_fraction < float(
        ratio["valid_fraction_peak_min"])
    if a_obs_peak >= a_obs_min and p_peak <= p_max:
        verdict = "SUPPORTED"
    elif a_obs_peak >= a_obs_min or p_peak <= p_max:
        verdict = "PARTIAL"
    else:
        verdict = "NOT_SUPPORTED"
    if coverage_shortfall and verdict == "SUPPORTED":
        verdict = "PARTIAL"

    gsl2 = {
        "gate": "G-SL2",
        "verdict": verdict,
        "primary_statistic": "lambda_peak_4_32",
        "A_obs_peak": a_obs_peak,
        "p_peak": p_peak,
        "n_valid_peak": int(valid_peak.sum()),
        "lambda_peak_valid_fraction": valid_peak_fraction,
        "sensitivity_centroid": {
            "A_obs_centroid": a_obs_centroid,
            "p_centroid": p_centroid,
            "n_valid_centroid": int(valid_star.sum()),
        },
        "thresholds": {"a_obs_min": a_obs_min, "p_max": p_max,
                       "d_int_tolerance": tolerance,
                       "valid_fraction_peak_min":
                           float(ratio["valid_fraction_peak_min"])},
        "null": {"n_perm": n_perm, "seed": perm_seed,
                 "shuffle_unit": list(units),
                 "A_null_peak_median": float(np.median(null_peak)),
                 "A_null_peak_p95": float(np.percentile(null_peak, 95)),
                 "A_null_centroid_median": float(np.median(null_centroid))},
        "peak_coverage_shortfall": bool(coverage_shortfall),
        "h1_side_informational": h1_side,
        "overlap_spearman_in_box": correlations,
        "interpretation_note": (
            "G-SL2 tests a hatch-related PERIODIC / integer-multiple scale "
            "only. Per 细则 §14 the word 'harmonic' is banned: in the strict "
            "Fourier sense the harmonics of h are h/n, not 2h or 3h. G-SL2 "
            "must be read jointly with G-SL3 before any H1/H2/H3 final "
            "judgement (上位规划 §17)."),
    }
    (summary_dir / "gsl2_evaluation.json").write_text(
        json.dumps(gsl2, ensure_ascii=False, indent=2), encoding="utf-8")

    p26.log(f"G-SL2 = {verdict} (A_obs={a_obs_peak:.4f}, p={p_peak:.4f}"
            + (", PEAK-COVERAGE SHORTFALL -> capped at PARTIAL"
               if coverage_shortfall else "") + ")")
    p26.log("Task 19 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
