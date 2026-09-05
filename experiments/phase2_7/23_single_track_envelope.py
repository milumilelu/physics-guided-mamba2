#!/usr/bin/env python3
"""Task 23 (G27-3): single-track envelope + finite-array observation model.

2.7r1 re-write: fixes all six implementation deviations identified in the
formal audit (81-line/32-phase/DOE-unit bootstrap/LOHO/direct-guard/
diagnostic output).  All scientific definitions frozen in 任务说明 v2.1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

# 2.7r2: load the phase2_7 library by explicit file location -- a bare
# `import _lib` is sys.path-order dependent and can be shadowed by another
# phase's _lib when this script is imported in-process (e.g. unit tests).
import importlib.util as _ilu  # noqa: E402

_spec_t23 = _ilu.spec_from_file_location(
    "phase2_7_lib_t23", Path(__file__).resolve().parent / "_lib.py")
p27 = _ilu.module_from_spec(_spec_t23)
_spec_t23.loader.exec_module(p27)
from src import data as sdata  # noqa: E402

EXPECTED = [
    "outputs/phase2_7/envelope/single_track_envelope.csv",
    "outputs/phase2_7/envelope/envelope_selection_compare.csv",
    "outputs/phase2_7/envelope/forward_model_diagnostic.csv",
    "outputs/phase2_7/envelope/bootstrap_delta_tv.csv",
    "outputs/phase2_7/envelope/envelope_selection_compare.csv",
    "outputs/phase2_7/envelope/phase_grid_sensitivity.csv",
    "outputs/phase2_7/summary/gsl27_3_evaluation.json",
]
# 2.7r2: forward_model_simulation.csv removed -- the frozen file was a stale
# r0 leftover (27 profiles, 16 phases) the r1 code never wrote; the live
# simulation record is forward_model_diagnostic.csv.


def weighted_loho_tv(q_obs: dict, q_m: dict, c_grid: list, n_h_all: dict,
                     h_levels: list) -> tuple[float, dict]:
    """LOHO evaluation with the frozen WEIGHTED statistic
    TV_w = sum_h (n_h/N) * TV_h(q_obs[h], q_M[(h, c*_held)]).

    The per-held c* selection statistic (unweighted sum over train h) is
    unchanged from r1.  2.7r2: the EVALUATION aggregation is weighted --
    r1 returned the unweighted macro mean over h, so the main delta and the
    bootstrap delta were different statistics."""
    c_assign = {}
    tv_w = 0.0
    for held in h_levels:
        train = [h for h in h_levels if h != held]
        c_star = min(c_grid, key=lambda c: sum(
            p27.tv(q_obs[h], q_m[(h, c)]) for h in train))
        c_assign[held] = c_star
        tv_w += (n_h_all[held] / 200) * p27.tv(q_obs[held],
                                               q_m[(held, c_star)])
    return float(tv_w), c_assign


def doe_unit_labels(manifest) -> "pd.Series":
    """Frozen bootstrap unit = (session_id, base_condition_group) per row
    (2.7r2; r1 used session only)."""
    key = (manifest["session_id"].astype(str) + "~~"
           + manifest["base_condition_group"].astype(str))
    return pd.factorize(key.to_numpy())[0]


def doe_stratum_units(manifest: "pd.DataFrame", h: float) -> dict:
    """Units per h x session stratum (resampling pool for the bootstrap)."""
    sub = manifest.loc[manifest["hatch_spacing_um"] == h]
    return {s: grp["_doe_unit"].unique()
            for s, grp in sub.groupby("session_id", sort=False)}


def main() -> int:
    cfg, quick = p27.load_config(__doc__)
    g3 = cfg["g27_3"]
    seed = int(cfg["meta"]["random_seed"])
    out = p27.output_dir(cfg, "envelope")
    summary_dir = p27.output_dir(cfg, "summary")

    geometry = pd.read_csv(REPO / cfg["paths"]["single_line_geometry"],
                           encoding="utf-8-sig")
    labels = pd.read_csv(REPO / cfg["paths"]["geometry_qa_labels"],
                         encoding="utf-8-sig")
    line_manifest = pd.read_csv(REPO / cfg["paths"]["single_line_manifest"],
                                encoding="utf-8-sig")
    view = pd.read_csv(REPO / cfg["paths"]["line_view_manifest"],
                       encoding="utf-8-sig").set_index("measurement_id")
    match = pd.read_csv(REPO / cfg["paths"]["direct_bridge"],
                        encoding="utf-8-sig")
    manifest = pd.read_csv(REPO / cfg["paths"]["phase2_manifest"])
    peak_valid_table = pd.read_csv(REPO / cfg["paths"]["lambda_over_hatch"],
                                   encoding="utf-8-sig").set_index("dataset_index")

    # 2.7r2: extraction moved to the canonical shared builder (src.data);
    # revisions: section selection uses plateau_stable_run MEMBERSHIP FLAGS
    # (bridged shallow positions no longer re-enter the mean profile) and
    # out-of-FOV lateral positions are fixed at 0 depth (no-material
    # convention; NaN previously propagated into the synthesized fields).
    lib = sdata.build_line_profile_library(
        {k: cfg["paths"][k] for k in ("line_cag", "line_view_manifest",
                                      "single_line_geometry",
                                      "single_line_manifest",
                                      "geometry_qa_labels")},
        lateral_samples=64, dy_um=0.278657, section_step_um=2.0,
        edge_frac_max=float(g3["edge_frac_max"]))
    profiles = lib["profiles"]
    population = lib["population"]
    h_levels = sorted(manifest["hatch_spacing_um"].unique().tolist())
    p27.log(f"Task 23 start | quick={quick} | population={len(population)}")

    # ---- envelope candidates (h_levels × m) -------------------------------- #
    v_pos = p27.lateral_positions(64, 0.278657)
    candidate_rows = []
    for h in h_levels:
        for m in (1, 2, 3):
            lam = m * h
            k = 1.0 / lam
            readings = [p27.hann_projection(
                profiles[lid]["profile"], v_pos, k)
                for lid in sorted(profiles)
                if profiles[lid]["suitable"]]
            candidate_rows.append({
                "h": h, "m": m, "lambda_um": lam, "k_per_um": k,
                "confidence": p27.cycles_level(lam),
                "mean_S_g": float(np.mean(readings)) if readings else np.nan,
                "n_lines_read": len(readings)})
    envelope = pd.DataFrame(candidate_rows)
    envelope.to_csv(out / "single_track_envelope.csv", index=False,
                    encoding="utf-8-sig")

    # ---- 3A: exact-match d_i guard ----------------------------------------- #
    match_cond = match[match["W_source"] == "estimable"].copy()
    _cond = match_cond["condition"].str.split(":", expand=True).astype(int)
    match_cond[["pulse_duration_fs", "frequency_kHz", "pass_count",
                "velocity_mm_s"]] = _cond
    match_cond["dataset_index_list_parsed"] = match_cond[
        "dataset_index_list"].apply(lambda s: [int(x) for x in str(s).split(";")])
    d_rows = []
    for _, group in match_cond.groupby(["pulse_duration_fs", "frequency_kHz",
                                        "pass_count", "velocity_mm_s"]):
        tau, f, n_, v_ = group.iloc[0][
            ["pulse_duration_fs", "frequency_kHz", "pass_count",
             "velocity_mm_s"]].astype(float)
        line_ids = population[
            (population["pulse_duration_fs"] == tau)
            & (population["frequency_kHz"] == f)
            & (population["pass_count"] == n_)
            & (population["velocity_mm_s"] == v_)]["single_line_id"]
        if not len(line_ids):
            continue
        line_id = int(line_ids.iloc[0])
        info = profiles.get(line_id)
        if info is None or not info["suitable"]:
            continue
        prof = info["profile"]
        h_hatch = float(manifest.loc[manifest["dataset_index"].isin(
            group["dataset_index_list_parsed"].iloc[0]),
            "hatch_spacing_um"].iloc[0])
        k_h = 1.0 / h_hatch
        s_h = p27.hann_projection(prof, v_pos, k_h)
        s_2h = (p27.hann_projection(prof, v_pos, 0.5 * k_h)
                if p27.cycles_level(2 * h_hatch) != "UNMEASURABLE" else np.nan)
        for ds in group["dataset_index_list_parsed"].iloc[0]:
            vp = bool(peak_valid_table.loc[ds, "lambda_peak_valid"])
            c_obs = (int(p27.assign_class(
                np.array([peak_valid_table.loc[ds, "lambda_peak_4_32_um"]
                          / h_hatch]),
                np.array([vp]))[0])) if vp else 0
            d_rows.append({"dataset_index": int(ds), "h": h_hatch,
                           "c_obs": c_obs, "S_g_at_h": s_h,
                           "S_g_at_2h": s_2h, "line_id": line_id,
                           "measurable_2h":
                               p27.cycles_level(2 * h_hatch) != "UNMEASURABLE"})
    compare = pd.DataFrame(d_rows)
    compare.to_csv(out / "envelope_selection_compare.csv", index=False,
                   encoding="utf-8-sig")
    p27.log(f"3A: {len(compare)} samples across "
            f"{match_cond.shape[0]} estimable conditions")

    # ---- 3B: simulation with full frozen contract --------------------------- #
    library_profiles = [profiles[lid]["profile"]
                        for lid in sorted(profiles)
                        if profiles[lid]["suitable"]]
    n_lib = len(library_profiles)
    p27.require(n_lib >= 60, f"3B library too small: {n_lib}")
    c_grid = [float(c) for c in g3["c_grid"]]
    n_phases_final = int(g3["phases_final"])
    sim_diag_rows = []

    def simulate_classes(h: float, c: float, n_phases: int) -> np.ndarray:
        phi_grid = np.arange(n_phases, dtype=float) * (
            (h if c == 0.0 else 2 * h) / n_phases)
        classes = np.empty(len(phi_grid) * n_lib, dtype=int)
        idx = 0
        for phi in phi_grid:
            for prof in library_profiles:
                field = p27.synth_field(prof, v_pos, h, phi, c,
                                        pixel_um=g3["pixel_um"],
                                        roi_um=g3["roi_um"])
                cls, lam = p27.field_class(field, h=h)
                classes[idx] = cls
                sim_diag_rows.append(
                    {"h": h, "c": c, "phi": phi,
                     "lambda_peak": lam, "class_code": cls})
                idx += 1
        return classes

    # c-grid scan (LOHO) — use frozen 32 phases for ALL c values (v2.1 收口)
    q_M = {}
    for h in h_levels:
        for c in c_grid:
            cls_arr = simulate_classes(h, c, n_phases_final)
            q_M[(h, c)] = p27.q_distribution(cls_arr)
    sim_diag = pd.DataFrame(sim_diag_rows)
    sim_diag.to_csv(out / "forward_model_diagnostic.csv", index=False,
                    encoding="utf-8-sig")
    p27.log(f"3B simulation: {len(sim_diag)} runs, "
            f"family rate = "
            f"{(sim_diag['class_code'] >= 2).mean():.3f}")

    # ---- five-class q_obs_h (INCLUDING INVALID) ---------------------------- #
    q_obs_h5 = {}
    n_h_all = {}
    for h in h_levels:
        sel = manifest["hatch_spacing_um"] == h
        n_h_all[h] = int(sel.sum())
        classes_h = np.array([
            (int(p27.assign_class(
                np.array([peak_valid_table.loc[ds, "lambda_peak_4_32_um"]
                          / h]),
                np.array([bool(peak_valid_table.loc[ds, "lambda_peak_valid"])]))[0]))
            for ds in manifest.loc[sel, "dataset_index"]])
        q_obs_h5[h] = p27.q_distribution(classes_h)

    # ---- five-class TV + LOHO ---------------------------------------------- #
    def tv_w_for(c: float, q_obs: dict) -> float:
        return sum((n_h_all[h] / 200) * p27.tv(q_obs[h], q_M[(h, c)])
                   for h in h_levels)

    def loho(q_obs: dict) -> tuple[float, float, dict]:
        tv_w, c_assign = weighted_loho_tv(q_obs, q_M, c_grid, n_h_all,
                                          h_levels)
        return tv_w, tv_w, c_assign

    tv_const = tv_w_for(0.0, q_obs_h5)
    tv_alt_mean, _, c_assign_full = loho(q_obs_h5)
    tv_alt = tv_alt_mean  # 2.7r2: now the weighted TV_w (same statistic as tv_const)
    delta_tv = tv_const - tv_alt
    p27.log(f"3B: TV_w(const)={tv_const:.4f} | TV_w(LOHO alt)={tv_alt:.4f} "
            f"| ΔTV={delta_tv:.4f}")

    # ---- h-level consistency (frozen H_eval = {4,6,8,10}) ------------------- #
    h_eval_list = [h for h in g3["h_consistency"]["h_eval"]
                   if n_h_all[h] >= g3["h_consistency"]["min_n_obs"]]
    wins = sum(1 for h in h_eval_list
               if p27.tv(q_obs_h5[h], q_M[(h, 0.0)])
               > p27.tv(q_obs_h5[h], q_M[(h, float(c_assign_full[h]))]))
    n_evaluable = len(h_eval_list)

    # ---- DOE-unit bootstrap with per-replicate LOHO ------------------------- #
    B = int(g3["bootstrap"]["B"])
    boot_seed = seed + int(cfg["seeds"]["bootstrap"])
    # 2.7r2 fix: bootstrap unit = (session_id, base_condition_group) per the
    # frozen contract (r1 used session only), resampled within h x session
    # strata so the h composition of each replicate is preserved exactly.
    manifest = manifest.copy()
    manifest["_doe_unit"] = doe_unit_labels(manifest)
    delta_boot = np.empty(B)
    rng = np.random.default_rng(boot_seed)
    for b in range(B):
        q_boot = {}
        for h in h_levels:
            take_parts = []
            for units_in in doe_stratum_units(manifest, h).values():
                chosen = rng.choice(units_in, size=len(units_in),
                                    replace=True)
                for u in chosen:
                    take_parts.append(
                        manifest.index[manifest["_doe_unit"] == u]
                        .to_numpy())
            ds_take = manifest.loc[np.concatenate(take_parts),
                                   "dataset_index"].to_numpy()
            classes_h = np.array([
                (int(p27.assign_class(
                    np.array([peak_valid_table.loc[ds, "lambda_peak_4_32_um"]
                              / h]),
                    np.array([bool(peak_valid_table.loc[ds,
                                                    "lambda_peak_valid"])]))[0]))
                for ds in ds_take])
            q_boot[h] = p27.q_distribution(classes_h)
        # per-replicate LOHO
        for held in h_levels:
            train = [hh for hh in h_levels if hh != held]
            c_star = min(c_grid, key=lambda c: sum(
                p27.tv(q_boot[hh], q_M[(hh, c)]) for hh in train))
            q_boot[("_pred", held)] = q_M[(held, c_star)]
        tv_const_b = tv_w_for(0.0, q_boot)
        tv_alt_b = sum((n_h_all[h] / 200)
                       * p27.tv(q_boot[h], q_boot[("_pred", h)])
                       for h in h_levels)
        delta_boot[b] = tv_const_b - tv_alt_b
    ci_low = float(np.quantile(delta_boot, 0.025))
    p_boot = float((1 + int((delta_boot <= 0).sum())) / (1 + B))
    pd.DataFrame({"delta_tv": delta_boot}).to_csv(
        out / "bootstrap_delta_tv.csv", index=False)

    # ---- 3A d_i (2.7r2: own-envelope measurement -> measurement) ------------ #
    # Each exact-match condition's OWN measured profile synthesizes q_{C,i}
    # (c=0) and q_{P2,i} (global c* = mode of the LOHO per-held c*, tie ->
    # smallest -- the 2.7 细则 §0.6 c_guard convention), so d_i compares two
    # MEASUREMENTS of the same condition instead of borrowing the 81-line
    # population q_M (r1 behaviour).  Condition-level aggregation (mean over
    # the condition's rows) under the frozen measurability gate
    # (cycles_level(2h) != UNMEASURABLE); rows with an invalid observed
    # class cannot contribute (no c_obs) and are counted separately.
    c_vals = [c_assign_full[h] for h in h_levels]
    c_global = float(min(sorted(set(c_vals)),
                         key=lambda c: (-c_vals.count(c), c)))

    def own_profile_q(prof: np.ndarray, h: float, c: float) -> np.ndarray:
        codes = []
        for phi in np.arange(n_phases_final, dtype=float) * (
                (h if c == 0.0 else 2 * h) / n_phases_final):
            field = p27.synth_field(prof, v_pos, h, phi, c,
                                    pixel_um=g3["pixel_um"],
                                    roi_um=g3["roi_um"])
            codes.append(p27.field_class(field, h=h)[0])
        return p27.q_distribution(np.array(codes))

    d_i_values = []
    own_skipped_invalid = 0
    for (h_key, line_key), grp in compare.groupby(["h", "line_id"]):
        if not bool(grp["measurable_2h"].all()):
            continue
        info = profiles.get(int(line_key))
        if info is None or not info["suitable"]:
            continue
        valid_rows = grp[grp["c_obs"] != p27.CODE_INVALID]
        own_skipped_invalid += int((grp["c_obs"] == p27.CODE_INVALID).sum())
        if not len(valid_rows):
            continue
        h = float(h_key)
        q_C = own_profile_q(info["profile"], h, 0.0)
        q_P2 = own_profile_q(info["profile"], h, c_global)
        row_ds = [float(q_P2[int(r.c_obs)] - q_C[int(r.c_obs)])
                  for r in valid_rows.itertuples(index=False)]
        d_i_values.append(float(np.mean(row_ds)))
    n_eval = len(d_i_values)
    n_contradict = sum(1 for d in d_i_values if d < 0)

    # ---- phase-grid sensitivity (frozen 16/32/64) ---------------------------- #
    # Registered implementation choice: sensitivity of the FINAL arms
    # (constant c=0; period-2 at the global c*) to the phase
    # marginalization.  Classes are computed ONCE on the 64-phase grid; the
    # 32- and 16-phase grids are nested subsets (phi_j = j*h/16 = (2j)*h/32
    # = (4j)*h/64) and are aggregated by stride.
    phase_sens = []
    if not quick:
        cls_by = {}
        for h in h_levels:
            for c in (0.0, c_global):
                cls_by[(h, c)] = simulate_classes(h, c, 64)
        for n_ph in (16, 32, 64):
            sel = np.arange(0, 64, 64 // n_ph)
            tv_c = tv_p = 0.0
            for h in h_levels:
                q0 = p27.q_distribution(np.concatenate(
                    [cls_by[(h, 0.0)][i * n_lib:(i + 1) * n_lib]
                     for i in sel]))
                qg = p27.q_distribution(np.concatenate(
                    [cls_by[(h, c_global)][i * n_lib:(i + 1) * n_lib]
                     for i in sel]))
                tv_c += (n_h_all[h] / 200) * p27.tv(q_obs_h5[h], q0)
                tv_p += (n_h_all[h] / 200) * p27.tv(q_obs_h5[h], qg)
            phase_sens.append({"n_phases": n_ph, "tv_w_constant": tv_c,
                               "tv_w_period2_cglobal": tv_p,
                               "delta_tv": tv_c - tv_p})
        sim_diag = pd.DataFrame(sim_diag_rows)
        sim_diag.to_csv(out / "forward_model_diagnostic.csv", index=False,
                        encoding="utf-8-sig")
        pd.DataFrame(phase_sens).to_csv(
            out / "phase_grid_sensitivity.csv", index=False,
            encoding="utf-8-sig")
        p27.log("phase-grid sensitivity: " + "; ".join(
            f"{r['n_phases']}ph dTV={r['delta_tv']:.4f}"
            for r in phase_sens))

    # ---- verdict (frozen order) --------------------------------------------- #
    tv_thresh = g3["tv"]
    inadequate = (tv_const > tv_thresh["inadequate"]
                  and tv_alt > tv_thresh["inadequate"])
    if inadequate:
        verdict = "MODEL_INADEQUATE"
    elif delta_tv <= 0 and (tv_const <= tv_thresh["inadequate"]
                            or tv_alt <= tv_thresh["inadequate"]):
        verdict = "NOT_SUPPORTED"
    else:
        cond = (delta_tv >= tv_thresh["delta_min"]
                and tv_alt <= tv_thresh["period2_max"]
                and ci_low > 0
                and wins >= g3["h_consistency"]["min_wins"]
                and n_evaluable >= g3["h_consistency"]["min_evaluable"])
        verdict = "SUPPORTED" if cond else "PARTIAL"
    if (n_eval >= g3["d_guard"]["n_eval_min"]
            and n_contradict / n_eval > g3["d_guard"]["contradiction_frac"]
            and verdict == "SUPPORTED"):
        verdict = "PARTIAL"

    evaluation = {
        "revision": "2.7r2",
        "r2_fixes": [
            "LOHO evaluation statistic unified to the frozen weighted "
            "TV_w = sum_h (n_h/N) TV_h (r1 returned the macro mean, so the "
            "main delta and the bootstrap were different statistics)",
            "bootstrap unit = (session_id, base_condition_group), "
            "resampled within h x session strata (r1 used session only)",
            "3A d_i = own-envelope measurement->measurement (own profile "
            "synthesizes q_C and q_P2; r1 borrowed the 81-line population "
            "q_M)",
            "profile extraction: plateau membership FLAGS (no bridged "
            "shallow re-entry) + out-of-FOV lateral positions fixed at 0 "
            "depth",
            "phase-grid sensitivity 16/32/64 executed (final arms)",
            "stale forward_model_simulation.csv (r0 leftover) removed",
        ],
        "population": {"3A": "13 exact-match conditions (own envelope, "
                             "2.7r2 own-envelope d_i)",
                       "3B": f"{n_lib}-line library × "
                             f"{n_phases_final} phases"},
        "c_global_mode_of_loho": c_global,
        "tv_w_constant": tv_const,
        "tv_w_period2_loho": tv_alt,
        "delta_tv": delta_tv,
        "bootstrap_ci_low": ci_low,
        "p_boot": p_boot,
        "bootstrap_unit": "(session_id, base_condition_group); strata "
                          "h x session",
        "phase_grid_sensitivity": phase_sens,
        "h_consistency": {"n_evaluable": n_evaluable, "period2_wins": wins,
                          "h_eval_list": h_eval_list},
        "d_guard": {"n_eval": n_eval, "n_contradictions": n_contradict,
                    "frac": (n_contradict / n_eval if n_eval else np.nan),
                    "definition": "own-envelope condition-level d_i "
                                  "(2.7r2); invalid-observed rows skipped: "
                                  + str(own_skipped_invalid)},
        "thresholds": g3["tv"],
        "G_SL27_3": verdict,
        "note": ("linear array model family insufficient; material "
                 "nonlinearity is one candidate, not established"
                 if verdict == "MODEL_INADEQUATE" else ""),
        "language": "two-line / period-doubled spatial organization; "
                    "harmonic forbidden; MODEL_INADEQUATE does not establish "
                    "material nonlinearity",
    }
    (summary_dir / "gsl27_3_evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8")
    p27.log(f"G27-3 = {verdict} | TV_w(const)={tv_const:.4f} "
            f"| TV_w(LOHO)={tv_alt:.4f} | ΔTV={delta_tv:.4f} "
            f"(CI low {ci_low:.4f}, p={p_boot:.4f})")
    p27.log("Task 23 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
