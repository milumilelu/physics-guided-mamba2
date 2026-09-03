#!/usr/bin/env python3
"""Phase 2 experiment 03: local neighbourhood audit (Phase 2A-3).

For every ordinary pair (shared-height-source pairs and the 49/50 sentinel
pair excluded) this script compares process distance in two spaces (raw five
controls; physics-motivated reduced derived (proxy) features) against morphology
distance in six metrics (total residual + four DCT bands via the Gram RMSE
definition of Phase 1.5-04, and the robust-z descriptor space).

Readouts per (space x metric):
  - Type I-IV classification at display thresholds (D_proc <= P10 of ordinary
    pairs; D_morph >= P90 of ordinary pairs) - display only;
  - T_lambda = median[D_morph | D_proc <= P10] - the primary continuous
    statistic, less dependent on the P90 binarisation;
  - Type II counts, also restricted to formal-only pairs.

Two permutation nulls (process rows only; the pair mask is structural and
stays fixed):
  - within-session_role (PRIMARY): rows permuted inside each session_role
    block, preserving the session composition (v2 §10.2);
  - global row permutation (sensitivity).
p_perm = (1 + #{null >= obs}) / (1 + n_perm).

Seed offsets: within-null = seed+300, global-null = seed+400.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _lib as p2

EXPECTED = ["neighborhood_pairs.csv", "process_near_morph_far_pairs.csv",
            "process_far_morph_near_pairs.csv", "neighborhood_summary.csv",
            "threshold_perturbation.csv", "phase2A_gate_answers.md"]

FIELDS = ["total", "DCT_8_16", "DCT_16_32", "DCT_32_64", "DCT_64_inf"]
METRICS = [("total", "D_morph_total_um"), ("DCT_8_16", "D_morph_DCT_8_16_um"),
           ("DCT_16_32", "D_morph_DCT_16_32_um"),
           ("DCT_32_64", "D_morph_DCT_32_64_um"),
           ("DCT_64_inf", "D_morph_DCT_64_inf_um"),
           ("desc", "D_morph_desc")]


def _condensed(D: np.ndarray, iu: np.ndarray, iv: np.ndarray) -> np.ndarray:
    return D[iu, iv]


def main() -> int:
    cfg, quick = p2.load_config(__doc__)
    t0 = time.time()
    out = p2.output_dir(cfg, "local_structure")
    n_perm = int(cfg["neighborhood"]["type2_permutations_quick"]
                 if quick else cfg["neighborhood"]["type2_permutations"])
    p2.log(f"== Phase 2 / 03: local neighbourhood audit "
           f"(n_perm={n_perm} x 2 nulls) ==")
    frozen = p2.l15.load_frozen(cfg)
    man = p2.read_manifest(cfg, require_loco=True)
    inv = pd.read_csv(p2.output_dir(cfg, "instability")
                      / "instability_inventory.csv")

    # ---- distance matrices --------------------------------------------------
    fields = p2.l15.multiscale_fields(frozen["R"], cfg)
    dcols = cfg["instability"]["descriptor_cols"]
    Zd = p2.robust_z(inv[dcols].to_numpy(float))
    Dm = {name: (p2.pairwise_gram_rmse(fields[f]) if f != "desc" else
                 np.sqrt(((Zd[:, None, :] - Zd[None, :, :]) ** 2).sum(-1)))
          for f, name in METRICS}
    Z_raw = p2.zscore(man[p2.PROC_RAW_COLS].to_numpy(float))
    Z_phys = p2.zscore(man[p2.PROC_PHYS_COLS].to_numpy(float))
    spaces = {"raw": Z_raw, "phys": Z_phys}

    ia, ib = p2.l15.sentinel_rows(man, cfg)
    iu, iv = p2.l15.ordinary_pair_mask(man, ia, ib)
    n_pairs = len(iu)
    p2.log(f"  ordinary pairs: {n_pairs} (of {200 * 199 // 2})")
    formal_pair = ((man["session_role"].to_numpy()[iu] == "formal")
                   & (man["session_role"].to_numpy()[iv] == "formal"))

    morph_ord = {name: _condensed(Dm[name], iu, iv) for _, name in METRICS}
    morph_thr = {name: float(np.quantile(v, cfg["neighborhood"]["morph_far_quantile"]))
                 for name, v in morph_ord.items()}
    sentinel_D = {"total": None, "DCT_8_16": None, "DCT_16_32": None,
                  "DCT_32_64": None, "DCT_64_inf": None, "desc": None}
    sent_tab = pd.read_csv(p2.l15.REPO
                           / "outputs/phase1_5/sentinel_multiscale_table.csv")
    for _, r in sent_tab.iterrows():
        if r["scale"] in sentinel_D:
            sentinel_D[r["scale"]] = float(r["sentinel_D_um"])

    proc_ord = {sp: _condensed(np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2)
                                       .sum(-1)), iu, iv)
                for sp, Z in spaces.items()}
    q_near = float(cfg["neighborhood"]["proc_near_quantile"])

    def _stats(dp: np.ndarray, dm: dict, formal_only: bool = False):
        """Per-metric (thr, T_lambda, type-II count) for one D_proc draw."""
        thr = float(np.quantile(dp, q_near))
        near = dp <= thr
        res = {}
        for _, name in METRICS:
            v = dm[name]
            if formal_only:
                near_m = near & formal_pair
                t = float(np.median(v[near_m])) if near_m.any() else np.nan
                cnt = int(np.count_nonzero(near_m & (v >= morph_thr[name])))
            else:
                t = float(np.median(v[near]))
                cnt = int(np.count_nonzero(near & (v >= morph_thr[name])))
            res[name] = (thr, t, cnt)
        return res

    obs = {sp: _stats(dp, morph_ord) for sp, dp in proc_ord.items()}
    obs_formal = {sp: _stats(dp, morph_ord, formal_only=True)
                  for sp, dp in proc_ord.items()}

    # ---- permutation nulls ---------------------------------------------------
    roles = man["session_role"].to_numpy()
    blocks = [np.flatnonzero(roles == r) for r in
              ("formal", "pass_main", "pass_supplement")]
    seed = int(cfg["random_seed"])
    nulls = {"within": {sp: {m: {"T": [], "II": [], "T_f": [], "II_f": []}
                             for _, m in METRICS} for sp in spaces},
             "global": {sp: {m: {"T": [], "II": [], "T_f": [], "II_f": []}
                             for _, m in METRICS} for sp in spaces}}
    for scheme, seed_off in (("within", 300), ("global", 400)):
        rng = np.random.default_rng(seed + seed_off)
        for _ in range(n_perm):
            perm = np.arange(200)
            if scheme == "within":
                for blk in blocks:
                    perm[blk] = blk[rng.permutation(len(blk))]
            else:
                perm = rng.permutation(200)
            dp = {sp: _condensed(np.sqrt(((Z[perm][:, None, :]
                                          - Z[perm][None, :, :]) ** 2)
                                         .sum(-1)), iu, iv)
                  for sp, Z in spaces.items()}
            for sp in spaces:
                st = _stats(dp[sp], morph_ord)
                st_f = _stats(dp[sp], morph_ord, formal_only=True)
                for _, name in METRICS:
                    d = nulls[scheme][sp][name]
                    d["T"].append(st[name][1])
                    d["II"].append(st[name][2])
                    d["T_f"].append(st_f[name][1])
                    d["II_f"].append(st_f[name][2])
        p2.log(f"  null [{scheme}] done ({n_perm} perms)")

    def _pval(null_list, obs_val, ge=True):
        arr = np.asarray([x for x in null_list if np.isfinite(x)])
        if not np.isfinite(obs_val):
            return np.nan
        c = int(np.count_nonzero(arr >= obs_val)) if ge else \
            int(np.count_nonzero(arr <= obs_val))
        return (1 + c) / (1 + len(arr))

    # ---- summary table -------------------------------------------------------
    rows = []
    for sp in spaces:
        for _, name in METRICS:
            thr, t_obs, cnt = obs[sp][name]
            _, t_f_obs, cnt_f = obs_formal[sp][name]
            nw = nulls["within"][sp][name]
            gg = nulls["global"][sp][name]
            dm = morph_ord[name]
            near = proc_ord[sp] <= thr
            far = dm >= morph_thr[name]
            rows.append({
                "space": sp, "metric": name, "n_pairs": n_pairs,
                "proc_thr_p10": thr, "morph_thr_p90": morph_thr[name],
                "n_near": int(near.sum()),
                "T_lambda": t_obs,
                "type_I": int(np.count_nonzero(near & ~far)),
                "type_II": cnt,
                "type_III": int(np.count_nonzero(~near & ~far)),
                "type_IV": int(np.count_nonzero(~near & far)),
                "p_perm_T_within": _pval(nw["T"], t_obs),
                "p_perm_type2_within": _pval(nw["II"], cnt),
                "p_perm_T_global": _pval(gg["T"], t_obs),
                "p_perm_type2_global": _pval(gg["II"], cnt),
                "formal_only_type2": cnt_f,
                "formal_only_T_lambda": t_f_obs,
                "formal_only_p_type2_within": _pval(nw["II_f"], cnt_f),
                "formal_only_p_T_within": _pval(nw["T_f"], t_f_obs),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "neighborhood_summary.csv", index=False)
    p2.log("  wrote neighborhood_summary.csv")

    # ---- wide pairs table ----------------------------------------------------
    pairs = pd.DataFrame({"i": iu, "j": iv,
                          "D_proc_raw": proc_ord["raw"],
                          "D_proc_phys": proc_ord["phys"]})
    for f, name in METRICS:
        pairs[name] = morph_ord[name]
        if sentinel_D[f] is not None:
            pairs[f"D_over_sentinel_{f}"] = p2.sentinel_normalize(
                morph_ord[name], sentinel_D[f])
    for sp in spaces:
        for _, name in METRICS:
            thr = obs[sp][name][0]
            near = proc_ord[sp] <= thr
            far = morph_ord[name] >= morph_thr[name]
            typ = np.where(near & far, "II",
                           np.where(near & ~far, "I",
                                    np.where(~near & ~far, "III", "IV")))
            pairs[f"type_{sp}_{name}"] = typ
    pairs.to_csv(out / "neighborhood_pairs.csv", index=False)
    p2.log(f"  wrote neighborhood_pairs.csv ({len(pairs)} rows)")

    type_cols = [c for c in pairs.columns if c.startswith("type_")]
    ii = pairs[(pairs[type_cols] == "II").any(axis=1)]
    ii.to_csv(out / "process_near_morph_far_pairs.csv", index=False)
    iii = pairs[(pairs[type_cols] == "III").any(axis=1)]
    iii.to_csv(out / "process_far_morph_near_pairs.csv", index=False)
    p2.log(f"  Type II rows (any space x metric): {len(ii)}; "
           f"Type III rows: {len(iii)}")

    # ---- threshold perturbation (Route-U robustness, 细则 §17) --------------
    # within-session null recomputed at P5/P95, P10/P90 and P15/P85 so the
    # Type-II / T_lambda readout is not anchored to one arbitrary threshold.
    thresh = ((0.05, 0.95), (0.10, 0.90), (0.15, 0.85))
    pert_obs = {}
    for qn, qf in thresh:
        for sp, dp0 in proc_ord.items():
            near = dp0 <= np.quantile(dp0, qn)
            for _, name in METRICS:
                dm = morph_ord[name]
                far = dm >= np.quantile(dm, qf)
                pert_obs[(sp, qn, name)] = (
                    float(np.median(dm[near])),
                    int(np.count_nonzero(near & far)))
    pert_null = {(sp, qn, name, stat): []
                 for sp in spaces for qn, _ in thresh for _, name in METRICS
                 for stat in ("T", "II")}
    rng_p = np.random.default_rng(seed + 500)
    for _ in range(n_perm):
        perm = np.arange(200)
        for blk in blocks:
            perm[blk] = blk[rng_p.permutation(len(blk))]
        for sp, Z in spaces.items():
            Dp = np.sqrt(((Z[perm][:, None, :] - Z[perm][None, :, :]) ** 2)
                         .sum(-1))
            dp = Dp[iu, iv]
            for qn, qf in thresh:
                near = dp <= np.quantile(dp, qn)
                for _, name in METRICS:
                    dm = morph_ord[name]
                    far = dm >= np.quantile(dm, qf)
                    pert_null[(sp, qn, name, "T")].append(
                        float(np.median(dm[near])))
                    pert_null[(sp, qn, name, "II")].append(
                        int(np.count_nonzero(near & far)))
    pert_rows = []
    for (sp, qn, name), (t_obs, c_obs) in sorted(pert_obs.items()):
        qf = dict(thresh)[qn]
        pert_rows.append({
            "space": sp, "metric": name, "q_near": qn, "q_far": qf,
            "T_lambda": t_obs, "type_II": c_obs,
            "p_perm_T_within": _pval(pert_null[(sp, qn, name, "T")], t_obs),
            "p_perm_type2_within": _pval(pert_null[(sp, qn, name, "II")],
                                         c_obs)})
    pd.DataFrame(pert_rows).to_csv(out / "threshold_perturbation.csv",
                                   index=False)
    p2.log("  wrote threshold_perturbation.csv "
           "(P5/P95, P10/P90, P15/P85 within-null)")

    # ---- gate answers template ------------------------------------------------
    rho_sq = spearmanr(inv["loco_total_pc1_deg"], inv["Sq_um"]).statistic
    rho_pv = spearmanr(inv["loco_total_pc1_deg"],
                       inv["peak_to_valley_p98p2_um"]).statistic
    best = summary.loc[summary["metric"] == "D_morph_total_um"]
    lines = [
        "# Phase 2A gate answers",
        "",
        "> 自动填数 by 03;结论行必须由 reviewer 人工填写后 gate 才算关闭(细则 §16)。",
        "",
        "## Q1 高 leverage 是否主要由 artifact 驱动?",
        f"- 自动证据: repair>0 样本 {int((inv['repair_pixel_count'] > 0).sum())}/200;"
        f" plane_rmse 中位 {man['plane_rmse_um'].median():.3f} um;"
        f" repair 最大连通域中位 {inv['repair_largest_component_px'].median():.0f}px。",
        "- 待两轮盲评(02 的 manual_review.csv)。",
        "- **人工结论:【待填写】**",
        "",
        "## Q2 高 leverage 是否集中在某类真实形貌结构?",
        "- 自动证据: 盲评 morphology_pattern 分布(待人工)。",
        "- **人工结论:【待填写】**",
        "",
        "## Q3 高 leverage 是否只是连续幅度尾部?",
        f"- Spearman(loco_total_pc1, Sq) = {rho_sq:.3f};"
        f" Spearman(loco_total_pc1, peak_to_valley) = {rho_pv:.3f}。",
        "- **人工结论:【待填写】**",
        "",
        "## Q4 process-near / morphology-far(Type II)是否真实存在?",
        "",
        "total 残差口径(primary continuous statistic T_lambda 与 Type II 计数):",
    ]
    for sp in spaces:
        r = best[best["space"] == sp].iloc[0]
        lines += [
            f"- [{sp}] T_lambda={r['T_lambda']:.3f} um "
            f"(within-null p={r['p_perm_T_within']:.3f}, "
            f"global-null p={r['p_perm_T_global']:.3f}); "
            f"TypeII={int(r['type_II'])} "
            f"(within-null p={r['p_perm_type2_within']:.3f}, "
            f"global-null p={r['p_perm_type2_global']:.3f}); "
            f"formal-only TypeII={int(r['formal_only_type2'])} "
            f"(p={r['formal_only_p_type2_within']:.3f})",
        ]
    lines += [
        "",
        f"- 全部 12 个 (space x metric) 组合见 neighborhood_summary.csv;"
        f" Type II 行明细见 process_near_morph_far_pairs.csv。",
        "- 判读规则: 以 within-session null 为主,global null 与 formal-only "
        "口径方向须并列呈现;Route U 还需通过 P15/P85 扰动(09/细则 §17)。",
        "- **人工结论:【待填写】**",
        "",
    ]
    gate_path = out / "phase2A_gate_answers.md"
    if gate_path.exists():
        # The canonical gate record is human/reviewer-maintained once closed;
        # never clobber it with the blank template on a rerun.
        p2.log("  phase2A_gate_answers.md already present (canonical gate "
               "record) — NOT overwritten; auto evidence lives in "
               "neighborhood_summary.csv / threshold_perturbation.csv")
    else:
        (out / "phase2A_gate_answers.md").write_text("\n".join(lines),
                                                     encoding="utf-8")
    missing = [f for f in EXPECTED if not (out / f).exists()]
    p2.require(not missing, f"missing outputs: {missing}")
    p2.log(f"03 done in {time.time() - t0:.1f}s; all {len(EXPECTED)} outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
