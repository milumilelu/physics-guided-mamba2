#!/usr/bin/env python3
"""Phase 2.5 Task 13: pseudo-pass spectral redistribution.

Cross-sectional pseudo-trajectories only (15 matched bases x N=1..4;
supplement 10 bases x N=5,6 as an INDEPENDENT check). Question: is there a
shared ILR-balance redistribution direction across matched process
conditions? The global exact sign-flip test enumerates ALL 2^15 / 2^10 sign
configurations; p_exact = #{T_null >= T_obs} / 2^B with NO Monte-Carlo
correction (the observed all-+1 configuration is inside the enumeration
space, 细则 §0.16). Coordinate-wise exact tests get Holm correction within
each step. N4->5 is session-confounded and refused (require).

Depth association (Δd vs Δz Spearman) is secondary and cross-sectional only.
No turning-cosine / reversal / dynamics language anywhere (Phase 1.5R
withdrawal + 规划 §40).

Seed offsets: none (exact enumeration).
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import _lib as p25

EXPECTED = ["pass_composition_table.csv", "pass_ilr_step_table.csv",
            "pass_step_global_test.csv", "pass_step_coordinate_tests.csv",
            "pass_depth_spectrum_association.csv",
            "pass_composition_trajectory.png", "pass_ilr_shift.png",
            "pass_sign_consistency.png", "README.md"]

STEPS = [(1, 2), (2, 3), (3, 4)]

README = """# pseudo-pass spectral redistribution (Task 13)

- Data: cross-sectional pseudo-trajectories (15 matched bases x N=1..4 from
  pass_main; 10 supplement bases x N=5..6 as an independent check).
  N4->5 is session-confounded and refused.
- `pass_step_global_test.csv`: exact sign-flip test on the mean ILR step
  vector per step — p_exact = count/2^B (full enumeration, no +1 correction).
- `pass_step_coordinate_tests.csv`: coordinate-wise exact two-sided tests
  with Holm correction within each step.
- `pass_depth_spectrum_association.csv`: Spearman(delta depth, delta z) —
  cross-sectional association only.
- All titles/labels say "cross-sectional pseudo-trajectory"; no dynamics,
  oscillation or reversal language anywhere.
"""


def _holm(pvals: list) -> list:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, p[idx] * (m - rank)))
        adj[idx] = running
    return adj.tolist()


def main() -> int:
    cfg, quick = p25.load_config(__doc__)
    t0 = time.time()
    out = p25.output_dir(cfg, "pseudopass")
    p25.log("== Phase 2.5 / 13: pseudo-pass spectral redistribution ==")
    man = p25.read_phase2_manifest(cfg)
    ilr = pd.read_csv(p25.output_dir(cfg, "spectral_composition")
                      / "ilr_coordinates.csv")
    comp = pd.read_csv(p25.output_dir(cfg, "spectral_composition")
                       / "spectral_composition.csv")
    zmap = ilr.set_index("dataset_index")
    pmap = comp.set_index("dataset_index")

    traj = man[man.session_role == "pass_main"]
    groups = sorted(traj["base_condition_group"].unique())
    p25.require(len(groups) == 15, f"pass_main bases {len(groups)} != 15")
    depth = man["median_depth_um"].to_numpy()

    def _collect(rows_mask, role):
        mats = []
        for g in groups if role == "main" else \
                sorted(man[man.session_role == "pass_supplement"]
                       ["base_condition_group"].unique()):
            sub = man[(man["base_condition_group"] == g)
                      & rows_mask].sort_values("pass_count")
            p25.require(len(sub) >= 2, f"base {g} has <2 passes")
            z = zmap.loc[sub["dataset_index"].to_numpy()].to_numpy()
            pp = pmap.loc[sub["dataset_index"].to_numpy()]
            counts = sub["pass_count"].to_numpy()
            mats.append((g, sub["dataset_index"].to_numpy(), z, pp, counts))
        return mats

    main_mats = _collect(man.session_role == "pass_main", "main")
    sup_mats = _collect(man.session_role == "pass_supplement", "supp")
    p25.log(f"  bases: main={len(main_mats)} (N1-4), supplement={len(sup_mats)} "
            "(N5-6 independent check)")

    comp_rows = []
    for g, idxs, z, pp, counts in main_mats + sup_mats:
        for r in range(len(idxs)):
            comp_rows.append({"base_condition_group": g,
                              "dataset_index": int(idxs[r]),
                              "pass_count": int(counts[r]),
                              "median_depth_um": float(depth[idxs[r]]),
                              **{f"p_{b}": float(pp.iloc[r][f"p_{b}"])
                                 for b in p25.ILR_BANDS},
                              **{f"ilr_z{j + 1}": float(z[r, j])
                                 for j in range(4)}})
    comp_table = pd.DataFrame(comp_rows).sort_values(
        ["base_condition_group", "pass_count"])
    comp_table.to_csv(out / "pass_composition_table.csv", index=False)

    # ---- ILR steps + exact tests --------------------------------------------
    step_rows, coord_rows, assoc_rows, ilr_step_rows = [], [], [], [],
    sign_png = []
    arms = [("N1-4", main_mats, STEPS), ("N5-6", sup_mats, [(5, 6)])]
    for arm, mats, steps in arms:
        for s_from, s_to in steps:
            dz = []
            dd = []
            for g, idxs, z, pp, counts in mats:
                c_from = np.flatnonzero(counts == s_from)
                c_to = np.flatnonzero(counts == s_to)
                if len(c_from) != 1 or len(c_to) != 1:
                    continue
                i_from, i_to = int(c_from[0]), int(c_to[0])
                dz.append(z[i_to] - z[i_from])
                dd.append(float(depth[idxs[i_to]] - depth[idxs[i_from]]))
            if not dz:
                continue
            dz = np.asarray(dz)
            p25.require_no_n4_to_5([(s_from, s_to)])
            for (g, idxs, z, pp, counts), row_dz, row_dd in zip(
                    [m for m in mats
                     if (int((m[4] == s_from).sum()) == 1
                         and int((m[4] == s_to).sum()) == 1)], dz, dd):
                ilr_step_rows.append({
                    "arm": arm, "base_condition_group": g,
                    "step": f"{s_from}->{s_to}",
                    **{f"d_z{j + 1}": float(row_dz[j]) for j in range(4)},
                    "d_depth_um": float(row_dd)})
            res = p25.exact_signflip_test(dz)
            holm = _holm([c["p_exact_two_sided"]
                          for c in res["coordinates"]])
            step_global = {"arm": arm, "step": f"{s_from}->{s_to}",
                           "n_bases": len(dz),
                           "T_obs": res["T_obs"],
                           "p_exact_global": res["p_exact_global"],
                           "n_configurations": res["n_configurations"]}
            step_rows.append(step_global)
            for c, h in zip(res["coordinates"], holm):
                coord_rows.append({"arm": arm, "step": f"{s_from}->{s_to}",
                                   "balance": f"z{c['coordinate'] + 1}",
                                   "mean_dz": c["mean_dz"],
                                   "p_exact_two_sided":
                                       c["p_exact_two_sided"],
                                   "p_holm_within_step": h})
            dd_arr = np.asarray(dd)
            for j in range(4):
                rho = spearmanr(dd_arr, dz[:, j]).statistic
                assoc_rows.append({"arm": arm,
                                   "step": f"{s_from}->{s_to}",
                                   "balance": f"z{j + 1}",
                                   "spearman_delta_d_vs_dz": float(rho)})
            sign_png.append((arm, f"{s_from}->{s_to}", dz))
            p25.log(f"  [{arm} {s_from}->{s_to}] T={res['T_obs']:.4f} "
                    f"p_exact={res['p_exact_global']:.5f} "
                    f"(n={len(dz)} bases)")

    pd.DataFrame(step_rows).to_csv(out / "pass_step_global_test.csv",
                                   index=False)
    pd.DataFrame(ilr_step_rows).to_csv(out / "pass_ilr_step_table.csv",
                                       index=False)
    pd.DataFrame(coord_rows).to_csv(out / "pass_step_coordinate_tests.csv",
                                    index=False)
    pd.DataFrame(assoc_rows).to_csv(out / "pass_depth_spectrum_association.csv",
                                    index=False)

    # ---- figures -------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    for ax, b in zip(axes, p25.ILR_BANDS):
        for g in groups:
            sub = comp_table[(comp_table.base_condition_group == g)
                             & (comp_table.pass_count <= 4)]
            ax.plot(sub["pass_count"], sub[f"p_{b}"], "-o", ms=3, lw=0.7,
                    alpha=0.6)
        ax.set_title(f"p_{b}", fontsize=9)
        ax.set_xlabel("N (pass count)")
    fig.suptitle("Composition across cross-sectional pseudo-trajectories "
                 "(15 bases, N=1-4)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "pass_composition_trajectory.png",
                dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for j, ax in enumerate(axes):
        for g in groups:
            sub = comp_table[(comp_table.base_condition_group == g)
                             & (comp_table.pass_count <= 4)]
            ax.plot(sub["pass_count"], sub[f"ilr_z{j + 1}"], "-o", ms=3,
                    lw=0.7, alpha=0.6)
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_title(f"ilr_z{j + 1}", fontsize=9)
        ax.set_xlabel("N")
    fig.suptitle("ILR balances across cross-sectional pseudo-trajectories",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "pass_ilr_shift.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ypos = []
    ylabels = []
    y = 0
    for arm, step, dz in sign_png:
        for j in range(4):
            vals = dz[:, j]
            frac_pos = float((vals > 0).mean())
            ax.barh(y, frac_pos - 0.5, left=0.5, color="C3", alpha=0.8)
            ax.barh(y, -(1 - frac_pos - 0.5), left=0.5, color="C0",
                    alpha=0.8)
            ypos.append(y)
            ylabels.append(f"{arm} {step} z{j + 1}")
            y += 1
    ax.axvline(0.5, color="0.4", lw=0.8)
    ax.set_yticks(ypos, ylabels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of bases with d_z > 0 (red right / blue left of 0.5)")
    ax.set_title("Sign consistency across matched bases "
                 "(cross-sectional pseudo-trajectory)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "pass_sign_consistency.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    (out / "README.md").write_text(README, encoding="utf-8")
    missing = [f for f in EXPECTED if not (out / f).exists()]
    p25.require(not missing, f"missing outputs: {missing}")
    p25.log(f"13 done in {time.time() - t0:.1f}s; all {len(EXPECTED)} outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
