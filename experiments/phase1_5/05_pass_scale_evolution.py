#!/usr/bin/env python3
"""Phase 1.5R experiment 05: morphology descriptors, per-scale pass-step
evolution (pseudo-trajectory / cross-sectional), and the
variability/repeatability summary.

1.5R: the previous "S(q) bootstrap stability" score is withdrawn; the summary
keeps across-sample SD/IQR and the 49/50 delta + percentile (repeatability).
Pass-step turning cosines for the two consecutive step pairs are reported
separately, with a trajectory-level bootstrap over the 15 base conditions.
"""

from __future__ import annotations

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import laplace
from scipy.stats import kurtosis, skew

import _lib

EXPECTED = ["morphology_descriptors.csv", "pass_step_stats.csv",
            "pass_scale_evolution.png", "variability_repeatability_summary.csv",
            "variability_repeatability_summary.png"]

PIXEL_UM = 0.5


def compute_descriptors(R: np.ndarray, dct_fields: dict) -> pd.DataFrame:
    n = R.shape[0]
    sq = np.sqrt(np.mean(R ** 2, axis=(1, 2)))
    sa = np.mean(np.abs(R), axis=(1, 2))
    ssk = np.zeros(n)
    sku = np.zeros(n)
    grad_rms = np.zeros(n)
    grad_x_rms = np.zeros(n)
    grad_y_rms = np.zeros(n)
    lap_rms = np.zeros(n)
    corr_len = np.zeros(n)
    pit_count = np.zeros(n, dtype=int)
    pit_depth = np.zeros(n)
    inv_e = 1.0 / np.e
    for i in range(n):
        Ri = R[i]
        ssk[i] = skew(Ri.ravel())
        sku[i] = kurtosis(Ri.ravel(), fisher=True)
        gy, gx = np.gradient(Ri, PIXEL_UM, PIXEL_UM)
        grad_x_rms[i] = np.sqrt(np.mean(gx ** 2))
        grad_y_rms[i] = np.sqrt(np.mean(gy ** 2))
        grad_rms[i] = np.sqrt(grad_x_rms[i] ** 2 + grad_y_rms[i] ** 2)
        lap_rms[i] = np.sqrt(np.mean(laplace(Ri) ** 2))
        F = np.fft.fft2(Ri)
        acf = np.fft.ifft2(F * np.conj(F)).real
        acf /= max(acf[0, 0], 1e-300)
        crosses = []
        for prof in (acf[:, 0], acf[0, :]):
            below = np.flatnonzero(prof[1:81] < inv_e)
            crosses.append((below[0] + 1) if below.size else 80)
        corr_len[i] = float(np.mean(crosses)) * PIXEL_UM
        mad = np.median(np.abs(Ri - np.median(Ri)))
        sig = 1.4826 * mad
        pit_count[i] = int(np.count_nonzero(Ri < -3.5 * sig))
        pit_depth[i] = float(-np.min(Ri))
    out = pd.DataFrame({
        "dataset_index": np.arange(n),
        "Sq_um": sq, "Sa_um": sa, "Ssk": ssk, "Sku": sku,
        "grad_rms_um_per_um": grad_rms,
        "aniso_gradx_over_y": grad_x_rms / np.maximum(grad_y_rms, 1e-300),
        "lap_rms_um": lap_rms, "corr_len_um": corr_len,
        "pit_count_per_roi": pit_count, "pit_depth_um": pit_depth,
    })
    var_R = np.maximum(np.mean(R ** 2, axis=(1, 2)), 1e-300)
    for name, fields in dct_fields.items():
        out[f"rms_{name}_um"] = np.sqrt(np.mean(fields ** 2, axis=(1, 2)))
        out[f"E_{name}_frac"] = np.mean(fields ** 2, axis=(1, 2)) / var_R
    return out


def main() -> int:
    t0 = time.time()
    cfg, quick = _lib.load_config(__doc__)
    out = _lib.output_dir(cfg)
    B = _lib.n_boot(cfg, quick)
    seed = int(cfg["random_seed"])
    _lib.log(f"== Phase 1.5R / 05: descriptors + pass pseudo-trajectory "
             f"evolution + variability/repeatability summary (B={B}) ==")
    frozen = _lib.load_frozen(cfg)
    man = frozen["man"]
    R = frozen["R"]
    dct_fields, _ = _lib.dct_band_fields(R, float(cfg["scales"]["pixel_um"]),
                                         cfg["scales"]["dct_bands_um"])

    desc = compute_descriptors(R, dct_fields)
    desc_out = desc.copy()
    desc_out.insert(1, "session_id", man["session_id"].to_numpy())
    desc_out.insert(2, "median_depth_um", man["median_depth_um"].to_numpy())
    desc_out.to_csv(out / "morphology_descriptors.csv", index=False)
    _lib.log(f"  wrote morphology_descriptors.csv ({desc_out.shape[1] - 1} "
             "quantities)")

    # ---- pass pseudo-trajectory evolution per DCT scale --------------------
    fields = {"total": _lib.to_2d(R)}
    fields.update({k: _lib.to_2d(v) for k, v in dct_fields.items()})
    traj = man[man["session_role"] == "pass_main"]
    groups = sorted(traj["design_group"].unique())
    _lib.require(len(groups) == 15, "trajectory count != 15")
    depth = man["median_depth_um"].to_numpy()
    rng = np.random.default_rng(seed + 99)

    def traj_step_stats(scale: str) -> dict:
        X = fields[scale]
        V = {s: [] for s in (1, 2, 3)}
        dd = {s: [] for s in (1, 2, 3)}
        cos12, cos23 = [], []
        for g in groups:
            gdf = traj[traj["design_group"] == g].sort_values("pass_count")
            idx = gdf["dataset_index"].to_numpy()
            for s in (1, 2, 3):
                V[s].append(X[idx[s]] - X[idx[s - 1]])
                dd[s].append(float(depth[idx[s]] - depth[idx[s - 1]]))
            def _cos(a, b):
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                return float(np.clip(a @ b / (na * nb), -1, 1)) \
                    if na > 0 and nb > 0 else np.nan
            cos12.append(_cos(V[1][-1], V[2][-1]))
            cos23.append(_cos(V[2][-1], V[3][-1]))
        return {s: np.asarray(V[s]) for s in (1, 2, 3)}, \
            {s: np.asarray(dd[s]) for s in (1, 2, 3)}, \
            np.asarray(cos12), np.asarray(cos23)

    traj_clusters = [np.array([i]) for i in range(15)]
    traj_boot = {f: _lib.build_resample_bank(traj_clusters, B,
                                             seed + 500 + i)
                 for i, f in enumerate(fields)}
    rows = []
    for si, (scale, X) in enumerate(fields.items()):
        Vs, dds, cos12, cos23 = traj_step_stats(scale)
        bank = traj_boot[scale]
        rms_q = {s: [] for s in (1, 2, 3)}
        c12_q, c23_q = [], []
        for idx_draw in bank:
            for s in (1, 2, 3):
                Vb = Vs[s][idx_draw]
                rms_q[s].append(float(np.median(np.sqrt(np.mean(Vb ** 2, axis=1)))))
            c12_q.append(float(np.nanmedian(cos12[idx_draw])))
            c23_q.append(float(np.nanmedian(cos23[idx_draw])))
        across = {s: [] for s in (1, 2, 3)}
        for s in (1, 2, 3):
            for i, j in itertools.combinations(range(15), 2):
                a, b = Vs[s][i], Vs[s][j]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 0 and nb > 0:
                    across[s].append(float(np.clip(a @ b / (na * nb), -1, 1)))
        for s in (1, 2, 3):
            q = np.percentile(rms_q[s], [25, 50, 75, 90, 95])
            rows.append((scale, s, *q,
                         float(np.nanmedian(cos12)), float(np.nanmedian(cos23)),
                         float(np.nanmedian(across[s])) if across[s] else np.nan,
                         float(np.median(dds[s]))))
        _lib.log(f"  [{scale}] step RMS Q50 (N1->2, 2->3, 3->4): "
                 + " ".join(f"{np.median([np.sqrt(np.mean(Vs[s][i] ** 2)) for i in range(15)]):.3f}"
                            for s in (1, 2, 3))
                 + " um | cos12/cos23 medians: "
                 + f"{np.nanmedian(cos12):.2f}/{np.nanmedian(cos23):.2f}"
                 + " | across-traj step cos: "
                 + " ".join(f"{'nan' if not across[s] else f'{np.median(across[s]):.2f}'}"
                            for s in (1, 2, 3)))
    pd.DataFrame(rows, columns=[
        "scale", "step", "step_rms_q25_um", "step_rms_q50_um",
        "step_rms_q75_um", "step_rms_q90_um", "step_rms_q95_um",
        "cos_step1_vs_2_q50", "cos_step2_vs_3_q50",
        "across_traj_same_step_cos_q50", "depth_step_median_um"
        ]).to_csv(out / "pass_step_stats.csv", index=False)
    _lib.log("  wrote pass_step_stats.csv")

    dpi = int(cfg["plot"]["dpi"])
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.0), dpi=dpi)
    colors = {"total": "black", "DCT_8_16": "tab:red", "DCT_16_32": "tab:purple",
              "DCT_32_64": "tab:brown", "DCT_64_inf": "tab:olive"}
    steps = (1, 2, 3)
    for scale in fields:
        sub = [r for r in rows if r[0] == scale]
        med = np.array([r[2] for r in sub])
        q25 = np.array([r[1] for r in sub])
        q75 = np.array([r[3] for r in sub])
        axes[0].plot(steps, med, "o-", color=colors[scale], lw=1.4, ms=4,
                     label=scale)
        axes[0].fill_between(steps, q25, q75, color=colors[scale], alpha=0.15)
    axes[0].set_xticks(steps, ["N1->2", "N2->3", "N3->4"])
    axes[0].set_ylabel("median step RMS [um] (Q50 line, Q25-Q75 band)")
    axes[0].set_title("Per-scale pass step magnitude\n"
                      "(trajectory-level bootstrap over 15 base conditions)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    pos = np.arange(len(fields))
    for i, scale in enumerate(fields):
        sub = [r for r in rows if r[0] == scale]
        axes[1].plot([i - 0.18], [sub[0][6]], "s", color=colors[scale], ms=6)
        axes[1].plot([i + 0.18], [sub[0][7]], "D", color=colors[scale], ms=5,
                     mfc="none")
    axes[1].set_xticks(pos, list(fields), fontsize=8, rotation=20)
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    axes[1].axhline(0.5, color="tab:red", ls=":", lw=1)
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[
        Line2D([], [], marker="s", ls="", color="0.3", label="cos step1 vs 2"),
        Line2D([], [], marker="D", ls="", mfc="none", color="0.3",
               label="cos step2 vs 3")], fontsize=8, loc="lower right")
    axes[1].set_ylabel("median consecutive-step direction cos")
    axes[1].set_title("Step-direction persistence (split by step pair)\n"
                      "pseudo-trajectory, cross-sectional")
    axes[1].grid(alpha=0.25, axis="y")
    dd_med, dd_q25, dd_q75 = [], [], []
    for s in (1, 2, 3):
        vals = []
        for g in groups:
            gdf = traj[traj["design_group"] == g].sort_values("pass_count")
            idx = gdf["dataset_index"].to_numpy()
            vals.append(float(depth[idx[s]] - depth[idx[s - 1]]))
        dd_med.append(float(np.median(vals)))
        dd_q25.append(float(np.percentile(vals, 25)))
        dd_q75.append(float(np.percentile(vals, 75)))
    dd_med, dd_q25, dd_q75 = map(np.asarray, (dd_med, dd_q25, dd_q75))
    axes[2].plot(steps, dd_med, "o-", color="tab:purple", lw=1.5, ms=5)
    axes[2].fill_between(steps, dd_q25, dd_q75, color="tab:purple", alpha=0.2)
    axes[2].set_xticks(steps, ["N1->2", "N2->3", "N3->4"])
    axes[2].set_ylabel("median depth step [um]")
    axes[2].set_title("Depth increment per pass step\n(pseudo-trajectory)")
    axes[2].grid(alpha=0.25)
    fig.suptitle("N=1-4 pass evolution by DCT scale — pseudo-trajectories "
                 "(cross-sectional, different positions per N; no dynamics "
                 "claimed)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "pass_scale_evolution.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote pass_scale_evolution.png")

    # ---- variability / repeatability summary -------------------------------
    i_a, i_b = _lib.sentinel_rows(man, cfg)
    pu, pv = _lib.ordinary_pair_mask(man, i_a, i_b)
    quantities: list[tuple[str, np.ndarray, str]] = [
        ("median_depth_um", man["median_depth_um"].to_numpy(), "um"),
        ("residual_Sq_um", man["residual_Sq_um"].to_numpy(), "um"),
        *[(f"rms_{k}_um", desc[f"rms_{k}_um"].to_numpy(), "um")
          for k in dct_fields],
        ("Sq_um", desc["Sq_um"].to_numpy(), "um"),
        ("Sa_um", desc["Sa_um"].to_numpy(), "um"),
        ("Ssk", desc["Ssk"].to_numpy(), "-"),
        ("Sku", desc["Sku"].to_numpy(), "-"),
        ("grad_rms", desc["grad_rms_um_per_um"].to_numpy(), "um/um"),
        ("aniso_x_over_y", desc["aniso_gradx_over_y"].to_numpy(), "-"),
        ("lap_rms_um", desc["lap_rms_um"].to_numpy(), "um"),
        ("corr_len_um", desc["corr_len_um"].to_numpy(), "um"),
        *[(f"E_{k}_frac", desc[f"E_{k}_frac"].to_numpy(), "-")
          for k in dct_fields],
        ("pit_count", desc["pit_count_per_roi"].to_numpy().astype(float), "px"),
        ("pit_depth_um", desc["pit_depth_um"].to_numpy(), "um"),
    ]
    map_rows = []
    for name, qv, unit in quantities:
        sd = float(np.std(qv))
        iqr = float(np.percentile(qv, 75) - np.percentile(qv, 25))
        diffs = np.abs(qv[pu] - qv[pv])
        sent = abs(float(qv[i_a]) - float(qv[i_b]))
        pct = float(np.mean(diffs < sent) * 100)
        map_rows.append((name, unit, sd, iqr, sent, pct))
    mdf = pd.DataFrame(map_rows, columns=[
        "observable", "unit", "sd_across_samples", "iqr_across_samples",
        "sentinel_abs_delta_49_50", "sentinel_pct_of_ordinary"])
    mdf.to_csv(out / "variability_repeatability_summary.csv", index=False)
    _lib.log("  wrote variability_repeatability_summary.csv")

    fig, ax = plt.subplots(figsize=(13.5, 8.2), dpi=dpi)
    ax.axis("off")
    hdr = (f"{'observable':<16}{'unit':>7}{'SD':>10}{'IQR':>10}"
           f"{'49/50 delta':>13}{'49/50 pct':>11}   repeatability")
    lines = [hdr, "-" * len(hdr)]
    for r in map_rows:
        r_cls = "high" if r[5] <= 5 else ("medium" if r[5] <= 20 else "low")
        lines.append(f"{r[0]:<16}{r[1]:>7}{r[2]:>10.3f}{r[3]:>10.3f}"
                     f"{r[4]:>13.3f}{r[5]:>11.1f}   {r_cls}")
    ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=9,
            va="top", transform=ax.transAxes)
    ax.set_title("Variability / repeatability summary (across-sample SD & "
                 "IQR; 49/50 single-pair sentinel percentile vs ordinary "
                 "pairs; high<=5%, medium<=20%)", fontsize=10,
                 family="monospace")
    fig.tight_layout()
    fig.savefig(out / "variability_repeatability_summary.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote variability_repeatability_summary.png")

    missing = [f for f in EXPECTED if not (out / f).exists()]
    _lib.require(not missing, f"missing outputs: {missing}")
    _lib.log(f"05 done in {_lib.elapsed(t0)}; all {len(EXPECTED)} outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
