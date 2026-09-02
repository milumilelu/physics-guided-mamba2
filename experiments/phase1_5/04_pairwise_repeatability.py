#!/usr/bin/env python3
"""Phase 1.5R experiment 04: pairwise morphology distance vs process distance,
session separability, and the multiscale 49/50 repeatability sentinel.

Scale fields are physical-wavelength DCT bands (plus the total residual).
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _lib

EXPECTED = ["pairwise_distance_summary.csv", "session_separability.csv",
            "sentinel_multiscale_table.csv", "morph_vs_process_distance.png",
            "sentinel_multiscale_49_50.png"]


def main() -> int:
    t0 = time.time()
    cfg, quick = _lib.load_config(__doc__)
    out = _lib.output_dir(cfg)
    _lib.log("== Phase 1.5R / 04: pairwise distance / sentinel / session ==")
    frozen = _lib.load_frozen(cfg)
    man = frozen["man"]
    all_fields = _lib.multiscale_fields(frozen["R"], cfg)
    scale_names = ["total", "DCT_8_16", "DCT_16_32", "DCT_32_64",
                   "DCT_64_inf"]
    fields = {s: all_fields[s] for s in scale_names}
    n_feat = fields["total"].shape[1]

    pcols = list(cfg["distance"]["process_params"])
    P = man[pcols].to_numpy(dtype=float)
    Pz = (P - P.mean(axis=0)) / P.std(axis=0)
    Dp = np.sqrt(np.maximum(
        (Pz ** 2).sum(1)[:, None] + (Pz ** 2).sum(1)[None, :] - 2 * Pz @ Pz.T,
        0.0))

    i_a, i_b = _lib.sentinel_rows(man, cfg)
    pu, pv = _lib.ordinary_pair_mask(man, i_a, i_b)
    dp_ord = Dp[pu, pv]
    _lib.log(f"  ordinary pairs: {pu.size}; sentinel pair ({i_a},{i_b})")

    dist = {}
    for s in scale_names:
        G = fields[s] @ fields[s].T
        dist[s] = _lib.pairwise_rmse_from_gram(G, n_feat)
    del fields

    dpi = int(cfg["plot"]["dpi"])
    cmap_div = plt.get_cmap(cfg["plot"]["diverging_cmap"]).copy()

    rows = []
    binned_curves = {}
    for s in scale_names:
        d = dist[s]
        d_ord = d[pu, pv]
        rho = float(spearmanr(dp_ord, d_ord).statistic)
        nb = int(cfg["distance"]["n_bins"])
        edges = np.quantile(dp_ord, np.linspace(0, 1, nb + 1))
        centers, meds = [], []
        for j in range(nb):
            m = (dp_ord >= edges[j]) & (dp_ord <= edges[j + 1] if j == nb - 1
                                        else dp_ord < edges[j + 1])
            if m.sum() >= 5:
                centers.append(float(dp_ord[m].mean()))
                meds.append(float(np.median(d_ord[m])))
        binned_curves[s] = (centers, meds)
        q10 = float(np.quantile(dp_ord, cfg["distance"]["near_quantile"]))
        near = d_ord[dp_ord <= q10]
        sent_d = float(d[i_a, i_b])
        pct = float(np.mean(d_ord < sent_d) * 100)
        rows.append((s, rho, q10, float(np.median(near)),
                     float(np.median(d_ord)),
                     float(np.median(near) / np.median(d_ord)),
                     float(np.percentile(near, 5)),
                     float(np.percentile(near, 95)), sent_d, pct,
                     float(np.percentile(d_ord, 5)),
                     float(np.percentile(d_ord, 95))))
        _lib.log(f"  [{s}] Spearman(D_process, D) = {rho:.3f} | near-pair "
                 f"(D_process<=Q10, n={near.size}) median D = "
                 f"{np.median(near):.3f} vs all {np.median(d_ord):.3f} um | "
                 f"sentinel D = {sent_d:.4f} um (percentile {pct:.2f})")
    pd.DataFrame(rows, columns=[
        "scale", "spearman_process_morph", "near_q10_process",
        "near_median_D_um", "all_median_D_um", "near_ratio",
        "near_p5_um", "near_p95_um", "sentinel_D_um", "sentinel_pct",
        "all_p5_um", "all_p95_um"]).to_csv(
        out / "pairwise_distance_summary.csv", index=False)
    pd.DataFrame(
        [(r[0], r[8], r[9], r[4], r[10], r[11]) for r in rows],
        columns=["scale", "sentinel_D_um", "sentinel_pct_of_ordinary",
                 "ordinary_median_D_um", "ordinary_p5_um",
                 "ordinary_p95_um"]).to_csv(
        out / "sentinel_multiscale_table.csv", index=False)
    _lib.log("  wrote pairwise_distance_summary.csv / "
             "sentinel_multiscale_table.csv")

    sess = man["session_id"].to_numpy()
    srows = []
    for s in scale_names:
        d = dist[s]
        within, between = [], []
        for se in np.unique(sess):
            m = (sess[pu] == se) & (sess[pv] == se)
            within.append((se, float(np.median(d[pu, pv][m])), int(m.sum())))
        for sa, sb in (("zro2_120_formal", "zro2_60_pass"),
                       ("zro2_120_formal", "zro2_20_supplement"),
                       ("zro2_60_pass", "zro2_20_supplement")):
            m = ((sess[pu] == sa) & (sess[pv] == sb)) | \
                ((sess[pu] == sb) & (sess[pv] == sa))
            between.append((f"{sa[:9]}..|{sb[:9]}..",
                            float(np.median(d[pu, pv][m])), int(m.sum())))
        w_med = float(np.median([w[1] for w in within]))
        b_med = float(np.median([b[1] for b in between]))
        srows.append((s, *[x for t in within for x in t],
                      *[x for t in between for x in t], w_med, b_med,
                      b_med / w_med))
        _lib.log(f"  [{s}] session: within median {w_med:.3f} um, between "
                 f"median {b_med:.3f} um, ratio {b_med / w_med:.2f}")
    pd.DataFrame(srows, columns=[
        "scale",
        "within_1_label", "within_1_um", "within_1_n",
        "within_2_label", "within_2_um", "within_2_n",
        "within_3_label", "within_3_um", "within_3_n",
        "between_1_label", "between_1_um", "between_1_n",
        "between_2_label", "between_2_um", "between_2_n",
        "between_3_label", "between_3_um", "between_3_n",
        "within_median_um", "between_median_um",
        "between_within_ratio"]).to_csv(out / "session_separability.csv",
                                        index=False)
    _lib.log("  wrote session_separability.csv")

    # ---- fig 5: morphology distance vs process distance --------------------
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), dpi=dpi)
    ax = axes[0]
    d_tot = dist["total"][pu, pv]
    ax.scatter(dp_ord, d_tot, s=3, alpha=0.05, color="0.3", edgecolors="none")
    centers, meds = binned_curves["total"]
    ax.plot(centers, meds, "o-", color="tab:red", ms=4, label="binned median")
    ax.set_xlabel("process distance D_process (z-scored 5 params)")
    ax.set_ylabel("residual-shape RMSE [um]")
    ax.set_title("D_process vs D_morphology (total residual, ordinary pairs)")
    ax.legend()
    ax.grid(alpha=0.25)
    ax = axes[1]
    colors = {"total": "black", "DCT_8_16": "tab:red",
              "DCT_16_32": "tab:purple", "DCT_32_64": "tab:brown",
              "DCT_64_inf": "tab:olive"}
    for s in scale_names:
        centers, meds = binned_curves[s]
        ax.plot(centers, meds, "o-", color=colors[s], ms=3, lw=1.2, label=s)
    ax.set_xlabel("process distance D_process")
    ax.set_ylabel("binned median shape RMSE [um]")
    ax.set_title("Binned median morphology distance by DCT wavelength band")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax = axes[2]
    d_tot_mat = dist["total"]
    box_data, box_lab = [], []
    for se in np.unique(sess):
        m = (sess[pu] == se) & (sess[pv] == se)
        box_data.append(d_tot_mat[pu, pv][m])
        box_lab.append(f"within\n{se.replace('zro2_', '')}\n(n={int(m.sum())})")
    mb = sess[pu] != sess[pv]
    box_data.append(d_tot_mat[pu, pv][mb])
    box_lab.append(f"between\n(n={int(mb.sum())})")
    bp = ax.boxplot(box_data, tick_labels=box_lab, showmeans=True,
                    patch_artist=True)
    for i in range(3):
        bp["boxes"][i].set_facecolor("lightblue")
    bp["boxes"][3].set_facecolor("lightyellow")
    ax.set_ylabel("residual-shape RMSE [um]")
    ax.set_title("Within- vs between-session shape distance (total)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out / "morph_vs_process_distance.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote morph_vs_process_distance.png")

    # ---- fig 6: multiscale 49/50 sentinel ----------------------------------
    band_maps = {"total": frozen["R"]}
    dct_only, _ = _lib.dct_band_fields(frozen["R"],
                                       float(cfg["scales"]["pixel_um"]),
                                       cfg["scales"]["dct_bands_um"])
    band_maps.update(dct_only)
    fig, axes = plt.subplots(len(scale_names), 3,
                             figsize=(13.0, 4.3 * len(scale_names)), dpi=dpi)
    for r, s in enumerate(scale_names):
        d = dist[s]
        sent_d = float(d[i_a, i_b])
        d_ord = d[pu, pv]
        pct = float(np.mean(d_ord < sent_d) * 100)
        for c, (idx, label) in enumerate(
                ((i_a, "DOE 49"), (i_b, "DOE 50"))):
            ax = axes[r, c]
            img = band_maps[s][idx]
            vmax = np.nanmax(np.abs(img))
            im = ax.imshow(img, cmap=cmap_div, vmin=-vmax, vmax=vmax)
            ax.set_title(f"{label}  {s} [um]", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax = axes[r, 2]
        diff = band_maps[s][i_a] - band_maps[s][i_b]
        vmax = np.nanmax(np.abs(diff))
        im = ax.imshow(diff, cmap=cmap_div, vmin=-vmax, vmax=vmax)
        ax.set_title(f"diff (RMS {sent_d:.4f} um,\npct {pct:.2f} of ordinary)",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Multiscale repeatability sentinel (DOE 49/50, "
                 "identical process, different position)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "sentinel_multiscale_49_50.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote sentinel_multiscale_49_50.png")

    missing = [f for f in EXPECTED if not (out / f).exists()]
    _lib.require(not missing, f"missing outputs: {missing}")
    _lib.log(f"04 done in {_lib.elapsed(t0)}; all {len(EXPECTED)} outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
