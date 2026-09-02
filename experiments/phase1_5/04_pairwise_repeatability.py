#!/usr/bin/env python3
"""Phase 1.5 experiment 04: pairwise morphology distance vs process distance,
session separability, and the multiscale 49/50 repeatability sentinel.
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
    _lib.log("== Phase 1.5 / 04: pairwise distance / sentinel / session ==")
    frozen = _lib.load_frozen(cfg)
    man = frozen["man"]
    bands = _lib.make_bands(frozen["R"], float(cfg["scales"]["sigma_low_px"]),
                            float(cfg["scales"]["sigma_high_px"]))
    fields = {"total": _lib.to_2d(frozen["R"]),
              "low": _lib.to_2d(bands["low"]),
              "mid": _lib.to_2d(bands["mid"]),
              "high": _lib.to_2d(bands["high"])}
    band_names = ["total", "low", "mid", "high"]
    n_feat = fields["total"].shape[1]

    # process distance on z-scored design parameters
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
    for band in band_names:
        G = fields[band] @ fields[band].T
        dist[band] = _lib.pairwise_rmse_from_gram(G, n_feat)
    del fields

    dpi = int(cfg["plot"]["dpi"])
    cmap_div = plt.get_cmap(cfg["plot"]["diverging_cmap"]).copy()

    rows = []
    binned_curves = {}
    for band in band_names:
        d = dist[band]
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
        binned_curves[band] = (centers, meds)
        q10 = float(np.quantile(dp_ord, cfg["distance"]["near_quantile"]))
        near = d_ord[dp_ord <= q10]
        sent_d = float(d[i_a, i_b])
        pct = float(np.mean(d_ord < sent_d) * 100)
        rows.append((band, rho, q10, float(np.median(near)),
                     float(np.median(d_ord)), float(np.median(near) / np.median(d_ord)),
                     float(np.percentile(near, 5)), float(np.percentile(near, 95)),
                     sent_d, pct, float(np.median(d_ord)),
                     float(np.percentile(d_ord, 5)), float(np.percentile(d_ord, 95))))
        _lib.log(f"  [{band}] Spearman(D_process, D) = {rho:.3f} | near-pair "
                 f"(D_process<=Q10, n={near.size}) median D = {np.median(near):.3f} "
                 f"vs all {np.median(d_ord):.3f} um | sentinel D = {sent_d:.4f} um "
                 f"(percentile {pct:.2f})")
    pd.DataFrame(rows, columns=[
        "band", "spearman_process_morph", "near_q10_process",
        "near_median_D_um", "all_median_D_um", "near_ratio",
        "near_p5_um", "near_p95_um", "sentinel_D_um", "sentinel_pct",
        "all_median_D_um_dup", "all_p5_um", "all_p95_um"]).to_csv(
        out / "pairwise_distance_summary.csv", index=False)
    pd.DataFrame(
        [(r[0], r[8], r[9], r[10], r[11], r[12]) for r in rows],
        columns=["band", "sentinel_D_um", "sentinel_pct_of_ordinary",
                 "ordinary_median_D_um", "ordinary_p5_um",
                 "ordinary_p95_um"]).to_csv(
        out / "sentinel_multiscale_table.csv", index=False)
    _lib.log("  wrote sentinel_multiscale_table.csv")

    # session separability
    sess = man["session_id"].to_numpy()
    srows = []
    for band in band_names:
        d = dist[band]
        within, between = [], []
        for s in np.unique(sess):
            m = (sess[pu] == s) & (sess[pv] == s)
            within.append((s, float(np.median(d[pu, pv][m])), int(m.sum())))
        for sa, sb in (("zro2_120_formal", "zro2_60_pass"),
                       ("zro2_120_formal", "zro2_20_supplement"),
                       ("zro2_60_pass", "zro2_20_supplement")):
            m = ((sess[pu] == sa) & (sess[pv] == sb)) | \
                ((sess[pu] == sb) & (sess[pv] == sa))
            between.append((f"{sa[:9]}..|{sb[:9]}..", float(np.median(d[pu, pv][m])),
                            int(m.sum())))
        w_med = float(np.median([w[1] for w in within]))
        b_med = float(np.median([b[1] for b in between]))
        srows.append((band, *[x for t in within for x in t],
                      *[x for t in between for x in t], w_med, b_med, b_med / w_med))
        _lib.log(f"  [{band}] session: within median {w_med:.3f} um, "
                 f"between median {b_med:.3f} um, ratio {b_med / w_med:.2f}")
    pd.DataFrame(srows, columns=[
        "band",
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
    colors = {"total": "black", "low": "tab:blue", "mid": "tab:green",
              "high": "tab:red"}
    for band in band_names:
        centers, meds = binned_curves[band]
        ax.plot(centers, meds, "o-", color=colors[band], ms=3, lw=1.2,
                label=band)
    ax.set_xlabel("process distance D_process")
    ax.set_ylabel("binned median shape RMSE [um]")
    ax.set_title("Binned median morphology distance by band")
    ax.grid(alpha=0.25)
    ax.legend()
    ax = axes[2]
    d_tot_mat = dist["total"]
    box_data, box_lab = [], []
    for s in np.unique(sess):
        m = (sess[pu] == s) & (sess[pv] == s)
        box_data.append(d_tot_mat[pu, pv][m])
        box_lab.append(f"within\n{s.replace('zro2_', '')}\n(n={int(m.sum())})")
    mb = ((sess[pu] != sess[pv]))
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
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 13.0), dpi=dpi)
    for r, band in enumerate(("low", "mid", "high")):
        d = dist[band]
        sent_d = float(d[i_a, i_b])
        d_ord = d[pu, pv]
        pct = float(np.mean(d_ord < sent_d) * 100)
        for c, (idx, label) in enumerate(((i_a, "DOE 49"), (i_b, "DOE 50"))):
            ax = axes[r, c]
            img = bands[band][idx]
            vmax = np.nanmax(np.abs(img))
            im = ax.imshow(img, cmap=cmap_div, vmin=-vmax, vmax=vmax)
            ax.set_title(f"{label}  R_{band} [um]", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax = axes[r, 2]
        diff = bands[band][i_a] - bands[band][i_b]
        vmax = np.nanmax(np.abs(diff))
        im = ax.imshow(diff, cmap=cmap_div, vmin=-vmax, vmax=vmax)
        ax.set_title(f"diff (RMS {sent_d:.4f} um,\npct {pct:.2f} of ordinary)",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Multiscale repeatability sentinel (DOE 49/50, "
                 "identical process, different position)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
