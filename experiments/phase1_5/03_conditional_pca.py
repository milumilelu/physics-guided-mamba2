#!/usr/bin/env python3
"""Phase 1.5 experiment 03: conditional residual PCA + size-matched baseline
+ depth sliding-window mode rotation.

H1 (regime mixing): does PCA become stable inside process/depth subsets?
H3 (nonlinear manifold): do modes rotate smoothly with depth?
Sample-size matched random baselines control the small-n artifact.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _lib

EXPECTED = ["conditional_pca_table.csv", "size_matched_baseline.csv",
            "conditional_stability_heatmap.png", "depth_window_table.csv",
            "depth_window_mode_rotation.png"]


def subset_positions(man_sub: pd.DataFrame, key: str) -> list[np.ndarray]:
    """Cluster membership as positions within the subset (rows are sorted)."""
    out = []
    for _, g in man_sub.groupby(key):
        out.append(np.searchsorted(man_sub["dataset_index"].to_numpy(),
                                   g["dataset_index"].to_numpy()))
    return out


def main() -> int:
    t0 = time.time()
    cfg, quick = _lib.load_config(__doc__)
    out = _lib.output_dir(cfg)
    B = _lib.n_boot(cfg, quick)
    seed = int(cfg["random_seed"])
    _lib.log(f"== Phase 1.5 / 03: conditional PCA (B={B}) ==")
    frozen = _lib.load_frozen(cfg)
    man = frozen["man"]

    bands3 = _lib.make_bands(frozen["R"], float(cfg["scales"]["sigma_low_px"]),
                             float(cfg["scales"]["sigma_high_px"]))
    fields = {"total": _lib.to_2d(frozen["R"]),
              "low": _lib.to_2d(bands3["low"]),
              "mid": _lib.to_2d(bands3["mid"]),
              "high": _lib.to_2d(bands3["high"])}
    band_names = ["total", "low", "mid", "high"]

    formal = man[man["session_role"] == "formal"]
    pass60 = man[man["session_role"] == "pass_main"]
    subsets: list[tuple[str, np.ndarray]] = [("global", man["dataset_index"].to_numpy())]
    for f in sorted(formal["frequency_kHz"].unique()):
        subsets.append((f"formal f={int(f)}kHz",
                        formal[formal["frequency_kHz"] == f]["dataset_index"].to_numpy()))
    for p in sorted(formal["pulse_duration_fs"].unique()):
        subsets.append((f"formal pulse={int(p)}fs",
                        formal[formal["pulse_duration_fs"] == p]["dataset_index"].to_numpy()))
    for n_ in sorted(formal["pass_count"].unique()):
        subsets.append((f"formal N={int(n_)}",
                        formal[formal["pass_count"] == n_]["dataset_index"].to_numpy()))
    depth = man["median_depth_um"].to_numpy()
    q = np.quantile(depth, np.linspace(0, 1, int(cfg["conditional"]["depth_quantiles"]) + 1))
    for qi in range(int(cfg["conditional"]["depth_quantiles"])):
        sel = (depth >= q[qi]) & (depth <= q[qi + 1]) if qi == len(q) - 2 else \
              (depth >= q[qi]) & (depth < q[qi + 1])
        subsets.append((f"depth Q{qi + 1}", man[sel]["dataset_index"].to_numpy()))
    for n_ in sorted(pass60["pass_count"].unique()):
        subsets.append((f"60pass N={int(n_)}",
                        pass60[pass60["pass_count"] == n_]["dataset_index"].to_numpy()))
    _lib.log(f"  {len(subsets)} subsets: " + ", ".join(
        f"{name}(n={len(idx)})" for name, idx in subsets))

    pool_clusters = _lib.cluster_lists(man, cfg["bootstrap"]["cluster_key"])
    n_draws = int(cfg["bootstrap"]["n_replicates_quick"] if quick else
                  cfg["bootstrap"]["n_replicates_matched"])
    inner_B = 5 if quick else 20
    _lib.log(f"  matched baseline: {n_draws} draws x inner B={inner_B}, "
             "cluster-count matched, cached per (band, n_clusters)")

    rows, base_rows = [], []
    cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for si, (name, rows_idx) in enumerate(subsets):
        man_sub = man.iloc[rows_idx]
        clusters_sub = subset_positions(man_sub, "shared_height_source_id")
        n_clusters = len(clusters_sub)
        for bi, band in enumerate(band_names):
            X = fields[band]
            Xs = X[rows_idx]
            ref, evr = _lib.gram_pca(Xs, 3)
            Gs = Xs @ Xs.T
            ang, _ = _lib.boot_angles(Gs, Xs, clusters_sub, ref, 3, B,
                                      seed + 1000 * si + bi)
            th1m, th1q = float(np.median(ang[:, 0])), float(np.percentile(ang[:, 0], 75)
                                                           - np.percentile(ang[:, 0], 25))
            th3m = float(np.median(ang[:, 2]))
            th3q = float(np.percentile(ang[:, 2], 75) - np.percentile(ang[:, 2], 25))

            # size-matched random baseline: distinct clusters drawn from the
            # global pool, own reference basis, identical bootstrap protocol
            key = (band, n_clusters)
            if key not in cache:
                rng_c = np.random.default_rng(seed + 555 + 101 * bi)
                r1, r3 = [], []
                for d in range(n_draws):
                    rng_d = np.random.default_rng(seed + 9000 + 131 * bi + d)
                    pick = rng_c.choice(len(pool_clusters), size=n_clusters,
                                        replace=False)
                    idx = np.sort(np.concatenate([pool_clusters[p] for p in pick]))
                    Y = X[idx]
                    ref_r, _ = _lib.gram_pca(Y, 3)
                    man_r = man.iloc[idx]
                    cl_r = subset_positions(man_r, "shared_height_source_id")
                    Gy = Y @ Y.T
                    ang_r, _ = _lib.boot_angles(Gy, Y, cl_r, ref_r, 3, inner_B,
                                                int(rng_d.integers(0, 10 ** 6)))
                    r1.append(np.median(ang_r[:, 0]))
                    r3.append(np.median(ang_r[:, 2]))
                cache[key] = (np.asarray(r1), np.asarray(r3))
            r1, r3 = cache[key]
            prank1 = float(np.mean(r1 > th1m))
            prank3 = float(np.mean(r3 > th3m))
            rows.append((name, band, len(rows_idx), float(evr[0]), float(evr[:3].sum()),
                         th1m, th1q, th3m, th3q,
                         float(np.median(r1)), float(np.percentile(r1, 25)),
                         float(np.percentile(r1, 75)), prank1,
                         float(np.median(r3)), prank3))
        _lib.log(f"  [{name}] done ({_lib.elapsed(t0)})")

    for (band, n_clusters), (r1, r3) in sorted(cache.items()):
        base_rows.append((band, n_clusters, len(r1),
                          float(np.median(r1)), float(np.percentile(r1, 25)),
                          float(np.percentile(r1, 75)),
                          float(np.median(r3)), float(np.percentile(r3, 25)),
                          float(np.percentile(r3, 75))))
    pd.DataFrame(base_rows, columns=[
        "band", "n_clusters", "n_draws", "rand_theta1_median_deg",
        "rand_theta1_p25_deg", "rand_theta1_p75_deg", "rand_theta3_median_deg",
        "rand_theta3_p25_deg", "rand_theta3_p75_deg"]).to_csv(
        out / "size_matched_baseline.csv", index=False)
    df = pd.DataFrame(rows, columns=[
        "subset", "band", "n", "evr_pc1", "evr_cum3",
        "theta1_median_deg", "theta1_iqr_deg", "theta3_median_deg",
        "theta3_iqr_deg", "rand_theta1_median_deg", "rand_theta1_p25_deg",
        "rand_theta1_p75_deg", "prank1", "rand_theta3_median_deg", "prank3"])
    df.to_csv(out / "conditional_pca_table.csv", index=False)
    _lib.log("  wrote conditional_pca_table.csv / size_matched_baseline.csv")

    strong = df[(df.prank1 >= 0.95) & (df.theta1_median_deg < df.rand_theta1_p25_deg)]
    if len(strong):
        _lib.log("  subsets clearly more stable than size-matched baseline:")
        for _, r in strong.iterrows():
            _lib.log(f"    {r['subset']} [{r['band']}]: theta1="
                     f"{r['theta1_median_deg']:.1f} deg vs random median "
                     f"{r['rand_theta1_median_deg']:.1f} deg (p-rank {r['prank1']:.2f})")
    else:
        _lib.log("  no subset beats its size-matched baseline decisively "
                 "(p-rank >= 0.95 and below baseline P25)")

    # ---- heatmap -----------------------------------------------------------
    dpi = int(cfg["plot"]["dpi"])
    piv1 = df.pivot(index="subset", columns="band", values="theta1_median_deg")
    piv3 = df.pivot(index="subset", columns="band", values="theta3_median_deg")
    order = [name for name, _ in subsets]
    piv1, piv3 = piv1.loc[order], piv3.loc[order]
    piv1 = piv1.reindex(columns=band_names)
    piv3 = piv3.reindex(columns=band_names)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.42 * len(order) + 2.6), dpi=dpi)
    for ax, piv, title in ((axes[0], piv1, "bootstrap theta (k=1) median [deg]"),
                           (axes[1], piv3, "bootstrap theta (k=1..3) median [deg]")):
        im = ax.imshow(piv.to_numpy(), cmap="RdYlGn_r", vmin=0, vmax=90,
                       aspect="auto")
        ax.set_xticks(range(len(band_names)), band_names, fontsize=8)
        ax.set_yticks(range(len(order)), order, fontsize=7)
        ax.set_title(title, fontsize=10)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.to_numpy()[i, j]:.0f}", ha="center",
                        va="center", fontsize=7,
                        color="black")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("Conditional PCA bootstrap stability by condition x band "
                 "(green = stable, red = unstable)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "conditional_stability_heatmap.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote conditional_stability_heatmap.png")

    # ---- depth sliding window (H3) ----------------------------------------
    w = int(cfg["conditional"]["window_size"])
    step = int(cfg["conditional"]["window_step"])
    depth = man["median_depth_um"].to_numpy()
    order = np.argsort(depth)
    wrows = []
    cos_pc1 = {b: [] for b in band_names}
    cos_pc13 = {b: [] for b in band_names}
    centers = []
    for band in band_names:
        X = fields[band]
        comps_seq, evr_seq, ctr_seq = [], [], []
        for s in range(0, 200 - w + 1, step):
            idx = order[s:s + w]
            ref_w, evr_w = _lib.gram_pca(X[idx], 3)
            comps_seq.append(ref_w)
            evr_seq.append(float(evr_w[0]))
            ctr_seq.append(float(np.mean(depth[idx])))
        centers = ctr_seq
        for a, b in zip(comps_seq[:-1], comps_seq[1:]):
            sv = _lib.principal_angles(a[:1].T, b[:1].T)
            cos_pc1[band].append(float(np.cos(np.radians(sv[-1]))))
            sv3 = _lib.principal_angles(a[:3].T, b[:3].T)
            cos_pc13[band].append(float(np.cos(np.radians(sv3[0]))))
        for j, ctr in enumerate(ctr_seq[:-1]):
            wrows.append((band, ctr, evr_seq[j], cos_pc1[band][j],
                          cos_pc13[band][j]))
    pd.DataFrame(wrows, columns=["band", "window_center_depth_um",
                                 "evr_pc1", "cos_pc1_next", "cos_pc13_next"]
                 ).to_csv(out / "depth_window_table.csv", index=False)
    _lib.log("  wrote depth_window_table.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=dpi)
    colors = {"total": "black", "low": "tab:blue", "mid": "tab:green",
              "high": "tab:red"}
    for band in band_names:
        axes[0].plot(centers[:-1], cos_pc1[band], "o-", color=colors[band],
                     ms=3, lw=1.2, label=band)
        axes[1].plot(centers[:-1], cos_pc13[band], "o-", color=colors[band],
                     ms=3, lw=1.2, label=band)
    for ax, title in ((axes[0], "|cos| adjacent-window PC1"),
                      (axes[1], "cos adjacent-window PC1-3 subspace")):
        ax.axhline(0.8, color="tab:blue", ls=":", lw=1)
        ax.axhline(0.5, color="tab:red", ls=":", lw=1)
        ax.set_xlabel("window centre depth [um]")
        ax.set_ylabel(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Depth sliding-window PCA (w={w}, step={step}): "
                 "mode rotation with depth", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "depth_window_mode_rotation.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote depth_window_mode_rotation.png")
    for band in band_names:
        m1 = float(np.median(cos_pc1[band]))
        _lib.log(f"  sliding-window [{band}]: median |cos| PC1 = {m1:.3f}")

    missing = [f for f in EXPECTED if not (out / f).exists()]
    _lib.require(not missing, f"missing outputs: {missing}")
    _lib.log(f"03 done in {_lib.elapsed(t0)}; all {len(EXPECTED)} outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
