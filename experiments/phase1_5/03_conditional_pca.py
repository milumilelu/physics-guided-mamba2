#!/usr/bin/env python3
"""Phase 1.5R experiment 03: conditional residual PCA with matched baselines,
LOCO influence, eigengaps, and depth-window (pseudo-trajectory-free) mode
rotation checks.

H1 (regime mixing): does PCA become stable inside process/depth subsets?
Baselines draw from the SAME session (global pool for cross-session subsets)
with matched ROI count and matched within-subset cluster-size pattern.
H3 diagnostics: adjacent-window subspace angles for overlapping windows,
non-overlapping windows, and a shuffled-depth overlap null.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _lib

EXPECTED = ["conditional_pca_table.csv", "baseline_matched.csv",
            "loco_top5_influencers.csv", "loco_top5_montage.png",
            "conditional_stability_heatmap.png", "depth_window_table.csv",
            "depth_window_null.csv", "depth_window_mode_rotation.png"]

ANOMALY_SUBSETS = ("formal N=1", "formal N=2", "depth Q1")


def subset_positions(man_sub: pd.DataFrame,
                     key: str = "shared_height_source_id") -> list[np.ndarray]:
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
    n_draws = int(cfg["conditional"]["baseline_draws"])
    inner_b = int(cfg["conditional"]["baseline_inner_b"])
    if quick:
        n_draws, inner_b = 10, 10
    _lib.log(f"== Phase 1.5R / 03: conditional PCA (B={B}, baseline "
             f"{n_draws} draws x inner B={inner_b}, same-session + occupancy "
             "matched) ==")
    frozen = _lib.load_frozen(cfg)
    man = frozen["man"]
    fields = _lib.multiscale_fields(frozen["R"], cfg)
    field_names = list(fields)
    _lib.log(f"  scales: {field_names}")

    formal = man[man["session_role"] == "formal"]
    pass60 = man[man["session_role"] == "pass_main"]
    subsets: list[tuple[str, np.ndarray]] = [
        ("global", man["dataset_index"].to_numpy())]
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
    q = np.quantile(depth, np.linspace(0, 1,
                                       int(cfg["conditional"]["depth_quantiles"]) + 1))
    for qi in range(int(cfg["conditional"]["depth_quantiles"])):
        sel = (depth >= q[qi]) & (depth <= q[qi + 1]) if qi == len(q) - 2 else \
              (depth >= q[qi]) & (depth < q[qi + 1])
        subsets.append((f"depth Q{qi + 1}",
                        man[sel]["dataset_index"].to_numpy()))
    for n_ in sorted(pass60["pass_count"].unique()):
        subsets.append((f"60pass N={int(n_)}",
                        pass60[pass60["pass_count"] == n_]["dataset_index"].to_numpy()))
    _lib.log(f"  {len(subsets)} subsets")

    pools = _lib.session_cluster_pools(man)
    global_pool = _lib.cluster_lists(man, cfg["bootstrap"]["cluster_key"])

    rows, base_rows, loco_rows = [], [], []
    total_top5 = []
    cache: dict = {}
    for si, (name, rows_idx) in enumerate(subsets):
        man_sub = man.iloc[rows_idx]
        clusters_sub = subset_positions(man_sub)
        cluster_ids = man_sub.iloc[[p[0] for p in clusters_sub]][
            "shared_height_source_id"].to_numpy()
        bank = _lib.build_resample_bank(clusters_sub, B, seed + 17 * si)
        sessions = list(man_sub["session_id"].unique())
        sig = _lib.occupancy_signature(man_sub)
        # baseline draws: same session when the subset is single-session,
        # otherwise the same per-session composition (count + occupancy)
        if len(sessions) == 1:
            pool = pools[sessions[0]]
            pool_label = str(sessions[0])
            comp_key: tuple = (sig,)

            def draw_one(rng_):
                return _lib.draw_matched_subset(pool, sig_holder[0], rng_)
        else:
            pool_label = "session_composition_matched"
            sess_parts = [(s, _lib.occupancy_signature(
                man_sub[man_sub["session_id"] == s])) for s in sessions]
            # per-session occupancy composition goes into the cache key so
            # that subsets with identical overall size patterns but different
            # session mixes never share a baseline
            comp_key = tuple(sess_parts)

            def draw_one(rng_):
                parts = [_lib.draw_matched_subset(pools[s], sg, rng_)
                         for s, sg in sess_parts]
                return np.sort(np.concatenate(parts))
        comp_str = ("single:" + "|".join(map(str, sig))) if len(sessions) == 1 \
            else " | ".join(f"{s}:{'|'.join(map(str, sg))}"
                            for s, sg in comp_key)
        sig_holder = [sig]

        for fi, fname in enumerate(field_names):
            X = fields[fname]
            Xs = X[rows_idx]
            ref, evr = _lib.gram_pca(Xs, 3)
            Gs = Xs @ Xs.T
            eig = _lib.gram_eigenvalues(Xs, 2)
            eigengap = float(eig[0] / max(eig[1], 1e-300))
            ang, _ = _lib.boot_angles_bank(Gs, Xs, bank, ref, 3)
            q1 = _lib.angle_quantiles(ang[:, 0])
            th3 = float(np.median(ang[:, 2]))

            key = (fname, pool_label, len(sig), comp_key)
            if key not in cache:
                rng_c = np.random.default_rng(seed + 424242 + 977 * fi)
                r1 = []
                for _ in range(n_draws):
                    idx = draw_one(rng_c)
                    Y = X[idx]
                    ref_r, _ = _lib.gram_pca(Y, 3)
                    man_r = man.iloc[idx]
                    cl_r = subset_positions(man_r)
                    Gy = Y @ Y.T
                    inner = _lib.build_resample_bank(cl_r, inner_b,
                                                     int(rng_c.integers(0, 10 ** 6)))
                    ang_r, _ = _lib.boot_angles_bank(Gy, Y, inner, ref_r, 3)
                    r1.append(float(np.median(ang_r[:, 0])))
                cache[key] = np.asarray(r1)
            rb = cache[key]
            prank1 = float(np.mean(rb > q1["q50"]))
            rand_q = {p: float(np.percentile(rb, p)) for p in (25, 50, 75, 90)}

            # LOCO influence (all fields for non-global subsets; global only
            # on the total field to keep the Gram refits affordable)
            if not (name == "global" and fname != "total"):
                loco = _lib.loco_angles(Xs, clusters_sub, k=1)
                loco_med, loco_max = float(np.median(loco)), float(np.max(loco))
                order_loco = np.argsort(loco)[::-1]
                for rank, ci in enumerate(order_loco[:5]):
                    cid = str(cluster_ids[ci])
                    members = man_sub.iloc[clusters_sub[ci]]
                    m_desc = "|".join(
                        f"{r.session_id}:s{int(r.sample_id)}"
                        f"(po{int(r.processing_order)},N{int(r.pass_count)},"
                        f"d{r.median_depth_um:.1f})"
                        for r in members.itertuples())
                    loco_rows.append((name, fname, rank + 1, cid,
                                      float(loco[ci]), m_desc))
                    if fname == "total":
                        total_top5.append((name, rank, cid, float(loco[ci]),
                                           members))
            else:
                loco_med = loco_max = np.nan

            # stability call: Q50 against Q50 AND Q90 against Q90 (baseline
            # stores both stats), plus LOCO influence agreement
            if prank1 >= 0.95 and q1["q50"] < rand_q[25]:
                if q1["q90"] < rand_q[90] and loco_max < 45:
                    call = "robust_stable"
                else:
                    call = "fragile_stable"
            else:
                call = "not_called"

            rows.append((name, fname, len(rows_idx), float(evr[0]),
                         float(evr[:3].sum()), eigengap,
                         float(q1["q25"]), float(q1["q50"]), float(q1["q75"]),
                         float(q1["q90"]), float(q1["q95"]), th3,
                         loco_med, loco_max,
                         pool_label, rand_q[50], rand_q[25], rand_q[75],
                         rand_q[90], prank1, call))
        base_rows.append((name, pool_label, comp_str, n_draws, inner_b))
        _lib.log(f"  [{name}] done ({_lib.elapsed(t0)})")

    df = pd.DataFrame(rows, columns=[
        "subset", "scale", "n", "evr_pc1", "evr_cum3", "eigengap_l1_over_l2",
        "theta1_q25_deg", "theta1_q50_deg", "theta1_q75_deg",
        "theta1_q90_deg", "theta1_q95_deg", "theta3_q50_deg",
        "loco_median_deg", "loco_max_deg", "baseline_pool",
        "rand_theta1_q50_deg", "rand_theta1_q25_deg", "rand_theta1_q75_deg",
        "rand_theta1_q90_deg", "prank1", "stable_call"])
    df.to_csv(out / "conditional_pca_table.csv", index=False)
    pd.DataFrame(loco_rows, columns=[
        "subset", "scale", "rank", "cluster_id", "loco_angle_deg",
        "members"]).to_csv(out / "loco_top5_influencers.csv", index=False)
    pd.DataFrame(base_rows, columns=["subset", "baseline_pool",
                                     "occupancy_signature", "n_draws",
                                     "inner_b"]).to_csv(
        out / "baseline_matched.csv", index=False)
    _lib.log("  wrote conditional_pca_table.csv / loco_top5_influencers.csv / "
             "baseline_matched.csv")

    for call in ("robust_stable", "fragile_stable"):
        sel = df[df.stable_call == call]
        _lib.log(f"  [{call}]: {len(sel)} (subset, scale) cells")
        for _, r in sel.head(24).iterrows():
            _lib.log(f"    {r['subset']} [{r['scale']}]: theta1 Q50="
                     f"{r['theta1_q50_deg']:.1f} Q90={r['theta1_q90_deg']:.1f} "
                     f"vs rand Q50={r['rand_theta1_q50_deg']:.1f} "
                     f"Q75={r['rand_theta1_q75_deg']:.1f} "
                     f"(p={r['prank1']:.2f}, locoMax={r['loco_max_deg']:.1f})")

    _lib.log("  anomaly check (N1/N2, depth Q1): eigengap / theta1 Q50 [Q25, "
             "Q75] / LOCO max / baseline Q50 (p-rank), per scale")
    for sub in ANOMALY_SUBSETS:
        sub_df = df[df.subset == sub]
        _lib.log(f"    -- {sub} --")
        for _, r in sub_df.iterrows():
            _lib.log(f"    {r['scale']:<12} gap={r['eigengap_l1_over_l2']:8.2f} "
                     f"th1={r['theta1_q50_deg']:6.2f} "
                     f"[{r['theta1_q25_deg']:6.2f},{r['theta1_q75_deg']:6.2f}] "
                     f"locoMax={r['loco_max_deg']:6.2f} "
                     f"rand={r['rand_theta1_q50_deg']:6.2f} "
                     f"(p={r['prank1']:.2f})")

    # ---- heatmap -----------------------------------------------------------
    dpi = int(cfg["plot"]["dpi"])
    piv1 = df.pivot(index="subset", columns="scale",
                    values="theta1_q50_deg").reindex(
        index=[n for n, _ in subsets], columns=field_names)
    piv3 = df.pivot(index="subset", columns="scale",
                    values="theta3_q50_deg").reindex(
        index=[n for n, _ in subsets], columns=field_names)
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 0.42 * len(subsets) + 2.6),
                             dpi=dpi)
    for ax, piv, title in ((axes[0], piv1, "bootstrap theta (k=1) Q50 [deg]"),
                           (axes[1], piv3, "bootstrap theta (k=1..3) Q50 [deg]")):
        im = ax.imshow(piv.to_numpy(), cmap="RdYlGn_r", vmin=0, vmax=90,
                       aspect="auto")
        ax.set_xticks(range(len(field_names)), field_names, fontsize=7,
                      rotation=30, ha="right")
        ax.set_yticks(range(len(subsets)), [n for n, _ in subsets], fontsize=7)
        ax.set_title(title, fontsize=10)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.to_numpy()[i, j]:.0f}", ha="center",
                        va="center", fontsize=6, color="black")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("Conditional PCA bootstrap stability: condition x scale "
                 "(green = stable, red = unstable)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "conditional_stability_heatmap.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote conditional_stability_heatmap.png")

    # ---- depth windows: overlap / non-overlap / shuffled-depth null --------
    w = int(cfg["conditional"]["window_size"])
    step = int(cfg["conditional"]["window_step"])
    step_non = int(cfg["conditional"]["window_nonoverlap_step"])
    depth = man["median_depth_um"].to_numpy()
    order = np.argsort(depth)
    win_fields = ["total", "G16", "DCT_8_16", "DCT_16_32", "DCT_32_64"]
    rng_w = np.random.default_rng(seed + 31337)

    def window_cosines(seq_order: np.ndarray, starts: list[int],
                       X: np.ndarray) -> tuple[list, list, list, list]:
        refs, evrs, ctrs = [], [], []
        for s in starts:
            idx = seq_order[s:s + w]
            ref_w, evr_w = _lib.gram_pca(X[idx], 3)
            refs.append(ref_w)
            evrs.append(float(evr_w[0]))
            ctrs.append(float(np.mean(depth[idx])))
        c1, c13 = [], []
        for a, b in zip(refs[:-1], refs[1:]):
            ang1 = _lib.principal_angles(a[:1].T, b[:1].T)[-1]
            c1.append(float(np.cos(np.radians(ang1))))
            ang3 = _lib.principal_angles(a[:3].T, b[:3].T)[-1]
            c13.append(float(np.cos(np.radians(ang3))))
        return ctrs[:-1], evrs[:-1], c1, c13

    wrows = []
    null_rows = []
    real_curves = {}
    null_stats = {}
    n_null = int(cfg["conditional"]["shuffle_null_perms"])
    if quick:
        n_null = min(n_null, 10)
    _lib.log(f"  shuffled-depth null: {n_null} permutations")
    for fname in win_fields:
        X = fields[fname]
        ctrs, evrs, c1, c13 = window_cosines(
            order, list(range(0, 200 - w + 1, step)), X)
        real_curves[fname] = (ctrs, c1, c13)
        for j, ctr in enumerate(ctrs):
            wrows.append(("overlap", fname, ctr, evrs[j], c1[j], c13[j]))
        ctrs_n, _, c1_n, c13_n = window_cosines(
            order, list(range(0, 200 - w + 1, step_non)), X)
        for j, ctr in enumerate(ctrs_n):
            wrows.append(("nonoverlap", fname, ctr, np.nan, c1_n[j], c13_n[j]))
        null_med = []
        for p in range(n_null):
            perm = rng_w.permutation(200)
            _, _, c1_p, _ = window_cosines(
                perm, list(range(0, 200 - w + 1, step)), X)
            med_p = float(np.median(c1_p))
            null_med.append(med_p)
            null_rows.append((fname, p + 1, med_p))
        null_stats[fname] = (float(np.median(null_med)),
                             float(np.percentile(null_med, 90)))
        _lib.log(f"  sliding-window [{fname}]: real |cos| PC1 median="
                 f"{np.median(np.abs(c1)):.3f} | shuffled-depth null median="
                 f"{null_stats[fname][0]:.3f} (Q90 {null_stats[fname][1]:.3f})"
                 f" | nonoverlap median |cos|={np.median(np.abs(c1_n)):.3f}"
                 f" | worst-case PC1-3 cos (overlap) median="
                 f"{np.median(c13):.3f}")
    dfw = pd.DataFrame(wrows, columns=["kind", "scale",
                                       "window_center_depth_um", "evr_pc1",
                                       "cos_pc1_next",
                                       "cos_pc13_next_worst"])
    dfw.to_csv(out / "depth_window_table.csv", index=False)
    pd.DataFrame(null_rows, columns=["scale", "permutation",
                                     "null_median_cos_pc1_next"]
                 ).to_csv(out / "depth_window_null.csv", index=False)
    _lib.log("  wrote depth_window_table.csv / depth_window_null.csv")

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.0), dpi=dpi)
    colors = {"total": "black", "G16": "tab:green", "DCT_8_16": "tab:red",
              "DCT_16_32": "tab:purple", "DCT_32_64": "tab:brown"}
    ax = axes[0]
    for fname in win_fields:
        ctrs, c1, _ = real_curves[fname]
        ax.plot(ctrs, np.abs(c1), "o-", color=colors[fname], ms=3.5, lw=1.1,
                label=fname)
        ax.axhline(null_stats[fname][0], color=colors[fname], ls=":", lw=0.9,
                   alpha=0.6)
    ax.set_xlabel("window centre depth [um]")
    ax.set_ylabel("|cos| adjacent-window PC1 (overlap w=50, step=10)")
    ax.set_title("Overlap windows vs shuffled-depth null (dotted)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    ax = axes[1]
    for fname in win_fields:
        X = fields[fname]
        ctrs_n, _, c1_n, _ = window_cosines(
            order, list(range(0, 200 - w + 1, step_non)), X)
        ax.plot(ctrs_n, np.abs(c1_n), "s-", color=colors[fname], ms=5, lw=1.2,
                label=fname)
    ax.set_xlabel("window centre depth [um]")
    ax.set_ylabel("|cos| adjacent-window PC1 (non-overlap)")
    ax.set_title(f"Non-overlapping windows (w={w}, step={step_non})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    ax = axes[2]
    for fname in win_fields:
        ctrs, _, c13 = real_curves[fname]
        ax.plot(ctrs, c13, "o-", color=colors[fname], ms=3.5, lw=1.1,
                label=fname)
    ax.set_xlabel("window centre depth [um]")
    ax.set_ylabel("worst-case cos PC1-3 subspaces (overlap)")
    ax.set_title("Adjacent-window 3D subspace alignment (max-angle cos)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.suptitle("Depth-window mode rotation (descriptive; pseudo-"
                 "depth ordering, no dynamics claimed)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "depth_window_mode_rotation.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote depth_window_mode_rotation.png")

    # ---- LOCO top-5 montage (total-residual PC1 influence) -----------------
    H, V = frozen["H"], frozen["V"]
    n_sub = len(subsets)
    fig, axes = plt.subplots(n_sub, 5, figsize=(11.5, 0.62 * n_sub + 1.2),
                             dpi=120)
    cmap_m = plt.get_cmap("viridis").copy()
    cmap_m.set_bad("0.82")
    for r, (name, rows_idx) in enumerate(subsets):
        cells = [t for t in total_top5 if t[0] == name]
        for c in range(5):
            ax = axes[r, c]
            if c < len(cells):
                _, rank, cid, ang, members = cells[c]
                dsi = int(members["dataset_index"].iloc[0])
                img = np.ma.masked_where(~V[dsi], H[dsi])
                ax.imshow(img, cmap=cmap_m)
                ax.set_title(f"#{rank} {ang:.0f}deg  {cid.split(':')[-1]}\n"
                             f"{members['session_id'].iloc[0].replace('zro2_', '')} "
                             f"s{int(members['sample_id'].iloc[0])} "
                             f"d={members['median_depth_um'].iloc[0]:.0f}um",
                             fontsize=4.6)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                axes[r, 0].set_ylabel(name, fontsize=5.5, rotation=0,
                                      ha="right", va="center")
    fig.suptitle("LOCO top-5 influential clusters per subset "
                 "(total-residual PC1; one member ROI shown)", fontsize=10)
    fig.subplots_adjust(wspace=0.02, hspace=0.28, top=0.955, bottom=0.005,
                        left=0.085, right=0.995)
    fig.savefig(out / "loco_top5_montage.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote loco_top5_montage.png")

    missing = [f for f in EXPECTED if not (out / f).exists()]
    _lib.require(not missing, f"missing outputs: {missing}")
    _lib.log(f"03 done in {_lib.elapsed(t0)}; all {len(EXPECTED)} outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
