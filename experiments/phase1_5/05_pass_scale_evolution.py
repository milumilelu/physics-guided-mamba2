#!/usr/bin/env python3
"""Phase 1.5 experiment 05: morphology descriptors, per-scale pass-step
evolution, and the deterministic-stochastic map.
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
            "pass_scale_evolution.png", "deterministic_stochastic_map.csv",
            "deterministic_stochastic_map.png"]

PIXEL_UM = 0.5


def compute_descriptors(R: np.ndarray, bands: dict[str, np.ndarray]) -> pd.DataFrame:
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
        prof_y, prof_x = acf[:, 0], acf[0, :]
        crosses = []
        for prof in (prof_y, prof_x):
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
    for band in ("low", "mid", "high"):
        out[f"rms_{band}_um"] = np.sqrt(np.mean(bands[band] ** 2, axis=(1, 2)))
        out[f"E_{band}_frac"] = (
            np.mean(bands[band] ** 2, axis=(1, 2))
            / np.maximum(np.mean(R ** 2, axis=(1, 2)), 1e-300))
    return out


def main() -> int:
    t0 = time.time()
    cfg, quick = _lib.load_config(__doc__)
    out = _lib.output_dir(cfg)
    B = _lib.n_boot(cfg, quick)
    seed = int(cfg["random_seed"])
    _lib.log(f"== Phase 1.5 / 05: descriptors + pass evolution + map (B={B}) ==")
    frozen = _lib.load_frozen(cfg)
    man = frozen["man"]
    R = frozen["R"]
    bands = _lib.make_bands(R, float(cfg["scales"]["sigma_low_px"]),
                            float(cfg["scales"]["sigma_high_px"]))

    desc = compute_descriptors(R, bands)
    desc_out = desc.copy()
    desc_out.insert(1, "session_id", man["session_id"].to_numpy())
    desc_out.insert(2, "median_depth_um", man["median_depth_um"].to_numpy())
    desc_out.to_csv(out / "morphology_descriptors.csv", index=False)
    _lib.log(f"  wrote morphology_descriptors.csv ({desc_out.shape[1] - 1} quantities)")

    # ---- pass evolution per band -------------------------------------------
    fields = {"total": _lib.to_2d(R),
              "low": _lib.to_2d(bands["low"]),
              "mid": _lib.to_2d(bands["mid"]),
              "high": _lib.to_2d(bands["high"])}
    traj = man[man["session_role"] == "pass_main"]
    groups = sorted(traj["design_group"].unique())
    _lib.require(len(groups) == 15, "trajectory count != 15")
    depth = man["median_depth_um"].to_numpy()
    step_rows = []
    for band, X in fields.items():
        steps_rms = {1: [], 2: [], 3: []}
        cos_vals = []
        cross = {1: [], 2: [], 3: []}
        dd = {1: [], 2: [], 3: []}
        vecs = {1: [], 2: [], 3: []}
        for g in groups:
            gdf = traj[traj["design_group"] == g].sort_values("pass_count")
            idx = gdf["dataset_index"].to_numpy()
            for s in (1, 2, 3):
                dv = X[idx[s]] - X[idx[s - 1]]
                vecs[s].append(dv)
                steps_rms[s].append(float(np.sqrt(np.mean(dv ** 2))))
                dd[s].append(float(depth[idx[s]] - depth[idx[s - 1]]))
            for s in (1, 2):
                a, b = vecs[s][-1], vecs[s + 1][-1]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 0 and nb > 0:
                    cos_vals.append(float(np.clip(a @ b / (na * nb), -1, 1)))
        for s in (1, 2, 3):
            V = np.asarray(vecs[s])
            for i, j in itertools.combinations(range(len(V)), 2):
                na, nb = np.linalg.norm(V[i]), np.linalg.norm(V[j])
                if na > 0 and nb > 0:
                    cross[s].append(float(np.clip(V[i] @ V[j] / (na * nb), -1, 1)))
            step_rows.append((band, s, float(np.median(steps_rms[s])),
                              float(np.percentile(steps_rms[s], 25)),
                              float(np.percentile(steps_rms[s], 75)),
                              float(np.median(cross[s])) if cross[s] else np.nan,
                              len(cross[s]),
                              float(np.median(dd[s])),
                              float(np.median(cos_vals))))
    pd.DataFrame(step_rows, columns=[
        "band", "step", "median_step_rms_um", "step_rms_q25", "step_rms_q75",
        "median_across_traj_cos", "n_across_pairs", "median_depth_step_um",
        "median_consecutive_cos"]).to_csv(out / "pass_step_stats.csv", index=False)
    _lib.log("  wrote pass_step_stats.csv")
    for band in fields:
        sub = [r for r in step_rows if r[0] == band]
        _lib.log(f"  [{band}] step RMS medians (N1->2, 2->3, 3->4): "
                 + " ".join(f"{r[2]:.3f}" for r in sub)
                 + " um | across-traj step-direction cos: "
                 + " ".join(f"{r[5]:.2f}" for r in sub))

    dpi = int(cfg["plot"]["dpi"])
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.9), dpi=dpi)
    colors = {"total": "black", "low": "tab:blue", "mid": "tab:green",
              "high": "tab:red"}
    steps = (1, 2, 3)
    for band in fields:
        sub = [r for r in step_rows if r[0] == band]
        med = np.array([r[2] for r in sub])
        q25 = np.array([r[3] for r in sub])
        q75 = np.array([r[4] for r in sub])
        axes[0].errorbar(steps, med, yerr=[med - q25, q75 - med], fmt="o-",
                         color=colors[band], lw=1.4, ms=4, capsize=3,
                         label=band)
    axes[0].set_xticks(steps, ["N1->2", "N2->3", "N3->4"])
    axes[0].set_ylabel("median step RMS [um]")
    axes[0].set_title("Per-scale pass step magnitude")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
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
    axes[2].set_title("Depth increment per pass step")
    axes[2].grid(alpha=0.25)
    cos_by_band = []
    for band, X in fields.items():
        vals = []
        for g in groups:
            gdf = traj[traj["design_group"] == g].sort_values("pass_count")
            idx = gdf["dataset_index"].to_numpy()
            vecs = [X[idx[s]] - X[idx[s - 1]] for s in (1, 2, 3)]
            for s in (1, 2):
                a, b = vecs[s - 1], vecs[s]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 0 and nb > 0:
                    vals.append(float(np.clip(a @ b / (na * nb), -1, 1)))
        cos_by_band.append(vals)
    bp = axes[1].boxplot(cos_by_band, tick_labels=list(fields), showmeans=True,
                         patch_artist=True)
    for i, band in enumerate(fields):
        bp["boxes"][i].set_facecolor("lightblue" if band != "total" else "0.9")
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    axes[1].axhline(0.5, color="tab:red", ls=":", lw=1)
    axes[1].set_ylabel("consecutive-step direction cos")
    axes[1].set_title("Step-direction persistence per band (15 trajs)")
    axes[1].grid(alpha=0.25, axis="y")
    fig.suptitle("N=1-4 pass evolution by spatial scale", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "pass_scale_evolution.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote pass_scale_evolution.png")

    # ---- deterministic-stochastic map --------------------------------------
    i_a, i_b = _lib.sentinel_rows(man, cfg)
    pu, pv = _lib.ordinary_pair_mask(man, i_a, i_b)
    clusters = _lib.cluster_lists(man, cfg["bootstrap"]["cluster_key"])
    rng = np.random.default_rng(seed + 42)
    draws = [np.concatenate([clusters[p] for p in
                             rng.integers(0, len(clusters), size=len(clusters))])
             for _ in range(B)]
    quantities: list[tuple[str, np.ndarray, str]] = [
        ("median_depth_um", man["median_depth_um"].to_numpy(), "um"),
        ("residual_Sq_um", man["residual_Sq_um"].to_numpy(), "um"),
        ("rms_low_um", desc["rms_low_um"].to_numpy(), "um"),
        ("rms_mid_um", desc["rms_mid_um"].to_numpy(), "um"),
        ("rms_high_um", desc["rms_high_um"].to_numpy(), "um"),
        ("Sq_um", desc["Sq_um"].to_numpy(), "um"),
        ("Sa_um", desc["Sa_um"].to_numpy(), "um"),
        ("Ssk", desc["Ssk"].to_numpy(), "-"),
        ("Sku", desc["Sku"].to_numpy(), "-"),
        ("grad_rms", desc["grad_rms_um_per_um"].to_numpy(), "um/um"),
        ("aniso_x_over_y", desc["aniso_gradx_over_y"].to_numpy(), "-"),
        ("lap_rms_um", desc["lap_rms_um"].to_numpy(), "um"),
        ("corr_len_um", desc["corr_len_um"].to_numpy(), "um"),
        ("E_low_frac", desc["E_low_frac"].to_numpy(), "-"),
        ("E_mid_frac", desc["E_mid_frac"].to_numpy(), "-"),
        ("E_high_frac", desc["E_high_frac"].to_numpy(), "-"),
        ("pit_count", desc["pit_count_per_roi"].to_numpy().astype(float), "px"),
        ("pit_depth_um", desc["pit_depth_um"].to_numpy(), "um"),
    ]
    map_rows = []
    for name, q, unit in quantities:
        sd = float(np.std(q))
        iqr = float(np.percentile(q, 75) - np.percentile(q, 25))
        boot_med = np.array([np.median(q[d]) for d in draws])
        se = float(np.std(boot_med))
        s_val = float(np.clip(1.0 - se / max(sd, 1e-300), 0.0, 1.0))
        diffs = np.abs(q[pu] - q[pv])
        sent = abs(float(q[i_a]) - float(q[i_b]))
        pct = float(np.mean(diffs < sent) * 100)
        map_rows.append((name, unit, sd, iqr, se, s_val, sent, pct))
    mdf = pd.DataFrame(map_rows, columns=[
        "observable", "unit", "sd_across_samples", "iqr_across_samples",
        "boot_se_of_median", "S_bootstrap_stability",
        "sentinel_abs_delta", "sentinel_pct_of_ordinary"])
    mdf.to_csv(out / "deterministic_stochastic_map.csv", index=False)
    _lib.log("  wrote deterministic_stochastic_map.csv")

    fig, ax = plt.subplots(figsize=(12.5, 7.6), dpi=dpi)
    ax.axis("off")
    hdr = (f"{'observable':<16}{'unit':>7}{'SD':>10}{'IQR':>10}"
           f"{'S(q)':>7}{'49/50 pct':>11}   class (S | repeatability)")
    lines = [hdr, "-" * len(hdr)]
    for r in map_rows:
        s_cls = "high" if r[5] >= 0.8 else ("medium" if r[5] >= 0.5 else "low")
        r_cls = ("high" if r[7] <= 5 else ("medium" if r[7] <= 20 else "low"))
        lines.append(f"{r[0]:<16}{r[1]:>7}{r[2]:>10.3f}{r[3]:>10.3f}"
                     f"{r[5]:>7.2f}{r[7]:>11.1f}   {s_cls} | {r_cls}")
    ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=9,
            va="top", transform=ax.transAxes)
    ax.set_title("Deterministic-stochastic map (S = 1 - bootSE/SD; "
                 "repeatability class: high<=5%, medium<=20% of ordinary pairs)",
                 fontsize=10, family="monospace")
    fig.tight_layout()
    fig.savefig(out / "deterministic_stochastic_map.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote deterministic_stochastic_map.png")

    missing = [f for f in EXPECTED if not (out / f).exists()]
    _lib.require(not missing, f"missing outputs: {missing}")
    _lib.log(f"05 done in {_lib.elapsed(t0)}; all {len(EXPECTED)} outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
