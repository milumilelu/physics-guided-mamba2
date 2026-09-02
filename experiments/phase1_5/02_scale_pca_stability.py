#!/usr/bin/env python3
"""Phase 1.5 experiment 02: per-scale residual PCA + cluster bootstrap.

For total residual and each band (low/mid/high) plus the sigma sweep of
low-pass fields, report EVR(k=1..10) and bootstrap principal-angle stability
(k=1..6, B=200, cluster = shared_height_source_id). Descriptive only.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _lib

EXPECTED = ["scale_pca_bootstrap.csv", "scale_evr_curves.png",
            "scale_bootstrap_angles.png"]


def main() -> int:
    t0 = time.time()
    cfg, quick = _lib.load_config(__doc__)
    out = _lib.output_dir(cfg)
    B = _lib.n_boot(cfg, quick)
    seed = int(cfg["random_seed"])
    k_max = int(cfg["pca"]["angle_k_max"])
    n_pcs = int(cfg["pca"]["n_pcs"])
    _lib.log(f"== Phase 1.5 / 02: per-scale PCA + cluster bootstrap (B={B}) ==")
    _lib._self_test()
    _lib.log("  fast-bootstrap self-test passed")

    frozen = _lib.load_frozen(cfg)
    R2 = _lib.to_2d(frozen["R"])
    bands = _lib.make_bands(frozen["R"], float(cfg["scales"]["sigma_low_px"]),
                            float(cfg["scales"]["sigma_high_px"]))
    fields = {"total": R2,
              "low": _lib.to_2d(bands["low"]),
              "mid": _lib.to_2d(bands["mid"]),
              "high": _lib.to_2d(bands["high"])}
    for sigma in cfg["scales"]["sigma_sweep_px"]:
        fields[f"lowpass_{int(sigma)}px"] = _lib.to_2d(
            _lib.gaussian_smooth(frozen["R"], float(sigma)))

    clusters = _lib.cluster_lists(frozen["man"], cfg["bootstrap"]["cluster_key"])
    _lib.log(f"  {len(clusters)} clusters, "
             f"{sum(len(c) for c in clusters)} rows total")

    rows = []
    curves = {}
    for name, X in fields.items():
        ref, evr = _lib.gram_pca(X, n_pcs)
        G = X @ X.T
        angles, evr_b = _lib.boot_angles(G, X, clusters, ref, k_max, B, seed)
        curves[name] = {"evr": evr, "angles": angles}
        for k in range(1, k_max + 1):
            v = angles[:, k - 1]
            rows.append((name, k, float(evr[k - 1]), float(np.median(v)),
                         float(np.percentile(v, 75) - np.percentile(v, 25)),
                         float(v.min()), float(v.max())))
        q1m, q2m, q3m = np.median(angles[:, 0]), None, None
        _lib.log(f"  [{name}] EVR PC1-3 = "
                 + " ".join(f"{e * 100:.2f}" for e in evr[:3])
                 + f"% | theta_boot(k=1) median = {np.median(angles[:, 0]):.1f} deg"
                 f" [{np.percentile(angles[:, 0], 25):.1f},"
                 f" {np.percentile(angles[:, 0], 75):.1f}]"
                 f" | theta(k=6) = {np.median(angles[:, -1]):.1f} deg")
    df = pd.DataFrame(rows, columns=["field", "k", "evr", "theta_median_deg",
                                     "theta_iqr_deg", "theta_min_deg",
                                     "theta_max_deg"])
    df.to_csv(out / "scale_pca_bootstrap.csv", index=False)
    _lib.log("  wrote scale_pca_bootstrap.csv")

    tot1 = float(df[(df.field == "total") & (df.k == 1)]["theta_median_deg"].iloc[0])
    tot6 = float(df[(df.field == "total") & (df.k == 6)]["theta_median_deg"].iloc[0])
    _lib.log(f"  cross-check vs Phase 1: total theta(k=1)={tot1:.1f} deg "
             f"(Phase 1 ~31), theta(k=6)={tot6:.1f} deg (Phase 1 ~71)")

    dpi = int(cfg["plot"]["dpi"])
    ks = np.arange(1, k_max + 1)
    ke = np.arange(1, n_pcs + 1)
    canon = {"total": "black", "low": "tab:blue", "mid": "tab:green",
             "high": "tab:red"}
    sweep = [n for n in fields if n.startswith("lowpass")]

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=dpi)
    for name in sweep:
        ax.plot(ke, curves[name]["evr"] * 100, "--", color="0.6", lw=1.0,
                label="_")
        ax.text(ke[-1] + 0.1, curves[name]["evr"][0] * 100,
                name.replace("lowpass_", "G"), fontsize=7, color="0.4",
                va="center")
    for name, color in canon.items():
        ax.plot(ke, curves[name]["evr"] * 100, "o-", color=color, lw=1.6,
                ms=4, label=name)
    ax.set_xlabel("principal component k")
    ax.set_ylabel("explained variance ratio [%]")
    ax.set_title("EVR(k): total residual vs spatial-scale bands")
    ax.set_xticks(ke)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "scale_evr_curves.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote scale_evr_curves.png")

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=dpi)
    for name, color in canon.items():
        med = np.array([df[(df.field == name) & (df.k == k)]["theta_median_deg"].iloc[0]
                        for k in ks])
        iqr = np.array([df[(df.field == name) & (df.k == k)]["theta_iqr_deg"].iloc[0]
                        for k in ks])
        ax.plot(ks, med, "o-", color=color, lw=1.8, ms=4, label=name)
        ax.fill_between(ks, med - iqr / 2, med + iqr / 2, color=color,
                        alpha=0.15)
    for name in sweep:
        med = np.array([df[(df.field == name) & (df.k == k)]["theta_median_deg"].iloc[0]
                        for k in ks])
        ax.plot(ks, med, "--", color="0.6", lw=1.0)
    ax.axhline(20, color="tab:blue", ls=":", lw=1)
    ax.axhline(40, color="tab:red", ls=":", lw=1)
    ax.text(k_max + 0.05, 20, "stable benchmark 20deg", fontsize=7,
            color="tab:blue", va="center")
    ax.text(k_max + 0.05, 40, "unstable benchmark 40deg", fontsize=7,
            color="tab:red", va="center")
    ax.set_xticks(ks)
    ax.set_xlim(0.8, k_max + 1.6)
    ax.set_xlabel("subspace dimension k")
    ax.set_ylabel("bootstrap max principal angle [deg]")
    ax.set_title("Bootstrap subspace stability by spatial scale "
                 f"(B={B}, cluster=shared source)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "scale_bootstrap_angles.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote scale_bootstrap_angles.png")

    missing = [f for f in EXPECTED if not (out / f).exists()]
    _lib.require(not missing, f"missing outputs: {missing}")
    _lib.log(f"02 done in {_lib.elapsed(t0)}; all {len(EXPECTED)} outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
