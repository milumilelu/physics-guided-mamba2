#!/usr/bin/env python3
"""Phase 1.5R experiment 02: per-scale residual PCA + cluster bootstrap.

Fields: total residual, Gaussian low-pass G2/G4/G8/G16, and DCT wavelength
bands. One pre-generated cluster resample bank (B>=1000) is shared across all
fields. The angle distribution is asymmetric, so Q25/Q50/Q75/Q90/Q95 are
reported and plotted as quantile bands (no median +/- IQR/2).
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
    _lib.log(f"== Phase 1.5R / 02: per-scale PCA + cluster bootstrap (B={B}, "
             "shared resample bank) ==")
    _lib._self_test()
    _lib.log("  self-test passed")

    frozen = _lib.load_frozen(cfg)
    fields = _lib.multiscale_fields(frozen["R"], cfg)
    clusters = _lib.cluster_lists(frozen["man"], cfg["bootstrap"]["cluster_key"])
    bank = _lib.build_resample_bank(clusters, B, seed)
    _lib.log(f"  {len(fields)} fields x {len(clusters)} clusters, "
             f"bank B={len(bank)}")

    rows = []
    curves = {}
    for name, X in fields.items():
        ref, evr = _lib.gram_pca(X, n_pcs)
        G = X @ X.T
        angles, _ = _lib.boot_angles_bank(G, X, bank, ref, k_max)
        q = _lib.angle_quantiles(angles)
        curves[name] = {"evr": evr, "q": q}
        for k in range(1, k_max + 1):
            rows.append((name, k, float(evr[k - 1]),
                         float(q["q25"][k - 1]), float(q["q50"][k - 1]),
                         float(q["q75"][k - 1]), float(q["q90"][k - 1]),
                         float(q["q95"][k - 1]),
                         float(angles[:, k - 1].min()),
                         float(angles[:, k - 1].max())))
        _lib.log(f"  [{name}] EVR PC1-3 = "
                 + " ".join(f"{e * 100:.2f}" for e in evr[:3])
                 + f"% | theta(k=1) Q50={q['q50'][0]:.1f} deg "
                 f"[Q25 {q['q25'][0]:.1f}, Q75 {q['q75'][0]:.1f}, "
                 f"Q90 {q['q90'][0]:.1f}, Q95 {q['q95'][0]:.1f}]"
                 f" | theta(k=6) Q50 = {q['q50'][-1]:.1f} deg")
    df = pd.DataFrame(rows, columns=["field", "k", "evr", "theta_q25_deg",
                                     "theta_q50_deg", "theta_q75_deg",
                                     "theta_q90_deg", "theta_q95_deg",
                                     "theta_min_deg", "theta_max_deg"])
    df.to_csv(out / "scale_pca_bootstrap.csv", index=False)
    _lib.log("  wrote scale_pca_bootstrap.csv")

    tot = df[(df.field == "total") & (df.k == 1)]["theta_q50_deg"].iloc[0]
    tot6 = df[(df.field == "total") & (df.k == 6)]["theta_q50_deg"].iloc[0]
    _lib.log(f"  cross-check vs Phase 1: total theta(k=1) Q50={tot:.1f} deg "
             f"(Phase 1 seed 20260901 gave 31.1), theta(k=6) Q50={tot6:.1f} "
             "deg (Phase 1 ~71)")

    dpi = int(cfg["plot"]["dpi"])
    ks = np.arange(1, k_max + 1)
    ke = np.arange(1, n_pcs + 1)
    main_fields = ["total", "G2", "G4", "G8", "G16",
                   "DCT_8_16", "DCT_16_32", "DCT_32_64", "DCT_64_inf"]
    colors = {"total": "black", "G2": "tab:cyan", "G4": "tab:blue",
              "G8": "tab:orange", "G16": "tab:green",
              "DCT_8_16": "tab:red", "DCT_16_32": "tab:purple",
              "DCT_32_64": "tab:brown", "DCT_64_inf": "tab:olive"}

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=dpi)
    for name in main_fields:
        ax.plot(ke, curves[name]["evr"] * 100, "o-", lw=1.4, ms=3.5,
                color=colors[name], label=name)
    ax.set_xlabel("principal component k")
    ax.set_ylabel("explained variance ratio [%]")
    ax.set_title("EVR(k): total residual vs filter-named scale fields")
    ax.set_xticks(ke)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "scale_evr_curves.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote scale_evr_curves.png")

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=dpi)
    for name in main_fields:
        q = curves[name]["q"]
        ax.plot(ks, q["q50"], "o-", color=colors[name], lw=1.6, ms=4,
                label=name)
        ax.fill_between(ks, q["q25"], q["q75"], color=colors[name], alpha=0.13)
    ax.axhline(20, color="tab:blue", ls=":", lw=1)
    ax.axhline(40, color="tab:red", ls=":", lw=1)
    ax.text(k_max + 0.05, 20, "stable benchmark 20deg", fontsize=7,
            color="tab:blue", va="center")
    ax.text(k_max + 0.05, 40, "unstable benchmark 40deg", fontsize=7,
            color="tab:red", va="center")
    ax.set_xticks(ks)
    ax.set_xlim(0.8, k_max + 1.9)
    ax.set_xlabel("subspace dimension k")
    ax.set_ylabel("bootstrap max principal angle [deg]")
    ax.set_title(f"Bootstrap subspace stability by scale (B={B}, shared bank; "
                 "Q50 line, Q25-Q75 band)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "scale_bootstrap_angles.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote scale_bootstrap_angles.png")

    missing = [f for f in EXPECTED if not (out / f).exists()]
    _lib.require(not missing, f"missing outputs: {missing}")
    _lib.log(f"02 done in {_lib.elapsed(t0)}; all {len(EXPECTED)} outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
