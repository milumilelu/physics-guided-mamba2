#!/usr/bin/env python3
"""Phase 1.5 experiment 01: spatial-scale decomposition of the residual.

Disjoint Gaussian band split R = R_low + R_mid + R_high, variance partition,
and a sigma sweep of low-pass energy. Descriptive only.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _lib

EXPECTED = ["scale_energy_table.csv", "scale_decomposition_examples.png"]


def main() -> int:
    t0 = time.time()
    cfg, quick = _lib.load_config(__doc__)
    out = _lib.output_dir(cfg)
    _lib.log("== Phase 1.5 / 01: scale decomposition ==")
    frozen = _lib.load_frozen(cfg)
    man, R = frozen["man"], frozen["R"]
    pixel_um = float(cfg["scales"]["pixel_um"])

    sig_lo = float(cfg["scales"]["sigma_low_px"])
    sig_hi = float(cfg["scales"]["sigma_high_px"])
    _lib.log(f"  bands: sigma_low={sig_lo:.0f}px ({sig_lo * pixel_um}um low-pass "
             f"for R_low), sigma_high={sig_hi:.0f}px ({sig_hi * pixel_um}um "
             f"high-pass for R_high)")
    bands = _lib.make_bands(R, sig_lo, sig_hi)
    recon_err = float(np.max(np.abs(R - (bands["low"] + bands["mid"]
                                         + bands["high"]))))
    _lib.require(recon_err < 1e-8, f"band reconstruction error {recon_err}")
    _lib.log(f"  band reconstruction max |R - (low+mid+high)| = {recon_err:.2e}")

    # per-sample variance fractions
    def var_frac(part: np.ndarray) -> np.ndarray:
        num = np.nanvar(part, axis=(1, 2))
        den = np.nanvar(R, axis=(1, 2))
        return num / np.maximum(den, 1e-300)

    rows = []
    band_fracs = {}
    for name, part in (("R_low", bands["low"]), ("R_mid", bands["mid"]),
                       ("R_high", bands["high"])):
        f = var_frac(part)
        band_fracs[name] = f.mean()
        rows.append((f"{name} (sigma {sig_lo:.0f}/{sig_hi:.0f}px)", f.mean(),
                     np.median(f), *np.percentile(f, [25, 75])))
        _lib.log(f"  variance fraction {name}: mean={f.mean():.4f}, "
                 f"median={np.median(f):.4f}")
    frac_sum = sum(band_fracs.values())
    _lib.log(f"  variance fraction sum (low+mid+high) = {frac_sum:.3f} "
             "(Gaussian bands are pointwise exact but not orthogonal, so the "
             "variance fractions do not add to 1; cross terms below)")
    _lib.require(0.3 <= frac_sum <= 1.8,
                 f"band variance fractions sum to {frac_sum}, bands mis-set")
    mean_var = float(np.mean(np.nanvar(R, axis=(1, 2))))
    for n1, n2 in (("R_low", "R_mid"), ("R_mid", "R_high"),
                   ("R_low", "R_high")):
        cross = np.mean([
            np.nanmean(bands[n1.split("_")[1].lower()][i]
                       * bands[n2.split("_")[1].lower()][i])
            for i in range(R.shape[0])]) / mean_var
        rows.append((f"cross({n1},{n2})/Var(R)", cross, np.nan, np.nan, np.nan))
        _lib.log(f"  cross-covariance {n1}-{n2}: {cross:+.4f} of Var(R)")
    for sigma in cfg["scales"]["sigma_sweep_px"]:
        smooth = _lib.gaussian_smooth(R, float(sigma))
        f = var_frac(smooth)
        rows.append((f"lowpass G_{sigma}px ({float(sigma) * pixel_um}um)",
                     f.mean(), np.median(f), *np.percentile(f, [25, 75])))
        _lib.log(f"  lowpass sigma={sigma}px: variance fraction mean="
                 f"{f.mean():.4f}")
    df = pd.DataFrame(rows, columns=["component", "frac_mean", "frac_median",
                                     "frac_q25", "frac_q75"])
    df.to_csv(out / "scale_energy_table.csv", index=False)
    _lib.log("  wrote scale_energy_table.csv")

    # representative examples: shallowest / median-depth / deepest
    depths = man["median_depth_um"].to_numpy()
    picks = [int(np.argmin(depths)), int(np.argsort(depths)[100]),
             int(np.argmax(depths))]
    cmap_div = plt.get_cmap(cfg["plot"]["diverging_cmap"]).copy()
    cmap_div.set_bad("0.82")
    dpi = int(cfg["plot"]["dpi"])
    fig, axes = plt.subplots(3, 5, figsize=(15.5, 9.6), dpi=dpi)
    cols = [("H (height_raw)", frozen["H"], frozen["V"], "viridis"),
            ("R (residual)", R, None, cfg["plot"]["diverging_cmap"]),
            (f"R_low (>{sig_lo * pixel_um}um scale)", bands["low"], None,
             cfg["plot"]["diverging_cmap"]),
            (f"R_mid ({sig_hi * pixel_um}-{sig_lo * pixel_um}um)",
             bands["mid"], None, cfg["plot"]["diverging_cmap"]),
            (f"R_high (<{sig_hi * pixel_um}um)", bands["high"], None,
             cfg["plot"]["diverging_cmap"])]
    for r, idx in enumerate(picks):
        for c, (label, data, valid, cmap_name) in enumerate(cols):
            ax = axes[r, c]
            img = data[idx]
            if valid is not None:
                m = np.ma.masked_where(~valid[idx], img)
                cmap = plt.get_cmap(cmap_name).copy()
                cmap.set_bad("0.82")
                im = ax.imshow(m, cmap=cmap)
            else:
                vmax = np.nanmax(np.abs(img))
                im = ax.imshow(img, cmap=plt.get_cmap(cmap_name).copy(),
                               vmin=-vmax, vmax=vmax)
            if r == 0:
                ax.set_title(label, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"idx {idx}\ndepth {depths[idx]:.1f} um",
                              fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Scale decomposition examples (per-panel colour scales, um)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "scale_decomposition_examples.png", dpi=dpi,
                bbox_inches="tight")
    plt.close(fig)
    _lib.log("  wrote scale_decomposition_examples.png")

    missing = [f for f in EXPECTED if not (out / f).exists()]
    _lib.require(not missing, f"missing outputs: {missing}")
    _lib.log(f"01 done in {_lib.elapsed(t0)}; all {len(EXPECTED)} outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
