#!/usr/bin/env python3
"""Phase 1.5R experiment 01: spatial-scale decomposition of the residual.

Scales are named by their filters, not by guessed µm band labels:
- Gaussian low-pass fields G2/G4/G8/G16 (sigma in pixels) with the analytic
  and numeric -3 dB wavelength of the Gaussian transfer function;
- DCT band fields defined by physical wavelength shells (um).

Descriptive only.
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
    _lib.log("== Phase 1.5R / 01: scale decomposition ==")
    _lib._self_test()
    _lib.log("  self-test passed")
    frozen = _lib.load_frozen(cfg)
    R = frozen["R"]
    pixel_um = float(cfg["scales"]["pixel_um"])

    rows = []
    for sigma in cfg["scales"]["sigmas_px"]:
        G = _lib.gaussian_smooth(R, float(sigma))
        frac = float(np.mean(np.nanvar(G, axis=(1, 2))
                             / np.nanvar(R, axis=(1, 2))))
        lam_ana = _lib.lambda_3db_px(float(sigma)) * pixel_um
        lam_num = _lib.numeric_lambda_3db_px(float(sigma)) * pixel_um
        rows.append((f"G{int(sigma)}", f"low-pass sigma={int(sigma)}px",
                     frac, lam_ana, lam_num))
        _lib.log(f"  G{int(sigma)}: var fraction mean={frac:.4f}, "
                 f"lambda_3dB analytic={lam_ana:.2f} um "
                 f"(numeric {lam_num:.2f} um)")
    dct_fields, coverage = _lib.dct_band_fields(
        R, pixel_um, cfg["scales"]["dct_bands_um"])
    for name, fields in dct_fields.items():
        frac = float(np.mean(np.nanvar(fields, axis=(1, 2))
                             / np.nanvar(R, axis=(1, 2))))
        rows.append((name, "DCT wavelength band", frac, np.nan, np.nan))
        _lib.log(f"  {name}: var fraction mean={frac:.4f}")
    _lib.log(f"  DCT band coverage of pixel grid: {coverage:.4f} "
             "(wavelengths below the first band edge are excluded)")
    df = pd.DataFrame(rows, columns=["scale", "kind", "var_frac_mean",
                                     "lambda_3db_um_analytic",
                                     "lambda_3db_um_numeric"])
    df.to_csv(out / "scale_energy_table.csv", index=False)
    _lib.log("  wrote scale_energy_table.csv")

    # representative examples: shallowest / median-depth / deepest
    depths = frozen["man"]["median_depth_um"].to_numpy()
    picks = [int(np.argmin(depths)), int(np.argsort(depths)[100]),
             int(np.argmax(depths))]
    cmap_div = plt.get_cmap(cfg["plot"]["diverging_cmap"]).copy()
    cmap_div.set_bad("0.82")
    dpi = int(cfg["plot"]["dpi"])
    fig, axes = plt.subplots(3, 5, figsize=(15.5, 9.6), dpi=dpi)
    cols = [("H (height_raw)", frozen["H"], frozen["V"], "viridis"),
            ("R (residual)", R, None, cfg["plot"]["diverging_cmap"]),
            ("G16 (low-pass, sigma=16px)",
             _lib.gaussian_smooth(R, 16.0), None,
             cfg["plot"]["diverging_cmap"]),
            ("DCT_16_32 (16-32 um)", dct_fields["DCT_16_32"], None,
             cfg["plot"]["diverging_cmap"]),
            ("DCT_8_16 (8-16 um)", dct_fields["DCT_8_16"], None,
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
