#!/usr/bin/env python3
"""Phase 2 experiment 07: scale predictability summary (Phase 2B-5).

Aggregates `cv_fold_results.csv` (05) into the scale-predictability curve
R2_CV(lambda) at the main deterministic variant (src_gkf), plus the by-input
(dR2 reduced-derived vs raw) and by-model (dR2 ExtraTrees vs Ridge) band
views. Depth is drawn as a reference line only — it never enters any
scale-dependence trigger (细则 §0.10/§17).

Seed offsets: none (reads 05/06 outputs only).
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _lib as p2

EXPECTED = ["scale_predictability_summary.csv",
            "scale_predictability_curve.png", "scale_predictability_by_input.png",
            "scale_predictability_by_model.png", "README.md"]

BANDS = ["8_16", "16_32", "32_64", "64_inf"]
BAND_LO = {"8_16": 8.0, "16_32": 16.0, "32_64": 32.0, "64_inf": 64.0}

README = """# scale predictability summary (Phase 2B-5)

- `scale_predictability_summary.csv`: fold-quantiles of R2 at src_gkf for
  every (target, input_set, model).
- Curve figure: band-RMS targets vs band lower wavelength (log x); depth and
  Sq drawn as horizontal REFERENCE lines only — depth never enters a
  scale-dependence trigger (细则 §17 Route S compares morphology bands only).
- by-input: dR2 = R2(input R) - R2(input A) per band (fold-paired median,
  from 06); by-model: dR2 = ExtraTrees - Ridge per band.
- All values are exploratory CV estimates on n=200 (细则 §18).
"""


def main() -> int:
    cfg, quick = p2.load_config(__doc__)
    t0 = time.time()
    out = p2.output_dir(cfg, "process_explainability")
    p2.log("== Phase 2 / 07: scale predictability summary ==")
    res = pd.read_csv(out / "cv_fold_results.csv")
    vkey = "src_gkf" if "src_gkf" in set(res["cv_variant"]) \
        else sorted(set(res["cv_variant"]))[0]
    main_res = res[res["cv_variant"] == vkey]

    summary = (main_res.groupby(["target_id", "family", "input_set", "model"])
               ["R2"].agg(R2_median="median",
                          R2_q10=lambda s: s.quantile(0.10),
                          R2_q25=lambda s: s.quantile(0.25),
                          R2_q75=lambda s: s.quantile(0.75),
                          R2_q90=lambda s: s.quantile(0.90),
                          n_folds="size").reset_index())
    summary.to_csv(out / "scale_predictability_summary.csv", index=False)

    def _val(df, tid, iname, mname, col="R2_median"):
        sel = df[(df["target_id"] == tid) & (df["input_set"] == iname)
                 & (df["model"] == mname)]
        return float(sel[col].iloc[0]) if len(sel) else np.nan

    # ---- Figure 5: scale predictability curve --------------------------------
    models = [m for m in ("ridge", "extratrees")
              if m in set(summary["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(6.4 * len(models), 4.4),
                             sharey=True, squeeze=False)
    for j, mname in enumerate(models):
        ax = axes[0, j]
        for k, b in enumerate(BANDS):
            med = _val(summary, f"rms_DCT_{b}_um", "A", mname)
            q25 = _val(summary, f"rms_DCT_{b}_um", "A", mname, "R2_q25")
            q75 = _val(summary, f"rms_DCT_{b}_um", "A", mname, "R2_q75")
            ax.errorbar([BAND_LO[b]], [med],
                        yerr=[[max(med - q25, 0)], [max(q75 - med, 0)]],
                        fmt="o-", color=f"C{k}", capsize=3, lw=1.2,
                        label=f"DCT {b.replace('_', '-')} um")
        for ref, col in (("median_depth_um", "tab:gray"),
                         ("Sq_um", "tab:olive")):
            v = _val(summary, ref, "A", mname)
            if np.isfinite(v):
                ax.axhline(v, color=col, lw=1.0, ls="--",
                           label=f"{ref} (ref)")
        ax.set_xscale("log")
        ax.set_xticks([8, 16, 32, 64],
                      ["8-16", "16-32", "32-64", ">=64"])
        ax.set_xlabel("spatial wavelength band [um]")
        if j == 0:
            ax.set_ylabel("CV R2 median (src_gkf, input A)")
        ax.set_title(f"model = {mname}", fontsize=10)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Scale-resolved process explainability "
                 "(exploratory, n=200; depth/Sq are references only)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "scale_predictability_curve.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    # ---- by-input / by-model band views --------------------------------------
    def _band_view(path, title, diff_csv, hue, hue_vals):
        if not diff_csv.empty:
            sub = diff_csv[diff_csv["family"] == "C"]
            sub = sub[sub["target_id"].str.startswith("rms_DCT_")]
            sub = sub[sub["cv_variant"] == vkey]
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
            for k, hval in enumerate(hue_vals):
                xs, ys = [], []
                for b in BANDS:
                    row = sub[(sub[hue] == hval)
                              & (sub["target_id"] == f"rms_DCT_{b}_um")]
                    if len(row):
                        xs.append(BAND_LO[b])
                        ys.append(float(row["dR2_median"].iloc[0]))
                ax.plot(xs, ys, "o-", color=f"C{k}", label=str(hval))
            ax.axhline(0.0, color="0.4", lw=0.8)
            ax.set_xscale("log")
            ax.set_xticks([8, 16, 32, 64],
                          ["8-16", "16-32", "32-64", ">=64"])
            ax.set_xlabel("spatial wavelength band [um]")
            ax.set_ylabel("fold-paired dR2 median")
            ax.set_title(f"{title}  [cv_variant={vkey}]", fontsize=10)
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / path, dpi=cfg["plot"]["dpi"])
            plt.close(fig)

    reparam = pd.read_csv(out / "raw_vs_reparam_coordinates.csv")
    _band_view("scale_predictability_by_input.png",
               "dR2 = reduced derived (R) - raw (A)", reparam, "model",
               [m for m in ("ridge", "extratrees")
                if m in set(reparam["model"])])
    nonlin = pd.read_csv(out / "ridge_vs_tree.csv")
    _band_view("scale_predictability_by_model.png",
               "dR2 = ExtraTrees - Ridge", nonlin, "input_set",
               sorted(nonlin["input_set"].unique())
               if not nonlin.empty else [])

    (out / "README.md").write_text(README, encoding="utf-8")
    expected = [f for f in EXPECTED
                if not (quick and f == "scale_predictability_by_model.png")]
    missing = [f for f in expected if not (out / f).exists()]
    p2.require(not missing, f"missing outputs: {missing}")
    p2.log(f"07 done in {time.time() - t0:.1f}s: {len(summary)} summary cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
