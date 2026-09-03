#!/usr/bin/env python3
"""Phase 2 experiment 06: physics-coordinate and model-increment comparison.

Fold-paired differences from `cv_fold_results.csv` (05):
  dR2_reparam = R2(input=R) - R2(input=A)   at fixed model / cv_variant / fold;
  dR2_nonlin  = R2(ExtraTrees) - R2(Ridge)  at fixed input / cv_variant / fold.
Reported per cell: fold median, Q25/Q75, sign-consistent folds / total folds.
Exploratory direction-consistency only — no p-value wrapping (细则 §8).

Interpretation guard (细则 §0.2): a positive dR2_reparam means the
reduced derived features fit the model's inductive bias better, NOT that
more physical information was added.

Seed offsets: none (reads 05 outputs only).
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _lib as p2

EXPECTED = ["raw_vs_reparam_coordinates.csv", "ridge_vs_tree.csv",
            "raw_vs_reparam_coordinates.png", "ridge_vs_tree.png", "README.md"]

README = """# coordinate / model increment comparison (Phase 2B-4)

- `raw_vs_reparam_coordinates.csv`: dR2 = R2(reduced derived R) - R2(raw A), fold
  paired, per (target, model, cv_variant) at the main deterministic variant.
- `ridge_vs_tree.csv`: dR2 = R2(ExtraTrees) - R2(Ridge), fold paired, per
  (target, input_set, cv_variant).
- `n_folds_pos` / `n_folds` is a sign-consistency count, not a test
  statistic; quick mode has no ExtraTrees and writes an empty nonlin table.
- Positive dR2_reparam = reparameterization fits the model inductive bias
  better; it does NOT mean added physical information (细则 §0.2/§18).
"""


def _diff_table(res: pd.DataFrame, vary_col: str, base: str, other: str,
                fixed_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold-paired `other - base` of R2 within
    (target, family, cv_variant, *fixed_cols).

    Returns (aggregated table, per-fold merged table)."""
    keep = ["target_id", "family", "cv_variant", "fold"] + fixed_cols
    base_df = res[res[vary_col] == base][keep + ["R2"]] \
        .rename(columns={"R2": "R2_base"})
    other_df = res[res[vary_col] == other][keep + ["R2"]] \
        .rename(columns={"R2": "R2_other"})
    m = base_df.merge(other_df, on=keep, validate="one_to_one")
    m["dR2"] = m["R2_other"] - m["R2_base"]
    grp_cols = ["target_id", "family", "cv_variant"] + fixed_cols
    agg = m.groupby(grp_cols)["dR2"].agg(
        dR2_median="median",
        dR2_q25=lambda s: s.quantile(0.25),
        dR2_q75=lambda s: s.quantile(0.75),
        n_folds="size").reset_index()
    agg["n_folds_pos"] = m.assign(pos=m["dR2"] > 0) \
        .groupby(grp_cols)["pos"].sum().to_numpy()
    agg["pair"] = f"{other}-{base}"
    return agg, m


def _bar_plot(ax, agg: pd.DataFrame, m: pd.DataFrame, hue: str, title: str,
              vkey: str):
    targets = sorted(agg["target_id"].unique())
    xpos = np.arange(len(targets))
    hues = sorted(agg[hue].unique())
    width = 0.8 / max(len(hues), 1)
    for k, hval in enumerate(hues):
        sub = agg[agg[hue] == hval].set_index("target_id").reindex(targets)
        ax.bar(xpos + (k - (len(hues) - 1) / 2) * width,
               sub["dR2_median"], width=width, label=str(hval), color=f"C{k}")
        mf = m[m[hue] == hval]
        for xi, tid in enumerate(targets):
            dvals = mf[mf["target_id"] == tid]["dR2"].to_numpy()
            ax.plot(np.full(len(dvals), xi + (k - (len(hues) - 1) / 2) * width),
                    dvals, ".", color="0.3", ms=2.5, alpha=0.5)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(xpos, [t.replace("_", " ") for t in targets], rotation=60,
                  fontsize=6.5)
    ax.set_ylabel("fold-paired dR2 (median; dots = folds)")
    ax.set_title(f"{title}  [cv_variant={vkey}]", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")


def main() -> int:
    cfg, quick = p2.load_config(__doc__)
    t0 = time.time()
    out = p2.output_dir(cfg, "process_explainability")
    p2.log("== Phase 2 / 06: coordinate & model increments ==")
    res = pd.read_csv(out / "cv_fold_results.csv")
    vkey = "src_gkf" if "src_gkf" in set(res["cv_variant"]) \
        else sorted(set(res["cv_variant"]))[0]

    agg_reparam, m_reparam = _diff_table(res, "input_set", "A", "R",
                                         fixed_cols=["model"])
    # CSVs cover ALL cv variants (细则 §8: per (target, variant)); the plot
    # shows only the main deterministic variant.
    agg_reparam.to_csv(out / "raw_vs_reparam_coordinates.csv", index=False)

    if "extratrees" in set(res["model"]):
        agg_nonlin, m_nonlin = _diff_table(res, "model", "ridge", "extratrees",
                                           fixed_cols=["input_set"])
    else:
        agg_nonlin = pd.DataFrame(
            columns=["target_id", "family", "cv_variant", "dR2_median",
                     "dR2_q25", "dR2_q75", "n_folds", "n_folds_pos", "pair"])
        m_nonlin = pd.DataFrame()
    agg_nonlin.to_csv(out / "ridge_vs_tree.csv", index=False)

    agg_reparam_plot = agg_reparam[agg_reparam["cv_variant"] == vkey]
    m_reparam_plot = m_reparam[m_reparam["cv_variant"] == vkey] \
        if len(m_reparam) else m_reparam
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    _bar_plot(ax, agg_reparam_plot, m_reparam_plot, "model",
              "dR2 = reduced derived (R) - raw (A), per target", vkey)
    fig.tight_layout()
    fig.savefig(out / "raw_vs_reparam_coordinates.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    if not agg_nonlin.empty:
        agg_nonlin_plot = agg_nonlin[agg_nonlin["cv_variant"] == vkey]
        m_nonlin_plot = m_nonlin[m_nonlin["cv_variant"] == vkey]
        fig, ax = plt.subplots(figsize=(12.5, 4.4))
        _bar_plot(ax, agg_nonlin_plot, m_nonlin_plot, "input_set",
                  "dR2 = ExtraTrees - Ridge, per target", vkey)
        fig.tight_layout()
        fig.savefig(out / "ridge_vs_tree.png", dpi=cfg["plot"]["dpi"])
        plt.close(fig)

    (out / "README.md").write_text(README, encoding="utf-8")
    expected = [f for f in EXPECTED
                if not (quick and f == "ridge_vs_tree.png")]
    missing = [f for f in expected if not (out / f).exists()]
    p2.require(not missing, f"missing outputs: {missing}")
    p2.log(f"06 done in {time.time() - t0:.1f}s; reparam cells "
           f"{len(agg_reparam)}, nonlin cells {len(agg_nonlin)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
