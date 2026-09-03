#!/usr/bin/env python3
"""Phase 2.5 Task 10: normalized spectral composition (dual-track).

Track A (reconciliation, provenance): replicate the Phase 1.5-05 frozen
convention exactly (E_b = mean(R_b^2) / mean(R^2), the DC-inside->=64-band
second-moment ratio) and reconcile against the committed descriptor CSV at
atol 1e-8, PLUS the rev2 algebraic identities
    E_b^frozen   = (1 - r_DC) * p_b           (DC-free bands)
    E_64^frozen  = r_DC + (1 - r_DC) * p_64   (>=64 band contains DC)
with r_DC = dc_offset_frac = mean(R)^2 / mean(R^2).

Track B (science): five-part clean composition from NON-DC coefficient
energies -> ILR balances Z1..Z4 -> radial log-lambda spectrum (24 bins on
[0.7, 160] um) -> spectrum descriptors -> amplitude-vs-fraction consistency
(frozen and clean identities) -> bin-count sensitivity (16/24/32).

Seed offsets: none (fully deterministic).
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _lib as p25

EXPECTED = ["spectral_composition.csv", "ilr_coordinates.csv",
            "radial_spectrum_long.csv", "radial_spectrum_matrix.npz",
            "spectrum_descriptor_summary.csv",
            "amplitude_vs_fraction_consistency.csv", "dct_reconciliation.csv",
            "radial_bin_sensitivity.csv", "mean_normalized_spectrum.png",
            "spectrum_by_process_quantiles.png", "README.md"]

BANDS5 = p25.ILR_BANDS
ATOL = None  # set from config in main

README = """# spectral composition (Task 10)

- Track A `dct_reconciliation.csv`: the frozen Phase 1.5 convention
  (E_b = mean(R_b^2)/mean(R^2); DC sits inside the >=64 um band) is replicated
  and reconciled against the committed descriptor CSV; the rev2 algebraic
  identities E_b^frozen = (1-r_DC) p_b / E_64^frozen = r_DC + (1-r_DC) p_64
  are asserted at 1e-8. `var_R` in 1.5-05 is the SECOND moment mean(R^2)
  despite its name.
- Track B `spectral_composition.csv` / `ilr_coordinates.csv`: clean NON-DC
  five-part composition + ILR balances Z1..Z4 + `dc_offset_frac`
  (= mean(R)^2/mean(R^2), a DC/mean-offset descriptor kept OUT of the
  composition; optional ratio column dc_to_non_dc_ratio).
- `radial_spectrum_long.csv` / `radial_spectrum_matrix.npz` (cache,
  gitignored): 24 log bins on [0.7, 160] um; `low_mode_count` bins (<20
  modes) are flagged and must be shaded in figures; they never carry
  pointwise inferential claims.
- `amplitude_vs_fraction_consistency.csv`: RMS_b = Sq*sqrt(E_b^frozen)
  (frozen identity) and RMS_{b,nonDC} = Sq*sqrt((1-r_DC) p_b) (clean) —
  the algebraic reason fraction-predictability and RMS-unpredictability
  coexist.
- `radial_bin_sensitivity.csv`: descriptor stability across 16/24/32 bins.
"""


def main() -> int:
    global ATOL
    cfg, quick = p25.load_config(__doc__)
    t0 = time.time()
    out = p25.output_dir(cfg, "spectral_composition")
    ATOL = float(cfg["spectrum"]["reconciliation_atol"])
    p2log = p25.log
    p2log("== Phase 2.5 / 10: normalized spectral composition ==")
    frozen = p25.l15.load_frozen(cfg)
    R = frozen["R"]
    pixel = float(cfg["scales"]["pixel_um"])

    # ---- Track A: frozen convention + reconciliation ------------------------
    E_frozen, coverage = p25.frozen_band_fractions(
        R, cfg["scales"]["dct_bands_um"], pixel)
    desc = pd.read_csv(p25.l15.REPO / "outputs/phase1_5/morphology_descriptors.csv")
    band4 = ["8_16", "16_32", "32_64", "64_inf"]
    recon_rows = []
    for b in band4:
        col = f"E_DCT_{b}_frac"
        ref = desc[col].to_numpy()
        resid = np.max(np.abs(E_frozen[f"DCT_{b}"] - ref))
        recon_rows.append({"band": b, "identity": "frozen_vs_phase15_csv",
                           "max_abs_residual": float(resid)})
        p25.require(resid <= ATOL, f"frozen fraction mismatch for {b}: {resid}")
    p2log(f"  frozen convention replicated (coverage {coverage:.4f}); "
          "matches Phase 1.5 CSV at 1e-8")

    # ---- Track B: clean five-part composition + ILR -------------------------
    p, dc_offset = p25.five_part_composition(R, pixel)
    P = np.column_stack([p[b] for b in BANDS5])
    p25.require(np.allclose(P.sum(axis=1), 1.0, atol=1e-12), "sum p != 1")
    p25.require(P.min() > float(cfg["composition"]["zero_threshold"]),
                "composition below zero threshold — replacement branch must "
                "be revisited (expected: never on this dataset)")
    r_dc = dc_offset
    for b, name in (("8_16", "DCT_8_16"), ("16_32", "DCT_16_32"),
                    ("32_64", "DCT_32_64")):
        resid = float(np.max(np.abs(E_frozen[f"DCT_{b}"]
                                    - (1 - r_dc) * p[b])))
        recon_rows.append({"band": b, "identity": "clean_identity",
                           "max_abs_residual": resid})
        p25.require(resid <= ATOL, f"clean identity failed for {b}: {resid}")
    resid64 = float(np.max(np.abs(
        E_frozen["DCT_64_inf"] - (r_dc + (1 - r_dc) * p["64_inf"]))))
    recon_rows.append({"band": "64_inf", "identity": "clean_identity",
                       "max_abs_residual": resid64})
    p25.require(resid64 <= ATOL, f"clean identity failed for 64_inf: {resid64}")
    p2log("  rev2 identities verified at 1e-8: E_b^frozen=(1-r_DC)p_b, "
          "E_64^frozen=r_DC+(1-r_DC)p_64")
    pd.DataFrame(recon_rows).to_csv(out / "dct_reconciliation.csv",
                                    index=False)

    comp = pd.DataFrame({"dataset_index": np.arange(len(R))})
    for b in BANDS5:
        comp[f"p_{b}"] = p[b]
    comp["dc_offset_frac"] = dc_offset
    comp["dc_to_non_dc_ratio"] = dc_offset / np.maximum(1 - dc_offset, 1e-300)
    comp.to_csv(out / "spectral_composition.csv", index=False)

    z = p25.ilr_transform(P)
    p25.require(np.allclose(p25.ilr_inverse(z), P, atol=1e-10),
                "ILR roundtrip failed")
    ilr = pd.DataFrame({"dataset_index": np.arange(len(R))})
    for j in range(4):
        ilr[f"ilr_z{j + 1}"] = z[:, j]
    ilr.to_csv(out / "ilr_coordinates.csv", index=False)
    p2log("  five-part composition + ILR written (p_lt8 median "
          f"{np.median(p['lt8']):.4f}, max {p['lt8'].max():.4f})")

    # ---- radial spectrum -----------------------------------------------------
    scfg = cfg["spectrum"]
    radial, edges = p25.radial_spectrum(R, pixel, int(scfg["radial_log_bins"]),
                                        float(scfg["lambda_lo_um"]),
                                        float(scfg["lambda_hi_um"]))
    p25.require(float(radial["uncovered_frac"].max()) < 1e-9,
                "radial bins do not cover the non-DC spectrum")
    long_rows = []
    lam_geo = radial["lambda_geo_um"]
    dlog = np.log(radial["lambda_hi_um"] / radial["lambda_lo_um"])
    for b in range(len(edges) - 1):
        e_frac = radial["energy_fraction"][:, b]
        long_rows.append(pd.DataFrame({
            "dataset_index": np.arange(len(R)),
            "bin": b, "lambda_lo_um": radial["lambda_lo_um"][b],
            "lambda_hi_um": radial["lambda_hi_um"][b],
            "lambda_geo_um": lam_geo[b], "n_modes": radial["n_modes"][0, b],
            "energy": radial["energy"][:, b],
            "energy_fraction": e_frac,
            "energy_density_per_loglambda": e_frac / dlog[b],
            "low_mode_count": radial["low_mode_count"][0, b]}))
    pd.concat(long_rows).to_csv(out / "radial_spectrum_long.csv", index=False)
    np.savez_compressed(out / "radial_spectrum_matrix.npz",
                        energy_fraction=radial["energy_fraction"],
                        lambda_geo_um=lam_geo, edges=edges,
                        low_mode_count=radial["low_mode_count"])

    desc_spec = p25.spectrum_descriptors(radial["energy_fraction"], lam_geo)
    spec_df = pd.DataFrame({"dataset_index": np.arange(len(R)), **desc_spec})
    spec_df.to_csv(out / "spectrum_descriptor_summary.csv", index=False)

    # ---- amplitude vs fraction consistency ----------------------------------
    Sq = desc["Sq_um"].to_numpy()
    cons_rows = []
    for b in band4:
        col = f"rms_DCT_{b}_um"
        rms = desc[col].to_numpy()
        pred_frozen = Sq * np.sqrt(E_frozen[f"DCT_{b}"])
        rel = np.abs(rms - pred_frozen) / rms
        cons_rows.append({"band": b, "convention": "frozen",
                          "max_abs_error_um": float(np.max(np.abs(rms - pred_frozen))),
                          "median_relative_error": float(np.median(rel))})
    for b in ("8_16", "16_32", "32_64"):
        rms = desc[f"rms_DCT_{b}_um"].to_numpy()
        pred_clean = Sq * np.sqrt((1 - r_dc) * p[b])
        rel = np.abs(rms - pred_clean) / rms
        cons_rows.append({"band": b, "convention": "clean_nonDC",
                          "max_abs_error_um": float(np.max(np.abs(rms - pred_clean))),
                          "median_relative_error": float(np.median(rel))})
    rms64 = desc["rms_DCT_64_inf_um"].to_numpy()
    pred64 = Sq * np.sqrt(r_dc + (1 - r_dc) * p["64_inf"])
    cons_rows.append({"band": "64_inf", "convention": "clean_nonDC",
                      "max_abs_error_um": float(np.max(np.abs(rms64 - pred64))),
                      "median_relative_error": float(np.median(
                          np.abs(rms64 - pred64) / rms64))})
    pd.DataFrame(cons_rows).to_csv(out / "amplitude_vs_fraction_consistency.csv",
                                   index=False)

    # ---- bin-count sensitivity (S5) ------------------------------------------
    sens_rows = []
    base = spec_df
    for nb in cfg["spectrum"]["radial_bin_sensitivity"]:
        r_nb, _ = p25.radial_spectrum(R, pixel, int(nb),
                                      float(scfg["lambda_lo_um"]),
                                      float(scfg["lambda_hi_um"]))
        d_nb = p25.spectrum_descriptors(r_nb["energy_fraction"],
                                        r_nb["lambda_geo_um"])
        for key in ("spectral_centroid_log_um", "spectral_entropy",
                    "effective_band_number"):
            rho = float(pd.Series(base[key]).corr(pd.Series(d_nb[key]),
                                                   method="spearman"))
            sens_rows.append({"n_bins": int(nb), "descriptor": key,
                              "spearman_vs_primary": rho})
    pd.DataFrame(sens_rows).to_csv(out / "radial_bin_sensitivity.csv",
                                   index=False)

    # ---- figures ---------------------------------------------------------------
    low = radial["low_mode_count"][0].astype(bool)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    q25, q50, q75 = np.percentile(radial["energy_fraction"], [25, 50, 75],
                                  axis=0)
    ax.plot(lam_geo, q50, "o-", lw=1.2, ms=3, label="median")
    ax.fill_between(lam_geo, q25, q75, alpha=0.25, label="Q25–Q75")
    for b in range(len(lam_geo)):
        if low[b]:
            ax.axvspan(radial["lambda_lo_um"][b], radial["lambda_hi_um"][b],
                       color="0.85", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("lambda [um] (shaded = low mode count)")
    ax.set_ylabel("normalized energy fraction")
    ax.set_title("Mean normalized radial spectrum (non-DC, 24 log bins)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out / "mean_normalized_spectrum.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    man = p25.read_phase2_manifest(cfg)
    depth = man["median_depth_um"].to_numpy()
    qs = np.quantile(depth, [0.25, 0.5, 0.75])
    lab = np.digitize(depth, qs)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, ql in enumerate(("Q1", "Q2", "Q3", "Q4")):
        m = lab == k
        ax.plot(lam_geo, radial["energy_fraction"][m].mean(axis=0), "o-",
                ms=3, lw=1.2, label=f"depth {ql}")
    for b in range(len(lam_geo)):
        if low[b]:
            ax.axvspan(radial["lambda_lo_um"][b], radial["lambda_hi_um"][b],
                       color="0.85", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("lambda [um] (shaded = low mode count)")
    ax.set_ylabel("mean energy fraction")
    ax.set_title("Spectrum by median-depth quartiles (cross-sectional)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out / "spectrum_by_process_quantiles.png",
                dpi=cfg["plot"]["dpi"])
    plt.close(fig)

    (out / "README.md").write_text(README, encoding="utf-8")
    missing = [f for f in EXPECTED if not (out / f).exists()]
    p25.require(not missing, f"missing outputs: {missing}")
    p2log(f"10 done in {time.time() - t0:.1f}s; all {len(EXPECTED)} outputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
