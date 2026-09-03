#!/usr/bin/env python3
"""Phase 2 experiment 04: build multiscale targets (Phase 2B-1).

Targets follow 细则 §6:
  family A (1): median_depth_um (manifest, valid-mask median depth);
  family B (12): ten Phase 1.5 descriptors + peak_to_valley_p98p2_um and
    deepest_negative_residual_um from the 01 inventory;
  family C (8): four DCT band RMS + energy fractions.
Family D (band PC scores) is fold-internal and is NOT stored here; the band
fields needed to build it are cached in `band_fields.npz` (gitignored,
rebuildable; 细则 §0.15).

Seed offsets: none (no stochastic step).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import _lib as p2

EXPECTED = ["multiscale_targets.csv", "targets_manifest.json",
            "band_fields.npz", "README.md"]

BANDS = ["8_16", "16_32", "32_64", "64_inf"]
B_DESC_COLS = ["Sq_um", "Sa_um", "Ssk_skewness", "kurtosis_excess_fisher",
               "grad_rms_um_per_um", "lap_rms_um_per_px2",
               "acf_e_fold_lag_um", "aniso_gradx_over_y",
               "pit_density_per_Mpx", "pit_depth_um"]

README = """# multiscale targets (Phase 2B-1)

- `multiscale_targets.csv`: 200 rows aligned with `dataset_index`; 21 targets
  (family A depth, family B descriptors, family C band amplitudes). Any NaN
  aborts the build (current data has valid_fraction = 1 everywhere).
- `band_fields.npz`: rebuildable LOCAL cache (gitignored) with R_total and the
  four DCT band fields, float32; used by 05 family-D fold-internal PCA and by
  the raw/repaired sensitivity rebuild in 09.
- Family D (band PC1..PC3) is never precomputed: PCA is fitted inside each
  training fold only (细则 §7.4).
- Note: median_depth_um describes the central 80x80 um ROI, not the full slot
  cross-section (v2 §12).
"""


def main() -> int:
    cfg, quick = p2.load_config(__doc__)
    t0 = time.time()
    out = p2.output_dir(cfg, "multiscale_targets")
    p2.log("== Phase 2 / 04: build multiscale targets ==")
    frozen = p2.l15.load_frozen(cfg)
    man = p2.read_manifest(cfg, require_loco=True)
    inv = pd.read_csv(p2.output_dir(cfg, "instability")
                      / "instability_inventory.csv")
    desc = pd.read_csv(p2.l15.REPO / "outputs/phase1_5/morphology_descriptors.csv")
    p2.require(list(desc["dataset_index"]) == list(range(200)),
               "descriptor row order != dataset_index")

    tgt = pd.DataFrame({"dataset_index": np.arange(200)})
    tgt["median_depth_um"] = man["median_depth_um"].to_numpy()
    for c in B_DESC_COLS:
        tgt[c] = desc[c].to_numpy()
    tgt["peak_to_valley_p98p2_um"] = inv["peak_to_valley_p98p2_um"].to_numpy()
    tgt["deepest_negative_residual_um"] = inv["deepest_negative_residual_um"] \
        .to_numpy()
    for b in BANDS:
        tgt[f"rms_DCT_{b}_um"] = desc[f"rms_DCT_{b}_um"].to_numpy()
        tgt[f"E_DCT_{b}_frac"] = desc[f"E_DCT_{b}_frac"].to_numpy()

    num = tgt.select_dtypes(include=[float]).columns
    p2.require(not tgt[list(num)].isna().any().any(),
               "NaN in multiscale targets")
    p2.require(len(tgt) == 200, "targets rows != 200")

    bands, coverage = p2.l15.dct_band_fields(
        frozen["R"], float(cfg["scales"]["pixel_um"]),
        cfg["scales"]["dct_bands_um"])
    np.savez_compressed(
        out / "band_fields.npz",
        R_total=frozen["R"].astype(np.float32),
        **{f"R_band_{b}": bands[f"DCT_{b}"].astype(np.float32)
           for b in BANDS})

    manifest = {
        "median_depth_um": {"family": "A", "definition": "valid-mask median "
                            "depth of the raw height field",
                            "unit": "um", "source": "exploration_manifest",
                            "nan_policy": "abort", "notes": "central 80x80 um "
                            "ROI, not full slot cross-section (v2 §12)"},
        **{c: {"family": "B", "definition": "Phase 1.5 morphology descriptor",
               "unit": "see descriptors", "source":
               "outputs/phase1_5/morphology_descriptors.csv",
               "nan_policy": "abort", "notes": ""} for c in B_DESC_COLS},
        "peak_to_valley_p98p2_um": {
            "family": "B", "definition": "P98-P2 of valid residual pixels",
            "unit": "um", "source": "phase2/instability/"
            "instability_inventory.csv", "nan_policy": "abort",
            "notes": "robust range; R_max_minus_min_um kept as remark only"},
        "deepest_negative_residual_um": {
            "family": "B", "definition": "-min(residual)",
            "unit": "um", "source": "phase2/instability/"
            "instability_inventory.csv", "nan_policy": "abort", "notes": ""},
        **{f"rms_DCT_{b}_um": {"family": "C",
                               "definition": f"DCT band {b} um RMS",
                               "unit": "um",
                               "source": "morphology_descriptors.csv",
                               "nan_policy": "abort",
                               "notes": "band definition frozen in 1.5"}
           for b in BANDS},
        **{f"E_DCT_{b}_frac": {"family": "C",
                               "definition": f"DCT band {b} um energy "
                               f"fraction of residual variance",
                               "unit": "1",
                               "source": "morphology_descriptors.csv",
                               "nan_policy": "abort", "notes": ""}
           for b in BANDS},
        "_band_fields_cache": {"family": "D", "definition": "R_total + 4 DCT "
                               "band fields for fold-internal PCA targets",
                               "unit": "um", "source": "height_raw via "
                               "l15.dct_band_fields", "nan_policy": "n/a",
                               "notes": f"band coverage {coverage:.4f}; "
                               "gitignored local cache"},
    }
    (out / "targets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tgt.to_csv(out / "multiscale_targets.csv", index=False)
    (out / "README.md").write_text(README, encoding="utf-8")

    missing = [f for f in EXPECTED if not (out / f).exists()]
    p2.require(not missing, f"missing outputs: {missing}")
    p2.log(f"04 done in {time.time() - t0:.1f}s: {tgt.shape[1] - 1} targets, "
           f"band coverage {coverage:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
