# multiscale targets (Phase 2B-1)

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
