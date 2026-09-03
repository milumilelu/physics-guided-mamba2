# sensitivity checks (Phase 2B-6)

- `sensitivity_fold_results.csv`: per-arm fold-level R2.
- `sensitivity_summary.csv`: arm medians vs the 05 main run (src_gkf), with
  descriptive |dR2| buckets (stable/moderate/strong). No arm result may be
  read as "these samples should be deleted" (规划 §23.3).
- `repaired` arm: median_depth_um intentionally unchanged (raw height is the
  authority, v2 §7); Sq and band RMS recomputed from the repaired residual.
- `dog` arm: DoG stds are octave-like band amplitudes (G2-G4, G4-G8, G8-G16,
  G16 low-pass); baseline pairs each DoG band with its DCT counterpart.
- `exclude_artifact_yes` arm implements the 2A gate PASS_WITH_FLAGS addendum.
