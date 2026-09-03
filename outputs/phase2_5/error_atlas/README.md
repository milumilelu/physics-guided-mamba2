# mechanism bridge + OOF error atlas (Task 14)

- `mechanism_feature_provenance.csv`: three-state provenance
  (APPLICABLE / NOT_APPLICABLE / REDUNDANT_WITH_C). The mechanism constants
  (threshold 16.3967 J/cm2, incubation S=0.79652, delta_eff lookup) are fixed
  dataclass values in the mechanism module; the fitted TorchPhysicsModel
  recursion is NOT used (label dependency). Caveat: the constants' original
  calibration predates this repo.
- `mechanism_bridge_summary.csv`: dQ2_mech = Q2(z ~ [A, m(u)]) - Q2(z ~ A),
  fold-paired on identical grouped folds. Positive only means the mechanistic
  transformation adds inductive bias — never "E1/E2/E5 verified".
- `oof_error_atlas.csv`: per-sample OOF errors (ridge primary, ET
  sensitivity) normalized by the training-fold IQR; composition error =
  Aitchison distance.
- Hotspot rule: "model-robust unresolved" requires the elevated-error pattern
  to persist under ET (error Spearman + top-10% Jaccard reported); targets
  with ET-Ridge >= 0.1 (>=4/5 folds) in Task 12 are labelled
  "linear-baseline hotspot" instead.
