# spectral process map (Task 12, main experiment)

- `cv_fold_results.csv`: fold-level long table. Composition rows report
  Q2_Aitchison / dA_median / MAE_p_* / R2_z1..z4; scalar rows report
  R2/MAE/RMSE/Spearman.
- `composition_oof_predictions.csv`: per-sample OOF z (and inverse-ILR p) for
  every (variant, model, input) — src_gkf rows are unique per sample
  (contract test #15). 14B primary model = ridge (细则 §0.7).
- `input_comparison.csv`: fold-paired dQ2(C-A) for ALL variants (rev2-06
  lesson); positive = derived combinations give simple models a better
  inductive bias — never "new experimental information".
- `nonlinear_comparison.csv`: dQ2(ET-spline) and dR2(ET-Ridge) per variant.
- `permutation_importance.csv`: ET permutation importance on the composition
  (Aitchison Q2 drop) — only meaningful where Q2 clearly beats dummy
  (median > 0.10, 细则 §14).
- Gates G1/G2b/G3a are evaluated from these files (细则 §12).
