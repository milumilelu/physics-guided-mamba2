# directional spectrum (Task 11)

- `directional_metrics.csv`: A2 / theta_k / theta_stripe / angular_entropy per
  sample x band. Orientation is IMAGE-FRAME relative — no scan/hatch pose
  metadata exists, so angles must not be interpreted as hatch-relative.
- `stripe_validation.csv`: G2a phenotype validation on the 28-sample enriched
  audit set (labels = blind pattern contains "periodic stripe"). AUROC /
  rank-biserial / permutation p. **Enriched selection — never read as
  population stripe prevalence.** p > 0.05 means INCONCLUSIVE_AT_AUDIT_SIZE,
  not "metric invalid".
- `directional_spectrum_long.csv`: per-sample x band x 36 folded-orientation
  bins of normalized power (figure input).
