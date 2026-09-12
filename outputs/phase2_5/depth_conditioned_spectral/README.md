# Depth-conditioned spectral/amplitude analysis (MAIN180)

Support: n=180 (MAIN180); excluded `pass_supplement` rows. Predictors are D=median depth (µm), N=pass count, and Q=log10(areal dose proxy).

Partial R² is a nested descriptive contribution: each process coordinate is added after the other two are fitted. Reverse columns report metric → D after N and Q are fitted. With the same linear model and controls, forward/reverse partial R² are mathematically symmetric; the reverse standardized beta and partial Spearman columns expose orientation and sign. Process-coordinate dependence is reported in `process_coordinate_dependence.csv` and `process_coordinate_vif.csv`. Signs/shape checks are in `conditional_contributions_long.csv`; depth-quartile associations are in `within_depth_quartile_associations.csv`.

## Largest conditional contributions
- `rms_DCT_16_32_um`: `D` partial R²=0.256, standardized beta=0.648, partial Spearman=0.625 (n=180).
- `p_32_64`: `D` partial R²=0.218, standardized beta=0.558, partial Spearman=0.500 (n=180).
- `E_DCT_32_64_frac`: `D` partial R²=0.211, standardized beta=0.550, partial Spearman=0.492 (n=180).
- `rms_DCT_32_64_um`: `D` partial R²=0.192, standardized beta=0.558, partial Spearman=0.597 (n=180).
- `A2_8_16`: `Q_log10` partial R²=0.184, standardized beta=-0.544, partial Spearman=-0.446 (n=180).
- `p_lt8`: `D` partial R²=0.180, standardized beta=-0.513, partial Spearman=-0.423 (n=180).
- `angular_entropy_8_16`: `Q_log10` partial R²=0.174, standardized beta=0.509, partial Spearman=0.480 (n=180).
- `Sq_um`: `D` partial R²=0.149, standardized beta=0.494, partial Spearman=0.510 (n=180).
- `p_16_32`: `D` partial R²=0.139, standardized beta=0.470, partial Spearman=0.401 (n=180).
- `E_DCT_16_32_frac`: `D` partial R²=0.130, standardized beta=0.454, partial Spearman=0.388 (n=180).
- `spectral_centroid_log_um`: `D` partial R²=0.126, standardized beta=0.434, partial Spearman=0.399 (n=180).
- `Sa_um`: `D` partial R²=0.120, standardized beta=0.445, partial Spearman=0.459 (n=180).

These are support-limited associations, not causal effects or evidence that a descriptor is a sufficient state. Any follow-up intervention must preserve the same depth/process support and measure repeated depth noise.
