# spectral composition (Task 10)

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
