# scale predictability summary (Phase 2B-5)

- `scale_predictability_summary.csv`: fold-quantiles of R2 at src_gkf for
  every (target, input_set, model).
- Curve figure: band-RMS targets vs band lower wavelength (log x); depth and
  Sq drawn as horizontal REFERENCE lines only — depth never enters a
  scale-dependence trigger (细则 §17 Route S compares morphology bands only).
- by-input: dR2 = R2(input R) - R2(input A) per band (fold-paired median,
  from 06); by-model: dR2 = ExtraTrees - Ridge per band.
- All values are exploratory CV estimates on n=200 (细则 §18).
