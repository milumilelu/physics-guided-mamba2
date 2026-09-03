# pseudo-pass spectral redistribution (Task 13)

- Data: cross-sectional pseudo-trajectories (15 matched bases x N=1..4 from
  pass_main; 10 supplement bases x N=5..6 as an independent check).
  N4->5 is session-confounded and refused.
- `pass_step_global_test.csv`: exact sign-flip test on the mean ILR step
  vector per step — p_exact = count/2^B (full enumeration, no +1 correction).
- `pass_step_coordinate_tests.csv`: coordinate-wise exact two-sided tests
  with Holm correction within each step.
- `pass_depth_spectrum_association.csv`: Spearman(delta depth, delta z) —
  cross-sectional association only.
- All titles/labels say "cross-sectional pseudo-trajectory"; no dynamics,
  oscillation or reversal language anywhere.
