# local regime probe (same held-out comparison, 细则 §10)

- `local_vs_global.csv`: fold × stratum × target × model rows with MAE_global,
  MAE_local, MAE_dummy (stratum-mean, local training rows) and
  Skill = 1 - MAE_model/MAE_dummy, all evaluated on the IDENTICAL held-out
  rows of the stratum.
- Strata: depth quartile / Sq quartile (edges from the training fold;
  own-variable targets excluded) and A_consensus median split.
- `delta_skill = Skill_local - Skill_global` is the only regime readout;
  positive values with a small local training set must be checked against the
  dummy column before any "local representation" language (细则 §18).
