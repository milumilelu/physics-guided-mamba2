# D1.1 scope

Frozen MAIN180 only: E00 outer and inner component partitions are reused after D0 binding. SUPP20 is excluded from every fit and is not an OOF claim.

Models are BU, U+{A,P,PE,G,T}, explicit U-by-morphology interaction Ridge, cross-fitted residual M|U (Ridge/GAM), and a fixed-budget ExtraTrees sensitivity baseline on raw log-U plus all morphology columns. U inputs are log transformed deterministically; Regressor scaling and GAM splines are fit inside each training fold. Residual targets use strict nested inner cross-fitting.

Metrics are component-balanced OOF MAE/RMSE/Q2 and fixed-OOF component bootstrap risk deltas. All results are predictive/descriptive; they do not identify mediation or causality. ExtraTrees uses a fixed exploratory hyperparameter budget and is not a confirmatory model comparison.
