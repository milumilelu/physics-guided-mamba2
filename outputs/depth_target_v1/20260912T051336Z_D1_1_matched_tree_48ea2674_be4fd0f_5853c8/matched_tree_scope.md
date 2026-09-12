# D1.1 matched Tree scope

Frozen MAIN180 only; E00 outer component partitions are reused and SUPP20 is excluded. Tree(U), Tree(P), Tree(M), Tree(U+P), and Tree(U+M) use the identical ExtraTrees budget: n_estimators=300, min_samples_leaf=3, max_features=1.0, n_jobs=1; only feature set changes. Process coordinates are log transformed; morphology uses median-residual features. Metrics are component-balanced OOF errors and fixed-OOF component bootstrap deltas vs Tree(U+M). Results are descriptive predictive controls and do not identify causal mediation.
