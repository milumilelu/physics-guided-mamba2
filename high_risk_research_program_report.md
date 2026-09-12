# High-risk research program: discovering the minimum state for ultrafast zirconia ablation

**Repository and data basis.** This report uses the current checkout of physics-guided-mamba2, the frozen D0/D1 and Phase 2.8r1 artifacts, the Task-State diagnostics, and the persisted research brief. It treats the reported metrics as discovery evidence, not independent confirmation.

## 1. What is scientifically unresolved?

The central unresolved issue is not whether a regressor can lower MAE. It is whether terminal topography is the material state that controls removal, an observation of another state, or a geometry-specific side effect.

Four observations must be explained together:

1. Full process inputs \(U\) predict terminal depth \(D\) strongly: D1 BU GAM MAE 6.53 µm, RMSE 8.39 µm, Q² .839 on grouped MAIN180 OOF.
2. DC-removed morphology \(M\) predicts D meaningfully but much less strongly: BM GAM MAE 12.99 µm, Q² .392; Ridge MAE 12.35 µm, Q² .440. Spectral composition BP is the best single block (GAM MAE 12.44 µm, Q² .407; Ridge 12.11 µm, Q² .455).
3. Adding all current morphology to process inputs does not improve the released linear/GAM fit: BUM GAM MAE 7.91 µm, Q² .779; Ridge MAE about 7.30 µm, Q² about .80. The D1.1 follow-up finds U+single blocks generally neutral or worse and only tiny, uncertainty-overlapping residual point gains.
4. Some morphology is highly process-predictable but weakly depth-related. Phase 2.8r1 source-grouped Q² is .641 for 8–16 µm directional amplitude and .662 for directional entropy, while depth is .552 and hatch-only depth is .044.

The deepest questions are therefore:

- Is the missing information in \(M\) a lost spatial/temporal state, or is it never present in the terminal surface?
- Does depth organize morphology as a reaction coordinate, or are both outputs driven by pass count, dose, focus, and history?
- Does morphology affect the next increment of removal, even when it cannot reconstruct already-realized terminal depth?
- Are there regime switches or non-commutative exposure histories hidden by static dose coordinates?
- What is the minimum state dimension needed before a recurrent/SSM/Mamba architecture is justified?
- Can any state be identified separately from a disguised encoding of \(U\), session, or measurement artifacts?

The canonical target is \(D=-\operatorname{median}_{\Omega}H\), with \(r=H-\operatorname{median}_{\Omega}H\) used for morphology. Because adding a constant to H changes D but not r, translation-invariant morphology cannot universally invert absolute depth. The 80 µm field and 0.5 µm pixels also limit the number of independent cycles in the ≥64 µm band. These are observability limits, not implementation details.

## 2. Competing causal and dynamical structures

| Structure | What it asserts | Current support | Main contradiction or limitation | Decisive observation |
|---|---|---|---|---|
| \(U\to D\) | Static process variables determine terminal depth | Strong D1 BU result | Does not reveal memory or explain morphology | Held-family/session performance and temporal-order intervention |
| \(U\to M\to D\) | Morphology mediates the process effect | M→D skill exists; spectral block strongest | U remains much stronger; U+M lacks stable gain; terminal data cannot identify mediation | Same-current-depth, different-M intervention followed by identical next pass |
| \(U\to\{M,D\}\) | Process has parallel observable outputs | Direction and depth show different process dependencies | Parallel outputs may still share hidden state | Joint state model with measured state and cross-family validation |
| \(U\to S\to\{M,D\}\) | Low-dimensional evolving material state produces both outputs | Heat accumulation, incubation, defocus and porosity are plausible; U predicts both | No independent S measurement; terminal data are observational | State-reset and observer experiment |
| \(U_t,S_t\to S_{t+1}\), \(S_t\to\{M_t,D_t\}\) | Removal is path-dependent state transition | Pseudo-pass clues and zirconia heat-accumulation literature motivate it | Current N=1–4 records are different positions, not same-location trajectories | Same-location per-pass metrology with ordered exposure |
| \(S_t\to\{M_t,D_t\}\) | M and D are sibling observations of state | Explains weak terminal M→D and distinct morphology blocks | State dimension and observability unknown | Joint latent-state model plus external thermal/optical/porosity measurements |
| \((D_t,M_t,U_{t+1})\to\Delta D_{t+1}\) | Morphology is useful as a state for future removal | Scientifically plausible, not tested | No genuine next-step target currently exists | Longitudinal next-increment OOF |
| \(U\to Z_D(U)\to D\) | A compact process state is sufficient for D | Full U is strong; dose/incubation candidates exist | \(Z_D\) may just be a fitted proxy; morphology relation unclear | Nested process-to-state-to-D comparison and state ablation |
| \(U\to S\to M\), \(D\) as progress coordinate | Depth parametrizes part of morphology evolution | E02/Phase 1.5 show selected spectral clues | Cross-sectional pseudo-time and common pass/session confounding | Matched-depth trajectories and true longitudinal sampling |
| \(M\) as diagnostic fingerprint only | M records history without controlling future D | Direction channel is strongly process/hatch-linked | Does not explain any future increment unless tested | Counterfactual initial morphology experiment |
| Terminal \(M\) as irreversible bottleneck | Distinct histories collapse to similar M or same D | U≫M and translation invariance make this plausible | Many-to-one counterexamples not yet quantified | Adversarial matched pairs and conditional entropy |
| Local spatial state | Global descriptors destroy depth-relevant heterogeneity | Historical residual spatial clustering; finite ROI | No patch-level depth target yet | Median-centered patch/pyramid models under grouped CV |
| Regime-switching removal law | One smooth model mixes shallow/deep or smooth/textured regimes | Non-monotonic fluence/roughness literature | No stable regime analysis released | Pre-registered mixture/change-point model with held-family tests |
| Non-commutative path dependence | \(F(U_a,U_b)\ne F(U_b,U_a)\) at matched exposure | Incubation/thermal history makes it plausible | No ordering experiment | Pulse-order reversal with same pulse multiset |
| Morphology-conditioned effective coupling | M provides empirical closure for missing absorptance/focus physics | Roughness/porosity feedback is plausible | Absorptance not measured; mechanisms material-dependent | Pre-roughened surfaces plus optical/thermal observer |
| Spectral redistribution state | Cross-scale energy allocation matters more than total roughness | P outperforms A; 8–16 µm direction is distinct | Band selection and finite-window sensitivity remain | Fixed/normalized bands, wavelets, and matched-amplitude pretexture |
| Defocus-depth state | Cavity depth changes effective focus and removal efficiency | Defocus literature and deep-cavity physics | No focus measurement in current data | Controlled depth/defocus sweep and observer |
| Long-memory state | Many previous passes matter beyond current D/M | Candidate incubation/redeposition mechanisms | Current pass order and waiting time absent | Same-location sequence with state-reset ablation |

The present evidence favors a parallel/shared-state interpretation over strict mediation, but cannot distinguish the shared-state alternatives without interventions or longitudinal observation.

## 3. High-risk hypotheses

| ID | Scientific question | Proposed mechanism | Supporting evidence | Contradicting evidence | Analysis | New data needed | Failure mode | Strongest paper claim | Minimum evidence |
|---|---|---|---|---|---|---|---|---|---|
| H1 | Are M and D sibling observations of S? | Incubation, thermal state, defocus, defects | U predicts both; M only partly predicts D | No measured S; U+M weak | Joint latent-state and state ablation | Per-pass M,D plus thermal/optical proxy | State encodes U/session | A compact observable state explains both outputs | Cross-family stable state with external correlate |
| H2 | Does M predict future removal? | Roughness, shielding, focus, coupling alter next pulse | Physical precedent | No longitudinal target | \(D_t,U_{t+1}\) vs \(D_t,M_t,U_{t+1}\) | Same-location per-pass data | D_t alone sufficient | Morphology is a predictive state for \(\Delta D\) | Pre-registered next-step gain in ≥4/5 family folds |
| H3 | Is there non-commutative history? | Incubation/thermal memory | Zirconia heat accumulation literature | Static DOE cannot test ordering | Order reversal and waiting-time models | Exact pulse order/wait | No order effect | Integrated dose is insufficient; processing is path-dependent | Matched-dose intervention with reproducible effect |
| H4 | Do regimes switch? | Threshold, melting/recast, shielding transitions | Non-monotonic fluence/roughness | No stable clusters yet | Mixture GAM/HMM/change-point | More points near transitions | Session shift masquerades as regime | Distinct removal regimes with physical covariates | Held-family regime recurrence |
| H5 | Is spectral redistribution the depth state? | Fine/coarse redistribution changes coupling | BP strongest block | Finite bands and session noise | Fixed/normalized DCT, wavelet, PLS | Higher-resolution repeatability | Band artifact or amplitude proxy | One scale-transition coordinate predicts D | Stable coordinate after amplitude/hatch controls |
| H6 | Is D a reaction coordinate for M? | Depth indexes accumulated state | E02/Phase 1.5 clues | Pseudo-time is cross-sectional | Matched-depth conditional models | Longitudinal validation | Pass/session confounding | Selected morphology scales collapse by D | D beats N/dose across families |
| H7 | Is terminal M an irreversible bottleneck? | Information lost through smoothing/redeposition | U≫M; offset non-invertibility | Counterexamples unquantified | Matched-pair/conditional entropy analysis | More repeat histories | Apparent ambiguity is metrology noise | History cannot be reconstructed from terminal M | Stable many-to-one pairs under noise model |
| H8 | Is state local/spatial? | Patch heterogeneity controls local removal | Spatial residual clustering | No patch D labels | Centered patches, pyramids, local spectra | Co-registered local depth | Overfit spatial texture | Local morphology recovers missing state | Beats global summaries under same budget |
| H9 | Is effective absorptance morphology-conditioned? | Roughness/porosity changes coupling | Zirconia nanopore and roughness literature | Absorptance unmeasured; material dependence | Closure residual on calibrated physics | Reflectance/thermal proxy | Closure is arbitrary black box | One interpretable closure corrects physics model | Held-family gain plus independent proxy |
| H10 | Is depth-dependent defocus essential? | Cavity changes focus/fluence | Defocus studies | No focus metadata | Explicit depth-defocus state | Focus sweep or in-situ focus | Defocus confounded with dose | Depth plus defocus is minimal state | State improves extrapolation, not interpolation only |
| H11 | Is incubation saturating? | Defect accumulation lowers threshold then saturates | Fused-silica and zirconia precedent | Material coefficients differ | Saturating vs power-law grouped CV | N sweep with repeated timing | Session or temperature confound | Saturating incubation state transfers across families | Parameter stability and held-out N |
| H12 | Is directionality only geometry? | Hatch/scan organizes directional spectrum | Q² .641/.662, hatch-only high | Scan axis missing | Residualize directional block against hatch | Recorded orientation | Hidden orientation/session | Geometry and removal channels separate | Direction residual has no D gain after controls |
| H13 | Can process histories form equivalence classes? | Different U produce same functional S | U/M patterns suggest compression | No stable metric yet | Supervised metric learning/clustering | Repeated histories | Arbitrary clusters | Recipe equivalence classes predict next response | Cluster stability and response equivalence |
| H14 | Is a low-dimensional \(Z_D\) sufficient? | Depth-relevant central subspace | BP and residual clues | Compression unrun | Nested PLS/SDR/sparse reductions | None initially | Reduction unstable or U-proxy | Minimal depth-oriented morphology bottleneck | Non-inferior risk at fixed k |
| H15 | Is pass memory longer than current D/M? | Thermal/redeposition memory persists | Heat accumulation precedent | No order/wait data | GRU/SSM with state-reset tests | Longitudinal sequences | Mamba memorizes recipe IDs | Measured long-memory state is necessary | Improvement under pass extrapolation and reset |
| H16 | Does material phase/defect chemistry matter? | Phase transformation/porosity changes removal | Zirconia excitation/nanopore studies | No phase/porosity measurements | Add independent phase/porosity observers | Raman/XRD/SEM/porosity | Correlation with dose only | Defect state explains residual depth | Observer adds stable conditional information |
| H17 | Are current negative U+M results regularization artifacts? | Shared scaling/linear basis hides interactions | UTREE sensitivity MAE 5.42, Q² .885 vs BU GAM, exploratory | No ExtraTrees(U) comparator | Matched U-only/M/U+M tree and boosting | None initially | Tree gain is tuning leakage/overfit | Nonlinear morphology interaction is real | Same-budget family-stable gain |
| H18 | Is terminal morphology sufficient for process control? | If M is state, inverse control possible | M→D skill | U≫M, non-invertibility | Control-relevant next-step test | Longitudinal interventions | Terminal M only fingerprint | Terminal morphology has/does not have control value | Prospective next-step success |
| H19 | Does scan order create spatial memory? | Local exposure order, plume shielding | Geometry/feedback plausibility | No scan path metadata | Recorded order and orientation perturbation | Exact path metadata | Unidentifiable from current data | Spatial order is a state input | Reproducible order effect |
| H20 | Can a small physical state match Mamba? | Low-dimensional physics memory | Static U already strong; Mamba unvalidated | E03 solver FAIL; terminal training failed | State hierarchy S0–S8 | Longitudinal/observer data | Architecture wins only interpolation | Physical memory dimension is small or large | Equal-budget extrapolation and state interpretability |

## 4. Top five moonshot directions

### Moonshot 1: morphology as a state variable for future removal

This changes the endpoint from terminal reconstruction to control-relevant prediction. Create matched current depths with different centered morphology, apply identical next-step U, and predict \(\Delta D\). A surprising result would be a large morphology effect after conditioning on D and U. A quick falsifier is a same-location two-step pilot with three repeated fields; if M adds no stable next-step skill, deprioritize mediation/state claims. Immediate implementation: only the data contract and analysis harness; the scientific test requires new longitudinal data.

### Moonshot 2: non-commutative laser processing

Match total dose and pulse multiset, reverse temporal order, and vary waiting time. If \(F(U_a,U_b)\ne F(U_b,U_a)\), the project becomes a path-dependent dynamical-system study and sequence models gain a real scientific role. A quick falsifier is a two-order, one-wait pilot at matched dose. Immediate implementation: pre-register dose/order and simulation bookkeeping; do not fit Mamba to static data.

### Moonshot 3: depth as a reaction coordinate for multiscale surface evolution

Test whether different histories collapse onto \(M_b=f(D)\) for spectral/amplitude blocks while directionality remains history-dependent. A surprising result would be a common depth-indexed manifold across process families. A quick falsifier is matched-depth nearest-neighbor analysis with support thresholds; if residual process dependence is as large as unconditional dependence, stop the reaction-coordinate narrative. Immediate implementation: D2 reverse models and manifold diagnostics on current data.

### Moonshot 4: a one- or two-dimensional morphology closure for missing physics

Use a calibrated depth/defocus/incubation model and let a supervised spectral coordinate correct its residual. The result would be stronger than a black-box gain if the closure aligns with an independent reflectance/thermal/porosity observer. A quick falsifier is a pre-roughened/polished matched-dose comparison with no increment difference. Immediate implementation: only after E03 finite-ROI and parameter calibration gates are repaired.

### Moonshot 5: minimal physical memory dimension

Build S0 static U, S1 D, S2 D+defocus, S3 incubation, S4 compact M, S5 joint explicit state, S6 GRU, S7 linear SSM, S8 Mamba-2. The surprising result could be either a scalar state matching Mamba or a reproducible need for long memory. A quick falsifier is a longitudinal dataset in which S1–S4 match S7/S8 on pass extrapolation. Immediate implementation: hierarchy specification and ablation protocol; model training waits for sequences.

## 5. Kill-fast experiments and analyses

1. **Residual D test:** cross-fit BU, regress \(D-\hat D_U\) on each M block; require fold-stable gain and predeclared practical margin.
2. **Matched-depth counterexamples:** find pairs with \(|D_i-D_j|\) within metrology tolerance but large M differences, and pairs with similar M but different D; quantify support and measurement noise.
3. **ExtraTrees comparator:** run ExtraTrees(U), ExtraTrees(M), ExtraTrees(U+M) with identical fold-local tuning; this is required before interpreting the D1.1 UTREE result.
4. **Interaction shuffle:** preserve U/M marginals but permute morphology within process families; compare interaction gain to a null distribution.
5. **Scale knockouts:** remove one DCT band at a time, compare absolute versus normalized energy, and test whether the BP signal survives amplitude/hatch residualization.
6. **Session and support stress:** report MAIN180 source-grouped, process-grouped, leave-session-out, and SUPP20 support-distance strata; do not pool them into one score.
7. **Depth coordinate adversary:** compare D, pass count N, total dose, and session as predictors of each morphology target under identical folds; use permutation tests for incremental D.
8. **Regime stability:** fit two- and three-regime GAM/mixture models with predeclared complexity; require recurrence across held-out families and no session-only separation.
9. **Spatial information test:** median-center local patches, spatial pyramids, and global summaries; compare under equal feature/tuning budgets and strict source grouping.
10. **State-reset simulation:** for any recurrent/SSM model, reset hidden state at specimen/session boundaries and permute pass order; a claimed memory gain must disappear under the appropriate destructive permutation.
11. **Metrology repeatability:** estimate feature noise from independent repeated scans or fields; attenuate apparent M→D gains by an errors-in-variables sensitivity analysis.
12. **Dose-coupling audit:** treat frequency and pulse energy as one coupled factor at fixed average power; test whether apparent independent effects vanish under reparameterization.

Kill criteria: stop a hypothesis after two preregistered representations, two grouping schemes, and support-matched analysis fail to reproduce its gain, unless the failure itself is the paper result.

## 6. Decisive new experiments

| Experiment | Controlled variables | Changed variable | Measured outputs | Competing predictions | Minimum design | Identification |
|---|---|---|---|---|---|---|
| A. Same depth, different M | Material, spot, current D, next U | Initial spectral/amplitude/directional pretexture | \(D_t,M_t,\Delta D\), reflectance | D-only model predicts equal \(\Delta D\); morphology-state model predicts differences | 3 histories × 3 repeats × 4 locations | Directly tests whether M modifies future removal |
| B. Same M, different history | Material, approximate M, next U | Prior order/wait/dose history | D_t, M_t, next \(\Delta D\), thermal/optical proxy | Sufficient-M model predicts equal response; hidden-state model predicts residual history effect | 4 history pairs × 3 repeats | Tests whether terminal M is sufficient |
| C. Order reversal | Total pulse multiset, total dose, material | \(U_a\to U_b\) versus \(U_b\to U_a\) | Per-pass D/M, final D/M, temperature | Memoryless model invariant; incubation/thermal state model non-commutative | 2 orders × 3 repeats × 4 matched fields | Causal temporal intervention |
| D. Spectral pretexture | RMS amplitude, material, dose, next U | Fine/coarse scale allocation at matched amplitude | D increments, DCT/wavelet, reflectance | Amplitude-only model equal; spectral-state model differs | 3 scale states × 3 repeats | Tests scale-specific coupling |
| E. Directional pretexture | Amplitude/spectrum, material, dose, recorded orientation | Orientation relative to scan | Direction block, D increments, scan-relative data | Passive-fingerprint model changes M only; coupling model changes \(\Delta D\) | 2 orientations × 4 hatch levels × 3 repeats | Requires scan-axis metadata |
| F. Wait-time sweep | U, dose, path, current D | Inter-pass wait time | Thermal decay, D_t/M_t, \(\Delta D\) | No memory predicts equal increments; cooling state predicts wait effect | 3 waits × 3 repeats | Causal thermal-history test |
| G. Observer augmentation | U and metrology fixed | Add calibrated reflectance/thermal/phase observer | Observer, D/M, uncertainty | If S is real, observer reduces residual and improves state transfer | 3 repeats/condition plus held-out families | Mechanistic observability, not automatic mediation |

Use at least three independent specimens or fields per condition as a starting design; calculate final power from D1 residual variance. Record exact pulse order, scan direction, waiting time, focus, and location. Fixed average power means frequency and pulse energy cannot be independently manipulated without changing the power constraint.

## 7. Model architecture only after scientific structure

The mandatory hierarchy is:

- **Level 0:** train-component mean.
- **Level 1:** dose/overlap-only model.
- **Level 2:** full static U (BU).
- **Level 3:** U plus hand-crafted M (D1.1 block and BUM).
- **Level 4:** U plus compact supervised \(Z_D\).
- **Level 5:** explicit physical state (depth, defocus, incubation).
- **Level 6:** explicit state plus learned residual closure.
- **Level 7:** small GRU or switching state-space model.
- **Level 8:** linear/nonlinear SSM.
- **Level 9:** Mamba-2.

For every architecture:

| Architecture | State represented | Why needed | Simpler baseline to defeat | Necessity ablation |
|---|---|---|---|---|
| GAM/Ridge | Static conditional mean | Establish interpretable floor | Null/dose/BU | Block and residual removal |
| Boosted trees/ExtraTrees | Static nonlinear U–M interactions | Test D1.1 regularization explanation | BU and matched tree(U) | U-only vs U+M, tuning parity |
| PLS/SDR/sparse \(Z_D\) | Depth-relevant morphology subspace | Test compression and sufficiency | BM/BP/BUM | k=1…6 and shuffled-target |
| Physics state model | D, defocus, incubation, coupling | Encode candidate mechanisms | BU and dose | Remove one state at a time |
| Physics + closure | Residual missing physics | Test empirical morphology closure | Calibrated physics alone | Remove closure and morphology |
| GRU | Learned finite-memory state | Establish whether recurrence helps | Level 5/6 | Sequence shuffle/reset and hidden-size curve |
| Linear SSM | Long-range state transition | Test long memory efficiently | GRU and explicit state | Context-length/pass extrapolation |
| Mamba-2 | Selective long-memory state | Only if order/history matters | All Levels 0–8 | Replace with static nonlinear model, reset/permute sequence |

Current status: Mamba-2 backend acceptance is an engineering result; terminal learning attempts failed dtype/device gates, and E03 physics registration remains failed. Therefore no architecture claim is presently justified. A model becomes scientifically necessary only if it improves a registered longitudinal or pass-range extrapolation task, survives state-reset/order permutations, and its state is at least partly observable.

## 8. Paper strategy, ranked from transformative to defensible

1. **Morphology predicts future ablation, not terminal depth.** Most transformative. Requires same-location \(M_t,D_t\), next-step \(\Delta D\), matched-depth interventions, and improvement over \(D_t,U_{t+1}\).
2. **Non-commutative path dependence in multipass zirconia ablation.** Requires matched-dose order/wait experiments, exact process logging, and a state model that predicts order effects.
3. **Depth as a reaction coordinate for multiscale morphology.** Requires cross-family collapse of selected \(M_b\) versus D, explicit residual history channels, and longitudinal confirmation.
4. **Minimal physics-guided state explains depth and morphology.** Requires calibrated defocus/incubation/thermal/defect observers, state ablations, and held-family transfer.
5. **Terminal morphology is an irreversible information bottleneck.** Requires robust many-to-one history/depth counterexamples, measurement-noise sensitivity, and adversarial matching.
6. **Multiscale spectral morphology as empirical closure.** Requires a stable \(Z_D\), physics residual correction, and independent coupling evidence.
7. **Geometry and removal channels separate.** Requires recorded scan orientation, hatch residualization, and matched-depth morphology comparisons.
8. **Large sequence models are unnecessary.** Most defensible negative result. Requires longitudinal data showing explicit low-dimensional states match GRU/SSM/Mamba under equal extrapolation tests.
9. **A compact depth-oriented morphology information bottleneck.** Requires nested k-curves, non-inferiority tolerance, and stability across source/process/session splits.
10. **Static process model is sufficient in the current domain.** Requires D1.2/D2 null results, SUPP20 support stress, and explicit statement that this is a domain-bounded result.

### Contradiction map

- Heat accumulation/incubation and morphology feedback predict history dependence and sibling outputs; the strong BU result and weak BUM gain suggest terminal U still contains unrecovered history.
- Hatch-dominant directional Q² predicts a geometry channel; it conflicts with any claim that all morphology is a depth mediator.
- Spectral-composition skill and E02 depth-conditioned ILR clues predict a depth-relevant scale channel; session-sensitive absolute energy and finite-window limits predict measurement attenuation.
- Defocus and porosity mechanisms predict latent-state behavior; absent focus/phase/thermal measurements make current identification impossible.
- Mamba/SSM literature demonstrates sequence-model capacity, not laser necessity; static nonlinear and explicit-state baselines must be defeated first.

## Repository evidence and literature anchors

The authoritative implementation and artifacts are:

- src/depth_target/features.py: median-centered residual features, 17 morphology inputs, translation-invariance guard.
- src/depth_target/splits.py and src/task_state_learning/grouping.py: source/process-family component isolation.
- experiments/depth_target_v1/01_depth_baselines.py: sealed D1 Ridge/GAM baselines.
- experiments/depth_target_v1/03_d1_1_block_residual.py: additive D1.1 block, strict residual, interaction, and tree sensitivity.
- outputs/depth_target_v1/20260912T041545Z_D1_1_block_residual_48ea2674_ea51976_0909dd/: latest D1.1 sealed run, 2,880 OOF rows, fixed-OOF intervals.
- outputs/phase2_8_r1/summary/gsl28_a_evaluation.json and predictability_spectrum.csv: corrected D/A/P/T grouped spectrum.
- outputs/task_state_v1/20260906T111859Z_E02_b901bea5_533009c_68a562/: separate Task-State morphology/depth-conditioned diagnostics.
- outputs/task_state_v1/e03_audit/ and E04/E06 artifacts: physics/SSM gate history and limitations.

Recent literature constraining the hypotheses includes zirconia heat accumulation and pulse-resolved modelling ([Bogatyrev et al., JMPT 2025](https://doi.org/10.1016/j.jmatprotec.2024.118668)); non-monotonic ultrashort ZrO2 removal and roughness ([JCPR 2023](https://doi.org/10.36410/jcpr.2023.24.2.230)); time-resolved zirconia excitation ([Applied Physics A 2024](https://doi.org/10.1007/s00339-023-07223-7)); incubation saturation in fused silica ([De Palo et al. 2022](https://doi.org/10.1364/OE.475592)); morphology/absorption/incubation coupling ([Song et al. 2024](https://doi.org/10.1016/j.jmapro.2024.04.047)); scale-dependent spectral evolution ([Kažukauskas et al. 2026](https://doi.org/10.1364/OE.588502)); depth/defocus precedent ([PMC11828434](https://pmc.ncbi.nlm.nih.gov/articles/PMC11828434/)); supervised information bottlenecks ([Ghosh 2022](https://doi.org/10.3390/e24020167)); and Mamba neural operators for dynamical systems ([Hu et al. 2024](https://arxiv.org/abs/2409.03231)). These studies differ in material, wavelength, pulse regime, outcome, or measurement and are mechanism precedents rather than transferable zirconia coefficients.

**Decision rule.** Do not select the winning narrative from current point estimates. First run the kill-fast tests, then the smallest discriminating intervention. If morphology adds no next-step information and no stable latent state is observed, the strongest result is a rigorously bounded terminal-observability limit. If it does, build the paper around the minimum state that survives held-family and pass-extrapolation tests; Mamba is only the final comparator.



## Expanded literature matrix and transferability cautions

| Reference | Material/regime | Outcome and mechanism | Method | Transferability |
|---|---|---|---|---|
| Bogatyrev et al., JMPT 2025, DOI 10.1016/j.jmatprotec.2024.118668 | 8YSZ, femtosecond dynamic grooves/3D processing | Micro/macro heat accumulation, interpulse transition to melting, nanoporosity and morphology feedback | Pulse-resolved surface-evolution model | Direct zirconia precedent; calibrate composition, wavelength, spot and kinematics |
| Ultrashort pulse laser processing of ZrO2 ceramics, JCPR 2023, DOI 10.36410/jcpr.2023.24.2.230 | ZrO2, 390 fs–6 ps, varied fluence/repetition | Non-monotonic removal, U-shaped roughness, heat accumulation and shielding/redeposition | Experimental process map | Direct material relevance; not the present ROI/depth definition |
| Zirconia electron-excitation study, Applied Physics A 2024, DOI 10.1007/s00339-023-07223-7 | Zirconia, ultrashort pulse, ps–ns observation | Carrier excitation and filament dynamics | Time-resolved mechanistic modelling | Candidate hidden carrier/energy state; no depth prediction |
| YSZ nanopores, JMRT 2023 | YSZ, near-threshold femtosecond pulses | Intergranular ablation and nanopore formation | Microscopy/process study | Candidate defect/porosity state; single-pulse regime |
| De Palo et al., Optics Express 2022, DOI 10.1364/OE.475592 | Fused silica, 1030 nm fs, 0.06–200 kHz | Threshold reduction saturates with pulse count (incubation) | Exponential fit | Functional form to test; coefficients not transferable |
| Song et al., JMP 2024, DOI 10.1016/j.jmapro.2024.04.047 | Fused silica, fs multipulse | Changing morphology/fluence absorption coupled to incubation | Phenomenological crater/groove model | Feedback precedent; material/optical mismatch |
| Thomae et al., Applied Surface Science Advances 2026, DOI 10.1016/j.apsadv.2025.100928 | Aluminium and steel, 500 fs/1040 nm | Roughness-enhanced absorption in Al but distinct steel mechanism | FDTD plus experiment | Demonstrates mechanism non-universality |
| Kažukauskas et al., Optics Express 2026, DOI 10.1364/OE.588502 | Femtosecond deep engraving of glass | Different spatial scales forget initial surface at different rates | Spectral evolution | Multiscale-state motivation; not zirconia law |
| Femtosecond cavity/defocus study, PMC11828434 | Femtosecond implant cavity | Depth changes focus and subsequent removal | Depth–defocus experiment | Supports explicit defocus state; coefficients nontransferable |
| Li et al., Machines 2026, DOI 10.3390/machines14050509 | Femtosecond single-scan grooves | Monotonic process–depth relation under constrained model | Attention ensemble plus monotonic constraint | Baseline/constraint precedent; no multipass state validation |
| Ghosh, Entropy 2022, DOI 10.3390/e24020167 | General regression | Sufficient reduction as information bottleneck | SDR/information theory | Formal basis for \(Z_D\), independent of laser material |
| Hu et al., arXiv:2409.03231 (2024) | PDE and pharmacology dynamics | Long-range state-space operator learning and extrapolation | Mamba neural operator | Architecture precedent; no laser evidence |
| Gu & Dao, arXiv:2312.00752 (2023); Dao & Gu, arXiv:2405.21060 (2024) | General sequence modelling | Selective SSM and Mamba-2 efficiency | SSM/SSD theory and benchmarks | Engineering basis only |
| Imai, Keele & Tingley, DOI 10.1037/a0020761 | Causal mediation methodology | Identification requires sequential ignorability, positivity and sensitivity | Causal mediation | Explains why terminal U/M/D regressions are insufficient |

The literature predicts several mutually inconsistent possibilities: heat/incubation papers predict path dependence; geometry/LIPSS papers predict a directional channel independent of depth; defocus papers predict depth-indexed focus loss; roughness studies predict material-dependent coupling; and information-bottleneck work predicts that a task-specific low-dimensional representation may outperform morphology reconstruction. This contradiction is a design guide: each mechanism needs a distinct intervention or observer.




## Beyond conventional regression

The scientific alternatives should be tested in this order, with a reason for each method:

| Framing | Hypothesis tested | Falsifier | Why needed | Simpler substitute |
|---|---|---|---|---|
| Latent dynamical-system identification | A small hidden S generates M and D | No stable state under held-family CV | Separates state from static correlation | Joint GAM with explicit depth/defocus |
| Nonlinear observability analysis | D and M jointly observe S | Large indistinguishable-state sets | Quantifies what terminal sensors can recover | Matched-pair analysis |
| Switching dynamics/HMM | Removal law changes by regime | Regime labels do not recur across families | Tests threshold/melting/shielding transitions | Piecewise GAM |
| Koopman/operator coordinates | A transformed coordinate evolves approximately linearly | No stable eigenfunctions or forecast gain | Tests whether a simple reaction coordinate exists | PLS/SDR \(Z_D\) |
| Universal differential equation / neural ODE | Known ablation law plus learned closure is adequate | Closure is unstable or violates constraints | Encodes continuous exposure and waiting time | Discrete state update |
| Invariant representation learning | A state transfers across sessions/material batches | Representation remains session-specific | Separates physics from nuisance | Grouped residualization |
| Predictive-state representation | Future observations are sufficient state | Past history improves after state conditioning | Directly targets control-relevant prediction | \(D_t,M_t\) feature model |
| Causal representation learning | Latent factors respond invariantly to interventions | Factors change under equivalent interventions | Connects representation to mechanism | Supervised SDR |
| Multiview/joint learning | Thermal/optical/phase views identify the same state | Views disagree or add no information | Tests latent-state observability | Single-view regression |

A method is necessary only when it makes a falsifiable scientific prediction that a simpler method cannot test. Benchmark improvement without a changed interpretation is insufficient.

## Adversarial self-critique matrix

| Positive observation | Simplest alternative | Adversarial test | Stop condition |
|---|---|---|---|
| A morphology feature predicts D | It proxies hatch/pass/session | Residualize against those variables; leave session/family out | Gain disappears under support matching |
| A latent state improves D | It encodes U or sample identity | Predict U/session from state; shuffle IDs; state-reset | State is not stable or is trivially decodable as nuisance |
| A sequence model improves | Static nonlinear interactions suffice | ExtraTrees/boosting with matched tuning and derived dose | Sequence gain vanishes |
| D predicts M | Both are driven by N or dose | Compare D, N, dose under identical folds | D has no incremental contribution |
| M predicts future \(\Delta D\) | Current D alone explains it | Compare \(D_t,U_{t+1}\) versus \(D_t,M_t,U_{t+1}\) | M increment is within practical null |
| A spectral ratio is important | It is amplitude or resolution artifact | Match amplitude, repeat metrology, fixed/normalized bands | Ratio unstable across scans |
| A regime switch appears | Session/domain shift | Hold out sessions and process families; preregister regimes | Regime is session-only |
| Physics closure improves | It is unconstrained overfit | Remove closure, enforce constraints, test independent observer | Gain fails physical/observer checks |
| Mamba improves | It exploits leakage or interpolation | Permute pass order, reset state, pass-range extrapolate | Gain exists only in interpolation |
| Similar D surfaces differ in M | Measurement noise explains it | Repeat scans and errors-in-variables sensitivity | Difference is below metrology noise |




## Moonshot scoring

Scores are 1–5; high risk is reported separately from novelty. “Current-data fit” is intentionally low for ideas requiring new sequences.

| Rank | Moonshot | Novelty | Scientific depth | Impact | Current-data fit | Identifiability | Falsifiability | Implementation difficulty | Publication potential |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Longitudinal morphology state for \(\Delta D\) | 5 | 5 | 5 | 1 | 4 | 5 | 5 | 5 |
| 2 | Non-commutative order/wait effects | 5 | 5 | 5 | 1 | 5 | 5 | 4 | 5 |
| 3 | Minimum explicit physical state + closure | 5 | 5 | 5 | 2 | 3 | 4 | 5 | 5 |
| 4 | Depth reaction coordinate plus history bottleneck | 5 | 5 | 4 | 4 | 3 | 5 | 3 | 5 |
| 5 | SSM/Mamba pass extrapolation | 4 | 4 | 4 | 1 | 3 | 4 | 4 | 4 |

The ranking favors hypotheses that could alter the scientific interpretation and still produce a valuable null result. It does not favor Mamba simply because it is available in the repository.

