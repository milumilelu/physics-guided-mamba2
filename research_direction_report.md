# Research-direction report: machining depth as the final target

> **Revision note (2026-09-12).** External review narrowed the active program to one flagship question: **when is current depth sufficient to represent the state relevant to the next removal step, and which measured morphology variables must be retained when it is not?** The shortlist below is therefore staged rather than parallel. Novelty scores are provisional opportunity scores; literature precedents constrain claims about heat accumulation, morphology feedback, scale-dependent evolution, order effects, and depth/defocus. Terminal regressions establish predictive associations and observability limits under a specified representation, not causal mediation or fundamental material forgetting.

**Repository audited:** `physics-guided-mamba2` (workspace checkout, 2026-09-12)  
**Pasted brief:** `C:\Users\RZF\.codex\attachments\bc708bd6-3119-4845-9fbf-32d90e17c222\pasted-text-1.txt`

## Executive judgment

The strongest defensible structure is currently:

\[
U\rightarrow D \quad\text{(dominant predictive route)},\qquad
M_{\mathrm{DC\ removed}}\rightarrow D \quad\text{(real but weaker route)}
\]

with morphology decomposing into at least two observable families:

- **spectral composition and amplitude**, which carry depth-related information;
- **directional texture**, which is strongly organized by hatch/scan geometry and is weakly related to terminal depth.

The data do **not** yet identify a strict serial causal chain \(U\to M\to D\), a unique latent physical state, or a morphology representation that replaces process parameters. The most promising paper is therefore a **depth-oriented state sufficiency study**: quantify what terminal morphology preserves under a fixed representation, what remains unobservable, which process factors remain necessary after observing morphology, and whether a compact spectral/amplitude representation is sufficient for depth and then for the next removal increment within the measured support.

The immediate next step is to interpret the sealed D1.1 conditional/residual result, then execute D2 factor absorption, reverse depth conditioning, compression, and the error atlas. A large Mamba model should be deferred until a genuine sequence target exists and simple state baselines establish an incremental need.

**Flagship execution order.** (A) Complete matched Tree(U), Tree(M), Tree(U+P), and Tree(U+M) baselines; the current UTREE (U+M) score is exploratory without Tree(U). (B) Build a support-matched depth/pass/dose/session conditional spectrum matrix. (C) Collect same-location histories with comparable (D_t), repeated metrology, two fixed next-step recipes, and compare depth-only, depth-plus-morphology, and depth-plus-history predictors. (D) Add observer and explicit depth/defocus/thermal states before testing GRU, SSM, or Mamba on pass extrapolation. This sequence tests whether history can be compressed into current depth plus a few morphology variables while preserving future response.

## Repository audit and evidence map

### Data, contracts, and executed protocols

- Main rectangular dataset: **200 ROI records**, comprising 120 formal, 60 pass-main, and 20 pass-supplement; **160 shared height sources** and **134 process groups**. The repository explicitly forbids treating all 200 ROI as independent.
- Primary depth experiment `outputs/depth_target_v1/20260911T041355Z_D1_depth_baselines_...` is a **PASS** run on MAIN180 (formal120 + pass-main60). SUPP20 was held out as domain stress and was not used for training.
- Morphology is computed from \(r=H-\operatorname{median}(H)\). The feature contract forbids canonical depth, absolute height mean/minimum, raw DC, target-normalized features, and target-equivalent volume. Translation-invariance tests are present.
- Frozen outer and inner partitions reuse the audited source/process-family components. The D1 run records 3,600 OOF prediction rows and a sealed source snapshot. The primary tuning loss is component-balanced MSE; MAE is the primary report metric. Intervals are descriptive fixed-OOF bootstrap intervals, not refit uncertainty or causal intervals.
- Phase 2.8r1 is the corrected spectrum protocol. Its acceptance file reports 19,232 component OOF rows, exact Q² reconstruction error \(1.3\times10^{-15}\), and all 200 rows covered. Historical Phase 2.8 mechanism claims must not be mixed with r1.

### Numerical evidence from the executed depth baseline

| D model | Morphology/process inputs | GAM MAE (µm) | GAM RMSE (µm) | GAM component-balanced Q² | Ridge MAE (µm) | Ridge Q² |
|---|---|---:|---:|---:|---:|---:|
| B0 | train mean | 19.03 | 20.93 | 0.000 | 19.03 | 0.000 |
| BA | amplitude block A | 16.63 | 18.70 | 0.201 | 17.32 | 0.143 |
| BG | geometry block G | 14.23 | 17.73 | 0.282 | 18.55 | −0.014 |
| BM | all morphology except process | 12.99 | 16.33 | 0.392 | 12.35 | 0.440 |
| BP | ILR spectral composition P | 12.44 | 16.12 | 0.407 | 12.11 | 0.455 |
| BPE | absolute spectral energies PE | 13.39 | 16.39 | 0.387 | 12.38 | 0.438 |
| BQ | dose proxy only | 14.83 | 17.48 | 0.302 | 14.79 | 0.323 |
| BT | directional texture T | 16.10 | 19.10 | 0.168 | 15.54 | 0.226 |
| BU | full process \(U\) | **6.53** | **8.39** | **0.839** | 7.00 | 0.811 |
| BUM | \(U+\) all morphology | 7.91 | 9.83 | 0.779 | 7.49 | 0.755 |

The result supports **M → D information**, especially spectral composition and amplitude, but does not show a stable improvement from adding all current morphology to \(U\). In the fixed-OOF conditional comparison, \(U\) remains strongly necessary after morphology; the GAM interval for \(\Delta_{M|U}=R(U)-R(U+M)\) is negative (adding all morphology worsened this regularized fit), while the Ridge interval includes zero. This is evidence of redundancy/representation and small-sample instability, not evidence that morphology is useless.

### Corrected Phase 2.8 spectrum evidence

Using grouped Q² (higher is better):

| Target | Full process | Hatch only | Process minus hatch | Interpretation |
|---|---:|---:|---:|---|
| Depth D (source-grouped) | 0.552 | 0.044 | 0.501 | depth is process-controlled but not hatch-dominated |
| Amplitude A | 0.160 | 0.001 | 0.142 | weak, multi-factor process dependence |
| Spectral composition P | 0.308 | 0.145 | 0.145 | multi-factor, with hatch contribution |
| Direction amplitude A2 | 0.641 | 0.615 | 0.005 | nearly hatch-sufficient in this support |
| Direction entropy | 0.662 | 0.582 | 0.015 | hatch-dominant but not exactly sufficient |

The direction result is scale-specific: the 8–16 µm band is highly hatch-associated, while the 16–32 µm direction target has Q² about 0.143. The spectral ILR coordinates represent log contrasts of five bands \((<8,8\!-\!16,16\!-\!32,32\!-\!64,\ge64\ \mu\mathrm m)\), not raw “energy in one band.” The corrected r1 bridge found no statistically validated model-family improvement (L2/L3a not achieved; L3b physical-invalid with one held-out failure). Negative fitted interaction signs are therefore not mechanistic evidence.

### Executed versus merely planned

**Executed and auditable:** D0/D1 depth contract and baselines; additive D1.1 conditional/residual follow-up; Phase 2.8r1 corrected predictability spectrum; extensive observer, grouping, finite-ROI, and backend unit/acceptance checks; task-state static morphology baseline; SUPP20 robustness bookkeeping; single-line pilot/QA packages.

**Diagnostic or engineering only:** E02 depth-conditioned morphology diagnostics; physics-unit/backend gates; finite-ROI solver registration; repair smoke tests; Mamba-2 reference backend acceptance. These do not establish a validated physics-guided predictor.

**Not completed as scientific evidence:** Route-P effect maps, unified error atlas, P/T residual/conditional analyses, support-matched confirmation, depth-oriented compression \(Z_D\), true longitudinal \(\Delta D\) prediction, validated physics calibration, virtual morphology transfer, and morphology-by-design.

**Historical contradictions to correct:** older wording claiming “independent controllable process attributes,” hierarchical controllability, or a uniquely meaningful negative interaction coefficient exceeds the corrected evidence. The repository’s own review documents these withdrawals.

## Hypothesis assessment

| Hypothesis | Status now | What supports it | Decisive next test |
|---|---|---|---|
| H1: compact depth-relevant morphology \(Z_D\) exists | **Partially supported** | BP/BA outperform null; P is strongest block | Nested grouped dimension–risk curves for \(k=1,2,\ldots\), stability across folds/sessions |
| H2: morphology absorbs process information for D | **Not established; redundancy is testable** | \(U\) is strong; \(U+M\) has no stable gain | Predefined per-factor \(\Delta_{j,\mathrm{before/after}}\), block-specific and regularization-separated |
| H3: depth is a progress coordinate for morphology | **Not yet tested** | D predicts some morphology components in E02 diagnostics | Compare \(M_b\leftarrow D\), \(M_b\leftarrow D+U\), \(M_b\leftarrow N\), \(M_b\leftarrow\) dose under support matching |
| H4: scales encode different mechanisms | **Partially supported** | 8–16 µm direction is hatch-dominant; P and D are not | Fixed and normalized bands, wavelet scales, SNR/metrology sensitivity, interaction tests |
| H5: sibling outputs of hidden state S | **Plausible, not identifiable** | Static U→D and U→M both work; terminal data are compatible with common history | Longitudinal same-location passes or measured thermal/optical state |
| H6: morphology improves future \(\Delta D\) | **Untested and high value** | No true pre-next-pass morphology target exists yet | Genuine \(D_t,M_t,U_{t+1}\to\Delta D\) experiment; retrospective pseudo-time only exploratory |
| H7: physics-guided SSM beats static models | **Not supported yet** | Mamba code is candidate/engineering infrastructure | First show validated state target and improvement over GAM/Ridge/GRU/SSM under pass-range extrapolation |

## Candidate directions

| Direction | Scientific question and hypothesis | Data/implementation | Expected result and failure mode | Novelty/value | Difficulty | Existing data? |
|---|---|---|---|---|---:|---|
| 1. Depth-oriented block decomposition | Which DC-removed morphology block predicts D? | Refit B0, BA, BP, BPE, BG, BT, BM, BU, BUM with identical nested grouped CV; report MAE/RMSE/Q² and fold distributions | P remains strongest; failure would bound terminal morphology | High interpretability; publishable negative result | Low | **Yes** |
| 2. Compact \(Z_D\) | Can 17 morphology features be compressed to 1–4 depth-relevant coordinates? | PLS, supervised SDR/SIR, sparse group lasso, monotone GAM; fit reduction inside each outer fold | Small k matches BM/BP; failure means distributed/noisy information | Strong information-bottleneck story | Medium | **Yes** |
| 3. Conditional process-factor absorption | Which U factors remain necessary after M? | Factor-wise leave-one-out \(R(U\setminus j)\), \(R(U\setminus j,M_b)\); group-level paired bootstrap and non-inferiority threshold set before analysis | Identify redundant versus irreducible factors; avoid causal language | Directly answers brief’s core question | Medium | **Yes** |
| 4. Depth-conditioned morphology absorption | Which morphology differences disappear at matched D? | \(M_b\leftarrow D\), \(D+U\), \(D+\)hatch; support-matched pairs and partial \(R^2\) | Amplitude/P partially collapse; direction remains history/geometry-dependent | Mechanistic diagnostic without causal overclaim | Medium | **Yes** |
| 5. Scale ablation and representation audit | Is the depth signal tied to absolute or normalized bands? | Absolute DCT bands, normalized compositions, wavelets, correlation/slope/topology; pre-register scales | Separate removal-state from geometry scales; failure exposes measurement limits | Novel if linked to D and support | Medium | **Yes** |
| 6. Residual learning | Does M explain residual depth after U? | Cross-fitted \(D_U\), model residual \(D-D_U\) from M with nested grouping | Positive residual skill is stronger evidence of added information than joint refit | Clear predictive criterion | Low–medium | **Yes** |
| 7. Nonlinear interaction audit | Are weak joint gains caused by linear/shared regularization? | Small GAM/tensor interactions, boosted-tree baseline, separate U/M penalties; same tuning budget | Positive only if fold-stable and beats pre-set margin | Resolves current ambiguity | Medium | **Yes** |
| 8. SUPP20/domain stress | Does morphology generalize outside MAIN180 support? | Freeze MAIN180 fits; evaluate SUPP20 with support flag and nearest-train distance | Likely degradation; quantifies domain boundary | Essential defensibility | Low | **Yes** |
| 9. Pseudo-temporal pass analysis | Does D organize morphology better than N or dose? | Use existing pass records only with explicit session/family caveats; compare coordinate models | Exploratory support for H3; cannot identify dynamics | Moderate | Medium | **Partly** |
| 10. True incremental-depth state | Does \(M_t\) improve prediction of next removal? | Same-location morphology after each pass; next-step \(\Delta D\); vary waiting time/order | Strong positive result would justify state model; null still informative | **Very high** | High | **No** |
| 11. Physics state + residual closure | What minimum physical state predicts D and M? | Calibrated dose/incubation/defocus state, monotone residual, finite-ROI verification | Simple state may beat black box; identifiability failure is valid | High if validated | High | Partly |
| 12. Mamba/SSM sequence model | Do long-range pass dependencies matter? | Only after longitudinal data; compare GRU, linear SSM, Mamba-2, neural ODE under pass extrapolation | Mamba must improve a defined extrapolation task, not just interpolation | High upside, weak current basis | Very high | **No** |

### Directions to deprioritize or abandon

1. **Claiming strict mediation \(U\to M\to D\) from terminal regressions.** The common-history and feedback alternatives are observationally indistinguishable here.
2. **Scaling up Mamba before a real sequence target exists.** Current pass count is not a recorded scan order, cooling schedule, or per-pass morphology trajectory.
3. **Morphology-by-design or inverse design before forward physics validation.** The corrected bridge already shows invalid/unstable candidate behavior; optimizing against an unvalidated generator would manufacture unsupported examples.

## Immediate re-analysis specification

All analyses below must reuse frozen source/process-family splits, fit preprocessing inside training folds, preserve SUPP20 as untouched stress data, and report physical units.

1. **Primary table:** B0, BQ, BU, BA, BP, BPE, BG, BT, BM, BUM under Ridge and GAM; add dose-only with clearly defined canonical dose and process-family coupling.
2. **Residual test:** obtain cross-fitted \( \hat D_U\), then predict \(D-\hat D_U\) from each morphology block. Use paired fold bootstrap; require improvement in at least four of five outer folds and a predeclared MAE/Q² margin.
3. **Factor absorption matrix:** rows = pulse duration, frequency/energy coupled factor, speed, hatch, pass count, areal dose/overlap factors; columns = A, P, PE, G, T, all M. Report before/after risk increments with uncertainty and support overlap. Do not separate frequency and pulse energy while average power is fixed.
4. **Depth-conditioned reverse matrix:** for each \(M_b\), compare D-only, U-only, D+U, D+hatch; report conditional \(R^2\), matched-depth residuals, and fold/session heterogeneity.
5. **Compression curve:** within each fold, fit PLS and sparse supervised reductions for \(k=1,\ldots,6\). Select the smallest k whose risk is non-inferior to full M using a metrology/application tolerance that is justified before analysis.
6. **Representation sensitivity:** fixed physical bands versus normalized bands; DCT versus wavelet; include RMS slope and correlation length; quantify repeatability and measurement noise.
7. **Error atlas:** map residuals against depth, pass count, hatch, frequency, role, support distance, and morphology SNR. A joint model that improves only in-support formal rows should not be presented as general improvement.
8. **Model comparison:** only after the above, compare GAM/Ridge, boosted trees, MLP, GRU, linear SSM, and Mamba under matched tuning budgets and a predefined extrapolation split.

## Staged roadmap with go/no-go criteria

### Phase 1 — Fast existing-data analyses

Interpret D1.1 block/residual results, then run the support audit, fixed/normalized scale ablation, and unified error atlas.

**Go:** at least one morphology block gives fold-stable residual skill beyond \(U\), or a compact \(Z_D\) is non-inferior to full M.  
**No-go:** no residual signal and all block gains vanish under support matching; write a terminal-morphology limit paper and proceed only to H3/H6 diagnostics.

### Phase 2 — Conditional and latent-state tests

Run factor absorption, reverse depth-conditioned morphology models, supervised compression, heteroscedasticity checks, and session/family random effects or hierarchical shrinkage.

**Go:** predefined factor redundancy or depth-conditioned collapse replicates across source- and process-grouped CV with uncertainty that excludes the practical-null margin.  
**No-go:** effects are fold- or role-specific, or support overlap is inadequate; report identifiability boundary.

### Phase 3 — Physics-guided modeling

Build a hierarchy: dose-only; dose + depth; depth + defocus; depth + incubation; depth + compact morphology; learned hidden state; SSM/Mamba. Use monotonicity/non-negativity where physically justified, but keep residual closure explicit.

**Go:** a low-dimensional state improves held-out MAE/RMSE or pass-range extrapolation over BU and BM, while the forward solver passes numerical/finite-ROI checks and parameters are identifiable enough for uncertainty.  
**No-go:** calibration is unstable, sequence metadata are missing, or gains occur only by interpolation; do not claim physics guidance.

### Phase 4 — Discriminating experiments

Minimum useful design:

- same nominal total dose, different temporal delivery;
- same pulse collection, different pulse ordering;
- varied inter-pass waiting time;
- same-location morphology after every pass;
- matched current depth with different prior histories;
- controlled pre-roughened surfaces followed by identical ablation;
- hatch/direction perturbations at matched dose;
- optional in-situ optical/thermal signals.

For each condition use at least 3 independent specimens or spatially independent fields, randomize order, record exact scan path and wait time, and reserve a complete process family for confirmation. Measure \(D_t\), \(M_t\), next-step \(\Delta D\), temperature/optical proxies if available, and metrology repeatability.

**Causal identification:** matched-depth history and temporal intervention can challenge common-history explanations; terminal cross-sectional data alone cannot.

## Paper-story candidates and evidence requirements

1. **“A depth-relevant morphology information bottleneck for femtosecond-machined zirconia.”**  
   Required: DC-safe features; nested grouped CV; compact \(Z_D\) with stable k; non-inferiority to full M; comparison with U and U+Z_D; SUPP20 stress test; no causal wording.

2. **“Depth-related versus geometry-related surface morphology.”**  
   Required: depth-conditioned reverse models; hatch-dominant direction bands; multi-scale ablation; support-matched comparisons; uncertainty showing which differences persist at matched D.

3. **“Terminal morphology has limited power to reconstruct ablation history.”**  
   Required: strong U→D, weaker M→D, no stable U+M gain, history residuals after matching D, and explicit demonstration that terminal morphology is non-unique under offset/history perturbations.

4. **“Depth as a progress coordinate for multiscale morphology.”**  
   Required: repeated-pass data or a carefully caveated pseudo-temporal analysis; D must outperform N and dose for selected bands; \(M_b\leftarrow D+U\) must reveal which process effects remain.

5. **“Morphology-enhanced prediction of subsequent material removal.”**  
   Required: genuine same-location longitudinal \(D_t,M_t,U_{t+1}\to\Delta D\), pre-registered next-step evaluation, and improvement over \(D_t,U_{t+1}\) under leave-family-out testing.

6. **“Minimum physical state for multipass ablation.”**  
   Required: validated forward solver, parameter identifiability, finite-ROI and domain checks, state ablations, uncertainty, and superiority or clearer extrapolation than static/sequence baselines. Mamba is an implementation option, not the claim.

## Literature and transferability

- A 2025 JMPT study of femtosecond ablation of zirconia-based ceramics reports micro/macro heat accumulation, an interpulse-period transition toward severe melting, nanoporosity linked to laser parameters, and a validated surface-evolution model. This directly motivates incubation and thermal-history states, but its geometry and calibration cannot be transferred without matching the present optical regime: [Bogatyrev et al., JMPT 2025, DOI 10.1016/j.jmatprotec.2024.118668](https://www.sciencedirect.com/science/article/pii/S0924013624003868).
- A 2026 Optics Express study of femtosecond deep engraving shows scale-dependent spectral roughness evolution and different rates of forgetting initial surface conditions. It supports multiscale state/forgetting tests, not a zirconia-specific law: [Kažukauskas et al., Optics Express 2026, DOI 10.1364/OE.588502](https://doi.org/10.1364/OE.588502).
- A 2023 review of 3Y-TZP surface modification distinguishes fs ablation from ns melting/recrystallization and documents strong dependence of feature depth and pattern definition on laser regime. It supports strict material/regime matching: [Review of 3Y-TZP topographical modification, JEurCeramSoc 2023, DOI 10.1016/j.jeurceramsoc.2023.02.043](https://doi.org/10.1016/j.jeurceramsoc.2023.02.043).
- A 2023 zirconia study comparing femtosecond, CO₂, and abrasive treatments demonstrates that surface roughening outcomes depend on treatment physics and measurement endpoint; adhesion is not equivalent to absolute removal depth: [Piulachs et al., Lasers in Medical Science 2023, DOI 10.1007/s10103-023-03859-2](https://doi.org/10.1007/s10103-023-03859-2).
- A 2022 information-theoretic treatment of sufficient dimension reduction connects central subspaces with information bottlenecks, supporting supervised rather than reconstruction-based morphology compression: [Ghosh, Entropy 2022, DOI 10.3390/e24020167](https://doi.org/10.3390/e24020167).
- A 2022 NeurIPS paper formalizes relevant versus residual information in high-dimensional regression, useful for treating small-sample joint-model gains cautiously: [Ngampruetikorn & Schwab, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3fbcfbc2b4009ae8dfa17a562532d123-Abstract-Conference.html).
- A 2024 paper introduces Mamba state-space neural operators and evaluates interpolation and extrapolation on dynamical systems. It supports Mamba as a candidate sequence architecture only after a valid dynamical target exists: [Hu et al., “State-space models are accurate and efficient neural operators for dynamical systems,” arXiv:2409.03231](https://arxiv.org/abs/2409.03231).
- Mamba-2’s state-space duality improves computational efficiency, but architecture efficiency does not establish a physical state or a better machining model: [Dao & Gu, “Transformers are SSMs,” arXiv:2405.21060](https://arxiv.org/abs/2405.21060).

## Final decision rule

Keep the program centered on the smallest defensible state that predicts **depth and, eventually, the next depth increment**. Treat morphology as an observable with task-dependent information content. A positive result, a robust null result, or an identifiability boundary can all form a paper; the unacceptable outcome is selecting a model or mechanism before the grouped evidence distinguishes it.



## Additional audit details that affect interpretation

The 72 single-pulse factorial file is excluded from the main depth analysis because it lacks a compatible design table. The stable rectangular package is \(200\times160\times160\) pixels at 0.5 µm/px; 52 samples involve repair masks, affecting only 0.109% of pixels. The main target distribution is D = 0.05–65.08 µm (median 26.75 µm, SD 19.29 µm), so a 6.5 µm MAE is materially better than the 19.0 µm mean baseline but still not metrology-level equivalence.

Phase 1.5 provides only cross-sectional pseudo-pass clues. It contains 15 trajectories with N=1–4 at different positions, not repeated measurements of the same location. Median depth increments are approximately 7.82, 7.05, and 5.29 µm for successive steps; morphology-step RMS first decreases and then rises, with heterogeneous/occasionally negative direction cosines. Depth-window PCA alignment is high for some 8–16 µm spectral components but varies by window. These observations motivate H3 and H6; they cannot establish a temporal state or next-step causality.

The task-state static morphology experiment (E06_STATIC) has a PASS engineering gate and 11,340 OOF rows, but it predicts morphology from process inputs, not depth. MLP_U gives exploratory Q² around 0.55 for A2, 0.60 for entropy, 0.45 for ILR z1, and only about 0.03–0.07 for amplitude/other ILRs; adding the physics input did not consistently improve it. Mamba-2 terminal training attempts failed dtype/device gates, while the backend acceptance test passed. E03 physics solver registration remains FAIL: coverage and finite-ROI checks expose unsupported parameter/domain cases, so no physical calibration or hybrid superiority result is available.

Spatial error diagnostics show residual clustering for amplitude/depth/direction (Moran’s I roughly 0.15–0.31 in historical phase-specific analyses), while ILR composition residuals are near-unclustered. This is a reason to test local/support effects and multiscale representations before increasing global model size. The absence of per-sample scan direction/fill axis makes scan-relative causal interpretation fundamentally unidentifiable.

## Literature extensions to include in a manuscript

Zirconia-specific work reports non-monotonic fluence/removal, roughness U-shapes, heat accumulation at high repetition, and pulse-width dependence in ZrO2 ceramics ([JCPR 2023](https://doi.org/10.36410/jcpr.2023.24.2.230)). Time-resolved electron excitation in zirconia provides a physical basis for carrier/energy latent states but is not a depth predictor ([Applied Physics A 2024](https://doi.org/10.1007/s00339-023-07223-7)). Femtosecond YSZ nanopore studies support defect/porosity as a candidate evolving absorptance state ([JMRT 2023](https://www.sciencedirect.com/science/article/pii/S2238785423000790)).

Incubation studies in fused silica show saturating threshold reduction with pulse count ([De Palo et al. 2022](https://doi.org/10.1364/OE.475592)); a 2024 phenomenological model couples morphology-dependent absorption and incubation ([Song et al. 2024](https://doi.org/10.1016/j.jmapro.2024.04.047)). A 2026 metal study demonstrates that roughness-enhanced absorption is material-dependent ([Applied Surface Science Advances 2026](https://doi.org/10.1016/j.apsadv.2025.100928)); this is a warning against assigning an optical-absorption mechanism to zirconia from morphology coefficients alone.

For model design, supervised sufficient-dimension reduction and information-bottleneck theory provide the right formal target: preserve \(E[D\mid M]\), rather than reconstruct all morphology ([Ghosh 2022](https://doi.org/10.3390/e24020167)). Physics-informed state-space work supports explicit constrained state updates, while Mamba neural-operator studies establish architecture feasibility for long dynamical sequences but not laser validity ([Hu et al. 2024](https://arxiv.org/abs/2409.03231)). Causal mediation requires sequential ignorability, positivity, consistency, and sensitivity analysis; these are not supplied by the terminal DOE ([Imai, Keele & Tingley 2010](https://doi.org/10.1037/a0020761)).


## Closing recommendation

Freeze the current D0/D1 and Phase 2.8r1 artifacts as the evidence base. Interpret D1.1 (including its fixed-budget tree sensitivity and the absence of an ExtraTrees U-only comparator), then execute D2 (factor absorption, reverse depth conditioning, compact Z_D, and support-aware error atlas) before any new architecture. If those tests fail to show stable complementary morphology information, the strongest paper is the scientifically useful limit: terminal morphology is depth-related but cannot reconstruct process history or replace U. If they succeed, use the smallest validated state and reserve Mamba for a genuinely longitudinal next-increment task.



## Evidence map for the competing process structures

| Structure | Evidence class | Decision |
|---|---|---|
| \(U\to D\) | **Supported within measured support** | BU is the dominant D1 predictor (GAM MAE 6.53 µm, Q² .839); not guaranteed outside MAIN180. |
| \(M\to D\) | **Partially supported** | BM/BP have real grouped skill, but errors are about twice BU and representation/session dependent. |
| \((U,M)\to D\) | **Not yet supported as an improvement** | BUM does not beat BU under the released protocol; separate penalties remain unrun, while D1.1 residual and interaction/tree sensitivities are descriptive only. |
| \(U\to M\) and \(U\to D\) in parallel | **Partially supported** | Process predicts spectral/directional morphology and depth through distinct patterns; terminal data cannot distinguish parallel outputs from a shared hidden state. |
| \(U\to S\to\{M,D\}\) | **Plausible, not identifiable** | Heat accumulation/incubation/defocus are physically credible candidates, but no validated measured S is available. |
| \(U\to Z\to D\), with Z morphology-related | **Not yet tested** | Requires supervised \(Z_D\) compression and conditional risk tests. |
| \(D\) as morphology progress coordinate | **Partially supported descriptive clue** | E02 and Phase 1.5 show selected spectral/amplitude relationships; pseudo-pass data are not longitudinal. |
| \(M\) as diagnostic fingerprint | **Partially supported** | M predicts D and process family to different degrees; realization and scan-axis effects are not fully observed. |
| \(M_t\) as state for subsequent increments | **Not tested** | Requires same-location per-pass morphology and next-step depth. |
| Strict causal mediation or optical absorption pathway | **Fundamentally unidentifiable with current terminal data** | No sequential ignorability, intervention, or independent absorptance/temperature measurement. |



## Requirement coverage and current re-analysis status

The following distinction is deliberate: the repository audit is complete, but the pasted brief’s requested new D1.1–D5 analyses are not all executed. Existing numbers are evidence; the planned rows below are work items.

| Required analysis | Current evidence/artifact | Status |
|---|---|---|
| \(U\to D\) | D1 depth_metrics.csv, BU | **Executed** |
| Dose-only \(\to D\) | D1 BQ | **Executed** |
| Each morphology block \(\to D\) | D1 BA/BP/BPE/BG/BT/BM | **Executed** |
| Selected morphology combinations \(\to D\) | BM and block definitions only | **Partial; no systematic combinations** |
| \(U+\) each morphology block \(\to D\) | D1.1 latest sealed run | **Executed descriptively** |
| \(U+\) all morphology \(\to D\) | D1 BUM | **Executed** |
| Residual \(D-\hat D_U\leftarrow M\) | D1.1 latest sealed run, strict nested residual CV | **Executed descriptively** |
| Factor before/after matrix | No released output | **Not executed** |
| \(D\to M_b\), \((D,U)\to M_b\) | E02 diagnostic OOF (Task-State, separate contract) | **Executed diagnostically; needs unified re-analysis** |
| Compact \(Z_D\), \(k=1,\ldots\) | No compression output for D target | **Not executed** |
| Separate U/M penalties, interactions, trees | D1.1 interaction and ExtraTrees sensitivity | **Partial; no separate penalties or U-only tree comparator** |
| Sample-size, metrology noise, session and support tests | Historical diagnostics only | **Partial; confirmation absent** |
| SUPP20 stress | E10/static tracking and support diagnostics | **Executed as stress only; never tuned** |
| True future \(\Delta D\) | No same-location pass sequence | **Fundamentally unavailable** |

E02 terminology in this report means the Task-State experiment 'outputs/task_state_v1/20260906T111859Z_E02...': it contains morphology targets and depth-conditioned diagnostics under its own protocol. It is not the depth_target_v1 D1 experiment and must not be presented as a new M→D benchmark.

## Explicit prioritized shortlist

Scores are 1 (low) to 5 (high); “defensible if null” measures whether a negative result still forms a useful paper. The rank is a decision aid, not a fitted statistical quantity.

| Rank | Direction | Novelty | Data fit | Preliminary support | Cost (inverse) | Interpretability | Defensible if null | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Block-specific residual \(M\to D-\hat D_U\) | 3 | 5 | 3 | 5 | 5 | 5 | **Low risk** |
| 2 | Factor absorption before/after morphology | 3 | 5 | 3 | 4 | 5 | 5 | **Low risk** |
| 3 | Depth-conditioned reverse morphology | 3 | 5 | 3 | 4 | 5 | 5 | **Low risk** |
| 4 | Compact supervised \(Z_D\) | 3 | 5 | 3 | 3 | 4 | 5 | **Low risk** |
| 5 | Scale/representation and error atlas | 4 | 5 | 3 | 4 | 4 | 5 | **Low risk** |
| 6 | Same-location next-increment morphology state | 3 | 1 | 4 | 1 | 5 | 5 | **High upside** |
| 7 | Defocus/incubation/absorptance physical state | 3 | 2 | 4 | 2 | 4 | 5 | **High upside** |
| 8 | Physics-informed residual-closure model | 3 | 2 | 2 | 2 | 3 | 4 | **High upside** |
| 9 | Mamba/SSM for pass extrapolation | 2 | 1 | 1 | 2 | 2 | 3 | **High upside only after data** |
| 10 | Strict terminal mediation claim | 2 | 5 | 0 | 4 | 1 | 1 | **Abandon** |
| 11 | Mamba-first scale-up without longitudinal targets | 2 | 1 | 0 | 1 | 1 | 1 | **Abandon** |
| 12 | Inverse/morphology-by-design before forward validation | 4 | 1 | 0 | 1 | 1 | 2 | **Abandon** |

The five low-risk directions are the first five rows. The four high-upside directions are rows 6–9, with rows 6–8 requiring new measurements or solver validation. The last three should be removed from the active work queue.

## Future-experiment matrix

| Experiment | Hypothesis tested | Controls | Changed variable | Outputs | Competing predictions | Minimum design | Causal status |
|---|---|---|---|---|---|---|---|
| Matched-dose temporal delivery | Incubation/thermal history matters beyond total dose | Material, spot, total pulse count, nominal dose | Pulse spacing or burst schedule | \(D_t,M_t,\Delta D\), temperature proxy | Dose-only predicts equal D; state model predicts schedule effect | Feasibility pilot, then power from repeated-process variance and a predeclared effect margin | Intervention identifies schedule effect within support; hidden-state mechanism still needs observer |
| Pulse-order reversal | Order affects accumulated state beyond current observable state | Same pulse multiset and total dose | Permute pulse energies/frequencies | Terminal D and M, per-pass profiles | Memoryless model invariant; nonlinear current-depth model may also be order-sensitive; residual effect supports extra state | Feasibility pilot, then powered matched orders | Causal order intervention within support, not proof of hidden memory by itself |
| Inter-pass waiting time | Cooling/relaxation changes depth increments | Same U, pass count, scan path | Wait 0.1/1/10 s (registered values) | \(D_t,M_t,\Delta D\), thermal decay | No waiting effect vs thermal-state effect | 3 specimens × 3 waits, repeated locations | Causal temporal intervention |
| Same-location per-pass metrology | M_t predicts next removal | Same coordinates and scan sequence | Pass number is observed, not substituted | Co-registered H/M after every pass | \(D_t,U\) sufficient vs M_t adds \(\Delta D\) skill | Repeat scans first; then locations/specimens sized for a predeclared practical margin | Needed for longitudinal predictive-state claim; not automatic causal morphology evidence |
| Matched-current-depth histories | Specified terminal observation does not uniquely define history | Current D within fixed, noise-based tolerance | Prior path/pass ordering | M differences at matched D | Unique-state model collapses; history model retains differences | History pairs and repeat scans selected before outcome inspection | Distinguishes an observability boundary from history dependence; not proof of irreversible material forgetting |
| Pre-roughened versus polished start | Initial morphology changes absorptance/incubation | Same subsequent U and dose | Controlled RMS/correlation-length bands | Initial M, D increments, final M | No initial effect vs morphology-mediated effect | 3 roughness levels × 3 repeats | Intervention; mediation still needs measured absorptance |
| Hatch/direction perturbation at matched dose | Directional texture is geometry channel | Dose, pulse count, material | Hatch and scan orientation | Direction blocks, D, residual D | Direction changes M only vs changes D too | 4 hatch levels × 2 orientations × 3 repeats | Causal geometry effect if orientation recorded |
| In-situ optical/thermal observer | Hidden state can be measured | Same process family and metrology | Add calibrated reflectance/thermal channel | \(S_t\) proxy, D/M, uncertainty | No observer gain vs state observability | 3 repeats per condition, calibration set held out | Mechanistic support; not automatically mediation |

The minimum counts are starting designs for workflow and variance estimation, not power calculations. Before execution, estimate repeated-scan and repeated-process variance and set a confirmatory holdout by complete process family. For an observed current depth \(\widetilde D_t=D_t+e_t\), the observed increment is \(\widetilde{\Delta D}=\Delta D+e_{t+1}-e_t\); under independent scan errors, \(\operatorname{Cov}(e_t,\widetilde{\Delta D})=-\operatorname{Var}(e_t)\). This shared-error term can create or distort apparent depth dependence and must be included in the analysis plan.

## Split semantics and reporting contract

Every result must label its generalization target:

- **Interpolation:** held-out samples from already represented process families and sessions.
- **New process-family generalization:** entire cv_process_group held out; shared-source components cannot cross folds.
- **New session generalization:** complete measurement session held out, with metrology calibration checked separately.
- **New pass-range extrapolation:** train on registered pass values and test on a higher pass range; current SUPP20 is confounded by family/session overlap and must not be called a pure N≥5 extrapolation.

For each model report MAE and RMSE in µm, component-balanced Q², fold/component distribution, support distance, and whether tuning used only training folds. Statistical non-significance is never converted into equivalence; a non-inferiority tolerance must be justified by metrology or application requirements.

## Evidence index for reproducibility

| Topic | Authoritative paths | What is actually established |
|---|---|---|
| Depth contract and leakage | config/depth_target_v1/protocol.yaml; experiments/depth_target_v1/00_contract.py; outputs/depth_target_v1/20260911T041330Z_D0_contract_... | D0 PASS, translation invariance and forbidden-feature checks |
| Depth baselines | experiments/depth_target_v1/01_depth_baselines.py; outputs/depth_target_v1/20260911T041355Z_D1_depth_baselines_... | Ridge/GAM D1 OOF and metrics |
| D1.1 conditional/residual follow-up | experiments/depth_target_v1/03_d1_1_block_residual.py; outputs/depth_target_v1/20260912T041545Z_D1_1_block_residual_48ea2674_ea51976_0909dd/ | U+block, strict residual, interaction and fixed ExtraTrees sensitivity; descriptive, no matched U-only tree comparator |
| Morphology extraction | src/depth_target/features.py; src/depth_target/models.py | 17 DC-removed features and B0–BUM design matrices |
| Grouping/splits | src/depth_target/splits.py; src/task_state_learning/grouping.py; E00 split manifests | Source/process-family component isolation |
| Spectral predictability | experiments/phase2_8_r1/24_information_decomposition.py; outputs/phase2_8_r1/predictability_spectrum.csv; acceptance_checks.json | Corrected D/A/P/T grouped Q²; no valid bridge interaction improvement |
| Scale/pass diagnostics | experiments/phase1_5/; outputs/phase1_5/; experiments/phase2_5/ | Cross-sectional scale and pseudo-pass clues, not longitudinal dynamics |
| Geometry/single-line | experiments/phase2_6/, outputs/phase2_6/ | Width/spectral bridge; scan-axis provenance unavailable |
| Task-state/physics/SSM | experiments/task_state_v1/; outputs/task_state_v1/ | Static morphology diagnostics and backend checks; physics calibration/terminal training not validated |
| Inverse design/virtual data | experiments/mechanism_virtual_augmentation/; phase2.8 bridge scripts | Design only; no validated forward generator or inverse result |

The reproducible D1 command is:
.\.venv\Scripts\python.exe experiments\depth_target_v1\01_depth_baselines.py --contract-run <PASS D0 run> --config config\depth_target_v1\protocol.yaml

The command must write a new run directory; existing sealed artifacts are never overwritten. D1.1 is an additive development extension and cannot retroactively change D1 frozen numbers. Its latest sealed run is outputs/depth_target_v1/20260912T041545Z_D1_1_block_residual_48ea2674_ea51976_0909dd.



## Measurement and historical-correction boundaries

The canonical target is \(D=-\operatorname{median}_{\Omega}H\), with \(\Omega\) the 80×80 µm ROI and 0.5 µm pixels. Morphology is \(r=H-\operatorname{median}_{\Omega}H\). The field is finite: the finest resolvable wavelength is about 1 µm, and the ≥64 µm DCT band has only a few independent spatial cycles in an 80 µm window. Any band result must therefore include finite-window and metrology sensitivity checks. Adding a constant to H leaves r unchanged while changing D, proving that r cannot universally recover absolute depth without a calibrated material/process prior.

MAIN180 contains 126 source/process components. SUPP20 stress evaluation purges 40 overlapping rows from training in the E00 protocol; “untouched” means not tuned or fitted, not an independent prospective confirmation, because all 200 records have participated in discovery choices. The current checkout is a dirty research worktree; released run manifests and source snapshots, rather than an unpinned working-tree state, are the reproducibility anchors.

The Phase 2.8r1 Q² values for D (~0.552) and the D1 BU Q² (~0.839) are not contradictory: they use different populations/estimators, fold weighting and skill aggregation. They must never appear as a single leaderboard.

| Historical issue | Correction | Residual implication |
|---|---|---|
| Positive-region physical constraint missed a −4.808 µm candidate | r1 rejects invalid pixels and retains held-out failures | L3b is physical-invalid, not evidence for a negative mechanism |
| L3a sampled only one h period | r1 uses a 2h phase grid | No valid interaction improvement |
| Training mean used where frozen protocol required median | r1 restores group-level median selection | Legacy scores are not interchangeable |
| One exact repeat over-interpreted | r1 labels realization diagnostic descriptive only | Repeatability floor is unknown |
| OOF/R² fields were incomplete | r1 stores observed/predicted/null and fold identifiers | Q² is reconstructible, but not causal evidence |
| Array-transfer DC singularity | finite-sum transfer implementation and tests | Historical bridge remains a model diagnostic |
| “Independent controllable attributes” wording | replace with observable-dependent predictability/conditional redundancy | Causal independence is not identified |

## Literature transfer matrix

| Paper/topic | Material and laser regime | Outcome/mechanism | Model or method | Transfer to this zirconia study |
|---|---|---|---|---|
| Bogatyrev et al., JMPT 2025 | 8YSZ, femtosecond; dynamic grooves/3D crown | Micro/macro heat accumulation, interpulse transition to melting, nanoporosity, morphology feedback | Pulse-resolved computational ablation model | **Direct mechanism precedent**, but calibrate composition, fluence, spot and kinematics |
| JCPR 2023 ultrashort ZrO2 | ZrO2 ceramic, 390 fs–6 ps, varied fluence/repetition | Non-monotonic removal, U-shaped roughness, heat accumulation, shielding/redeposition | Experimental parameter study | **Direct material relevance**; endpoint and optical setup differ |
| Applied Physics A 2024 zirconia excitation | Zirconia, ultrashort pulse, time-resolved | Electron excitation/filament dynamics from ps–ns | Mechanistic time-resolved simulation/analysis | Candidate carrier/energy state; **not a depth predictor** |
| YSZ nanopore JMRT 2023 | YSZ, femtosecond near threshold | Nanopores/intergranular ablation | Microscopy and process map | Supports porosity/defect state; single-pulse regime limits transfer |
| De Palo et al., Optics Express 2022 | Fused silica, 1030 nm fs, 0.06–200 kHz | Threshold decreases then saturates with N (incubation) | Exponential incubation fit | Functional form to test; **material analogue only** |
| Song et al., JMP 2024 | Fused silica, fs multipulse | Morphology-dependent absorption + incubation | Phenomenological crater/groove model | Supports feedback hypothesis; coefficients nontransferable |
| Applied Surface Science Advances 2026 incubation | Aluminium and steel, 500 fs/1040 nm | Roughness absorption in Al but different mechanism in steel | FDTD + experiment | Demonstrates mechanism non-universality; do not infer zirconia absorptance |
| Kažukauskas et al., Optics Express 2026 | Fused silica deep engraving, fs multilayer | Spatial scales forget initial surface at different rates | Spectral evolution analysis | Strong multiscale-state motivation; not zirconia law |
| 3Y-TZP review 2023 | Zirconia, ns/fs patterning across wavelengths | fs ablation versus ns melt/recrystallization; depth/definition trade-offs | Review of topography studies | Supports strict regime/material matching |
| Li et al., Machines 2026 groove-depth model | Femtosecond grooves, attention + monotonic constraints | Predicts groove depth under single-scan conditions | Attention ensemble with monotonic constraints | Useful baseline/constraint idea; does not establish multipass state |
| Ghosh, Entropy 2022 | Statistical methodology | Sufficient dimension reduction as information bottleneck | SDR/information-theoretic framework | Directly supports \(Z_D\) objective |
| Hu et al., 2024 SSM neural operator | Simulated PDE and pharmacology dynamics | Long-range/extrapolation sequence learning | Mamba SSM neural operator | Architecture precedent only; no laser validation |
| Imai et al., 2010 mediation | General causal methodology | Sequential ignorability, positivity, sensitivity | Causal mediation framework | Explains why terminal U/M/D regressions cannot identify mediation |

## Optional process-to-\(Z_D\)-to-depth chain

A process-informed compression route is testable but must be evaluated as a nested predictive construction:

1. Fit \(Z_D=g(U)\) only within each training fold, or fit \(Z_D\) from morphology with a supervised target and cross-fit it.
2. Compare \(D\leftarrow U\), \(D\leftarrow Z_D\), \(D\leftarrow U+Z_D\), and \(D\leftarrow U+M\).
3. Require \(U+Z_D\) to be non-inferior to \(U+M\), and require the intermediate to remain stable across process-family and session splits.
4. Treat a learned \(Z_D\) as a predictive state unless it maps to independently measured depth, defocus, incubation, temperature, porosity, or absorptance. A latent coordinate alone does not establish a physical mechanism.



## D1.1 executed follow-up (latest sealed run)

A new additive run was executed after the initial audit: outputs/depth_target_v1/20260912T041545Z_D1_1_block_residual_48ea2674_ea51976_0909dd/. It uses the frozen E00 outer/inner component partitions, 2,880 OOF rows across 16 models, strict nested residual-target construction, and a fixed-budget ExtraTrees sensitivity model. The gate is PASS, but the scope remains MAIN180 descriptive only; SUPP20 is excluded.

| Model | MAE (µm) | RMSE (µm) | Q² | Interpretation |
|---|---:|---:|---:|---|
| BU GAM | 6.528 | 8.386 | .839 | Frozen process baseline |
| BU Ridge | 7.002 | 9.089 | .811 | Frozen process baseline |
| U+A / U+P / U+PE / U+G / U+T, GAM | 7.16–7.42 | 9.05–9.10 | .811–.813 | All worse in pooled point estimates |
| U+A / U+P / U+PE / U+G / U+T, Ridge | 7.058–7.224 | 9.05–9.28 | .803–.813 | No stable gain |
| Residual M\|U, GAM | 6.458 | 8.349 | .841 | Tiny point improvement; fixed-OOF risk CI includes zero |
| Residual M\|U, Ridge | 6.887 | 9.004 | .815 | Tiny point improvement; fixed-OOF risk CI includes zero |
| Explicit U×M interaction, Ridge | 7.057 | 8.984 | .816 | No gain |
| UTREE, ExtraTrees on log-U + all M | 5.419 | 7.110 | .885 | Exploratory nonlinear sensitivity; no U-only ExtraTrees comparator |

The unadjusted fixed-OOF component bootstrap intervals show GAM U+P, U+PE, and U+G risk increases with lower bounds above zero; residual-model intervals include zero. UTREE is better than BU Ridge (Δrisk −32.05 µm², 95% CI [−62.50, −5.90]) but its comparison with BU GAM crosses zero (Δrisk −19.77 µm², 95% CI [−45.83, 4.11]). Because UTREE has no matched U-only ExtraTrees baseline and its hyperparameters were fixed rather than selected under the same budget, it is evidence for a nonlinear modeling opportunity, not proof that morphology adds information. A confirmatory tree comparison must include ExtraTrees(U), ExtraTrees(M), and ExtraTrees(U+M) with fold-local tuning.




A directly relevant machine-learning precedent is Guangxian Li, Luyang Ding, Meng Liu, Hui Xie and Songlin Ding, “Prediction of Groove Depth in Femtosecond Laser Ablation via Attention Mechanism and Monotonic Constraint,” Machines 14(5):509 (2026). It uses monotonic constraints for single-scan groove depth, but does not validate cumulative multipass 3D morphology; it is a baseline and constraint precedent, not evidence for this repository’s state model: [DOI 10.3390/machines14050509](https://doi.org/10.3390/machines14050509).

The 2026 incubation paper is Nicolas Thomae, Maximilian Spellauge, David Redka and Heinz P. Huber, “Deciphering the driving mechanisms of incubation in ultrashort pulse laser ablation,” Applied Surface Science Advances 32, 100928. Its aluminium/steel comparison is useful precisely because it finds material-dependent incubation mechanisms; it should be cited as a transferability warning: [DOI 10.1016/j.apsadv.2025.100928](https://doi.org/10.1016/j.apsadv.2025.100928).



Additional transferability notes: depth-dependent defocus has been measured in femtosecond implant-cavity work, where cavity depth changes the focal condition and lowers subsequent removal; this supports a depth→defocus state test, not a transferable coefficient ([PMC11828434](https://pmc.ncbi.nlm.nih.gov/articles/PMC11828434/)). Physics-informed state-space neural networks provide a learned-closure pattern in which constrained state updates are corrected by data residuals ([arXiv:2309.12211](https://arxiv.org/abs/2309.12211)); the present repository has no validated closure or independently observed state. Variational physics-informed state-space models add uncertainty-aware latent-state inference ([NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/file/b2913cff905a649c5bda3ce2cd19088c-Paper-Conference.pdf)), but this remains methodological precedent rather than laser evidence. Dynamic morphology feedback is directly motivated by the zirconia JMPT model and fused-silica incubation model above; no current dataset measures feedback within a pass.



The D1.1 run was reproduced with:

.\.venv\Scripts\python.exe experiments\depth_target_v1\03_d1_1_block_residual.py --contract-run outputs\depth_target_v1\20260911T041330Z_D0_contract_48ea2674_ced4068_e7407e --config config\depth_target_v1\protocol.yaml

The depth-target unit/release suite passes 9/9 tests. D1.1’s PASS gate means execution, coverage and contract checks passed; it is not a hypothesis-acceptance gate.



## Process-factor absorption matrix to populate in D2

The current data establish the candidate rows and identifiability constraints, but not the entries. The released D2 table should have this form, with each cell reporting \(\Delta R\) before and after morphology plus a fixed-OOF confidence interval:

| Process factor row | A | P | PE | G | T | All M | Identifiability note |
|---|---|---|---|---|---|---|---|
| Pulse duration | pending | pending | pending | pending | pending | pending | Separate only if independently varied |
| Frequency / pulse-energy coupled factor | pending | pending | pending | pending | pending | pending | Fixed average power makes f and \(E_p\) deterministically coupled |
| Scan speed | pending | pending | pending | pending | pending | pending | Check overlap with dose and pulse overlap |
| Hatch spacing | pending | pending | pending | pending | pending | pending | Strong direction-channel candidate |
| Pass count | pending | pending | pending | pending | pending | pending | Session/order confounding must be quantified |
| Areal dose / overlap | pending | pending | pending | pending | pending | pending | Derived factors are not independent rows |

An entry may be called predictive redundancy only when the post-morphology increment is within a predeclared application/metrology tolerance and stable across grouping choices. It must not be labeled causal absorption or optical absorption.



The E02 Task-State reverse diagnostics contain 22,680 OOF rows. For morphology targets, the GAM risk summary is approximately U-only .850, D-only .935, D+hatch .757; Ridge is U-only .791, D-only .932, D+hatch .812. Per-target D-only GAM Q² is A_med .168, ILR z1 .332, ILR z2 −.061, A2 −.002, entropy .085; adding hatch raises A2 to .483 and entropy to .536. These numbers support depth-related spectral/amplitude structure and hatch-related directional structure, but they are not a terminal M→D result and overall U+D additions were uncertain. HIST200/support-matched reruns remain unfinished.

