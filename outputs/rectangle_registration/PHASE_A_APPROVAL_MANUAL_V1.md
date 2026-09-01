# Phase A Approval (manual_v1)

Status: BLOCKED

Automatic decision: BLOCKED

Samples: 200 manual_v1 rows (all manual geometry gates PASS); 180 exported H_reg/H_200; 20 samples have no registered export and therefore fail the all-sample gates.

Automatic checks (evaluated over all 200 samples):
- manual_geometry_gate_200_of_200: PASS
- theta_d4_frozen_confirmed: PASS
- paired_gate_all_passed: PASS
- all_200_samples_have_registered_exports: FAIL
- l_reg_ge_260um_all_samples: FAIL
- final_leveling_all_pass: FAIL
- final_leveling_reference_sufficient: FAIL
- registered_npz_complete_all_samples: FAIL
- no_samplewise_method_mixing: PASS
- provenance_hash_chain_closed: FAIL

Failed automatic checks: ['all_200_samples_have_registered_exports', 'l_reg_ge_260um_all_samples', 'final_leveling_all_pass', 'final_leveling_reference_sufficient', 'registered_npz_complete_all_samples', 'provenance_hash_chain_closed']

Blocking conditions (Phase A cannot be approved until these are resolved):
- [policy:zro2_20_supplement] zro2_20_supplement (20 samples, remeasurement_required): Paired acquisition layout provides only 198.38 um common centred FOV under manual_v1 centres (13 of 20 samples cannot support a full 200 um square without extrapolation), below the 260 um registered-grid minimum.
- [BLK-1] zro2_20_supplement (20 samples, blocker): zro2_20_supplement cannot be exported with the registered grid, so 20 of 200 samples fail the all-sample common-canvas and mask gates. Resolution requires remeasurement of the session or a formally amended acceptance criterion. Until then Phase A stays BLOCKED and must not be switched to PASS manually.

Required resolution: remeasure the affected session, or amend the acceptance protocol formally and re-run the full chain. Do not switch this file to PASS manually while a blocker stands.

Known conditions for the reviewer:
- gate_coverage: "all 200 manual_v1 samples are in scope for every gate; a sample without a registered export fails the gate, it is never excused by the exclusion policy"
- samples_without_registered_export: {"count": 20, "keys": ["zro2_20_supplement:1", "zro2_20_supplement:10", "zro2_20_supplement:11", "zro2_20_supplement:12", "zro2_20_supplement:13", "zro2_20_supplement:14", "zro2_20_supplement:15", "zro2_20_supplement:16", "zro2_20_supplement:17", "zro2_20_supplement:18", "zro2_20_supplement:19", "zro2_20_supplement:2", "zro2_20_supplement:20", "zro2_20_supplement:3", "zro2_20_supplement:4", "zro2_20_supplement:5", "zro2_20_supplement:6", "zro2_20_supplement:7", "zro2_20_supplement:8", "zro2_20_supplement:9"], "policy": "config\\phase_a_exclusions_manual_v1.yaml", "policy_sha256": "B9FBE46095AC4B9749B363E439C90B06C6D97A3190A2805ED6AB709125C75E44"}
- zro2_60_pass_samples_53_54: "re-included by config/phase_a_exclusions_manual_v1.yaml: the available centred square is 338.65 um and 276.35 um with manual centres, both at or above the 260 um minimum, so the legacy measurement-27 exclusion was not carried over"
- session_canvas_status: {"zro2_120_formal": "PASS", "zro2_20_supplement": "EXCLUDED", "zro2_60_pass": "PASS"}
- npz_provenance_detail: {"h_reg_path": true, "h_200_path": true, "mask_path": true}
- manual_box_corner_overflow_observations: 17

Review both montages and the individual QA images under qa/manual_v1/ before deciding. The v6 centres/statuses shown as overlays are QA comparators only; manual_v1 uses manual centres for every sample with no per-sample fallback.

This file must be reviewed and changed by a human; scripts are forbidden from marking it PASS.
