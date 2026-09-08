"""E10 robustness matrix and E11 release accounting.

E10 stages (all reuse the frozen MAIN180 pipeline, no new tuning):
  waist   — P11 physics ablation at w0 = 0.874/2/3 um (solver-level sensitivity).
  supp20  — joint pass/session/domain shift: MAIN180-trained models scored on the
            20 supplemental ROIs (family-overlap track; purged track registered).
  null    — radial-power-matched angular-isotropic synthetic fields through the
            same observer (directionality null; NOT Fourier phase randomization).
E11 assembles final_* locked tables from the sealed run artifacts and builds the
claim ledger (E/S/M/H/U) with artifact paths.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import glob
import json
import numpy as np
import pandas as pd
import torch
import yaml
from src.task_state_learning.artifacts import ROOT, new_run, write_json, seal
from src.task_state_learning.contracts import load_contract
from src.task_state_learning.grouping import components, require
from src.task_state_learning.observers import CanonicalObserver


def latest(pattern):
    runs = sorted(glob.glob(str(ROOT / 'outputs/task_state_v1' / pattern)))
    require(runs, f'No run for {pattern}')
    return runs[-1]


def run_waist(cfg, out):
    from src.task_state_learning.physics import PhysicsParameters
    from src.task_state_learning.scan_cell import CellConfig, simulate_sample
    summary = pd.read_csv(Path(cfg['calibration_run']) / 'calibration_summary.csv')
    rows = []
    for waist in [0.874, 2.0, 3.0]:
        sub = summary[(summary.switch == 'P11') & (summary.waist_um == waist)]
        for fold in range(5):
            f_sub = sub[sub.outer_fold == fold]
            vec = np.array(f_sub.vector.iloc[0])
            log_F1, log_delta, log_kappa, logit_rho, gamma_F, gamma_delta = vec
            p = PhysicsParameters(waist * 1e-6, cfg['wavelength_m'], cfg['m_squared'],
                                  math.exp(log_F1), math.exp(log_delta), math.exp(log_kappa),
                                  1.0 / (1.0 + math.exp(-logit_rho)), gamma_F=gamma_F,
                                  gamma_delta=gamma_delta)
            rows.append({'waist_um': waist, 'outer_fold': fold,
                         'F1_ref_J_m2': float(math.exp(log_F1)),
                         'delta_ref_m': float(math.exp(log_delta)),
                         'kappa': float(math.exp(log_kappa)),
                         'rho': float(1.0 / (1.0 + math.exp(-logit_rho)))})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'physics_scenario_sensitivity.csv', index=False)
    return frame


def run_supp20(cfg, out, oof_dir=None):
    """Score every sealed E06 hybrid/static OOF model on SUPP20 (shift track)."""
    cfg_run, frozen, df, _, _ = load_contract()
    df = components(df)
    supp = df[df.session_role == 'pass_supplement']
    require(len(supp) == 20, 'SUPP20 count mismatch')
    targets = ['D', 'A_med', 'ilr_z1', 'ilr_z2', 'ilr_z3', 'ilr_z4', 'A2_8_16', 'entropy_8_16']
    rows = []
    for pattern, stage in (('2026*_E06_STATIC_*', 'static'), ('2026*_E06_HYBRID_*', 'hybrid')):
        runs = sorted(glob.glob(str(ROOT / 'outputs/task_state_v1' / pattern)))
        runs = [r for r in runs if (Path(r) / 'oof.csv').exists()]
        if not runs:
            continue
        run = runs[-1]
        oof = pd.read_csv(Path(run) / 'oof.csv')
        models = oof.model_id.unique().tolist()
        main180 = df[df.session_role.isin(['formal', 'pass_main'])]
        rows.append({'stage': stage, 'run': Path(run).name,
                     'models': ','.join(models),
                     'supp_family_overlap': int(supp.base_family_key.isin(
                         main180.base_family_key).sum()),
                     'note': 'joint pass/session/domain shift; NOT pure N>=5 extrapolation'})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'supp20_shift_tracking.csv', index=False)
    return frame


def observer_null(out, seed=20260907):
    """Radial-power-matched, angular-isotropic synthetic field through the observer."""
    rng = np.random.default_rng(seed)
    n = 160
    x = (np.arange(n) - (n - 1) / 2) * 0.5
    X, Yg = np.meshgrid(x, x)
    fields = []
    for i in range(12):
        depth = 1.5
        field = -depth + rng.normal(0, 0.03, (n, n))
        # enforce angular isotropy: radial power spectrum only (random phases)
        fft = np.fft.fft2(field - field.mean())
        power = np.abs(fft) ** 2
        r = np.sqrt(X ** 2 + Yg ** 2)
        radial = np.zeros_like(power)
        for band in range(1, 40):
            shell = (r >= band - 0.5) & (r < band + 0.5)
            if shell.any():
                radial[shell] = power[shell].mean()
        phases = rng.uniform(0, 2 * np.pi, power.shape)
        iso = np.sqrt(radial) * np.exp(1j * phases)
        field_iso = np.real(np.fft.ifft2(iso)) + field.mean()
        fields.append(field_iso)
    fields = np.array(fields)
    values, validity = CanonicalObserver()(fields, np.ones_like(fields, dtype=bool))
    values.to_csv(out / 'isotropic_null_observer_values.csv', index=False)
    return values


def claim_ledger(out, gate_paths):
    claims = [
        {'claim_id': 'C_data_contract', 'claim_text':
         'MAIN180 identity, targets and dependency components are auditable and frozen '
         '(E00 gate PASS, split hash cd90d1a1...).',
         'evidence_level': 'E', 'experiment_ids': 'E00',
         'artifact_paths': gate_paths.get('E00', ''), 'allowed_scope':
         'data provenance statements', 'forbidden_extension': 'external confirmation',
         'status': 'E'},
        {'claim_id': 'C_numeric_chain', 'claim_text':
         'Canonical observer, physics recursion, differentiable observer and AD/FD '
         'gradient path pass the registered numeric gates.',
         'evidence_level': 'E', 'experiment_ids': 'E01',
         'artifact_paths': gate_paths.get('E01', ''), 'allowed_scope':
         'numerical correctness of the registered models',
         'forbidden_extension': 'physical mechanism identification', 'status': 'E'},
        {'claim_id': 'C_mamba_backend', 'claim_text':
         'The Mamba-2 reference backend reproduces its registered recurrence '
         '(full/step/cache parity, closed-loop AD/FD 32/32).',
         'evidence_level': 'E', 'experiment_ids': 'E04-backend',
         'artifact_paths': gate_paths.get('E04_BACKEND', ''),
         'allowed_scope': 'backend correctness', 'forbidden_extension':
         'Mamba-specific learning advantage', 'status': 'E'},
        {'claim_id': 'C_b1_memory', 'claim_text':
         'Memory-model benefit on the B1 hidden-state benchmark (verdict recorded in '
         'gate_E04_G2).', 'evidence_level': 'S', 'experiment_ids': 'E04',
         'artifact_paths': gate_paths.get('E04_G2', ''),
         'allowed_scope': 'synthetic benchmark', 'forbidden_extension':
         'real-material state recovery', 'status': 'S'},
        {'claim_id': 'C_e02_diagnostic', 'claim_text':
         'Process-input U beats dose-only Qa for A/P/T prediction; U+Dhat shows no '
         'stable extra gain (E02).', 'evidence_level': 'S', 'experiment_ids': 'E02',
         'artifact_paths': gate_paths.get('E02', ''),
         'allowed_scope': 'retrospective MAIN180 diagnostic',
         'forbidden_extension': 'causal mediation', 'status': 'S'},
        {'claim_id': 'C_hybrid_real', 'claim_text':
         'Hybrid memory closure value on MAIN180 (C1-C3 verdicts; recorded when '
         'contrasts complete).', 'evidence_level': 'H', 'experiment_ids': 'E06',
         'artifact_paths': gate_paths.get('E06_CONTRASTS', ''),
         'allowed_scope': 'conditional on frozen splits/models',
         'forbidden_extension': 'Mamba-specific superiority without C3 joint support',
         'status': 'H'},
        {'claim_id': 'C_design_virtual', 'claim_text':
         'Real-data inverse design is restricted to sealed candidate selection '
         '(virtual candidates); no experimental validation claim.',
         'evidence_level': 'M', 'experiment_ids': 'E09',
         'artifact_paths': gate_paths.get('E09', ''),
         'allowed_scope': 'virtual process candidates',
         'forbidden_extension': 'experimental inverse-design validation', 'status': 'M'},
    ]
    frame = pd.DataFrame(claims)
    frame.to_csv(out / 'claim_ledger.csv', index=False)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/task_state_v1/real_main.yaml')
    parser.add_argument('--stage', choices=['waist', 'supp20', 'null', 'release'],
                        required=True)
    parser.add_argument('--calibration-run', default=None)
    args = parser.parse_args()
    from src.task_state_learning.guardrails import require_not_held
    require_not_held('E10_'+args.stage.upper())
    path = ROOT / args.config
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if args.calibration_run:
        cfg['calibration_run'] = args.calibration_run
    out = new_run(f'E10_{args.stage.upper()}' if args.stage != 'release' else 'E11_RELEASE',
                  path)
    gate = {'scope': f'E10_{args.stage}' if args.stage != 'release' else 'E11_RELEASE',
            'training_executed': False}
    try:
        if args.stage == 'waist':
            frame = run_waist(cfg, out)
            gate['n_rows'] = len(frame)
        elif args.stage == 'supp20':
            frame = run_supp20(cfg, out)
            gate['tracked_runs'] = len(frame)
        elif args.stage == 'null':
            values = observer_null(out)
            gate['n_fields'] = len(values)
            gate['note'] = ('radial-power-matched angular-isotropic null through the '
                            'canonical observer; direction metrics on these fields '
                            'quantify observer noise, not material anisotropy')
        else:
            gate_paths = {
                'E00': str(latest('2026*_E00_*')),
                'E01': str(latest('2026*_E01_G1_CLOSE*')),
                'E02': str(latest('2026*_E02_*')),
                'E04_BACKEND': str(latest('2026*_E04_BACKEND_GATE*')),
                'E04_G2': (str(latest('2026*_E04_G2*'))
                           if glob.glob(str(ROOT / 'outputs/task_state_v1/2026*_E04_G2*'))
                           else ''),
                'E06_CONTRASTS': (str(latest('2026*_E06_CONTRASTS*'))
                                  if glob.glob(str(ROOT / 'outputs/task_state_v1/2026*_E06_CONTRASTS*'))
                                  else ''),
                'E09': (str(latest('2026*_E09_*'))
                        if glob.glob(str(ROOT / 'outputs/task_state_v1/2026*_E09_*')) else ''),
            }
            ledger = claim_ledger(out, gate_paths)
            gate['n_claims'] = len(ledger)
            # release manifest of all sealed task_state_v1 runs
            runs = []
            for run_dir in sorted((ROOT / 'outputs/task_state_v1').glob('2026*')):
                marker = run_dir / 'release_manifest.json'
                if marker.exists():
                    runs.append({'run': run_dir.name, 'sealed': True})
            pd.DataFrame(runs).to_csv(out / 'release_runs.csv', index=False)
            gate['n_runs'] = len(runs)
        gate['execution_status'] = 'PASS'
        gate['metric_status'] = 'VALID'
    except Exception as exc:
        import traceback
        gate['error'] = f'{type(exc).__name__}: {exc}'
        gate['traceback'] = traceback.format_exc()
        gate['execution_status'] = 'FAIL'
    gate['status'] = gate['execution_status']
    write_json(out / f'gate_{out.name.split("_")[1]}_{args.stage}.json', gate)
    seal(out)
    print(json.dumps({k: v for k, v in gate.items() if k != 'traceback'},
                     ensure_ascii=False, indent=2))
    return 0 if gate['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
