"""E06 verification gate: closure=0 hybrid rollout vs validated cell solver.

Scope: forward-model agreement (dense line-sweep approximation vs per-pulse
lattice) and per-line drift, on sampled MAIN180 recipes with a fixed mid-domain
parameter vector. This gate must pass before any hybrid training run. It is NOT
the E03 calibration and uses no measured labels.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import math
import numpy as np
import pandas as pd
import torch
import yaml
from src.task_state_learning.artifacts import ROOT, new_run, write_json, seal
from src.task_state_learning.contracts import load_contract
from src.task_state_learning.grouping import require
from src.task_state_learning.laser_rollout import (RolloutConfig, line_schedule, rollout,
                                                   upsample_depth, build_batch_plan)


def rollout_recipe(row, params_vec, cfg, waist_m=0.874e-6):
    log_F1, log_delta, log_kappa, logit_rho, gamma_F, gamma_delta = params_vec
    schedules = [line_schedule(row.pulse_duration_fs * 1e-15, row.frequency_kHz * 1e3,
                               row.velocity_mm_s * 1e-3, row.hatch_spacing_um * 1e-6,
                               int(row.pass_count), 5.3333, waist_m, cfg.tail_waists)]
    line_matrix, line_mask, counts = build_batch_plan(schedules, cfg)
    B = 1
    d0 = torch.zeros(B, cfg.grid_n, cfg.grid_n, dtype=torch.float64)
    q0 = torch.zeros_like(d0)
    P = counts.shape[1]
    commands = torch.log(torch.stack([
        torch.full((B, P), float(row.pulse_duration_fs) * 1e-15),
        torch.full((B, P), float(row.frequency_kHz) * 1e3),
        torch.full((B, P), float(row.velocity_mm_s) * 1e-3),
        torch.full((B, P), float(row.hatch_spacing_um) * 1e-6)])).permute(1, 2, 0)
    params = {'w0': torch.full((B,), waist_m), 'zr': torch.full((B,), 3.71e-6),
              'F1': torch.full((B,), math.exp(log_F1)),
              'delta': torch.full((B,), math.exp(log_delta)),
              'kappa': torch.full((B,), math.exp(log_kappa)),
              'rho': torch.tensor(1.0 / (1.0 + math.exp(-logit_rho))),
              'Ep': schedules[0]['Ep_J'], 'tau_ref': 1e-12,
              'gamma_F': gamma_F, 'gamma_delta': gamma_delta}
    d, q, drift = rollout(d0, q0, schedules, params, commands, None, cfg, dtype=torch.float64)
    field = upsample_depth(d)[0].detach().numpy()
    d_med = float(np.median(-field) * 1e6)
    a_rms = float(np.sqrt(np.mean(((-field * 1e6) - np.median(-field * 1e6)) ** 2)))
    return d_med, a_rms, drift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/task_state_v1/real_main.yaml')
    parser.add_argument('--calibration-run', default=None,
                        help='E03 run dir with calibrated P11 vectors; else mid-domain vector')
    args = parser.parse_args()
    path = ROOT / args.config
    c = yaml.safe_load(path.read_text(encoding='utf-8'))
    out = new_run('E06_HYBRID_VERIFY', path)
    gate = {'scope': 'E06_HYBRID_FORWARD_VERIFICATION', 'training_executed': False}
    try:
        cfg_run, frozen, df, _, _ = load_contract()
        main180 = df[df.session_role.isin(['formal', 'pass_main'])]
        rng = np.random.default_rng(20260907)
        sample_rows = main180.iloc[rng.choice(len(main180), 24, replace=False)]
        if args.calibration_run:
            summary = pd.read_csv(Path(args.calibration_run) / 'calibration_summary.csv')
            vec = np.array(summary[(summary.switch == 'P11') &
                                   (summary.waist_um == 0.874)].vector.iloc[0])
        else:
            vec = np.array([math.log(2e8), math.log(1e-8), math.log(0.02), 0.0, 0.0, 0.0])
        cfg = RolloutConfig(max_packets=c['rollout']['max_packets'],
                            tail_waists=c['rollout']['tail_waists'])
        rows = []
        for _, row in sample_rows.iterrows():
            from src.task_state_learning.scan_cell import CellConfig, simulate_sample
            from src.task_state_learning.physics import PhysicsParameters
            log_F1, log_delta, log_kappa, logit_rho, gamma_F, gamma_delta = vec
            p = PhysicsParameters(0.874e-6, 1.03e-6, 1.2, math.exp(log_F1), math.exp(log_delta),
                                  math.exp(log_kappa), 1.0 / (1.0 + math.exp(-logit_rho)),
                                  gamma_F=gamma_F, gamma_delta=gamma_delta)
            d_cell, a_cell, info = simulate_sample(
                p, cfg=CellConfig(), tau_s=row.pulse_duration_fs * 1e-15,
                f_Hz=row.frequency_kHz * 1e3, v_m_s=row.velocity_mm_s * 1e-3,
                h_m=row.hatch_spacing_um * 1e-6, passes=int(row.pass_count),
                power_W=5.3333, switch='P11')
            d_roll, a_roll, drift = rollout_recipe(row, vec, cfg)
            rel_d = abs(d_roll - d_cell * 1e6) / max(abs(d_cell * 1e6), 1e-3)
            rel_a = abs(a_roll - a_cell * 1e6) / max(abs(a_cell * 1e6), 1e-3)
            rows.append({'dataset_index': int(row.dataset_index),
                         'D_cell_um': d_cell * 1e6, 'D_rollout_um': d_roll,
                         'A_cell_um': a_cell * 1e6, 'A_rollout_um': a_roll,
                         'rel_error_D': rel_d, 'rel_error_A': rel_a,
                         'max_line_dd_over_zr': drift['max_line_dd_over_zr'],
                         'max_line_log_threshold_drift': drift['max_line_log_threshold_drift']})
        frame = pd.DataFrame(rows)
        frame.to_csv(out / 'forward_agreement.csv', index=False)
        gate.update(max_rel_error_D=float(frame.rel_error_D.max()),
                    max_rel_error_A=float(frame.rel_error_A.max()),
                    max_drift_log_threshold=float(frame.max_line_log_threshold_drift.max()),
                    max_drift_dd_over_zr=float(frame.max_line_dd_over_zr.max()),
                    tolerance=0.02, drift_bound=0.05)
        ok = frame.rel_error_D.max() <= 0.02 and frame.rel_error_A.max() <= 0.02
        gate['execution_status'] = 'PASS' if ok else 'FAIL'
        gate['metric_status'] = 'VALID'
        gate['evidence_boundary'] = ('Forward-model agreement on sampled recipes; not a '
                                     'calibration or performance claim.')
    except Exception as exc:
        import traceback
        gate['error'] = f'{type(exc).__name__}: {exc}'
        gate['traceback'] = traceback.format_exc()
        gate['execution_status'] = 'FAIL'
    gate['status'] = gate['execution_status']
    write_json(out / 'gate_E06_VERIFY.json', gate)
    seal(out)
    print(json.dumps({k: v for k, v in gate.items() if k != 'traceback'},
                     ensure_ascii=False, indent=2))
    return 0 if gate['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
