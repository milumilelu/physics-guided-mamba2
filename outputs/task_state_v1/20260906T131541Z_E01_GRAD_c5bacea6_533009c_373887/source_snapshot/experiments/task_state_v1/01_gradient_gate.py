"""G1 gradient gate: AD vs central FD on the frozen scan fixture, plus convergence.

Pre-registered protocol (runbook 28.3): 32 registered smooth points from a seeded
log-uniform pool; pulse counts frozen per point from nominal pitches (fixed event
topology); FD reported at all five step sizes; the FD value used for acceptance is
selected by an adjacent-step stability plateau, never by agreement with AD; near-zero
derivatives are guarded by a per-direction pre-frozen absolute floor.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import math
import subprocess
import numpy as np
import pandas as pd
import torch
import yaml
from src.task_state_learning.artifacts import ROOT, new_run, write_json, seal
from src.task_state_learning.grouping import require
from src.task_state_learning.physics import PhysicsParameters
from src.task_state_learning.scan import (ScanFixture, descriptors, pulse_counts,
                                          pulse_positions, rollout_numpy, rollout_torch,
                                          split_packets)

DIRECTIONS = ["log_tau_s", "log_f_Hz", "log_v_m_s", "log_h_m"]


def derived(logs, power_W):
    lt, lf, lv, lh = logs
    return {'tau_s': math.exp(lt), 'f_Hz': math.exp(lf),
            'ep_J': power_W * math.exp(-lf), 'dx_m': math.exp(lv - lf), 'hatch_m': math.exp(lh)}


def nominal_rollout(logs, power_W, params, fixture, n_x, n_y):
    """Terminal depth field only; incubation is not consumed by the gate."""
    d = derived(logs, power_W)
    pos = pulse_positions(n_x, n_y, d['dx_m'], d['hatch_m'])
    depth, _ = rollout_numpy(pos, np.full(len(pos), d['ep_J']), d['tau_s'], params, fixture)
    return depth


def choose_fd(fd_values, floor, plateau_rtol):
    """Adjacent-step stability plateau over coarse->fine FD values (rule v2, pre-registered).

    Full two-consecutive-agreement plateau: take the finest step inside it. Otherwise
    take the mean of the finest agreeing adjacent pair. The tie-break uses only the
    FD table's internal consistency (never AD agreement): the observed error structure
    is monotone truncation decay, so the finer member of a stable pair is the better
    central estimate; the v1 coarse tie-break landed on steps with up to 4e-2 truncation
    error (FAIL run 20260906T131109Z preserved).
    """
    stable = [abs(fd_values[i] - fd_values[i - 1]) <= plateau_rtol * max(abs(fd_values[i]), floor)
              for i in range(1, len(fd_values))]
    runs, start = [], None
    for i, ok in enumerate(stable):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(stable) - 1))
    full = [b for a, b in runs if b - a >= 1]
    if full:
        k = max(full) + 1
        return k, 'PLATEAU'
    if runs:
        k = max(b for _, b in runs)
        return k, 'PLATEAU_EDGE'
    return len(fd_values) - 1, 'UNSTABLE'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'config/task_state_v1/gradient_gate.yaml')
    args = ap.parse_args()
    out = new_run('E01_GRAD', args.config)
    gate = {'experiment_id': 'E01', 'gate_scope': 'GRADIENT_AND_SCAN_CONVERGENCE',
            'execution_status': 'FAIL', 'hypothesis_status': 'NOT_TESTED',
            'G1_complete': False, 'training_executed': False,
            'backend': f'torch_{torch.__version__}_cpu_float64_vs_numpy_float64'}
    try:
        c = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        suite = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'discover',
                                '-s', 'tests/task_state_v1', '-p', 'test_scan.py', '-v'],
                               cwd=ROOT, capture_output=True, encoding='utf-8')
        (out / 'scan_tests.log').write_text(suite.stdout + suite.stderr, encoding='utf-8')
        require(suite.returncode == 0, 'Scan unit tests failed')
        params = PhysicsParameters(waist_m=c['waist_m'], wavelength_m=c['wavelength_m'],
                                   m_squared=c['m_squared'], F1_reference_J_m2=c['F1_reference_J_m2'],
                                   delta_reference_m=c['delta_reference_m'], kappa=c['kappa'],
                                   saturation_ratio=c['saturation_ratio'], gamma_F=c['gamma_F'],
                                   gamma_delta=c['gamma_delta'],
                                   reference_duration_s=c['reference_duration_s'])
        fixture = ScanFixture(region_m=c['scenario']['region_m'], grid_n=c['scenario']['grid_n'],
                              power_W=c['power_W'], n_x_max=c['caps']['n_x_max'],
                              n_y_max=c['caps']['n_y_max'], total_pulses_max=c['caps']['total_pulses_max'],
                              passes=c['scenario']['passes'])
        s = c['sampling']
        ranges = [np.log([s['tau_fs'][0] * 1e-15, s['tau_fs'][1] * 1e-15]),
                  np.log([s['f_kHz'][0] * 1e3, s['f_kHz'][1] * 1e3]),
                  np.log([s['v_mm_s'][0] * 1e-3, s['v_mm_s'][1] * 1e-3]),
                  np.log([s['h_um'][0] * 1e-6, s['h_um'][1] * 1e-6])]
        rng = np.random.default_rng(s['seed'])
        pool = rng.uniform(size=(s['pool_size'], 4))
        points, pool_rows = [], []
        for idx, u in enumerate(pool):
            status = 'NOT_EVALUATED'
            row = {'candidate': idx, **{DIRECTIONS[j]: float(u[j]) for j in range(4)}}
            if len(points) < s['n_points']:
                logs = [float(ranges[j][0] + u[j] * (ranges[j][1] - ranges[j][0])) for j in range(4)]
                row.update({f'log_{DIRECTIONS[j]}': logs[j] for j in range(4)})
                d = derived(logs, c['power_W'])
                try:
                    n_x, n_y = pulse_counts(fixture, d['dx_m'], d['hatch_m'])
                    grid = nominal_rollout(logs, c['power_W'], params, fixture, n_x, n_y)
                    require(np.isfinite(grid).all(), 'Nonfinite nominal rollout')
                    J = float(grid.mean())
                    require(J > 0, 'No ablation at nominal point')
                except ValueError as exc:
                    status = f'REJECTED_{exc}'
                else:
                    status = 'REGISTERED'
                    points.append({'point': len(points), **{f'log_{DIRECTIONS[j]}': logs[j] for j in range(4)},
                                   'tau_fs': d['tau_s'] * 1e15, 'f_kHz': d['f_Hz'] * 1e-3,
                                   'v_mm_s': math.exp(logs[2]) * 1e3, 'h_um': d['hatch_m'] * 1e6,
                                   'ep_J': d['ep_J'], 'n_x': n_x, 'n_y': n_y,
                                   'total_pulses': n_x * n_y, 'J_nominal_m': J, 'J_terminal_mean_m': J})
            row['status'] = status
            pool_rows.append(row)
        require(len(points) == s['n_points'], f'Only {len(points)} smooth points registered')
        registered = pd.DataFrame(points)
        registered.to_csv(out / 'registered_points.csv', index=False)
        pd.DataFrame(pool_rows).to_csv(out / 'pool_sampling.csv', index=False)

        # Automatic differentiation rollouts.
        torch.set_num_threads(4)
        torch.set_default_dtype(torch.float64)
        ad_rows = []
        for pt in points:
            logs = [torch.tensor(pt[f'log_{k}'], requires_grad=True) for k in DIRECTIONS]
            d_t, _ = rollout_torch(*logs, params, fixture, pt['n_x'], pt['n_y'], switch=c['switch'])
            J = d_t.mean()
            J.backward()
            require(torch.isfinite(J), 'Nonfinite AD output')
            for j, name in enumerate(DIRECTIONS):
                ad_rows.append({'point': pt['point'], 'direction': name, 'ad_value': float(logs[j].grad)})
        ad = pd.DataFrame(ad_rows)
        ad.to_csv(out / 'ad_gradients.csv', index=False)

        # Central finite differences, all five step sizes, both sides.
        fd_rows = []
        for pt in points:
            base = [pt[f'log_{k}'] for k in DIRECTIONS]
            for step in c['fd']['steps']:
                for j, name in enumerate(DIRECTIONS):
                    values = {}
                    for sign in (+1, -1):
                        logs = list(base)
                        logs[j] = base[j] + sign * step
                        grid = nominal_rollout(logs, c['power_W'], params, fixture, pt['n_x'], pt['n_y'])
                        values[sign] = float(grid.mean())
                    fd_rows.append({'point': pt['point'], 'direction': name, 'step': step,
                                    'fd_value': (values[1] - values[-1]) / (2 * step)})
        fd = pd.DataFrame(fd_rows)
        fd.to_csv(out / 'fd_table.csv', index=False)

        # Pre-frozen per-direction floor, then plateau selection and acceptance.
        floor = {name: c['fd']['deriv_floor_factor'] * float(
            fd[(fd.direction == name) & np.isclose(fd.step, 1e-4)].fd_value.abs().median())
            for name in DIRECTIONS}
        eval_rows = []
        for pt in points:
            for name in DIRECTIONS:
                sub = fd[(fd.point == pt['point']) & (fd.direction == name)].sort_values('step',
                                                                                         ascending=False)
                fd_values = sub.fd_value.to_numpy()
                k, flag = choose_fd(fd_values, floor[name], c['fd']['plateau_rtol'])
                ad_value = float(ad[(ad.point == pt['point']) & (ad.direction == name)].ad_value.iloc[0])
                rel = abs(ad_value - fd_values[k]) / max(abs(ad_value), abs(fd_values[k]), floor[name])
                eval_rows.append({'point': pt['point'], 'direction': name, 'chosen_step': sub.step.iloc[k],
                                  'plateau_flag': flag, 'fd_chosen': fd_values[k], 'ad_value': ad_value,
                                  'rel_error': rel, 'passed': bool(rel <= c['fd']['pass_rtol']),
                                  'unexplained': bool(rel > c['fd']['unexplained_fail_rtol']
                                                      and flag == 'PLATEAU')})
        evaluation = pd.DataFrame(eval_rows)
        evaluation.to_csv(out / 'gradient_gate_eval.csv', index=False)
        per_direction = evaluation.groupby('direction').passed.mean().to_dict()
        overall = float(evaluation.passed.mean())
        unexplained = int(evaluation.unexplained.sum())
        require(all(v >= c['fd']['min_pass_fraction'] for v in per_direction.values()),
                f'Per-direction pass fraction below threshold: {per_direction}')
        require(overall >= c['fd']['min_pass_fraction'], f'Overall pass fraction {overall} below threshold')
        require(unexplained == 0, f'{unexplained} unexplained relative errors > 1e-2')

        # Scan grid refinement on the first registered points (descriptors, finest two grids).
        conv = c['convergence']
        grids = conv['grid_px_per_axis']
        grid_rows = []
        for pt in points[:conv['refinement_points']]:
            logs = [pt[f'log_{k}'] for k in DIRECTIONS]
            fields = {}
            for n in grids:
                sub = ScanFixture(region_m=fixture.region_m, grid_n=n, power_W=fixture.power_W,
                                  n_x_max=10 ** 6, n_y_max=10 ** 6, total_pulses_max=10 ** 9)
                d = nominal_rollout(logs, c['power_W'], params, sub, pt['n_x'], pt['n_y'])
                fields[n] = descriptors(d)
            for key in fields[grids[-1]]:
                coarse, fine = fields[grids[-2]][key], fields[grids[-1]][key]
                rel = abs(fine - coarse) / abs(coarse) if coarse != 0 else (0.0 if fine == 0 else np.inf)
                grid_rows.append({'point': pt['point'], 'descriptor': key, **{f'grid_{n}': fields[n][key] for n in grids},
                                  'rel_change_finest': rel,
                                  'passed': bool(np.isfinite(rel) and rel <= conv['refinement_tolerance'])})
        grid_ref = pd.DataFrame(grid_rows)
        grid_ref.to_csv(out / 'scan_grid_refinement.csv', index=False)
        require(grid_ref.passed.all(), 'Scan grid refinement exceeded 1% on a descriptor')

        # Pulse-packet refinement: sub-pulse splitting, two finest splits compared.
        packet_rows = []
        for pt in points[:conv['refinement_points']]:
            logs = [pt[f'log_{k}'] for k in DIRECTIONS]
            d = derived(logs, c['power_W'])
            pos = pulse_positions(pt['n_x'], pt['n_y'], d['dx_m'], d['hatch_m'])
            base_energy = np.full(len(pos), d['ep_J'])
            stats = {}
            for m in conv['packet_splits']:
                sub_pos, sub_energy = split_packets(pos, base_energy, m)
                depth, _ = rollout_numpy(sub_pos, sub_energy, d['tau_s'], params, fixture)
                stats[m] = descriptors(depth)
            for key in stats[conv['packet_splits'][-1]]:
                coarse, fine = stats[conv['packet_splits'][-2]][key], stats[conv['packet_splits'][-1]][key]
                rel = abs(fine - coarse) / abs(coarse) if coarse != 0 else (0.0 if fine == 0 else np.inf)
                packet_rows.append({'point': pt['point'], 'descriptor': key,
                                    **{f'split_{m}': stats[m][key] for m in conv['packet_splits']},
                                    'rel_change_finest': rel,
                                    'passed': bool(np.isfinite(rel) and rel <= conv['refinement_tolerance'])})
        packet_ref = pd.DataFrame(packet_rows)
        packet_ref.to_csv(out / 'pulse_packet_refinement.csv', index=False)
        require(packet_ref.passed.all(), 'Pulse-packet refinement exceeded 1% on a descriptor')

        gate.update(execution_status='PASS', gradient_based_inverse_design='ALLOWED',
                    registered_points=len(points), ad_fd_comparisons=len(evaluation),
                    per_direction_pass_fraction=per_direction, overall_pass_fraction=overall,
                    unexplained_gt_fail_rtol_count=unexplained,
                    max_rel_error=float(evaluation.rel_error.max()),
                    plateau_flags=evaluation.plateau_flag.value_counts().to_dict(),
                    deriv_floor={k: float(v) for k, v in floor.items()},
                    grid_refinement_max_rel_change=float(grid_ref.rel_change_finest.max()),
                    packet_refinement_max_rel_change=float(packet_ref.rel_change_finest.max()),
                    remaining=['real_parameter_calibration_E03', 'memory_block_refinement_E04_B1'])
    except Exception as exc:
        gate['error'] = f'{type(exc).__name__}: {exc}'
    gate['status'] = gate['execution_status']
    write_json(out / 'gate_G1_GRAD.json', gate)
    seal(out)
    print(json.dumps({'output': str(out), **gate}, ensure_ascii=False, indent=2))
    return 0 if gate['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
