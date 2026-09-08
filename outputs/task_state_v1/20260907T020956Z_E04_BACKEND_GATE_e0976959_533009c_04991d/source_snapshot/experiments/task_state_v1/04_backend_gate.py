"""E04 backend acceptance: Mamba-2 reference behind the common closed-loop API.

Scope (runbook 28.3 acceptance of the actual training path):
  1. closed-loop AD vs central FD on x (control influence through the full
     physics -> token -> memory -> closure -> physics loop) and on a readout
     parameter, with pre-registered points, five FD steps, plateau rule v2;
  2. cache continuation: naive_forward on split segments equals one full pass;
  3. float32 training path finiteness and float64->float32 consistency;
  4. state-budget registration (cache scalars/bytes, parameters);
  5. memory-block/token resampling sensitivity is DESCRIPTIVE (architecture
     sensitivity, not solver convergence).
No B1 training and no real-data model run in this script.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import subprocess
import numpy as np
import pandas as pd
import torch
import yaml
from src.task_state_learning.artifacts import ROOT, new_run, write_json, seal
from src.task_state_learning.grouping import require
from src.task_state_learning.models import B1ClosureModel, b1_resolved_step


def closed_loop_loss(model, control, dt, mask, return_state=False):
    """End-to-end loss on terminal x; gradient must reach tokens and parameters."""
    x = model(control, dt, mask, return_state=return_state)
    if return_state:
        x, state = x
        return x.pow(2).sum(), state
    return x.pow(2).sum()


def plateau_fd(values, floor, rtol):
    stable = [abs(values[i] - values[i - 1]) <= rtol * max(abs(values[i]), floor)
              for i in range(1, len(values))]
    fine = None
    for i in range(len(stable) - 2, -1, -1):
        if stable[i] and stable[i + 1]:
            fine = i + 2
            break
    if fine is None:
        for i in range(len(stable) - 1, -1, -1):
            if stable[i]:
                fine = i + 1
                break
    return values[fine if fine is not None else 0], (fine is not None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/task_state_v1/b1_backend_gate.yaml')
    args = parser.parse_args()
    path = ROOT / args.config
    c = yaml.safe_load(path.read_text(encoding='utf-8'))
    out = new_run('E04_BACKEND_GATE', path)
    gate = {'scope': c['scope'], 'mamba_backend': c['mamba_backend'],
            'architecture': c['architecture'], 'training_executed': False}
    try:
        suite = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest'] +
                               [f'tests.task_state_v1.{m}' for m in c['regression_test_modules']],
                               cwd=ROOT, capture_output=True, encoding='utf-8')
        (out / 'regression_tests.log').write_text(suite.stdout + suite.stderr, encoding='utf-8')
        require(suite.returncode == 0, 'Mamba-2 backend regression tests failed')
        torch.manual_seed(c['seed'])
        arch = c['architecture']
        rows = []

        # ---- 1. closed-loop AD vs FD ----
        base = B1ClosureModel('MAMBA2', d_state=arch['d_state']).double()
        gen = torch.Generator().manual_seed(c['seed'] + 1)
        n_fam, n_tok = c['closed_loop']['families'], c['closed_loop']['token_count']
        control = (torch.rand(n_fam, n_tok, 2, generator=gen) * 2 - 1)
        dt = torch.rand(n_fam, n_tok, generator=gen) * 0.07 + 0.01
        mask = torch.ones(n_fam, n_tok, dtype=torch.bool)

        point_id = 0
        for family in range(c['closed_loop']['points'] % n_fam + 2):
            for direction in c['closed_loop']['directions']:
                if point_id >= c['closed_loop']['points'] * 2:
                    break
                if direction == 'control_early_u1_token5':
                    token = 5
                    def loss_at(delta, params=None):
                        cc = control.clone()
                        cc[family, token, 0] += delta
                        return float(closed_loop_loss(base, cc, dt, mask))
                    unit = 1.0
                else:
                    param = dict(base.readout.named_parameters())
                    weight = param['2.weight']

                    def loss_at(delta, params=None):
                        cc = control.clone()
                        with torch.no_grad():
                            saved = weight[0, 0].item()
                            weight[0, 0] = saved + delta
                            value = float(closed_loop_loss(base, cc, dt, mask))
                            weight[0, 0] = saved
                        return value
                    unit = float(weight[0, 0].detach().abs()) + 1e-6
                ad = torch.autograd.grad if False else None
                # AD value
                cc = control.clone().requires_grad_(direction == 'control_early_u1_token5')
                if direction == 'control_early_u1_token5':
                    loss = closed_loop_loss(base, cc, dt, mask)
                    (grad,) = torch.autograd.grad(loss, cc)
                    ad_value = float(grad[family, token, 0])
                else:
                    weight.requires_grad_(True)
                    base.zero_grad()
                    loss = closed_loop_loss(base, control, dt, mask)
                    (grad,) = torch.autograd.grad(loss, weight)
                    ad_value = float(grad[0, 0])
                    weight.requires_grad_(False)
                fd_values = []
                for step in c['closed_loop']['fd_relative_steps']:
                    h = step * unit
                    fd_values.append((loss_at(h) - loss_at(-h)) / (2 * h))
                chosen, stable = plateau_fd(fd_values, c['closed_loop']['deriv_floor'][direction],
                                            c['closed_loop']['plateau_rtol'])
                denom = max(abs(ad_value), c['closed_loop']['deriv_floor'][direction])
                rel = abs(chosen - ad_value) / denom
                rows.append({'point': point_id, 'direction': direction, 'family': family,
                             'ad': ad_value, 'fd_chosen': chosen, 'stable_plateau': stable,
                             'fd_table': fd_values, 'rel_error': rel,
                             'passed': bool(rel <= c['closed_loop']['fail_rtol']),
                             'unexplained': bool(rel > c['closed_loop']['unexplained_fail_rtol'])})
                point_id += 1
        grad_df = pd.DataFrame(rows)
        grad_df.to_csv(out / 'closed_loop_ad_fd.csv', index=False)
        pass_fraction = float(grad_df.passed.mean())
        unexplained = int(grad_df.unexplained.sum())

        # ---- 2. continuation ----
        cont_rows = []
        model = B1ClosureModel('MAMBA2').double()
        for frac in c['continuation']['split_fractions']:
            k = int(n_tok * frac)
            with torch.no_grad():
                full, state_full = closed_loop_loss(model, control, dt, mask, return_state=True)
                first = model(control[:, :k], dt[:, :k], mask[:, :k], return_state=True)
                second = model(control[:, k:], dt[:, k:], mask[:, k:],
                               initial_state={'x': first[0], 'memory': first[1]['memory']})
            err = float((second - full).abs().max() / full.abs().max())
            cont_rows.append({'split_fraction': frac, 'relative_error': err,
                              'passed': bool(err <= c['continuation']['relative_tolerance'])})
        cont_df = pd.DataFrame(cont_rows)
        cont_df.to_csv(out / 'continuation_checks.csv', index=False)

        # ---- 3. float32 path ----
        model32 = B1ClosureModel('MAMBA2').float()
        with torch.no_grad():
            out32 = model32(control.float(), dt.float(), mask)
        finite32 = bool(torch.isfinite(out32).all())
        model64 = B1ClosureModel('MAMBA2').double()
        model32.load_state_dict(model64.state_dict())
        with torch.no_grad():
            o64 = model64(control, dt, mask)
            o32 = model32(control.float(), dt.float(), mask).double()
        rel32 = float((o32 - o64).abs().max() / o64.abs().max())

        # ---- 4. state budget ----
        budgets = {}
        for name in ['H0', 'GRU', 'CTSSM', 'MAMBA2']:
            budgets[name] = B1ClosureModel(name).budget()

        # ---- 5. token resampling sensitivity (descriptive) ----
        token_rows = []
        base_desc = B1ClosureModel('MAMBA2').double()
        for count in c['memory_block_refinement']['token_counts']:
            reps = count // n_tok
            ctrl_res = control.repeat_interleave(reps, dim=1)[:, :count]
            dt_res = dt.repeat_interleave(reps, dim=1)[:, :count] / reps
            mask_res = torch.ones(n_fam, count, dtype=torch.bool)
            with torch.no_grad():
                value = float(closed_loop_loss(base_desc, ctrl_res, dt_res, mask_res))
            token_rows.append({'token_count': count, 'same_T_terminal_sq_sum': value})
        token_df = pd.DataFrame(token_rows)
        token_df.to_csv(out / 'memory_block_resampling.csv', index=False)

        gate.update(
            closed_loop_points=len(grad_df),
            closed_loop_pass_fraction=pass_fraction,
            closed_loop_unexplained=unexplained,
            continuation_max_rel_error=float(cont_df.relative_error.max()),
            float32_finite=finite32, float32_vs_float64_rel_error=rel32,
            state_budget=budgets,
            token_resampling_descriptive=token_df.to_dict('records'))
        ok = (pass_fraction >= c['closed_loop']['pass_fraction_required'] and unexplained == 0
              and cont_df.passed.all() and finite32 and rel32 <= c['float32']['gate_tolerance'])
        gate['execution_status'] = 'PASS' if ok else 'FAIL'
        gate['metric_status'] = 'VALID'
        gate['hypothesis_status'] = 'NOT_TESTED'
        gate['training_executed'] = False
        gate['evidence_boundary'] = ('Backend acceptance of the Mamba-2 reference on the '
                                     'actual closed-loop training path. No learning result.')
    except Exception as exc:
        gate['error'] = f'{type(exc).__name__}: {exc}'
        gate['execution_status'] = 'FAIL'
    gate['status'] = gate['execution_status']
    write_json(out / 'gate_E04_BACKEND.json', gate)
    seal(out)
    print(json.dumps(gate, ensure_ascii=False, indent=2, default=str))
    return 0 if gate['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
