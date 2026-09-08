"""E04 G2 verdict: family-paired memory comparison, null control, DP regret,
continuation consistency. Reads only sealed E04_B1_TERMINAL artifacts; the
continuation evaluator uses hidden arrays exclusively here, never in training.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
import torch
import yaml
from src.task_state_learning.artifacts import ROOT, new_run, write_json, verify_seal, seal
from src.task_state_learning.benchmarks import load_terminal_only
from src.task_state_learning.grouping import require
from src.task_state_learning.models import B1ClosureModel

MEMORY_MODELS = {'HM': 'MAMBA2', 'HG': 'GRU', 'HCT': 'CTSSM'}


def latest_run(pattern):
    runs = sorted(glob.glob(str(ROOT / 'outputs/task_state_v1' / pattern)))
    require(runs, f'No sealed run matches {pattern}')
    return verify_seal(runs[-1])


def seed_mean_loss(metrics, track, model_id):
    rows = metrics[(metrics.track == track) & (metrics.model_id == model_id)]
    require(len(rows) >= 1, f'No metrics for {track}/{model_id}')
    return float(rows.test_mse.mean())


def per_family(metrics, oof, track, model_id):
    """Family-paired squared error, averaged over seeds first (runbook 18.4)."""
    sub = oof[(oof.track == track) & (oof.model_id == model_id)]
    grouped = sub.groupby('family_id').sq_error.mean()
    return grouped


def paired_interval(a, b, level=0.9833, bootstrap=5000, seed=20260907):
    """Paired percentile interval for mean(a-b), resampling families."""
    require(len(a) == len(b) and len(a) >= 2, 'Paired vectors must align')
    diff = (a - b).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(diff)
    stats = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(bootstrap)])
    alpha = (1 - level) / 2
    return {'mean': float(diff.mean()), 'low': float(np.quantile(stats, alpha)),
            'high': float(np.quantile(stats, 1 - alpha)), 'n_families': n}


def continuation_pairs(ref, track, cfg):
    """Evaluator-side pairing: close terminal observable, separated hidden q."""
    test = load_terminal_only(ref / 'terminal_only' / f'test_{track}.npz')
    hidden = np.load(ref / 'evaluator_only' / f'test_{track}.npz', allow_pickle=False)
    terminal = test['terminal_x']
    scale = float(terminal.std(axis=0).max())
    q = hidden['hidden'][:, -1, 2:]
    pairs = []
    for i in range(len(terminal)):
        for j in range(i + 1, len(terminal)):
            obs_gap = float(np.abs(terminal[i] - terminal[j]).max())
            q_gap = float(np.abs(q[i] - q[j]).max()) / scale
            if obs_gap <= cfg['continuation']['observable_rel_tolerance'] and \
                    q_gap >= cfg['continuation']['q_separation']:
                pairs.append((i, j, obs_gap, q_gap))
    pairs.sort(key=lambda p: -p[3])
    return pairs[:cfg['continuation']['future_pairs']], test, hidden


def continuation_metrics(run_dir, ref, track, cfg, model_id, seed):
    pairs, test, hidden = continuation_pairs(ref, track, cfg)
    if not pairs:
        return pd.DataFrame([{'model_id': model_id, 'n_pairs': 0,
                              'continuation_mse': np.nan, 'ordering_accuracy': np.nan}])
    data = {k: torch.as_tensor(np.asarray(test[k]).astype(np.float32) if k != 'mask'
                               else np.asarray(test[k]).astype(bool))
            for k in ('control', 'physical_dt', 'mask')}
    model = B1ClosureModel(model_id.replace('_DP', ''), memory_seed=seed)
    state_path = run_dir / f'state_{model_id}_{track}_{seed}.pt'
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, weights_only=True))
    model.eval()
    future_dt = float(np.asarray(test['physical_dt']).astype(np.float32)[0, 0])
    steps = cfg['continuation']['future_steps']
    rows = []
    with torch.no_grad():
        first = model(data['control'], data['physical_dt'], data['mask'], return_state=True)
        future_u = torch.zeros(len(data['control']), steps, 2)
        future_dt_t = torch.full((len(data['control']), steps), future_dt)
        future_mask = torch.ones(len(data['control']), steps, dtype=torch.bool)
        cont = model(future_u, future_dt_t, future_mask,
                     initial_state={'x': first['x'], 'memory': first['memory']}).numpy()
    true_future = np.zeros_like(cont)
    q_all = hidden['hidden'][:, -1, :]
    for idx, (i, j, _, _) in enumerate(pairs):
        for col, family in ((0, i), (1, j)):
            # evaluator truth: continue the true hidden state under zero control
            x1, x2, q1, q2 = q_all[family]
            nx1 = x1 * np.exp(-0.4 * future_dt * steps)
            nx2 = x2 * np.exp(-0.7 * future_dt * steps)
            true_future[family] = [nx1, nx2]
    for idx, (i, j, _, _) in enumerate(pairs):
        mse = float(((cont[i] - true_future[i]) ** 2).mean()
                    + ((cont[j] - true_future[j]) ** 2).mean()) / 2
        order_true = np.sign(true_future[i, 0] - true_future[j, 0])
        order_pred = np.sign(cont[i, 0] - cont[j, 0])
        rows.append({'model_id': model_id, 'pair_index': idx, 'continuation_mse': mse,
                     'ordering_correct': bool(order_true != 0 and order_true == order_pred)})
    frame = pd.DataFrame(rows)
    summary = frame.groupby('model_id').agg(
        n_pairs=('pair_index', 'count'),
        continuation_mse=('continuation_mse', 'mean'),
        ordering_accuracy=('ordering_correct', 'mean')).reset_index()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/task_state_v1/b1_training.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8'))
    out = new_run('E04_G2', ROOT / args.config)
    gate = {'scope': 'E04_G2_VERDICT', 'training_executed': False}
    try:
        run_dir = latest_run('2026*_E04_B1_TERMINAL_*')
        metrics = pd.read_csv(run_dir / 'all_seed_metrics.csv')
        oof = pd.read_csv(run_dir / 'oof_per_family.csv')
        require(set(metrics.track) <= {'coupled', 'null'}, 'Unexpected tracks')
        h0 = per_family(metrics, oof, 'coupled', 'H0')
        memory_rows = []
        for tag, model_id in MEMORY_MODELS.items():
            family = per_family(metrics, oof, 'coupled', model_id)
            common = h0.index.intersection(family.index)
            memory_rows.append({
                'contrast': f'{model_id}_vs_H0',
                **paired_interval(family.loc[common], h0.loc[common],
                                  cfg['g2']['paired_interval'], cfg['g2']['bootstrap'],
                                  cfg['g2']['bootstrap_seed'])})
        # null track: no memory-specific gain should appear
        h0_null = per_family(metrics, oof, 'null', 'H0')
        for tag, model_id in MEMORY_MODELS.items():
            family = per_family(metrics, oof, 'null', model_id)
            common = h0_null.index.intersection(family.index)
            memory_rows.append({
                'contrast': f'null_{model_id}_vs_H0',
                **paired_interval(family.loc[common], h0_null.loc[common],
                                  cfg['g2']['paired_interval'], cfg['g2']['bootstrap'],
                                  cfg['g2']['bootstrap_seed'] + 1)})
        contrasts = pd.DataFrame(memory_rows)
        contrasts.to_csv(out / 'g2_prediction_contrasts.csv', index=False)
        # DP design regret contrasts
        design_frames = []
        for track in ['coupled', 'null']:
            path = run_dir / f'design_regret_{track}.csv'
            if path.exists():
                frame = pd.read_csv(path)
                frame['track'] = track
                design_frames.append(frame)
        design = pd.concat(design_frames, ignore_index=True)
        design.to_csv(out / 'g2_design_regret.csv', index=False)
        dp_rows = []
        for track in ['coupled', 'null']:
            for dp_model, t_model in (('MAMBA2_DP', 'MAMBA2'), ('GRU_DP', 'GRU')):
                a = design[(design.track == track) & (design.model_id == dp_model)]
                b = design[(design.track == track) & (design.model_id == t_model)]
                if len(a) and len(b):
                    am = a.groupby('pair_id').regret.mean()
                    bm = b.groupby('pair_id').regret.mean()
                    common = am.index.intersection(bm.index)
                    dp_rows.append({'contrast': f'{dp_model}_vs_{t_model}', 'track': track,
                                    **paired_interval(am.loc[common], bm.loc[common],
                                                      cfg['g2']['paired_interval'],
                                                      cfg['g2']['bootstrap'],
                                                      cfg['g2']['bootstrap_seed'] + 2)})
        dp_contrasts = pd.DataFrame(dp_rows)
        dp_contrasts.to_csv(out / 'g2_dp_contrasts.csv', index=False)
        # continuation (evaluator-only hidden access)
        cont_frames = []
        ref = ROOT / 'outputs/task_state_v1' / cfg['reference_run']
        for track in ['coupled', 'null']:
            for model_id in ['MAMBA2', 'GRU', 'H0']:
                cont_frames.append(continuation_metrics(run_dir, ref, track, cfg, model_id,
                                                        cfg['final_seeds'][0]))
        continuation = pd.concat(cont_frames, ignore_index=True)
        continuation.to_csv(out / 'g2_continuation.csv', index=False)
        # verdicts
        def supported(frame, contrast):
            row = frame[frame.contrast == contrast]
            return bool(len(row) and row['high'].iloc[0] < 0)
        memory_supported = any(supported(contrasts, f'{m}_vs_H0') for m in MEMORY_MODELS)
        null_spurious = any(supported(contrasts, f'null_{m}_vs_H0') for m in MEMORY_MODELS)
        dp_supported = any(supported(dp_contrasts[dp_contrasts.track == 'coupled'],
                                     f'{d}_vs_{t}') for d, t in (('MAMBA2_DP', 'MAMBA2'),
                                                                 ('GRU_DP', 'GRU')))
        verdicts = {'memory_prediction_benefit': memory_supported,
                    'null_spurious_memory_gain': null_spurious,
                    'decision_aware_regret_benefit': dp_supported}
        if memory_supported and not null_spurious:
            hypothesis = 'SUPPORTED'
        elif memory_supported and null_spurious:
            hypothesis = 'INCONCLUSIVE'
        else:
            hypothesis = 'NOT_SUPPORTED'
        gate.update(verdicts=verdicts, hypothesis_status=hypothesis,
                    execution_status='PASS', metric_status='VALID',
                    run_used=str(run_dir.relative_to(ROOT / 'outputs/task_state_v1')),
                    continuation_registered=bool(len(continuation)),
                    evidence_boundary='B1 synthetic benchmark verdict; no real-data claim.')
    except Exception as exc:
        import traceback
        gate['error'] = f'{type(exc).__name__}: {exc}'
        gate['traceback'] = traceback.format_exc()
        gate['execution_status'] = 'FAIL'
    gate['status'] = gate['execution_status']
    write_json(out / 'gate_E04_G2.json', gate)
    seal(out)
    print(json.dumps({k: v for k, v in gate.items() if k != 'traceback'}, ensure_ascii=False, indent=2))
    return 0 if gate['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
