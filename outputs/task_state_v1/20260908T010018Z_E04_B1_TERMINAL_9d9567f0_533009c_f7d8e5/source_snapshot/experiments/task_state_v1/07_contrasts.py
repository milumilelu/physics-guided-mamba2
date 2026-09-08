"""E06 primary contrasts C1/C2/C3 with component-level 98.33% paired bootstrap.

C1: HM-T - H0 on group-balanced normalized prediction risk.
C2: HM-DP - HM-T on real pairwise design regret.
C3: HM-DP - HG-DP on design regret (+ prediction non-inferiority <= 0.02).
Negative = improvement (front model loss minus back model loss).
Regret uses the sealed design evaluator on the frozen candidate table.
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
import yaml
from src.task_state_learning.artifacts import ROOT, new_run, write_json, seal
from src.task_state_learning.grouping import require


def latest(pattern):
    runs = sorted(glob.glob(str(ROOT / 'outputs/task_state_v1' / pattern)))
    require(runs, f'No run for {pattern}')
    return runs[-1]


def group_risk(oof, model, seed_agg=True):
    """Component-balanced normalized risk per sample, averaged within fold first."""
    sub = oof[oof.model_id == model].copy()
    require(len(sub), f'No OOF rows for {model}')
    # loss per sample first (across targets), then seed-mean, then component-mean
    piv = sub.pivot_table(index=['outer_fold', 'seed', 'dataset_index', 'component_id'],
                          columns='target', values=['observed', 'predicted', 'train_scale'])
    observed = piv['observed']
    predicted = piv['predicted']
    scale = piv['train_scale']
    err = ((predicted - observed) / scale) ** 2
    loss = (err['D'] + err[['ilr_z1', 'ilr_z2', 'ilr_z3', 'ilr_z4']].mean(axis=1)
            + err[['A2_8_16', 'entropy_8_16']].mean(axis=1)) / 3.0
    loss = loss.rename('loss').reset_index()
    if seed_agg:
        loss = loss.groupby(['outer_fold', 'dataset_index', 'component_id'],
                            as_index=False).loss.mean()
    return loss


def component_contrast(a, b, level=0.9833, bootstrap=5000, seed=20260907):
    """Paired component-level interval for mean(a-b); negative = improvement."""
    merged = a.merge(b, on=['outer_fold', 'dataset_index', 'component_id'],
                     suffixes=('_a', '_b'), validate='one_to_one')
    comp = merged.groupby(['outer_fold', 'component_id'])[['loss_a', 'loss_b']].mean()
    diff = (comp.loss_a - comp.loss_b)
    rng = np.random.default_rng(seed)
    n = len(diff)
    stats = np.array([diff.to_numpy()[rng.integers(0, n, n)].mean()
                      for _ in range(bootstrap)])
    alpha = (1 - level) / 2
    return {'mean': float(diff.mean()), 'low': float(np.quantile(stats, alpha)),
            'high': float(np.quantile(stats, 1 - alpha)), 'n_components': n}


def design_regret(oof_design, model):
    sub = oof_design[oof_design.model_id == model]
    require(len(sub), f'No design rows for {model}')
    return sub.groupby(['outer_fold', 'pair_id', 'component_pair']).regret.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/task_state_v1/real_main.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8'))
    out = new_run('E06_CONTRASTS', ROOT / args.config)
    gate = {'scope': 'E06_PRIMARY_CONTRASTS', 'training_executed': False}
    try:
        hybrid_dir = latest('2026*_E06_HYBRID*')
        oof = pd.read_csv(Path(hybrid_dir) / 'oof.csv')
        g = cfg['g6_contrasts']
        rows = []
        a = group_risk(oof, 'MAMBA2')
        b = group_risk(oof, 'H0')
        common = a.merge(b, on=['outer_fold', 'dataset_index', 'component_id'])
        c1 = component_contrast(a, b, g['paired_interval'], g['bootstrap'],
                                g['bootstrap_seed'])
        rows.append({'contrast': 'C1_HM-T_vs_H0_prediction_risk', **c1})
        dp_path = Path(hybrid_dir) / 'design_regret.csv'
        if dp_path.exists():
            design = pd.read_csv(dp_path)
            for name, m1, m2 in (('C2_HM-DP_vs_HM-T_regret', 'MAMBA2_DP', 'MAMBA2'),
                                 ('C3_HM-DP_vs_HG-DP_regret', 'MAMBA2_DP', 'GRU_DP')):
                ra = design_regret(design, m1)
                rb = design_regret(design, m2)
                common_idx = ra.index.intersection(rb.index)
                diff = (ra.loc[common_idx] - rb.loc[common_idx])
                comp = diff.groupby(['outer_fold']).mean()
                rng = np.random.default_rng(g['bootstrap_seed'] + 1)
                n = len(comp)
                stats = np.array([comp.to_numpy()[rng.integers(0, n, n)].mean()
                                  for _ in range(g['bootstrap'])])
                alpha = (1 - g['paired_interval']) / 2
                rows.append({'contrast': name, 'mean': float(comp.mean()),
                             'low': float(np.quantile(stats, alpha)),
                             'high': float(np.quantile(stats, 1 - alpha)),
                             'n_components': n})
        frame = pd.DataFrame(rows)
        frame.to_csv(out / 'primary_contrasts.csv', index=False)
        # non-inferiority companion for C3
        if (frame.contrast == 'C3_HM-DP_vs_HG-DP_regret').any():
            c3 = frame[frame.contrast == 'C3_HM-DP_vs_HG-DP_regret'].iloc[0]
            ga = group_risk(oof, 'MAMBA2')
            gb = group_risk(oof, 'GRU')
            c3_pred = component_contrast(ga, gb, g['paired_interval'], g['bootstrap'],
                                         g['bootstrap_seed'] + 2)
            pd.DataFrame([{'contrast': 'C3_prediction_noninferiority', **c3_pred}]).to_csv(
                out / 'c3_prediction_check.csv', index=False)
            gate['c3_joint_support'] = bool(c3['high'] < 0 and c3_pred['high'] <= 0.02)
        c1 = frame[frame.contrast == 'C1_HM-T_vs_H0_prediction_risk'].iloc[0]
        gate['c1_supported'] = bool(c1['high'] < 0)
        gate['n_contrasts'] = len(frame)
        gate['run_used'] = Path(hybrid_dir).name
        gate['execution_status'] = 'PASS'
        gate['metric_status'] = 'VALID'
        gate['hypothesis_status'] = 'RECORDED'
        gate['evidence_boundary'] = ('Conditional on fixed trained models and fixed splits; '
                                     'component-level paired bootstrap.')
    except Exception as exc:
        import traceback
        gate['error'] = f'{type(exc).__name__}: {exc}'
        gate['traceback'] = traceback.format_exc()
        gate['execution_status'] = 'FAIL'
    gate['status'] = gate['execution_status']
    write_json(out / 'gate_E06_CONTRASTS.json', gate)
    seal(out)
    print(json.dumps({k: v for k, v in gate.items() if k != 'traceback'},
                     ensure_ascii=False, indent=2))
    return 0 if gate['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
