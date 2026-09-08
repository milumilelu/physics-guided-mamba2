"""E04 B1 terminal-only training: H0/HG/HCT/HM comparison, null, DP, continuation.

Information firewall: learners load terminal_only files only (loader rejects
hidden arrays). The evaluator reads evaluator_only hidden states solely for
continuation pairing after sealed choices. Final refits use the frozen median
epoch count from inner tuning; the test split is never used for selection.
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
from src.task_state_learning.artifacts import ROOT, new_run, write_json, verify_seal, seal
from src.task_state_learning.benchmarks import load_terminal_only, b1_reference
from src.task_state_learning.design import seal_choices, evaluate_sealed, pairwise_regret
from src.task_state_learning.grouping import require
from src.task_state_learning.losses import PairSampler, terminal_scale, farthest_point_anchors
from src.task_state_learning.models import B1ClosureModel
from src.task_state_learning.static_baseline import StaticMLP
from src.task_state_learning.training import fit, predict

MODEL_DEVICES = {'STATIC': 'cuda', 'H0': 'cuda', 'GRU': 'cuda', 'CTSSM': 'cuda',
                 'MAMBA2': 'cuda', 'MAMBA2_DP': 'cuda', 'GRU_DP': 'cuda'}


def make_model(model_id, seed=0):
    torch.manual_seed(seed)
    if model_id == 'STATIC':
        return StaticMLP()
    base = model_id.replace('_DP', '')
    return B1ClosureModel(base, memory_seed=seed)


def tensors(data):
    out = {}
    for k, v in data.items():
        if k in ('control', 'physical_dt', 'mask', 'terminal_x'):
            arr = np.asarray(v)
            if k == 'mask':
                out[k] = torch.as_tensor(arr.astype(bool))
            else:
                out[k] = torch.as_tensor(arr.astype(np.float32))
    return out


def sealed_design(model, test_data, anchors, scale, train_terminal, run_dir, model_id,
                  seed_bundle):
    """Choices from predictions only; evaluator joins labels after hash check."""
    gen = torch.Generator().manual_seed(seed_bundle)
    n = len(test_data['terminal_x'])
    i = torch.randint(0, n, (256,), generator=gen)
    j = torch.randint(0, n - 1, (256,), generator=gen)
    j = torch.where(j >= i, j + 1, j)
    anchor_pick = torch.randint(0, len(anchors), (256,), generator=gen)
    pred = predict(model, test_data)
    rows = []
    for pair in range(256):
        a, b, t = int(i[pair]), int(j[pair]), int(anchor_pick[pair])
        ja = float(((pred[a] - anchors[t]) / scale).pow(2).sum())
        jb = float(((pred[b] - anchors[t]) / scale).pow(2).sum())
        chosen = a if ja <= jb else b
        rows.append({'pair_id': pair, 'anchor_id': t, 'candidate_id': ('A', a),
                     'chosen_id': ('A', a) if chosen == a else ('B', b)})
        rows.append({'pair_id': pair, 'anchor_id': t, 'candidate_id': ('B', b),
                     'chosen_id': ('A', a) if chosen == a else ('B', b)})
    choices = pd.DataFrame([{'pair_id': r['pair_id'], 'anchor_id': r['anchor_id'],
                             'candidate_id': f"{r['candidate_id'][0]}{r['candidate_id'][1]}",
                             'chosen_id': f"{r['chosen_id'][0]}{r['chosen_id'][1]}"} for r in rows])
    digest = seal_choices(run_dir / f'sealed_choices_{model_id}.csv', choices)
    measured = pd.DataFrame({'candidate_id': [f'A{k}' for k in range(n)] + [f'B{k}' for k in range(n)],
                             'true_terminal_x1': list(test_data['terminal_x'][:, 0]) * 2,
                             'true_terminal_x2': list(test_data['terminal_x'][:, 1]) * 2})
    merged = evaluate_sealed(run_dir / f'sealed_choices_{model_id}.csv', measured, 'candidate_id')
    anchor_term = anchors.numpy()
    merged['objective_true'] = [
        float((((np.array([r.true_terminal_x1, r.true_terminal_x2]) - anchor_term[r.anchor_id]) / scale.numpy()) ** 2).sum())
        for r in merged.itertuples()]
    regret = pairwise_regret(merged, 'objective_true')
    regret['model_id'] = model_id
    regret['family_a'] = i.numpy()
    regret['family_b'] = j.numpy()
    return regret, digest


# Continuation is evaluated independently by 05_g2_gate.py.

def main():
    from src.task_state_learning.guardrails import require_not_held
    require_not_held('E04_B1_TERMINAL')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/task_state_v1/b1_training.yaml')
    parser.add_argument('--tracks', nargs='*', default=None)
    parser.add_argument('--resume-run', type=Path)
    args = parser.parse_args()
    path = ROOT / args.config
    c = yaml.safe_load(path.read_text(encoding='utf-8'))
    if args.resume_run:
        out=args.resume_run.resolve()
        require(out.is_relative_to(ROOT/'outputs/task_state_v1'),'Resume path outside namespace')
        require(not (out/'release_manifest.json').exists(),'Cannot mutate a sealed run')
        require((out/'config_source.yaml').read_bytes()==path.read_bytes(),'Resume config mismatch')
        source=json.loads((out/'source_snapshot.json').read_text(encoding='utf-8'))
        from src.task_state_learning.artifacts import sha256
        for row in source['files']:
            require(sha256(ROOT/row['path'])==row['sha256'],'Resume source mismatch: '+row['path'])
    else:
        out = new_run('E04_B1_TERMINAL', path)
    (out/'config_resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
    tracks = args.tracks or c['tracks']
    require(tracks and set(tracks)<= {'coupled','null'},'Unknown or YAML-null track name')
    ref = ROOT / 'outputs/task_state_v1' / c['reference_run']
    verify_seal(ref)
    gate = {'scope': c['scope'], 'tracks': tracks, 'training_executed': True,
            'mamba_backend': 'reference_pure_torch_ssd_v1'}
    try:
        device_pref = c['device']
        all_results = []
        oof_rows = []
        for track in tracks:
            data = {}
            for split in ['train', 'validation', 'test']:
                data[split] = load_terminal_only(ref / 'terminal_only' / f'{split}_{track}.npz')
            # Hidden arrays are not opened by the training runner.
            train_tensors = tensors(data['train'])
            val_tensors = tensors(data['validation'])
            test_tensors = tensors(data['test'])
            scale = terminal_scale(train_tensors['terminal_x'])
            anchors = farthest_point_anchors(train_tensors['terminal_x'], scale, c['n_anchors'])
            dp_bundle = None
            sampler = PairSampler(len(train_tensors['terminal_x']), c['pairs_per_epoch'],
                                  c['anchor_seed'])
            dp_bundle = {'anchors': anchors, 'scale': scale, 'sampler': sampler}
            # ---- inner tuning ----
            inner_rows = []
            for model_id in c['models'] + c['decision_aware']:
                for lr in c['learning_rates']:
                    model = make_model(model_id, c['tuning_seed'])
                    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
                        if device_pref == 'auto' else device_pref
                    use_dp = model_id.endswith('_DP')
                    result = fit(model, train_tensors, val_tensors, lr=lr,
                                 seed=c['tuning_seed'], device=device,
                                 dp=dp_bundle if use_dp else None,
                                 max_epochs=c['max_epochs'],patience=c['patience'],batch=c['batch'],accum=c['accum'],
                                 checkpoint_dir=out/'jobs'/f'{track}_{model_id}_tune_{lr}',
                                 epoch_callback=lambda r: print(f"{track}/{model_id}/tune epoch {r['epochs_completed']}: {r['val_loss']:.6g}",flush=True))
                    inner_rows.append({'model_id': model_id, 'lr': lr,
                                       'best_epoch': result['best_epoch'],
                                       'val_loss': result['val_loss']})
            inner = pd.DataFrame(inner_rows)
            inner.to_csv(out / f'inner_tuning_{track}.csv', index=False)
            chosen = inner.loc[inner.groupby('model_id').val_loss.idxmin()]
            epochs = {r.model_id: int(r.best_epoch)+1 for r in chosen.itertuples()}
            lrs = {r.model_id: r.lr for r in chosen.itertuples()}
            # ---- outer refits (frozen epochs, no early stopping) ----
            test_metrics = []
            for model_id in c['models'] + c['decision_aware']:
                for seed in c['final_seeds']:
                    model = make_model(model_id, seed)
                    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
                        if device_pref == 'auto' else device_pref
                    use_dp = model_id.endswith('_DP')
                    result = fit(model, train_tensors, val_tensors, lr=lrs[model_id],
                                 seed=seed, device=device, fixed_epochs=epochs[model_id],
                                 dp=dp_bundle if use_dp else None,
                                 batch=c['batch'],accum=c['accum'],
                                 checkpoint_dir=out/'jobs'/f'{track}_{model_id}_refit_{seed}',
                                 epoch_callback=lambda r: print(f"{track}/{model_id}/seed{seed} epoch {r['epochs_completed']}",flush=True))
                    from src.task_state_learning.training import _atomic_save
                    _atomic_save(result['model'].state_dict(),out/f'state_{model_id}_{track}_{seed}.pt')
                    pred = predict(result['model'], test_tensors)
                    mse = float((pred - test_tensors['terminal_x']).pow(2).mean())
                    per_family = (pred - test_tensors['terminal_x']).pow(2).sum(1)
                    test_metrics.append({'track': track, 'model_id': model_id, 'seed': seed,
                                         'test_mse': mse, 'epochs': epochs[model_id],
                                         'lr': lrs[model_id]})
                    for i in range(len(per_family)):
                        oof_rows.append({'track': track, 'model_id': model_id, 'seed': seed,
                                         'family_id': data['test']['family_id'][i],
                                         'sq_error': float(per_family[i]),
                                         'pred_x1': float(pred[i, 0]), 'pred_x2': float(pred[i, 1]),
                                         'true_x1': float(test_tensors['terminal_x'][i, 0]),
                                         'true_x2': float(test_tensors['terminal_x'][i, 1])})
            # Reuse all registered refits; no extra training or seed selection.
            design_rows = []
            for model_id in c['models'] + c['decision_aware']:
                for seed in c['final_seeds']:
                    model=make_model(model_id,seed)
                    state_path=out/f'state_{model_id}_{track}_{seed}.pt'
                    require(state_path.exists(),'Missing refit checkpoint')
                    model.load_state_dict(torch.load(state_path,map_location='cpu',weights_only=True))
                    regret,digest=sealed_design(model,test_tensors,anchors,scale,
                        train_tensors['terminal_x'],out,f'{model_id}_{track}_{seed}',c['anchor_seed']+1)
                    regret['model_id']=model_id;regret['track']=track;regret['seed']=seed
                    design_rows.append(regret)
            pd.concat(design_rows).to_csv(out / f'design_regret_{track}.csv', index=False)
            pd.DataFrame(test_metrics).to_csv(out / f'test_metrics_{track}.csv', index=False)
            all_results.extend(test_metrics)
        pd.DataFrame(oof_rows).to_csv(out / 'oof_per_family.csv', index=False)
        pd.DataFrame(all_results).to_csv(out / 'all_seed_metrics.csv', index=False)
        gate['execution_status'] = 'PASS'
        gate['metric_status'] = 'VALID'
        gate['hypothesis_status'] = 'PENDING_G2_EVALUATION'
        gate['evidence_boundary'] = ('Terminal-only B1 training results. Hypothesis verdict '
                                     'assigned by 05_g2_gate.py on the sealed metrics.')
    except Exception as exc:
        import traceback
        gate['error'] = f'{type(exc).__name__}: {exc}'
        gate['traceback'] = traceback.format_exc()
        gate['execution_status'] = 'FAIL'
    gate['status'] = gate['execution_status']
    write_json(out / 'gate_E04_B1_TERMINAL.json', gate)
    seal(out)
    print(json.dumps({k: gate[k] for k in gate if k != 'traceback'}, ensure_ascii=False, indent=2))
    return 0 if gate['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
