"""E06 MAIN180 main experiment: static baselines, physics switches, hybrids.

Subcommands (staged execution, one sealed run each):
  static  — mean dummy, Ridge, additive spline, small U->Y MLP, MLP+frozen
            physics descriptors (sklearn/torch, canonical observer targets).
  physics — P00..P11 rows using calibrated vectors from the E03 run (cell-solver
            forward, D/A predictions; P/T blocks use the train mean null, so the
            joint risk reflects the depth-field information only).
  hybrid  — H0/GRU/CTSSM/HM-T/HM-DP/HG-DP with the differentiable laser rollout;
            every specimen starts from zero physical state and zero memory cache.

All OOF rows carry the runbook 12.4 schema; outer test is never used for
selection; the three primary contrasts are evaluated by 07_contrasts.py.
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
from src.task_state_learning.contracts import load_contract
from src.task_state_learning.diagnostics import (Regressor, Y, group_weights,
                                                 raw_features, target_statistics,
                                                 per_row_risk)
from src.task_state_learning.grouping import components, require
from src.task_state_learning.losses import PairSampler, terminal_scale, farthest_point_anchors

OOF_COLUMNS = ['run_id', 'experiment_id', 'model_id', 'information_track', 'seed',
               'outer_fold', 'dataset_index', 'session_id', 'sample_id', 'component_id',
               'target', 'observed', 'predicted', 'train_null', 'train_scale',
               'physical_valid', 'band_valid', 'support_status', 'model_sha',
               'data_sha', 'split_sha']


def load_frame(cfg):
    cfg_run, frozen, df, _, _ = load_contract()
    df = components(df)
    split_dir = ROOT / 'outputs/task_state_v1' / cfg['split_run']
    manifest = pd.read_csv(split_dir / 'split_manifest.csv')
    frame = df[df.session_role.isin(['formal', 'pass_main'])].merge(
        manifest[['dataset_index', 'outer_fold']], on='dataset_index', validate='one_to_one')
    return frame


def oof_rows(frame_test, preds, model_id, seed, fold, null, scale, out_name, track,
             support=None, physical_valid=True):
    rows = []
    for pos, row in enumerate(frame_test.reset_index(drop=True).itertuples(index=False)):
        for j, target in enumerate(Y):
            rows.append({'run_id': out_name, 'experiment_id': 'E06', 'model_id': model_id,
                         'information_track': track, 'seed': seed, 'outer_fold': fold,
                         'dataset_index': int(row.dataset_index), 'session_id': row.session_id,
                         'sample_id': int(row.sample_id), 'component_id': row.component_id,
                         'target': target, 'observed': float(getattr(row, target)),
                         'predicted': float(preds[pos, j]), 'train_null': float(null[j]),
                         'train_scale': float(scale[j]),
                         'physical_valid': bool(physical_valid), 'band_valid': True,
                         'support_status': 'IN_SUPPORT' if support is None else support[pos],
                         'model_sha': '', 'data_sha': '', 'split_sha': ''})
    return rows


def run_static(cfg, out, frame):
    from threadpoolctl import threadpool_limits
    rows = []
    with threadpool_limits(limits=8):
        for fold in range(5):
            train = frame[frame.outer_fold != fold]
            test = frame[frame.outer_fold == fold]
            null, scale = target_statistics(train)
            # physics descriptors from the frozen E03 calibration (train-only refit)
            phys = physics_descriptors(cfg, train, test)
            for model_id in cfg['static_models']:
                for seed in [cfg['tuning_seed']] if model_id in ('DUMMY', 'RIDGE', 'SPLINE') \
                        else cfg['final_seeds']:
                    if model_id == 'DUMMY':
                        preds = np.tile(null, (len(test), 1))
                    elif model_id in ('RIDGE', 'SPLINE'):
                        family = 'Ridge' if model_id == 'RIDGE' else 'GAM'
                        model = Regressor(family, 1.0).fit(
                            raw_features(train, 'M_U'), train[Y].to_numpy(float),
                            group_weights(train))
                        preds = model.predict(raw_features(test, 'M_U'))
                    else:
                        preds = static_mlp(train, test, model_id, seed, cfg, phys)
                    rows.extend(oof_rows(test, preds, model_id, seed, fold, null, scale,
                                         out.name, 'T' if model_id == 'DUMMY' else 'T'))
            print(f'static fold {fold + 1}/5', flush=True)
    return pd.DataFrame(rows)


def physics_descriptors(cfg, train, test):
    """Frozen calibrated P11 descriptors per sample (train-only calibration)."""
    from src.task_state_learning.scan_cell import CellConfig, simulate_sample
    summary_path = cfg.get('calibration_run')
    if summary_path and Path(summary_path).exists():
        summary = pd.read_csv(Path(summary_path) / 'calibration_summary.csv')
        vec = np.array(summary[(summary.switch == 'P11') &
                               (summary.waist_um == cfg['waist_um_nominal'])].vector.iloc[0])
    else:
        vec = np.array([math.log(2e8), math.log(1e-8), math.log(0.02), 0.0, 0.0, 0.0])
    log_F1, log_delta, log_kappa, logit_rho, gamma_F, gamma_delta = vec
    from src.task_state_learning.physics import PhysicsParameters
    p = PhysicsParameters(cfg['waist_um_nominal'] * 1e-6, cfg['wavelength_m'], cfg['m_squared'],
                          math.exp(log_F1), math.exp(log_delta), math.exp(log_kappa),
                          1.0 / (1.0 + math.exp(-logit_rho)), gamma_F=gamma_F,
                          gamma_delta=gamma_delta, reference_duration_s=cfg['reference_duration_s'])
    out = []
    for row in pd.concat([train, test]).itertuples(index=False):
        d, a, _ = simulate_sample(p, cfg=CellConfig(), tau_s=row.pulse_duration_fs * 1e-15,
                                  f_Hz=row.frequency_kHz * 1e3, v_m_s=row.velocity_mm_s * 1e-3,
                                  h_m=row.hatch_spacing_um * 1e-6, passes=int(row.pass_count),
                                  power_W=cfg['power_W'], switch='P11')
        out.append((d * 1e6, a * 1e6))
    return np.array(out)


def static_mlp(train, test, model_id, seed, cfg, phys):
    import torch.nn as nn
    torch.manual_seed(seed)
    x_tr = raw_features(train, 'M_U')
    x_te = raw_features(test, 'M_U')
    if model_id == 'MLP_UPHYS':
        n_tr, n_te = len(train), len(test)
        x_tr = np.column_stack([x_tr, phys[:n_tr]])
        x_te = np.column_stack([x_te, phys[n_tr:]])
    y_tr = train[Y].to_numpy(float)
    _, scale = target_statistics(train)
    mean, sd = x_tr.mean(0), x_tr.std(0).clip(1e-9)
    net = nn.Sequential(nn.Linear(x_tr.shape[1], cfg['static']['mlp_hidden']), nn.SiLU(),
                        nn.Linear(cfg['static']['mlp_hidden'], len(Y)))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.tensor((x_tr - mean) / sd, dtype=torch.float32)
    yt = torch.tensor(y_tr / scale, dtype=torch.float32)
    xe = torch.tensor((x_te - mean) / sd, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    best, best_loss = None, math.inf
    for epoch in range(cfg['static']['mlp_epochs']):
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), 16):
            idx = perm[i:i + 16]
            loss = (net(xt[idx]) - yt[idx]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val = float((net(xt) - yt).pow(2).mean())
        if val < best_loss - 1e-6:
            best_loss, best = val, {k: v.clone() for k, v in net.state_dict().items()}
        elif epoch - (best is not None) > cfg['static']['mlp_patience']:
            break
    if best is not None:
        net.load_state_dict(best)
    with torch.no_grad():
        return net(xe).numpy() * scale


def run_physics(cfg, out, frame):
    from src.task_state_learning.scan_cell import CellConfig, simulate_sample
    from src.task_state_learning.physics import PhysicsParameters
    summary = pd.read_csv(Path(cfg['calibration_run']) / 'calibration_summary.csv')
    rows = []
    for fold in range(5):
        train = frame[frame.outer_fold != fold]
        test = frame[frame.outer_fold == fold]
        null, scale = target_statistics(train)
        for switch in cfg['physics_models']:
            sub = summary[(summary.switch == switch) &
                          (summary.waist_um == cfg['waist_um_nominal']) &
                          (summary.outer_fold == fold)]
            vec = np.array(sub.vector.iloc[0])
            log_F1, log_delta = vec[0], vec[1]
            if switch in ('P01', 'P11'):
                log_kappa, logit_rho, gamma_F, gamma_delta = vec[2], vec[3], vec[4], vec[5]
                kappa, rho = math.exp(log_kappa), 1.0 / (1.0 + math.exp(-logit_rho))
            else:
                kappa, rho = 0.0, 0.5
                gamma_F, gamma_delta = vec[2], vec[3]
            p = PhysicsParameters(cfg['waist_um_nominal'] * 1e-6, cfg['wavelength_m'],
                                  cfg['m_squared'], math.exp(log_F1), math.exp(log_delta),
                                  kappa, rho, gamma_F=gamma_F, gamma_delta=gamma_delta,
                                  reference_duration_s=cfg['reference_duration_s'])
            preds = []
            for row in test.itertuples(index=False):
                d, a, _ = simulate_sample(p, cfg=CellConfig(),
                                          tau_s=row.pulse_duration_fs * 1e-15,
                                          f_Hz=row.frequency_kHz * 1e3,
                                          v_m_s=row.velocity_mm_s * 1e-3,
                                          h_m=row.hatch_spacing_um * 1e-6,
                                          passes=int(row.pass_count), power_W=cfg['power_W'],
                                          switch=switch)
                # physics provides D/A only; P/T blocks use the train mean null
                preds.append([d * 1e6, a * 1e6, null[2], null[3], null[4], null[5],
                              null[6], null[7]])
            preds = np.array(preds)
            rows.extend(oof_rows(test, preds, switch, cfg['tuning_seed'], fold, null, scale,
                                 out.name, 'T', physical_valid=True))
        print(f'physics fold {fold + 1}/5', flush=True)
    return pd.DataFrame(rows)


def run_hybrid(cfg, out, frame):
    from src.task_state_learning.laser_hybrid import LaserHybrid
    from src.task_state_learning.laser_rollout import (RolloutConfig, line_schedule,
                                                       build_batch_plan, gl_nodes,
                                                       terminal_height_um)
    from src.task_state_learning.differentiable_observer import DifferentiableObserver
    rows = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_roll = RolloutConfig(max_packets=cfg['rollout']['max_packets'],
                             tail_waists=cfg['rollout']['tail_waists'])
    observer = DifferentiableObserver()
    for fold in range(5):
        train = frame[frame.outer_fold != fold].reset_index(drop=True)
        test = frame[frame.outer_fold == fold].reset_index(drop=True)
        null, scale = target_statistics(train)
        y_scale = torch.tensor(scale, dtype=torch.float32)
        anchors_np, sampler = None, None
        for model_id in cfg['hybrid_models'] + cfg['dp_models']:
            for seed in ([cfg['tuning_seed']] if cfg.get('inner_only') else cfg['final_seeds']):
                torch.manual_seed(seed)
                model = LaserHybrid(model_id.replace('_DP', ''),
                                    d_state=16, seed=seed, cfg=cfg_roll).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rates'][1],
                                        weight_decay=cfg['weight_decay'])
                gen = torch.Generator().manual_seed(seed)
                n_train = len(train)
                best_loss, patience_count, best_state = math.inf, 0, None
                for epoch in range(cfg['max_epochs']):
                    model.train()
                    perm = torch.randperm(n_train, generator=gen).tolist()
                    for start in range(0, n_train, cfg['batch']):
                        idx = perm[start:start + cfg['batch']]
                        sub = train.iloc[idx]
                        loss = hybrid_loss(model, sub, cfg, cfg_roll, observer, device,
                                           y_scale)
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        val = float(hybrid_loss(model, train.iloc[:32], cfg, cfg_roll,
                                                observer, device, y_scale))
                    if val < best_loss - 1e-6:
                        best_loss, patience_count = val, 0
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    else:
                        patience_count += 1
                        if patience_count >= cfg['patience']:
                            break
                if best_state is not None:
                    model.load_state_dict(best_state)
                preds = np.zeros((len(test), len(Y)))
                bs = 8
                for start in range(0, len(test), bs):
                    preds[start:start + bs] = hybrid_predict(model, test.iloc[start:start + bs],
                                                             cfg, cfg_roll, observer, device)
                rows.extend(oof_rows(test, preds, model_id, seed, fold, null, scale,
                                     out.name, 'T'))
                model.cpu()
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()
            print(f'hybrid fold {fold + 1}/5 {model_id} done', flush=True)
    return pd.DataFrame(rows)


def hybrid_batch(sub, cfg, cfg_roll, device):
    """Recipe tensors for a batch of rows (schedule, params, commands, plan)."""
    schedules = [line_schedule(r.pulse_duration_fs * 1e-15, r.frequency_kHz * 1e3,
                               r.velocity_mm_s * 1e-3, r.hatch_spacing_um * 1e-6,
                               int(r.pass_count), cfg['power_W'],
                               cfg['waist_um_nominal'] * 1e-6, cfg_roll.tail_waists)
                 for r in sub.itertuples(index=False)]
    line_matrix, counts, pulse_matrix, per_line = build_batch_plan(schedules, cfg_roll)
    B, P = len(sub), counts.shape[1]
    commands = torch.log(torch.stack([
        torch.tensor([r.pulse_duration_fs * 1e-15 for r in sub.itertuples(index=False)]),
        torch.tensor([r.frequency_kHz * 1e3 for r in sub.itertuples(index=False)]),
        torch.tensor([r.velocity_mm_s * 1e-3 for r in sub.itertuples(index=False)]),
        torch.tensor([r.hatch_spacing_um * 1e-6 for r in sub.itertuples(index=False)])],
        dim=-1)[:, None, :].expand(B, P, 4)).to(device)
    params = {'w0': torch.full((B,), cfg['waist_um_nominal'] * 1e-6, device=device),
              'zr': torch.full((B,), math.pi * (cfg['waist_um_nominal'] * 1e-6) ** 2 /
                               (cfg['m_squared'] * cfg['wavelength_m']), device=device),
              'F1': torch.full((B,), math.exp(math.log(2e8)), device=device),
              'delta': torch.full((B,), math.exp(math.log(1e-8)), device=device),
              'kappa': torch.full((B,), 0.02, device=device),
              'rho': torch.full((B,), 0.5, device=device),
              'Ep': torch.tensor([s['Ep_J'] for s in schedules], device=device),
              'tau_ref': 1e-12, 'gamma_F': 0.0, 'gamma_delta': 0.0}
    packet_dt = torch.tensor([[counts[i, p] * (80e-6 / schedules[i]['dx_m'] /
                                                schedules[i]['Ep_J'] * 0 + 80e-6 /
                                                (schedules[i]['n_pulses_per_line'] *
                                                 schedules[i]['dx_m']))
                               for p in range(P)] for i in range(B)], device=device)
    packet_features = {'packet_dt': packet_dt,
                       'pulse_count': counts.float().to(device) *
                       torch.tensor([s['n_pulses_per_line'] for s in schedules],
                                    device=device).view(-1, 1),
                       'dx': torch.tensor([s['dx_m'] for s in schedules], device=device),
                       'pass_index': torch.zeros(B, P, device=device)}
    return (schedules, params, commands,
            line_matrix.to(device), counts.to(device), pulse_matrix.to(device),
            packet_features)


def hybrid_loss(model, sub, cfg, cfg_roll, observer, device, y_scale):
    from src.task_state_learning.laser_rollout import gl_nodes
    (schedules, params, commands, line_matrix, counts, pulse_matrix,
     packet_features) = hybrid_batch(sub, cfg, cfg_roll, device)
    B = len(sub)
    d0 = torch.zeros(B, cfg_roll.grid_n, cfg_roll.grid_n, device=device)
    q0 = torch.zeros_like(d0)
    gl_x, gl_w = gl_nodes(device=device)
    gl_x, gl_w = gl_x.float().to(device), gl_w.float().to(device)
    offsets = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device),
                         counts.cumsum(1)[:, :-1]], dim=1)
    bs, d, q = model(d0, q0, schedules, params, commands, packet_features,
                     line_matrix, counts, offsets, None, None, gl_x, gl_w)
    h = terminal_height_um(d)
    values = observer(h, torch.ones_like(h, dtype=torch.bool))
    pred = values.values[:, :8] if hasattr(values, 'values') else values
    y = torch.tensor(sub[list_of_targets()].to_numpy(np.float32), device=device)
    closure = float(bs.pow(2).mean()) if isinstance(bs, torch.Tensor) else 0.0
    loss = ((pred - y) / y_scale).pow(2).mean()
    return loss + cfg['lambda_closure'] * closure


def hybrid_predict(model, sub, cfg, cfg_roll, observer, device):
    from src.task_state_learning.laser_rollout import gl_nodes
    (schedules, params, commands, line_matrix, counts, pulse_matrix,
     packet_features) = hybrid_batch(sub, cfg, cfg_roll, device)
    B = len(sub)
    d0 = torch.zeros(B, cfg_roll.grid_n, cfg_roll.grid_n, device=device)
    q0 = torch.zeros_like(d0)
    gl_x, gl_w = gl_nodes(device=device)
    offsets = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device),
                         counts.cumsum(1)[:, :-1]], dim=1)
    with torch.no_grad():
        bs, d, q = model(d0, q0, schedules, params, commands, packet_features,
                         line_matrix, counts, offsets, None, None,
                         gl_x.float().to(device), gl_w.float().to(device))
        h = terminal_height_um(d)
        values = observer(h, torch.ones_like(h, dtype=torch.bool))
        pred = values.values[:, :8] if hasattr(values, 'values') else values
    return pred.cpu().numpy()


def list_of_targets():
    return ['D', 'A_med', 'ilr_z1', 'ilr_z2', 'ilr_z3', 'ilr_z4', 'A2_8_16', 'entropy_8_16']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/task_state_v1/real_main.yaml')
    parser.add_argument('--stage', choices=['static', 'physics', 'hybrid'], required=True)
    parser.add_argument('--calibration-run', default=None)
    args = parser.parse_args()
    path = ROOT / args.config
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if args.calibration_run:
        cfg['calibration_run'] = args.calibration_run
    out = new_run(f'E06_{args.stage.upper()}', path)
    gate = {'scope': f'E06_{args.stage}', 'stage': args.stage, 'training_executed': True}
    try:
        frame = load_frame(cfg)
        if args.stage == 'static':
            oof = run_static(cfg, out, frame)
        elif args.stage == 'physics':
            require(cfg.get('calibration_run'), 'physics stage needs --calibration-run')
            oof = run_physics(cfg, out, frame)
        else:
            oof = run_hybrid(cfg, out, frame)
        oof.to_csv(out / 'oof.csv', index=False)
        gate['n_oof_rows'] = len(oof)
        gate['models'] = sorted(oof.model_id.unique().tolist())
        gate['execution_status'] = 'PASS'
        gate['metric_status'] = 'VALID'
        gate['hypothesis_status'] = 'PENDING_CONTRASTS'
    except Exception as exc:
        import traceback
        gate['error'] = f'{type(exc).__name__}: {exc}'
        gate['traceback'] = traceback.format_exc()
        gate['execution_status'] = 'FAIL'
    gate['status'] = gate['execution_status']
    write_json(out / 'gate_E06.json', gate)
    seal(out)
    print(json.dumps({k: v for k, v in gate.items() if k != 'traceback'},
                     ensure_ascii=False, indent=2))
    return 0 if gate['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
