"""E03 Track R: per-fold effective-physics calibration over MAIN180, no labels from test.

For each outer fold x switch x waist scenario the effective parameter vector is
calibrated on the outer-train subset only (objective on a deterministic whole-
component subset of 64 recipes, then refined on the full outer-train). Outer-test
recipes are simulated once with the frozen parameters for the E06 physics
ablation. Identifiability: coordinate profiles (P11/nominal), boundary hits,
optimizer-path flatness. Single-line Track SL is a separate registered run.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from src.task_state_learning.artifacts import ROOT, new_run, write_json, seal
from src.task_state_learning.contracts import load_contract
from src.task_state_learning.grouping import require
from src.task_state_learning.physics import PhysicsParameters
from src.task_state_learning.scan_cell import CellConfig, simulate_sample, simulate_sample_checked

BASE_PARAMS = dict(wavelength_m=1.03e-6, m_squared=1.2, reference_duration_s=1e-12)
BOUNDS = {'log_F1_J_m2': (4.605, 16.118), 'log_delta_m': (-27.631, -13.816),
          'log_kappa': (-9.210, 0.693), 'logit_rho': (-6.0, 6.0),
          'gamma_F': (-1.0, 1.0), 'gamma_delta': (-1.0, 1.0)}


def decode(vector, switch):
    if switch in ('P00', 'P10'):
        log_F1, log_delta, gamma_F, gamma_delta = vector
        kappa, rho = 0.0, 0.3
    else:
        log_F1, log_delta, log_kappa, logit_rho, gamma_F, gamma_delta = vector
        kappa = math.exp(log_kappa)
        rho = 1.0 / (1.0 + math.exp(-logit_rho))
    return log_F1, log_delta, kappa, rho, gamma_F, gamma_delta


def physics_params(vector, switch, waist_m, cfg):
    log_F1, log_delta, kappa, rho, gamma_F, gamma_delta = decode(vector, switch)
    return PhysicsParameters(waist_m, cfg['wavelength_m'], cfg['m_squared'],
                             math.exp(log_F1), math.exp(log_delta), kappa, rho,
                             gamma_F=gamma_F, gamma_delta=gamma_delta,
                             reference_duration_s=cfg['reference_duration_s'])


def simulate_row(params, row, switch, power_W):
    d, a, info = simulate_sample(
        params, tau_s=row.pulse_duration_fs * 1e-15, f_Hz=row.frequency_kHz * 1e3,
        v_m_s=row.velocity_mm_s * 1e-3, h_m=row.hatch_spacing_um * 1e-6,
        passes=int(row.pass_count), power_W=power_W, switch=switch)
    return d * 1e6, a * 1e6, info


class Objective:
    def __init__(self, frame, switch, waist_m, cfg):
        self.frame = frame.reset_index(drop=True)
        self.switch = switch
        self.waist_m = waist_m
        self.cfg = cfg
        self.n = 0
        self.failures = 0
        d = self.frame.D.to_numpy()
        a = self.frame.A_med.to_numpy()
        self.sD = max(float(d.std()), 1e-6)
        self.sA = max(float(a.std()), 1e-6)

    def __call__(self, vector):
        self.n += 1
        try:
            params = physics_params(vector, self.switch, self.waist_m, self.cfg)
        except (ValueError, OverflowError):
            return 1e6
        total, count = 0.0, 0
        for row in self.frame.itertuples():
            try:
                d, a, _ = simulate_row(params, row, self.switch, self.cfg['power_W'])
                if not (math.isfinite(d) and math.isfinite(a)):
                    raise ValueError('nonfinite')
            except (ValueError, FloatingPointError, OverflowError):
                self.failures += 1
                return 1e6
            total += ((d - row.D) / self.sD) ** 2 + ((a - row.A_med) / self.sA) ** 2
            count += 1
        return total / (2 * count)


def starts(switch, rng):
    mid = {k: (v[0] + v[1]) / 2 for k, v in BOUNDS.items()}
    base = [mid['log_F1_J_m2'], mid['log_delta_m']]
    if switch in ('P01', 'P11'):
        base += [mid['log_kappa'], mid['logit_rho']]
    base += [0.0, 0.0]
    starts = [np.array(base)]
    for _ in range(2):
        perturb = base.copy()
        for i, key in enumerate(['log_F1_J_m2', 'log_delta_m'] +
                                (['log_kappa', 'logit_rho'] if switch in ('P01', 'P11') else []) +
                                ['gamma_F', 'gamma_delta']):
            lo, hi = BOUNDS[key]
            perturb[i] = rng.uniform(lo + 0.15 * (hi - lo), hi - 0.15 * (hi - lo))
        starts.append(np.array(perturb))
    return starts


def calibrate(frame, switch, waist_m, cfg, seed, maxfev, subset_ids=None):
    if subset_ids is not None:
        frame = frame[frame.dataset_index.isin(subset_ids)]
    rng = np.random.default_rng(seed)
    best = None
    for start in starts(switch, rng):
        obj = Objective(frame, switch, waist_m, cfg)
        result = minimize(obj, start, method='Nelder-Mead',
                          options={'maxfev': maxfev, 'xatol': 1e-3, 'fatol': 1e-4})
        if best is None or result.fun < best.fun:
            best = result
            best_info = {'evals': obj.n, 'failures': obj.failures}
    return best, best_info


def job(payload):
    fold, switch, waist_um, cfg, train_ids, test_ids, subset_ids = payload
    cfg_run, frozen, df, _, _ = load_contract()
    main = df[df.session_role.isin(['formal', 'pass_main'])].copy()
    train = main[main.dataset_index.isin(train_ids)]
    test = main[main.dataset_index.isin(test_ids)]
    waist_m = waist_um * 1e-6
    best, info = calibrate(train, switch, waist_m, cfg, seed=hash((fold, switch, waist_um)) % 2**31,
                           maxfev=cfg['calibration']['maxfev_search'],
                           subset_ids=subset_ids)
    vector = best.x
    full_obj = Objective(train, switch, waist_m, cfg)
    result = minimize(full_obj, vector, method='Nelder-Mead',
                      options={'maxfev': cfg['calibration']['maxfev_refine_full_train'],
                               'xatol': 1e-3, 'fatol': 1e-4})
    final_vector = result.x if result.fun <= best.fun else vector
    params = physics_params(final_vector, switch, waist_m, cfg)
    rows = []
    for split_name, frame in (('outer_train', train), ('outer_test', test)):
        for row in frame.itertuples():
            d, a, info = simulate_row(params, row, switch, cfg['power_W'])
            rows.append({'outer_fold': fold, 'switch': switch, 'waist_um': waist_um,
                         'dataset_index': row.dataset_index, 'split': split_name,
                         'D_sim_um': d, 'A_sim_um': a, 'D_meas_um': row.D,
                         'A_meas_um': row.A_med,
                         'finite': bool(math.isfinite(d) and math.isfinite(a))})
    lo = {k: v[0] for k, v in BOUNDS.items()}
    hi = {k: v[1] for k, v in BOUNDS.items()}
    hits = []
    names = (['log_F1_J_m2', 'log_delta_m', 'log_kappa', 'logit_rho', 'gamma_F', 'gamma_delta']
             if switch in ('P01', 'P11') else
             ['log_F1_J_m2', 'log_delta_m', 'gamma_F', 'gamma_delta'])
    tol = cfg['identifiability']['boundary_tolerance']
    for i, name in enumerate(names):
        span = hi[name] - lo[name]
        for bound, side in ((lo[name], 'lower'), (hi[name], 'upper')):
            if abs(final_vector[i] - bound) <= tol * span:
                hits.append({'outer_fold': fold, 'switch': switch, 'waist_um': waist_um,
                             'parameter': name, 'boundary': side,
                             'value': float(final_vector[i])})
    summary = {'outer_fold': fold, 'switch': switch, 'waist_um': waist_um,
               'objective_subset': float(best.fun), 'objective_full_train': float(min(result.fun, best.fun)),
               'search_evals': info['evals'], 'search_failures': info['failures'],
               'vector': [float(v) for v in final_vector],
               'n_boundary_hits': len(hits)}
    return rows, hits, summary


def profile_job(payload):
    fold, cfg, train_ids, subset_ids = payload
    cfg_run, frozen, df, _, _ = load_contract()
    main = df[df.session_role.isin(['formal', 'pass_main'])]
    train = main[main.dataset_index.isin(train_ids)]
    frame = train[train.dataset_index.isin(subset_ids)]
    waist_um = cfg['identifiability']['waist_um']
    waist_m = waist_um * 1e-6
    best, info = calibrate(train, 'P11', waist_m, cfg, seed=fold, maxfev=cfg['calibration']['maxfev_search'],
                           subset_ids=subset_ids)
    base = np.array(best.x)
    names = ['log_F1_J_m2', 'log_delta_m', 'log_kappa', 'logit_rho', 'gamma_F', 'gamma_delta']
    rows = []
    for i, name in enumerate(names):
        lo, hi = BOUNDS[name]
        center = np.clip(base[i], lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo))
        for delta in cfg['identifiability']['profile_grid']:
            value = float(np.clip(center + delta * (hi - lo) / 1.2, lo, hi))
            vector = base.copy()
            vector[i] = value
            obj = Objective(frame, 'P11', waist_m, cfg)
            score = obj(vector)
            rows.append({'outer_fold': fold, 'parameter': name, 'offset': delta,
                         'value': value, 'objective': float(score),
                         'at_boundary': bool(abs(value - lo) < 1e-9 or abs(value - hi) < 1e-9)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/task_state_v1/physics_calibration.yaml')
    args = parser.parse_args()
    path = ROOT / args.config
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    out = new_run('E03_CALIBRATION', path)
    gate = {'scope': cfg['scope'], 'training_executed': False,
            'single_line_track': cfg['single_line_track']}
    try:
        cfg_run, frozen, df, _, _ = load_contract()
        main = df[df.session_role.isin(['formal', 'pass_main'])].copy()
        split_dir = ROOT / 'outputs/task_state_v1' / cfg['split_run']
        manifest = pd.read_csv(split_dir / 'split_manifest.csv')
        inner = pd.read_csv(split_dir / 'inner_split_manifest.csv')
        require(set(manifest.dataset_index) == set(range(200)), 'Split manifest identity')
        comp = manifest[['dataset_index', 'component_id']]
        train_pool = main[main.dataset_index.isin(
            manifest.loc[manifest.outer_fold != 0, 'dataset_index'])]
        rng = np.random.default_rng(cfg['calibration']['subset_seed'])
        order = list(train_pool.groupby('component_id').size().sort_values(ascending=False).index)
        picked, count = [], 0
        for key in order:
            members = train_pool[train_pool.component_id == key].dataset_index.tolist()
            if count + len(members) <= cfg['calibration']['subset_recipes'] or not picked:
                picked.extend(members)
                count += len(members)
        subset_ids = sorted(picked)
        payloads = []
        for fold in range(5):
            train_ids = manifest.loc[manifest.outer_fold != fold, 'dataset_index'].tolist()
            test_ids = manifest.loc[manifest.outer_fold == fold, 'dataset_index'].tolist()
            inner_ids = inner.loc[inner.outer_fold == fold, 'dataset_index'].tolist()
            require(set(inner_ids) <= set(train_ids), 'Inner split outside outer train')
            for switch in cfg['switches']:
                for waist_um in cfg['waist_um']:
                    payloads.append((fold, switch, waist_um, cfg, train_ids, test_ids, subset_ids))
        all_rows, all_hits, summaries = [], [], []
        with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            for rows, hits, summary in pool.map(job, payloads):
                all_rows.extend(rows)
                all_hits.extend(hits)
                summaries.append(summary)
        pd.DataFrame(all_rows).to_csv(out / 'physics_calibration_by_fold.csv', index=False)
        pd.DataFrame(all_hits).to_csv(out / 'parameter_boundary_hits.csv', index=False)
        pd.DataFrame(summaries).to_csv(out / 'calibration_summary.csv', index=False)
        profile_rows = []
        prof_payloads = [(fold, cfg, manifest.loc[manifest.outer_fold != fold, 'dataset_index'].tolist(),
                          subset_ids) for fold in range(5)]
        with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            for frame in pool.map(profile_job, prof_payloads):
                profile_rows.append(frame)
        profiles = pd.concat(profile_rows, ignore_index=True)
        profiles.to_csv(out / 'parameter_profiles.csv', index=False)
        # flat profile detection per fold/parameter
        flat = []
        for (fold, name), group in profiles.groupby(['outer_fold', 'parameter']):
            spread = group.objective.max() - group.objective.min()
            flat.append({'outer_fold': fold, 'parameter': name, 'objective_spread': float(spread),
                         'flat': bool(spread <= cfg['identifiability']['flat_profile_rtol'])})
        pd.DataFrame(flat).to_csv(out / 'flat_profiles.csv', index=False)
        # switch ablation table (per-sample sim vs measured)
        calib = pd.DataFrame(all_rows)
        ablation = calib.groupby(['outer_fold', 'switch', 'waist_um', 'split']).apply(
            lambda g: pd.Series({
                'D_rmse_um': float(np.sqrt(np.mean((g.D_sim_um - g.D_meas_um) ** 2))),
                'A_rmse_um': float(np.sqrt(np.mean((g.A_sim_um - g.A_meas_um) ** 2))),
                'D_MAE_um': float(np.mean(np.abs(g.D_sim_um - g.D_meas_um))),
                'n': len(g), 'finite_rate': float(g.finite.mean())})).reset_index()
        ablation.to_csv(out / 'physics_switch_ablation.csv', index=False)
        gate['execution_status'] = 'PASS'
        gate['metric_status'] = 'VALID'
        gate['hypothesis_status'] = 'PENDING_IDENTIFIABILITY_REVIEW'
        gate['n_jobs'] = len(payloads)
        gate['failures_retained'] = True
    except Exception as exc:
        import traceback
        gate['error'] = f'{type(exc).__name__}: {exc}'
        gate['traceback'] = traceback.format_exc()
        gate['execution_status'] = 'FAIL'
    gate['status'] = gate['execution_status']
    write_json(out / 'gate_E03.json', gate)
    seal(out)
    print(json.dumps({k: v for k, v in gate.items() if k != 'traceback'}, ensure_ascii=False, indent=2))
    return 0 if gate['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
