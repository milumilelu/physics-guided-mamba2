"""E03 Track R: per-fold effective-physics calibration over MAIN180, no labels from test.

For each outer fold x switch x waist scenario the effective parameter vector is
calibrated on its own training subset (whole-component search, followed by
full-subset refinement). Three inner calibration units accompany each outer refit. Outer-test
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
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from src.task_state_learning.artifacts import ROOT, new_run, write_json, seal, verify_seal
from src.task_state_learning.contracts import load_contract
from src.task_state_learning.grouping import components, require
from src.task_state_learning.physics import PhysicsParameters



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
    names=parameter_names(switch)
    limits=[cfg['parameter_bounds'][name] for name in names]
    require(len(vector)==len(limits) and np.isfinite(vector).all(),'Invalid calibration vector')
    require(all(lo<=value<=hi for value,(lo,hi) in zip(vector,limits)),'Calibration parameter out of bounds')
    log_F1, log_delta, kappa, rho, gamma_F, gamma_delta = decode(vector, switch)
    return PhysicsParameters(waist_m, cfg['wavelength_m'], cfg['m_squared'],
                             math.exp(log_F1), math.exp(log_delta), kappa, rho,
                             gamma_F=gamma_F, gamma_delta=gamma_delta,
                             reference_duration_s=cfg['reference_duration_s'])


def parameter_names(switch):
    require(switch in ('P00','P10','P01','P11'),'Unknown switch')
    return ['log_F1_J_m2','log_delta_m']+(['log_kappa','logit_rho'] if switch in ('P01','P11') else [])+['gamma_F','gamma_delta']


def stable_seed(*identity):
    return int.from_bytes(hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).digest()[:4],'big')


def simulate_row(params,row,switch,power_W,cfg):
    from dataclasses import replace
    from src.task_state_learning.finite_scan import FiniteScanConfig,simulate_roi
    finite_cfg=FiniteScanConfig()
    recipe=dict(tau_s=row.pulse_duration_fs*1e-15,f_Hz=row.frequency_kHz*1e3,
                v_m_s=row.velocity_mm_s*1e-3,h_m=row.hatch_spacing_um*1e-6,
                passes=row.pass_count,power_W=power_W,switch=switch)
    values=[]
    for tail in (6.,8.):
        height,_,info=simulate_roi(params,cfg=replace(finite_cfg,tail_waists=tail),**recipe)
        median=float(np.median(height))
        values.append(np.array([-median,np.sqrt(np.mean((height-median)**2))]))
    errors=np.abs(values[0]-values[1])/np.maximum(np.abs(values[1]),1e-6)
    require(np.isfinite(errors).all() and (errors<=.01).all(),'Finite ROI tail convergence failed')
    info['tail_convergence_validated']=True
    info['tail_relative_errors']=errors.tolist()
    return float(values[1][0]),float(values[1][1]),info


class Objective:
    def __init__(self, frame, switch, waist_m, cfg):
        self.frame = frame.reset_index(drop=True)
        self.switch = switch
        self.waist_m = waist_m
        self.cfg = cfg
        self.cell_cfg = None
        require(len(self.frame)>0,'Empty calibration training subset')
        counts=self.frame.groupby('component_id').dataset_index.transform('count').to_numpy()
        self.weights=1./(self.frame.component_id.nunique()*counts)
        self.n = 0
        self.failures = 0
        d = self.frame.D.to_numpy()
        a = self.frame.A_med.to_numpy()
        self.sD = max(float(np.sqrt(np.sum(self.weights*(d-np.sum(self.weights*d))**2))), 1e-6)
        self.sA = max(float(np.sqrt(np.sum(self.weights*(a-np.sum(self.weights*a))**2))), 1e-6)

    def __call__(self, vector):
        self.n += 1
        try:
            params = physics_params(vector, self.switch, self.waist_m, self.cfg)
        except (ValueError, OverflowError):
            return 1e6
        total, count = 0.0, 0
        for weight,row in zip(self.weights,self.frame.itertuples()):
            try:
                d, a, _ = simulate_row(params, row, self.switch, self.cfg['power_W'],
                                       self.cell_cfg)
                if not (math.isfinite(d) and math.isfinite(a)):
                    raise ValueError('nonfinite')
            except (ValueError, FloatingPointError, OverflowError):
                self.failures += 1
                return 1e6
            total += weight*(((d - row.D) / self.sD) ** 2 + ((a - row.A_med) / self.sA) ** 2)
            count += 1
        return total / 2


def starts(switch, rng, cfg):
    mid = {k: (v[0] + v[1]) / 2 for k, v in cfg['parameter_bounds'].items()}
    base = [mid['log_F1_J_m2'], mid['log_delta_m']]
    if switch in ('P01', 'P11'):
        base += [mid['log_kappa'], mid['logit_rho']]
    base += [0.0, 0.0]
    starts = [np.array(base)]
    for _ in range(cfg['calibration']['n_starts']-1):
        perturb = base.copy()
        for i, key in enumerate(['log_F1_J_m2', 'log_delta_m'] +
                                (['log_kappa', 'logit_rho'] if switch in ('P01', 'P11') else []) +
                                ['gamma_F', 'gamma_delta']):
            lo, hi = cfg['parameter_bounds'][key]
            perturb[i] = rng.uniform(lo + 0.15 * (hi - lo), hi - 0.15 * (hi - lo))
        starts.append(np.array(perturb))
    return starts


def calibrate(frame, switch, waist_m, cfg, seed, maxfev, subset_ids=None):
    if subset_ids is not None:
        frame = frame[frame.dataset_index.isin(subset_ids)]
    rng = np.random.default_rng(seed)
    best = None
    best_info = {'evals': 0, 'failures': 0}
    for start in starts(switch, rng, cfg):
        obj = Objective(frame, switch, waist_m, cfg)
        result = minimize(obj, start, method='Nelder-Mead',
                          bounds=[cfg['parameter_bounds'][k] for k in parameter_names(switch)],
                          options={'maxfev': maxfev, 'xatol': 1e-3, 'fatol': 1e-4})
        if best is None or result.fun < best.fun:
            best = result
        best_info['evals'] += obj.n
        best_info['failures'] += obj.failures
    return best, best_info


def job(payload):
    import numba
    numba.set_num_threads(1)
    fold, switch, waist_um, cfg, train_ids, test_ids, subset_ids, refine_ids = payload
    cfg_run, frozen, df, _, _ = load_contract()
    df=components(df)
    main = df[df.session_role.isin(['formal','pass_main'])].copy()
    train = main[main.dataset_index.isin(train_ids)]
    test = main[main.dataset_index.isin(test_ids)]
    waist_m = waist_um * 1e-6
    best, info = calibrate(train, switch, waist_m, cfg, seed=stable_seed(fold,switch,waist_um,sorted(train_ids)),
                           maxfev=cfg['calibration']['maxfev_search'],
                           subset_ids=subset_ids)
    vector = best.x
    refine_obj = Objective(train, switch, waist_m, cfg)
    baseline_full=refine_obj(vector)
    result = minimize(refine_obj, vector, method='Nelder-Mead',
                      bounds=[cfg['parameter_bounds'][k] for k in parameter_names(switch)],
                      options={'maxfev': cfg['calibration']['maxfev_refine_full_train'],
                               'xatol': 1e-3, 'fatol': 1e-4})
    final_vector = result.x if result.fun <= baseline_full else vector
    params = physics_params(final_vector, switch, waist_m, cfg)
    rows = []
    cell_cfg = None
    for split_name, frame in (('outer_train', train), ('outer_test', test)):
        for row in frame.itertuples():
            try:
                d, a, sim_info = simulate_row(params, row, switch, cfg['power_W'], cell_cfg)
            except (ValueError, FloatingPointError, OverflowError) as exc:
                d, a, sim_info = float('nan'), float('nan'), {'solver_failure':
                                                              f'{type(exc).__name__}'}
            rows.append({'outer_fold': fold, 'switch': switch, 'waist_um': waist_um,
                         'dataset_index': row.dataset_index, 'split': split_name,
                         'D_sim_um': d, 'A_sim_um': a, 'D_meas_um': row.D,
                         'A_meas_um': row.A_med,
                         'finite': bool(math.isfinite(d) and math.isfinite(a))})
    lo = {k: v[0] for k, v in cfg['parameter_bounds'].items()}
    hi = {k: v[1] for k, v in cfg['parameter_bounds'].items()}
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
               'objective_subset': float(best.fun), 'objective_full_train': float(min(result.fun, baseline_full)),
               'search_evals': info['evals'], 'refine_evals':refine_obj.n, 'refine_failures':refine_obj.failures,
               'seed':stable_seed(fold,switch,waist_um,sorted(train_ids)),
               'optimizer_success':bool(result.success), 'optimizer_message':str(result.message),
               'n_train':len(train), 'search_failures': info['failures'],
               'vector': [float(v) for v in final_vector],
               'n_boundary_hits': len(hits)}
    return rows, hits, summary


def profile_job(payload):
    fold, cfg, train_ids, final_vector = payload
    cfg_run, frozen, df, _, _ = load_contract()
    df=components(df)
    main = df[df.session_role.isin(['formal', 'pass_main'])]
    train = main[main.dataset_index.isin(train_ids)]
    frame = train
    waist_um = cfg['identifiability']['waist_um']
    waist_m = waist_um * 1e-6
    base = np.array(final_vector)
    names = ['log_F1_J_m2', 'log_delta_m', 'log_kappa', 'logit_rho', 'gamma_F', 'gamma_delta']
    rows = []
    for i, name in enumerate(names):
        lo, hi = cfg['parameter_bounds'][name]
        center = np.clip(base[i], lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo))
        for delta in cfg['identifiability']['profile_grid']:
            value = float(np.clip(center + delta * (hi - lo) / 1.2, lo, hi))
            vector = base.copy()
            vector[i] = value
            obj = Objective(frame, 'P11', waist_m, cfg)
            free=[j for j in range(len(base)) if j!=i]
            def restricted(free_vector):
                current=vector.copy();current[free]=free_vector
                return obj(current)
            result=minimize(restricted,vector[free],method='Nelder-Mead',
                            bounds=[cfg['parameter_bounds'][names[j]] for j in free],
                            options={'maxfev':cfg['calibration']['maxfev_refine_full_train'],
                                     'xatol':1e-3,'fatol':1e-4})
            score=result.fun
            rows.append({'outer_fold': fold, 'parameter': name, 'offset': delta,
                         'value': value, 'objective': float(score), 'nuisance_refitted':True, 'valid_score':bool(score<1e6 and np.isfinite(score)),
                         'optimizer_success':bool(result.success),'nfev':int(result.nfev),
                         'at_boundary': bool(abs(value - lo) < 1e-9 or abs(value - hi) < 1e-9)})
    return pd.DataFrame(rows)


def build_jobs(main,manifest,inner,cfg):
    payloads=[]
    for fold in range(5):
        train_ids=manifest.loc[manifest.outer_fold!=fold,'dataset_index'].tolist()
        test_ids=manifest.loc[manifest.outer_fold==fold,'dataset_index'].tolist()
        nested=inner[inner.outer_fold==fold]
        require(set(nested.dataset_index)==set(train_ids),'Inner split coverage mismatch')
        levels=[(-1,train_ids,test_ids)]
        for inner_fold in sorted(nested.inner_fold.unique()):
            levels.append((int(inner_fold),nested.loc[nested.inner_fold!=inner_fold,'dataset_index'].tolist(),
                           nested.loc[nested.inner_fold==inner_fold,'dataset_index'].tolist()))
        require(len(levels)==4,'Expected three inner folds plus outer refit')
        for inner_fold,fit_ids,eval_ids in levels:
            fit=main[main.dataset_index.isin(fit_ids)]
            validation=main[main.dataset_index.isin(eval_ids)]
            require(not set(fit.component_id)&set(validation.component_id),'Component leakage')
            selected=[]
            for _,group in fit.groupby('component_id',sort=True):
                if len(selected)+len(group)<=cfg['calibration']['subset_recipes'] or not selected:
                    selected.extend(group.dataset_index.tolist())
            for switch in cfg['switches']:
                for waist in cfg['waist_um']:
                    ident=f'outer{fold}_inner{inner_fold}_{switch}_waist{waist}'
                    payloads.append((ident,inner_fold,(fold,switch,waist,cfg,fit_ids,eval_ids,selected,fit_ids)))
    require(len(payloads)==240,'Incomplete outer/inner scenario matrix')
    return payloads


def main():
    from src.task_state_learning.guardrails import require_not_held
    require_not_held('E03_CALIBRATION')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/task_state_v1/physics_calibration.yaml')
    args = parser.parse_args()
    path = ROOT / args.config
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    out = new_run('E03_CALIBRATION', path)
    gate = {'scope': cfg['scope'], 'training_executed': False,
            'single_line_track': cfg['single_line_track'], 'E03_complete':False}
    try:
        registration=cfg.get('solver_registration_run')
        require(registration,'Missing finite ROI solver registration; calibration remains prohibited')
        registered=verify_seal(ROOT/'outputs/task_state_v1'/registration)
        solver_gate=json.loads((registered/'solver_registration_gate.json').read_text(encoding='utf-8'))
        require(solver_gate.get('status')=='PASS' and solver_gate.get('allowed_to_calibrate') is True,
                'Solver registration does not authorize calibration')
        from src.task_state_learning.registration import calibration_fingerprint
        require(solver_gate.get('calibration_semantic_sha256') == calibration_fingerprint(cfg),
                'Calibration configuration differs from solver registration')
        inputs = json.loads((registered/'registration_inputs.json').read_text(encoding='utf-8'))
        require(inputs['sha256'] == solver_gate.get('input_sha256'), 'Registration input identity mismatch')
        for item in inputs['files']:
            require(hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest() == item['sha256'],
                    'Input changed after solver registration: '+item['path'])
        snapshot = json.loads((registered/'source_snapshot.json').read_text(encoding='utf-8'))
        for item in snapshot['files']:
            if item['path'].startswith('src/task_state_learning/') or item['path'] in (
                    'experiments/task_state_v1/03_physics_calibration.py',
                    'experiments/task_state_v1/03_register_solver.py'):
                require(hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest() == item['sha256'],
                        'Implementation changed after solver registration: '+item['path'])
        cfg_run, frozen, df, _, _ = load_contract()
        df = components(df)  # identity-stable dependency components (no labels used)
        main = df[df.session_role.isin(['formal', 'pass_main'])].copy()
        split_dir = ROOT / 'outputs/task_state_v1' / cfg['split_run']
        manifest = pd.read_csv(split_dir / 'split_manifest.csv')
        inner = pd.read_csv(split_dir / 'inner_split_manifest.csv')
        require(set(manifest.dataset_index) == set(main.dataset_index), 'Split manifest identity')
        require(len(manifest) == 180 and manifest.outer_fold.nunique() == 5, 'Split shape')
        payloads=build_jobs(main,manifest,inner,cfg)
        all_rows,all_hits,summaries=[],[],[]
        (out/'jobs').mkdir(exist_ok=True)
        write_json(out/'job_plan.json',[{'job_id':i,'inner_fold':n,'train_ids':p[4],'validation_ids':p[5]} for i,n,p in payloads])
        with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            futures={pool.submit(job,p):(ident,inner_fold) for ident,inner_fold,p in payloads}
            for future in as_completed(futures):
                ident,inner_fold=futures[future]
                try:
                    rows,hits,summary=future.result()
                    for row in rows+hits+[summary]:row['inner_fold']=inner_fold;row['job_id']=ident
                    if inner_fold!=-1:
                        for row in rows:row['split']='inner_train' if row['split']=='outer_train' else 'inner_validation'
                    # Failures stay explicit and JSON-valid; no successful row is dropped.
                    for row in rows:
                        if not row['finite']:row['D_sim_um']=None;row['A_sim_um']=None
                    result={'status':'PASS' if all(r['finite'] for r in rows) else 'FAIL',
                            'rows':rows,'hits':hits,'summary':summary}
                    write_json(out/'jobs'/f'{ident}.json',result)
                    all_rows.extend(rows);all_hits.extend(hits);summaries.append(summary)
                except Exception as exc:
                    write_json(out/'jobs'/f'{ident}.json',{'status':'FAIL','error':f'{type(exc).__name__}: {exc}'})
                print(f'completed {ident} ({len(summaries)}/{len(payloads)} returned)',flush=True)
        require(len(summaries)==len(payloads),'One or more calibration jobs failed; see jobs/')
        pd.DataFrame(all_rows).to_csv(out / 'physics_calibration_by_fold.csv', index=False)
        pd.DataFrame(all_hits).to_csv(out / 'parameter_boundary_hits.csv', index=False)
        pd.DataFrame(summaries).to_csv(out / 'calibration_summary.csv', index=False)
        profile_rows = []
        prof_payloads=[(row['outer_fold'],cfg,
                        manifest.loc[manifest.outer_fold!=row['outer_fold'],'dataset_index'].tolist(),row['vector'])
                       for row in summaries if row['inner_fold']==-1 and row['switch']=='P11'
                       and row['waist_um']==cfg['identifiability']['waist_um']]
        with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            for frame in pool.map(profile_job, prof_payloads):
                profile_rows.append(frame)
        profiles = pd.concat(profile_rows, ignore_index=True)
        profiles.to_csv(out / 'parameter_profiles.csv', index=False)
        # flat profile detection per fold/parameter
        flat = []
        for (fold, name), group in profiles.groupby(['outer_fold', 'parameter']):
            valid=group[group.valid_score]
            spread = valid.objective.max() - valid.objective.min() if len(valid)>=3 else None
            flat.append({'outer_fold': fold, 'parameter': name, 'objective_spread': None if spread is None else float(spread),
                         'metric_status':'VALID' if spread is not None else 'NOT_ESTIMABLE',
                         'flat': None if spread is None else bool(spread/max(abs(valid.objective.min()),1e-12) <= cfg['identifiability']['flat_profile_rtol'])})
        pd.DataFrame(flat).to_csv(out / 'flat_profiles.csv', index=False)
        # switch ablation table (per-sample sim vs measured)
        calib = pd.DataFrame(all_rows)
        ablation = calib.groupby(['outer_fold','inner_fold','switch','waist_um','split']).apply(
            lambda g: pd.Series({
                'D_rmse_um': float(np.sqrt(np.mean((g.D_sim_um - g.D_meas_um) ** 2))),
                'A_rmse_um': float(np.sqrt(np.mean((g.A_sim_um - g.A_meas_um) ** 2))),
                'D_MAE_um': float(np.mean(np.abs(g.D_sim_um - g.D_meas_um))),
                'n': len(g), 'finite_rate': float(g.finite.mean())})).reset_index()
        ablation.to_csv(out / 'physics_switch_ablation.csv', index=False)
        require(calib.finite.all(),'Nonfinite predictions prevent calibration PASS')
        require(all(s['objective_full_train']<1e6 for s in summaries),'Invalid calibration objective')
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
