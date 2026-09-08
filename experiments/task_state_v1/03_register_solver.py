"""E03 solver registration: MAIN180 recipe coverage + calibration parameter domain.

This is the hard entry gate for E03 calibration. It performs NO parameter
fitting. It answers two registration questions that the 2026-09-08 audit left
open (`remaining: [MAIN180_recipe_coverage, calibration_parameter_domain,
all_switch_grid_refinement]`):

1. Recipe coverage - for every MAIN180 recipe, every physics switch and every
   registered waist, does the finite-ROI solver return finite canonical
   descriptors whose tail truncation (6 w vs 8 w) is converged?
2. Parameter domain - over the frozen calibration box (vertices + Latin
   hypercube), does the solver stay finite and converged, and does
   `03_physics_calibration.physics_params` actually reject out-of-box vectors?

Supporting checks: grid refinement (640 -> 1280) for all four switches, and
bitwise determinism of a repeated case.

Output: outputs/task_state_v1/<run>/solver_registration_gate.json. The
calibration runner refuses to start unless status == PASS and
allowed_to_calibrate is true.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import importlib.util
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.stats.qmc import LatinHypercube

from src.task_state_learning.artifacts import ROOT, new_run, write_json, seal, input_manifest, verify_seal
from src.task_state_learning.contracts import load_contract
from src.task_state_learning.finite_scan import FiniteScanConfig, simulate_roi, simulate_points
from src.task_state_learning.grouping import require
from src.task_state_learning.observers import CanonicalObserver
from src.task_state_learning.registration import calibration_fingerprint, complete_jobs

TAU_REF = 1.0e-12


def load_calibration_module():
    """Import the calibration runner so registration uses its exact domain logic."""
    path = ROOT / 'experiments/task_state_v1/03_physics_calibration.py'
    spec = importlib.util.spec_from_file_location('e03_calibration_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observe(height_um):
    values, valid = CanonicalObserver()(height_um[None], np.ones((1, 160, 160), bool))
    return values.iloc[0].to_dict(), valid


def compare(reference, other, scales, tolerance):
    rows = []
    for key, scale in scales.items():
        x, y = float(reference[key]), float(other[key])
        if not (math.isfinite(x) and math.isfinite(y)):
            rows.append({'descriptor': key, 'error': None, 'passed': False,
                         'reason': 'UNDEFINED_DESCRIPTOR'})
        else:
            error = abs(x - y) / max(abs(y), scale)
            rows.append({'descriptor': key, 'error': error,
                         'passed': bool(error <= tolerance), 'reason': ''})
    return rows


def recipe_from_row(row, switch, power_W):
    return dict(tau_s=float(row.pulse_duration_fs) * 1e-15,
                f_Hz=float(row.frequency_kHz) * 1e3,
                v_m_s=float(row.velocity_mm_s) * 1e-3,
                h_m=float(row.hatch_spacing_um) * 1e-6,
                passes=int(row.pass_count), power_W=power_W, switch=switch)


def nominal_vector(switch, fixture):
    base = [fixture['log_F1_J_m2'], fixture['log_delta_m']]
    if switch in ('P01', 'P11'):
        base += [fixture['log_kappa'], fixture['logit_rho']]
    return np.array(base + [fixture['gamma_F'], fixture['gamma_delta']], dtype=float)


# --------------------------------------------------------------------------- #
# worker side
# --------------------------------------------------------------------------- #
def _worker_init(threads, calibration_cfg):
    import numba
    numba.set_num_threads(threads)
    global CAL, CAL_CFG
    CAL = load_calibration_module()
    CAL_CFG = calibration_cfg


def _simulate(params, recipe, tails):
    """Return {tail: (height, descriptors, validity, info)} for each truncation."""
    out = {}
    for tail in tails:
        height, q, info = simulate_roi(params, cfg=replace(FiniteScanConfig(), tail_waists=tail), **recipe)
        row, valid = observe(height)
        out[tail] = (height, row, valid, info, q)
    return out


def recipe_case(payload):
    """MAIN180 x switch x waist coverage at the nominal registration fixture."""
    ident, row_values, switch, waist_um, fixture, cfg = payload
    row = pd.Series(row_values)
    recipe = recipe_from_row(row, switch, cfg['power_W'])
    vector = nominal_vector(switch, fixture)
    params = CAL.physics_params(vector, switch, waist_um * 1e-6, CAL_CFG)
    base_rows, detail = [], {}
    sensitivity = cfg.get('sensitivity_tail_waists')
    try:
        sims = _simulate(params, recipe, cfg['tail_waists'])
        require(all(s[3]['max_window_attempts'] <= cfg['max_window_attempts_allowed']
                    for s in sims.values()), 'Window attempt registration budget exceeded')
        lo, hi = sorted(cfg['tail_waists'])
        rows = compare(sims[hi][1], sims[lo][1], cfg['near_zero_scales'], cfg['tail_relative_tolerance'])
        rows.append(state_tail_check(sims, lo, hi, cfg['tail_relative_tolerance']))
        for item in rows:
            item.update({'job': ident, 'switch': switch, 'waist_um': waist_um,
                         'dataset_index': int(row.dataset_index), 'check': 'tail'})
        base_rows = rows
        if sensitivity:
            low = _simulate(params, recipe, [float(sensitivity)])[float(sensitivity)]
            for item in compare(sims[lo][1], low[1], cfg['near_zero_scales'],
                                cfg['tail_relative_tolerance']):
                item.update({'job': ident, 'switch': switch, 'waist_um': waist_um,
                             'dataset_index': int(row.dataset_index), 'check': 'sensitivity_3w_vs_6w'})
                base_rows.append(item)
        detail = {'D_um': float(sims[hi][1]['D']), 'A_um': float(sims[hi][1]['A_med']),
                  'implausible_depth': bool(abs(float(sims[hi][1]['D'])) > float(cfg.get('implausible_depth_um', math.inf))),
                  'max_window_attempts': int(sims[hi][3]['max_window_attempts']),
                  'physical_pulses': int(sims[hi][3]['physical_pulses']),
                  'evaluated_point_pulse_updates': int(sims[hi][3]['evaluated_point_pulse_updates']),
                  'band_valid': bool(sims[hi][2].band_valid.all()),
                  'all_finite': bool(np.isfinite(list(sims[hi][1].values())[1:]).all())}
    except Exception as exc:                                    # noqa: BLE001 - recorded, not repaired
        base_rows = [{'job': ident, 'switch': switch, 'waist_um': waist_um,
                      'dataset_index': int(row.dataset_index), 'check': 'tail',
                      'descriptor': 'ALL', 'error': None, 'passed': False,
                      'reason': f'{type(exc).__name__}: {exc}'}]
        detail = {'solver_failure': f'{type(exc).__name__}: {exc}'}
    return ident, base_rows, detail


def domain_case(payload):
    """One sampled vector of the frozen parameter box at the representative recipe."""
    ident, switch, vector, recipe_values, waist_um, cfg = payload
    recipe = dict(recipe_values)
    rows, detail = [], {'vector': [float(v) for v in vector]}
    try:
        params = CAL.physics_params(np.asarray(vector, dtype=float), switch, waist_um * 1e-6, CAL_CFG)
        sims = _simulate(params, recipe, cfg['tail_waists'])
        require(all(s[3]['max_window_attempts'] <= cfg['max_window_attempts_allowed']
                    for s in sims.values()), 'Window attempt registration budget exceeded')
        lo, hi = sorted(cfg['tail_waists'])
        rows = compare(sims[hi][1], sims[lo][1], cfg['near_zero_scales'], cfg['tail_relative_tolerance'])
        rows.append(state_tail_check(sims, lo, hi, cfg['tail_relative_tolerance']))
        for item in rows:
            item.update({'job': ident, 'switch': switch, 'check': 'tail'})
        detail.update({'D_um': float(sims[hi][1]['D']), 'A_um': float(sims[hi][1]['A_med']),
                       'implausible_depth': bool(abs(float(sims[hi][1]['D'])) > float(cfg.get('implausible_depth_um', math.inf))),
                       'all_finite': bool(np.isfinite(list(sims[hi][1].values())[1:]).all()),
                       'accepted': True})
    except Exception as exc:                                    # noqa: BLE001
        rows = [{'job': ident, 'switch': switch, 'check': 'tail', 'descriptor': 'ALL',
                 'error': None, 'passed': False, 'reason': f'{type(exc).__name__}: {exc}'}]
        detail.update({'accepted': False, 'failure': f'{type(exc).__name__}: {exc}'})
    return ident, rows, detail


def domain_enforcement_case(payload):
    """Out-of-box coordinates must be rejected by the calibration entry point."""
    ident, switch, vector, waist_um = payload
    try:
        CAL.physics_params(np.asarray(vector, dtype=float), switch, waist_um * 1e-6, CAL_CFG)
        accepted = True
        error = ''
    except Exception as exc:                                    # noqa: BLE001
        accepted = False
        error = f'{type(exc).__name__}: {exc}'
    return ident, {'job': ident, 'switch': switch, 'out_of_bounds_accepted': bool(accepted),
                   'rejected_as_required': bool(not accepted), 'message': error,
                   'vector': [float(v) for v in vector]}


def grid_case(payload):
    """640 -> 1280 point-grid refinement, mapped back to the canonical 160 grid."""
    ident, row_values, switch, waist_um, fixture, cfg = payload
    row = pd.Series(row_values)
    recipe = recipe_from_row(row, switch, cfg['power_W'])
    params = CAL.physics_params(nominal_vector(switch, fixture), switch, waist_um * 1e-6, CAL_CFG)
    coords = (np.arange(160) - 79.5) * .5e-6
    xx, yy = np.meshgrid(coords, coords)
    mapped, direct = [], None
    for n in cfg['grid_refinement']['grids']:
        fine = (np.arange(n) - (n - 1) / 2) * 80e-6 / n
        xf, yf = np.meshgrid(fine, fine)
        d, _, _ = simulate_points(params, xf, yf,
                                  cfg=replace(FiniteScanConfig(), tail_waists=max(cfg['tail_waists'])),
                                  **recipe)
        mapped.append(observe(RegularGridInterpolator((fine, fine), -d * 1e6)(
            np.c_[yy.ravel(), xx.ravel()]).reshape(160, 160))[0])
    tail = max(cfg['tail_waists'])
    direct = observe(simulate_roi(params, cfg=replace(FiniteScanConfig(), tail_waists=tail), **recipe)[0])[0]
    rows = []
    for item in compare(mapped[-1], mapped[-2], cfg['near_zero_scales'], cfg['tail_relative_tolerance']):
        item.update({'job': ident, 'switch': switch, 'waist_um': waist_um,
                     'dataset_index': int(row.dataset_index), 'check': 'grid_finest'})
        rows.append(item)
    for item in compare(direct, mapped[-1], cfg['near_zero_scales'], cfg['tail_relative_tolerance']):
        item.update({'job': ident, 'switch': switch, 'waist_um': waist_um,
                     'dataset_index': int(row.dataset_index), 'check': 'grid_to_direct_points'})
        rows.append(item)
    return ident, rows, {'dataset_index': int(row.dataset_index), 'switch': switch}


def determinism_case(payload):
    """Bitwise repeatability of one full case."""
    ident, row_values, switch, waist_um, fixture, cfg = payload
    row = pd.Series(row_values)
    recipe = recipe_from_row(row, switch, cfg['power_W'])
    params = CAL.physics_params(nominal_vector(switch, fixture), switch, waist_um * 1e-6, CAL_CFG)
    digests = []
    for _ in range(int(cfg['determinism_repeats'])):
        height, _, _ = simulate_roi(params, cfg=replace(FiniteScanConfig(),
                                                        tail_waists=sorted(cfg['tail_waists'])[-1]),
                                    **recipe)
        digests.append(hashlib.sha256(np.ascontiguousarray(height).tobytes()).hexdigest())
    return ident, {'job': ident, 'switch': switch, 'waist_um': waist_um,
                   'dataset_index': int(row.dataset_index), 'digests': digests,
                   'identical': bool(len(set(digests)) == 1)}


CAL = None
CAL_CFG = None


def state_tail_check(sims, lo, hi, tolerance):
    """Check the retained incubation state, including subthreshold exposure."""
    ref, other = sims[hi][4], sims[lo][4]
    finite = bool(np.isfinite(ref).all() and np.isfinite(other).all())
    error = float(np.max(np.abs(ref - other) / np.maximum(np.abs(ref), 1e-12))) if finite else None
    return {'descriptor': 'q_state', 'error': error,
            'passed': bool(finite and error <= tolerance), 'reason': '' if finite else 'NONFINITE_STATE'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/task_state_v1/solver_registration.yaml')
    parser.add_argument('--limit-recipes', type=int, default=0,
                        help='>0 restricts recipe coverage to the first N MAIN180 rows '
                             '(smoke test only; such a run cannot authorize calibration)')
    parser.add_argument('--workers', type=int, default=0, help='override config workers')
    parser.add_argument('--threads', type=int, default=0, help='override config threads_per_worker')
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    if args.workers:
        cfg['workers'] = args.workers
    if args.threads:
        cfg['threads_per_worker'] = args.threads

    global CAL, CAL_CFG
    CAL = load_calibration_module()
    cal_path = ROOT / cfg['calibration_config']
    CAL_CFG = yaml.safe_load(cal_path.read_text(encoding='utf-8'))
    bounds = CAL_CFG['parameter_bounds']

    out = new_run('E03_SOLVER_REGISTRATION', cfg_path)
    (out / 'calibration_config_source.yaml').write_text(cal_path.read_text(encoding='utf-8'), encoding='utf-8')
    gate = {'scope': cfg['scope'], 'training_executed': False, 'parameter_fitting_executed': False,
            'calibration_semantic_sha256': calibration_fingerprint(CAL_CFG),
            'calibration_config': cfg['calibration_config'],
            'calibration_config_sha256': hashlib.sha256(cal_path.read_bytes()).hexdigest()}
    checks = {'recipe_coverage': [], 'parameter_domain': [], 'domain_enforcement': [],
              'grid_refinement': [], 'determinism': []}
    try:
        cfg_run, frozen, df, _, input_paths = load_contract()
        main180 = df[df.session_role.isin(['formal', 'pass_main'])].copy().reset_index(drop=True)
        require(len(main180) == 180, 'MAIN180 size mismatch')
        frame = main180[['dataset_index', 'pulse_duration_fs', 'frequency_kHz', 'velocity_mm_s',
                         'hatch_spacing_um', 'pass_count']].copy()
        split_dir = verify_seal(ROOT / 'outputs/task_state_v1' / cfg['split_run'])
        input_paths.extend([split_dir / 'split_manifest.csv', split_dir / 'inner_split_manifest.csv'])
        input_rows, input_digest = input_manifest(input_paths)
        write_json(out / 'registration_inputs.json', {'files': input_rows, 'sha256': input_digest})
        gate['input_sha256'] = input_digest
        manifest = pd.read_csv(split_dir / 'split_manifest.csv')
        require(set(manifest.dataset_index) == set(frame.dataset_index), 'MAIN180 identity mismatch')

        fixture = cfg['nominal_physics']
        cfg['implausible_depth_um'] = float(main180.D.max() * float(cfg['implausible_depth_factor']))
        limit = int(args.limit_recipes)
        if limit > 0:
            require(limit <= len(frame), 'Recipe limit exceeds MAIN180')
            frame = frame.head(limit).reset_index(drop=True)
        rows = frame.to_dict('records')

        # ---------------- parameter domain sampling ---------------- #
        pulses = ((200e-6 / (frame.velocity_mm_s.to_numpy() * 1e-3 /
                             (frame.frequency_kHz.to_numpy() * 1e3))).round().astype(int)
                  * (200e-6 / (frame.hatch_spacing_um.to_numpy() * 1e-6)).round().astype(int)
                  * frame.pass_count.to_numpy())
        order = np.argsort(pulses, kind='stable')
        n = len(order)
        picks = sorted({0, int(n // 4), int(n // 2), int(3 * n // 4), n - 1})
        selected = [rows[int(i)] for i in order[picks]]
        cols = ['pulse_duration_fs', 'frequency_kHz', 'velocity_mm_s', 'hatch_spacing_um', 'pass_count']
        z = frame[cols].to_numpy(dtype=float)
        z = (z - z.mean(axis=0)) / np.maximum(z.std(axis=0), 1e-12)
        representative = rows[int(np.argmin(np.abs(z).sum(axis=1)))]
        rep_recipe = recipe_from_row(pd.Series(representative), 'P11', cfg['power_W'])
        rep_switchless = {k: v for k, v in rep_recipe.items() if k != 'switch'}

        domain_jobs, enforce_jobs = [], []
        for switch in cfg['switches']:
            names = CAL.parameter_names(switch)
            limits = np.array([bounds[name] for name in names], dtype=float)
            vectors = []
            if cfg['domain_sampling']['include_vertices']:
                mesh = np.array(np.meshgrid(*[[0, 1]] * len(names))).T.reshape(-1, len(names))
                vectors.append(limits[..., 0] + mesh * (limits[..., 1] - limits[..., 0]))
            sampler = LatinHypercube(len(names), seed=int(cfg['domain_sampling']['seed']))
            unit = sampler.random(int(cfg['domain_sampling']['lhs_points']))
            vectors.append(limits[..., 0] + unit * (limits[..., 1] - limits[..., 0]))
            matrix = np.vstack(vectors)
            for i, vector in enumerate(matrix):
                for waist in cfg['waist_um']:
                    # Cross parameter corners with sparse, middle and dense exposure.
                    for record in [rows[int(order[j])] for j in sorted({0, n // 2, n - 1})]:
                        domain_jobs.append((f"domain_{switch}_{i}_w{waist}_r{record['dataset_index']}",
                                            switch, vector.tolist(),
                                            recipe_from_row(pd.Series(record), switch, cfg['power_W']),
                                            waist, cfg))
            frac = float(cfg['domain_sampling']['out_of_bounds_probe_fraction'])
            for i, name in enumerate(names):
                span = bounds[name][1] - bounds[name][0]
                for side, value in (('lower', bounds[name][0] - frac * span),
                                    ('upper', bounds[name][1] + frac * span)):
                    vector = (limits[..., 0] + limits[..., 1]) / 2
                    vector = vector.copy()
                    vector[i] = value
                    enforce_jobs.append((f'outofbounds_{switch}_{name}_{side}', switch,
                                         vector.tolist(), cfg['waist_um'][0]))

        recipe_jobs = []
        for switch in cfg['switches']:
            for waist in cfg['waist_um']:
                for record in rows:
                    recipe_jobs.append((f"recipe_{record['dataset_index']}_{switch}_w{waist}",
                                        record, switch, waist, fixture, cfg))
        grid_jobs = []
        for switch in cfg['switches']:
            for record in selected:
                for waist in cfg['waist_um']:
                    grid_jobs.append((f"grid_{record['dataset_index']}_{switch}_w{waist}",
                                      record, switch, waist, fixture, cfg))
        det_jobs = [(f"det_{selected[0]['dataset_index']}_{switch}_w{cfg['waist_um'][0]}",
                     selected[0], switch, cfg['waist_um'][0], fixture, cfg)
                    for switch in cfg['switches']]

        write_json(out / 'job_plan.json', {
            'n_recipe_coverage_cases': len(recipe_jobs),
            'n_parameter_domain_cases': len(domain_jobs),
            'n_domain_enforcement_cases': len(enforce_jobs),
            'n_grid_refinement_cases': len(grid_jobs),
            'n_determinism_cases': len(det_jobs),
            'representative_recipe_dataset_index': int(representative['dataset_index']),
            'grid_refinement_dataset_indices': [int(r['dataset_index']) for r in selected],
            'parameter_bounds_registered': bounds,
            'nominal_physics_fixture': fixture})

        coverage_rows, coverage_detail = [], []
        domain_rows, domain_detail = [], []
        grid_rows, det_rows, enforce_rows = [], [], []
        plans = [(recipe_case, recipe_jobs, 'recipe_coverage'),
                 (domain_case, domain_jobs, 'parameter_domain'),
                 (domain_enforcement_case, enforce_jobs, 'domain_enforcement'),
                 (grid_case, grid_jobs, 'grid_refinement'),
                 (determinism_case, det_jobs, 'determinism')]
        completed = {label: [] for _, _, label in plans}
        (out / 'jobs').mkdir()
        with ProcessPoolExecutor(max_workers=int(cfg['workers']),
                                 initializer=_worker_init,
                                 initargs=(int(cfg['threads_per_worker']), CAL_CFG)) as pool:
            for func, jobs, label in plans:
                futures = {pool.submit(func, job): job[0] for job in jobs}
                done = 0
                for future in as_completed(futures):
                    ident = futures[future]
                    try:
                        result = future.result()
                        require(result[0] == ident, 'Worker returned mismatched job identity')
                        write_json(out / 'jobs' / (ident + '.json'), {'stage': label, 'result': result})
                        completed[label].append(ident)
                        if label == 'recipe_coverage':
                            _, rows_out, detail = future.result()
                            coverage_rows.extend(rows_out)
                            coverage_detail.append({'job': ident, **detail})
                        elif label == 'parameter_domain':
                            _, rows_out, detail = future.result()
                            domain_rows.extend(rows_out)
                            domain_detail.append({'job': ident, **detail})
                        elif label == 'domain_enforcement':
                            _, row = future.result()
                            enforce_rows.append(row)
                        elif label == 'grid_refinement':
                            _, rows_out, detail = future.result()
                            grid_rows.extend(rows_out)
                        else:
                            _, row = future.result()
                            det_rows.append(row)
                    except Exception as exc:                    # noqa: BLE001
                        failure = f'{type(exc).__name__}: {exc}'
                        checks[label].append({'job': ident, 'error': failure})
                        if label in ('recipe_coverage', 'parameter_domain'):
                            coverage_rows.append({'job': ident, 'check': label, 'descriptor': 'ALL',
                                                  'error': None, 'passed': False, 'reason': failure})
                        else:
                            checks[label].append({'job': ident, 'error': failure})
                    done += 1
                    if done % 100 == 0 or done == len(jobs):
                        print(f'{label}: {done}/{len(jobs)}', flush=True)
                    if label == 'recipe_coverage':
                        pd.DataFrame(coverage_rows).to_csv(out / 'recipe_coverage_checks.csv', index=False)
                        pd.DataFrame(coverage_detail).to_csv(out / 'recipe_coverage_summary.csv', index=False)

        coverage = pd.DataFrame(coverage_rows)
        domain = pd.DataFrame(domain_rows)
        grid = pd.DataFrame(grid_rows)
        enforcement = pd.DataFrame(enforce_rows)
        determinism = pd.DataFrame(det_rows)
        coverage.to_csv(out / 'recipe_coverage_checks.csv', index=False)
        pd.DataFrame(coverage_detail).to_csv(out / 'recipe_coverage_summary.csv', index=False)
        domain.to_csv(out / 'parameter_domain_checks.csv', index=False)
        pd.DataFrame(domain_detail).to_csv(out / 'parameter_domain_summary.csv', index=False)
        grid.to_csv(out / 'grid_refinement_checks.csv', index=False)
        enforcement.to_csv(out / 'domain_enforcement.csv', index=False)
        determinism.to_csv(out / 'determinism.csv', index=False)

        n_cases = len(recipe_jobs)
        gate_coverage = coverage[coverage.check.eq('tail')] if len(coverage) else coverage
        sensitivity = coverage[~coverage.check.eq('tail')] if len(coverage) else coverage
        failed_jobs = sorted(set(gate_coverage.loc[~gate_coverage.passed, 'job'])) if len(gate_coverage) else []
        window_attempts = [int(d.get('max_window_attempts', 0)) for d in coverage_detail]
        gate.update({
            'recipe_coverage': {
                'cases': n_cases, 'waists': cfg['waist_um'], 'switches': cfg['switches'],
                'gate_descriptor_checks': int(len(gate_coverage)),
                'failed_descriptor_checks': int((~gate_coverage.passed).sum()) if len(gate_coverage) else None,
                'failed_cases': int(len(failed_jobs)),
                'failed_case_ids': failed_jobs[:50],
                'solver_failures': int(sum('solver_failure' in d for d in coverage_detail)),
                'nonfinite_cases': int(sum(not d.get('all_finite', False) for d in coverage_detail)),
                'implausible_depth_cases': int(sum(bool(d.get('implausible_depth')) for d in coverage_detail)),
                'implausible_depth_threshold_um': float(cfg['implausible_depth_um']),
                'max_window_attempts_observed': int(max(window_attempts)) if window_attempts else None,
                'max_window_attempts_allowed': int(cfg['max_window_attempts_allowed']),
                'window_self_consistent': bool(window_attempts
                                               and max(window_attempts) <= int(cfg['max_window_attempts_allowed'])
                                               and not any('solver_failure' in d for d in coverage_detail)),
                'sensitivity_3w_vs_6w_checks': int(len(sensitivity)),
                'sensitivity_3w_vs_6w_max_error': (None if not len(sensitivity)
                                                   else float(sensitivity.error.max()))},
            'parameter_domain': {
                'cases': len(domain_jobs),
                'descriptor_checks': int(len(domain)),
                'failed_descriptor_checks': int((~domain.passed).sum()),
                'accepted_vectors': int(sum(bool(d.get('accepted')) and d.get('all_finite', False)
                                            for d in domain_detail)),
                'rejected_or_nonfinite': int(sum(not (bool(d.get('accepted')) and d.get('all_finite', False))
                                                 for d in domain_detail)),
                'implausible_depth_cases': int(sum(bool(d.get('implausible_depth')) for d in domain_detail)),
                'implausible_depth_threshold_um': float(cfg['implausible_depth_um']),
                'D_um_max': float(max(d.get('D_um', float('nan')) for d in domain_detail)),
                'bounds_registered': bounds},
            'domain_enforcement': {
                'probes': int(len(enforcement)),
                'rejected_as_required': int(enforcement.rejected_as_required.sum()) if len(enforcement) else 0,
                'wrongly_accepted': int((~enforcement.rejected_as_required).sum()) if len(enforcement) else 0},
            'grid_refinement': {
                'cases': len(grid_jobs),
                'descriptor_checks': int(len(grid)),
                'failed_descriptor_checks': int((~grid.passed).sum()) if len(grid) else None},
            'determinism': {'cases': int(len(determinism)),
                            'identical': bool(determinism.identical.all()) if len(determinism) else False}})

        conditions = {
            'all_jobs_accounted': all(complete_jobs([job[0] for job in jobs],
                                                    completed[label], checks[label])
                                      for _, jobs, label in plans),
            'full_recipe_coverage': bool(limit == 0),
            'window_self_consistent': bool(gate['recipe_coverage']['window_self_consistent']),
            'recipe_coverage_finite_and_converged': bool(len(gate_coverage) and gate_coverage.passed.all()),
            'parameter_domain_finite_and_converged': bool(len(domain) and domain.passed.all()),
            'out_of_bounds_rejected': bool(len(enforcement) and enforcement.rejected_as_required.all()),
            'grid_refinement_converged': bool(len(grid) and grid.passed.all()),
            'determinism_bitwise': bool(len(determinism) and determinism.identical.all())}
        gate['conditions'] = conditions
        gate['worker_errors'] = checks
        passed = all(conditions.values())
        gate['status'] = 'PASS' if passed else 'FAIL'
        gate['allowed_to_calibrate'] = bool(passed)
        gate['recipe_coverage_scope'] = 'full_MAIN180' if limit == 0 else f'smoke_first_{limit}_rows'
        gate['metric_status'] = 'VALID' if passed else 'INVALID'
        gate['scenario_status'] = 'FINITE_ROI_REGISTERED_FIXTURE'
        gate['remaining'] = [] if passed else [k for k, v in conditions.items() if not v]
    except Exception as exc:                                    # noqa: BLE001
        import traceback
        gate.update({'status': 'FAIL', 'allowed_to_calibrate': False,
                     'error': f'{type(exc).__name__}: {exc}',
                     'traceback': traceback.format_exc()})
    write_json(out / 'solver_registration_gate.json', gate)
    seal(out)
    print(json.dumps({k: v for k, v in gate.items() if k != 'traceback'}, ensure_ascii=False, indent=2))
    return 0 if gate.get('status') == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
