"""G1 backend acceptance: DifferentiableObserver vs CanonicalObserver parity.

Validity masks and undefined reasons are compared BEFORE any finite value, per
runbook section 28.3. float32-input parity means the same float32 values are
compared after promotion to float64 on both paths.
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
from src.task_state_learning.artifacts import ROOT, new_run, write_json, verify_seal, seal
from src.task_state_learning.contracts import load_contract
from src.task_state_learning.observers import CanonicalObserver
from src.task_state_learning.differentiable_observer import DifferentiableObserver, TARGET_NAMES
from src.task_state_learning.grouping import require


def canonical_flags(validity, i):
    rows = validity[validity.dataset_index == i]
    blocks = {r.target_block: r for _, r in rows.iterrows()}
    if "ALL" in blocks:
        return False, False, False, [False, False], "EMPTY_OR_NONFINITE_VALID_HEIGHT"
    if "SPECTRAL" in blocks:
        return True, False, False, [False, False], "INCOMPLETE_GRID_NO_REGISTERED_IMPUTATION"
    comp = bool(blocks["ILR"].band_valid)
    bands = [bool(blocks[b].band_valid) for b in ("8_16", "16_32")]
    if not comp:
        reason = "ZERO_NON_DC_ENERGY"
    elif not all(bands):
        reason = "ZERO_DIRECTION_BAND_ENERGY"
    else:
        reason = ""
    return True, True, comp, bands, reason


def _replacement_flag(canonical_values, i):
    """Rows that exit before the spectral path never set the flag; missing/NaN means False."""
    if "composition_replacement_used" not in canonical_values.columns:
        return False
    value = canonical_values["composition_replacement_used"].iloc[i]
    return False if pd.isna(value) else bool(value)


def compare_batch(canonical_values, canonical_validity, obs, tags, atol, rtol):
    value_rows, flag_rows = [], []
    values = obs.values.detach().numpy()
    valid = obs.valid.detach().numpy()
    for i, tag in enumerate(tags):
        usable, full, comp, bands, reason = canonical_flags(canonical_validity, i)
        expected = [usable, usable, comp, comp, comp, comp, bands[0], bands[0], bands[1], bands[1]]
        match = bool(np.array_equal(valid[i].astype(bool), np.asarray(expected, dtype=bool)))
        flag_rows.append({"tag": tag, "canonical_reason": reason, "differentiable_reason": obs.reasons[i],
                          "validity_mask_match": match,
                          "replacement_match": _replacement_flag(canonical_values, i)
                                               == bool(obs.replacement_used[i])})
        require(match and obs.reasons[i] == reason, f'Validity/reason mismatch at {tag}: '
                f'{reason} vs {obs.reasons[i]}')
        for j, target in enumerate(TARGET_NAMES):
            a, b = float(values[i, j]), float(canonical_values[target].iloc[i])
            if np.isnan(a) and np.isnan(b):
                value_rows.append({"tag": tag, "target": target, "canonical": b, "differentiable": a,
                                   "abs_error": np.nan, "passed": True, "both_undefined": True})
                continue
            require(not (np.isnan(a) or np.isnan(b)), f'Validity disagreement at {tag}/{target}')
            err = abs(a - b)
            value_rows.append({"tag": tag, "target": target, "canonical": b, "differentiable": a,
                               "abs_error": err, "passed": bool(err <= atol + rtol * abs(b)),
                               "both_undefined": False})
    return pd.DataFrame(value_rows), pd.DataFrame(flag_rows)


def synthetic_fields(seed):
    x = np.arange(160) * .5
    ax = np.broadcast_to(np.cos(2 * np.pi * x / 10), (160, 160)).copy()
    rng = np.random.default_rng(seed)
    return [("constant_plane", np.full((160, 160), -3.0)), ("single_x_cosine", ax),
            ("single_y_cosine", ax.T), ("orthogonal_two_cosine", ax + ax.T),
            ("gaussian_random_field", rng.normal(size=(160, 160)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-run', required=True, type=Path)
    ap.add_argument('--config', type=Path, default=ROOT / 'config/task_state_v1/diff_backend.yaml')
    args = ap.parse_args()
    audit = verify_seal(args.audit_run)
    require(json.loads((audit / 'gate_G0.json').read_text(encoding='utf-8'))['status'] == 'PASS', 'G0 not PASS')
    out = new_run('E01_BACKEND_OBS', args.config)
    gate = {'experiment_id': 'E01', 'gate_scope': 'DIFFERENTIABLE_OBSERVER_BACKEND',
            'execution_status': 'FAIL', 'hypothesis_status': 'NOT_TESTED',
            'G1_complete': False, 'training_executed': False,
            'backend': f'torch_{torch.__version__}_cpu_float64'}
    try:
        c = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        (out / 'config_resolved.yaml').write_text(yaml.safe_dump(c, sort_keys=False), encoding='utf-8')
        suite = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'discover',
                                '-s', 'tests/task_state_v1', '-p', 'test_*.py', '-v'],
                               cwd=ROOT, capture_output=True, encoding='utf-8')
        (out / 'task_state_tests.log').write_text(suite.stdout + suite.stderr, encoding='utf-8')
        require(suite.returncode == 0, 'Task-state unit tests failed')
        cfg, frozen, df, _, _ = load_contract()
        expected = pd.read_csv(audit / 'frozen_targets.csv')
        require(np.array_equal(df.dataset_index, expected.dataset_index), 'Upstream identities changed')
        torch.set_num_threads(4)
        torch.set_default_dtype(torch.float64)
        # The two observers use different positional argument orders; bind by keyword.
        observer = DifferentiableObserver(theta_bins=c['theta_bins'],
                                          zero_threshold=c['zero_threshold'],
                                          replacement_delta=c['replacement_delta'])
        canonical = CanonicalObserver(zero_threshold=c['zero_threshold'],
                                      replacement_delta=c['replacement_delta'],
                                      theta_bins=c['theta_bins'])
        height, mask = frozen['H'], frozen['V']

        # Chain check: fresh canonical vs frozen targets reproduces the E01_OBS acceptance.
        chain = []
        y, validity = canonical(height, mask)
        for target in TARGET_NAMES:
            a, b = y[target].to_numpy(), expected[target].to_numpy()
            ok = np.isclose(a, b, atol=c['atol_float64'], rtol=c['rtol_float64'])
            chain.extend(bool(v) for v in ok)
            require(ok.all(), f'Canonical drifted from frozen targets: {target}')
        # Primary acceptance: differentiable vs canonical on all real ROIs, float64.
        h64 = torch.from_numpy(np.ascontiguousarray(height)).clone().requires_grad_(True)
        v64 = torch.from_numpy(np.ascontiguousarray(mask.astype(bool)))
        obs64 = observer(h64, v64)
        tags = [f'real_{i}' for i in range(200)]
        parity64, flags64 = compare_batch(y, validity, obs64, tags, c['atol_float64'], c['rtol_float64'])
        parity64.to_csv(out / 'parity_backend_float64.csv', index=False)
        flags64.to_csv(out / 'validity_backend_float64.csv', index=False)
        require(parity64.passed.all(), 'float64 backend parity failed')

        # float32-input parity: same float32 values promoted to float64 on both paths.
        h32 = height.astype(np.float32)
        y32, validity32 = canonical(h32.astype(np.float64), mask)
        obs32 = observer(torch.from_numpy(h32), v64)
        parity32, flags32 = compare_batch(y32, validity32, obs32, tags,
                                          c['atol_float32_input'], c['rtol_float32_input'])
        parity32.to_csv(out / 'parity_backend_float32input.csv', index=False)
        flags32.to_csv(out / 'validity_backend_float32input.csv', index=False)
        require(parity32.passed.all(), 'float32-input backend parity failed')

        # Synthetic contract cases: both implementations must agree, including rejections.
        fields = synthetic_fields(c['synthetic_seed'])
        stack = np.stack([f for _, f in fields])
        y_syn, validity_syn = canonical(stack, np.ones_like(stack, dtype=bool))
        obs_syn = observer(torch.from_numpy(stack), None)
        parity_syn, flags_syn = compare_batch(y_syn, validity_syn, obs_syn, [n for n, _ in fields],
                                              c['atol_float64'], c['rtol_float64'])
        parity_syn.to_csv(out / 'parity_backend_synthetic.csv', index=False)
        flags_syn.to_csv(out / 'validity_backend_synthetic.csv', index=False)
        require(parity_syn.passed.all() and flags_syn.validity_mask_match.all()
                and (flags_syn.canonical_reason == flags_syn.differentiable_reason).all(),
                'Synthetic backend parity failed')

        # Gradient sanity on the real batch through the differentiable observer.
        finite = torch.where(obs64.valid, obs64.values, torch.zeros_like(obs64.values))
        finite.sum().backward()
        grad = h64.grad
        require(grad is not None and torch.isfinite(grad).all(), 'Nonfinite observer gradient')

        max64 = float(parity64.abs_error.max())
        max32 = float(parity32.abs_error.max())
        gate.update(execution_status='PASS',
                    compared_values=int(len(parity64) + len(parity32) + len(parity_syn)),
                    real_coverage=200, float64_max_abs_error=max64, float32input_max_abs_error=max32,
                    synthetic_max_abs_error=float(parity_syn.abs_error.max()),
                    chain_check_values=len(chain), canonical_vs_frozen_max_abs_error=
                    float(np.max([abs(a - b) for t in TARGET_NAMES
                                  for a, b in zip(y[t].to_numpy(), expected[t].to_numpy())])))
    except Exception as exc:
        gate['error'] = f'{type(exc).__name__}: {exc}'
    gate['status'] = gate['execution_status']
    write_json(out / 'gate_G1_BACKEND_OBS.json', gate)
    seal(out)
    print(json.dumps({'output': str(out), **gate}, ensure_ascii=False, indent=2))
    return 0 if gate['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
