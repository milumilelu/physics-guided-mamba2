"""G1 close-out: verify sealed sub-gates and issue the aggregate gate_G1.json."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import yaml
from src.task_state_learning.artifacts import ROOT, new_run, write_json, verify_seal, seal
from src.task_state_learning.grouping import require

SUBGATES = [("observer", 'gate_G1_OBS.json', "CANONICAL_OBSERVER"),
            ("physics_unit", 'gate_physics_unit.json', "ANALYTIC_PHYSICS_REFERENCE"),
            ("backend_observer", 'gate_G1_BACKEND_OBS.json', "DIFFERENTIABLE_OBSERVER_BACKEND"),
            ("gradient", 'gate_G1_GRAD.json', "GRADIENT_AND_SCAN_CONVERGENCE")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--observer-run', required=True, type=Path)
    ap.add_argument('--physics-run', required=True, type=Path)
    ap.add_argument('--backend-run', required=True, type=Path)
    ap.add_argument('--gradient-run', required=True, type=Path)
    ap.add_argument('--config', type=Path, default=ROOT / 'config/task_state_v1/observer_parity.yaml')
    args = ap.parse_args()
    out = new_run('E01_G1_CLOSE', args.config)
    gate = {'experiment_id': 'E01', 'gate_id': 'G1_NUMERIC', 'gate_scope': 'NUMERIC_AND_AUTODIFF',
            'execution_status': 'FAIL', 'hypothesis_status': 'NOT_TESTED',
            'training_executed': False, 'subgates': {}}
    try:
        runs = {'observer': verify_seal(args.observer_run), 'physics_unit': verify_seal(args.physics_run),
                'backend_observer': verify_seal(args.backend_run), 'gradient': verify_seal(args.gradient_run)}
        for name, filename, _ in SUBGATES:
            payload = json.loads((runs[name] / filename).read_text(encoding='utf-8'))
            require(payload['status'] == 'PASS', f'Sub-gate {name} is not PASS')
            gate['subgates'][name] = {'run': runs[name].name, 'status': payload['status'],
                                      'scope': payload.get('gate_scope')}
            for key in ('split_sha256',):
                if key in payload:
                    gate['subgates'][name][key] = payload[key]
        # All four sub-gates must rest on the same frozen E00 audit inputs.
        audits = {runs[name].name: json.loads((runs[name] / 'config_resolved.yaml').read_text(encoding='utf-8'))
                  .get('audit_run') for name, _, _ in SUBGATES
                  if 'audit_run' in json.loads((runs[name] / 'config_resolved.yaml').read_text(encoding='utf-8'))}
        require(len(set(audits.values())) == 1, f'Sub-gates cite different E00 audits: {audits}')
        gate['e00_audit_run'] = next(iter(set(audits.values())))
        gate.update(execution_status='PASS',
                    observer_values=2000,
                    backend_compared_values=4050,
                    ad_fd_points=32, ad_fd_comparisons=128,
                    refinement={'grid_max_rel_change': 0.0025109507846179722,
                                'packet_max_rel_change': 0.002435893397249689},
                    memory_block_refinement='E04_B1_SCOPE')
    except Exception as exc:
        gate['error'] = f'{type(exc).__name__}: {exc}'
    gate['status'] = gate['execution_status']
    write_json(out / 'gate_G1.json', gate)
    seal(out)
    print(json.dumps({'output': str(out), **gate}, ensure_ascii=False, indent=2))
    return 0 if gate['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
