"""G1 close-out: verify sealed sub-gates and issue the aggregate gate_G1.json."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import yaml
import numpy as np
import pandas as pd
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
        payloads = {}
        for name, filename, expected_scope in SUBGATES:
            payload = json.loads((runs[name] / filename).read_text(encoding='utf-8'))
            require(payload['status'] == 'PASS', f'Sub-gate {name} is not PASS')
            require(payload.get('gate_scope') == expected_scope, f'Wrong scope for {name}')
            payloads[name] = payload
            gate['subgates'][name] = {'run': runs[name].name, 'status': payload['status'],
                                      'scope': payload.get('gate_scope')}
            for key in ('split_sha256',):
                if key in payload:
                    gate['subgates'][name][key] = payload[key]
        # Record explicit provenance; synthetic fixtures need not cite a data audit.
        audits = {}
        unresolved = []
        for name, _, _ in SUBGATES:
            resolved = yaml.safe_load((runs[name] / 'config_resolved.yaml').read_text(encoding='utf-8'))
            if resolved.get('audit_run'):
                audits[runs[name].name] = resolved['audit_run']
            else:
                unresolved.append(name)
        require(len(set(audits.values())) == 1, f'Sub-gates cite different E00 audits: {audits}')
        gate['e00_audit_run'] = next(iter(set(audits.values())))
        gate['subgates_without_explicit_audit_reference'] = unresolved
        gate['common_e00_chain_complete'] = not unresolved
        grad_config = yaml.safe_load((runs['gradient']/'config_resolved.yaml').read_text(encoding='utf-8'))
        table = pd.read_csv(runs['gradient']/'gradient_gate_eval.csv')
        require(np.isfinite(table.rel_error).all(), 'Nonfinite gradient evidence')
        require(not (table.rel_error>grad_config['fd']['unexplained_fail_rtol']).any(),
                'Unexplained gradient mismatch (plateau labels are not exemptions)')
        packet = pd.read_csv(runs['gradient']/'pulse_packet_refinement.csv')
        sizes = sorted(set(grad_config['convergence']['packet_splits']))
        require(len(sizes)>=2 and sizes[0]==1, 'Missing exact pulse reference')
        fine, coarse = packet[f'packet_{sizes[0]}'].to_numpy(),packet[f'packet_{sizes[1]}'].to_numpy()
        rel = np.divide(np.abs(fine-coarse),np.abs(coarse),out=np.full_like(fine,np.inf),where=coarse!=0)
        rel[(fine==0)&(coarse==0)]=0.
        require(np.isfinite(rel).all() and (rel<=grad_config['convergence']['refinement_tolerance']).all(),
                'Corrected finest packet comparison failed')
        packet['corrected_rel_change_finest']=rel
        packet.to_csv(out/'corrected_packet_refinement.csv',index=False)
        gate.update(execution_status='PASS',
                    observer_values=payloads['observer']['compared_values'],
                    backend_compared_values=payloads['backend_observer']['compared_values'],
                    ad_fd_points=int(table.point.nunique()), ad_fd_comparisons=len(table),
                    refinement={'grid_max_rel_change': payloads['gradient']['grid_refinement_max_rel_change'],
                                'packet_max_rel_change': float(rel.max()),
                                'packet_sizes_finest': sizes[:2]},
                    applies_to='sealed_synthetic_numeric_fixtures_and_observer_parity_only',
                    E03_cell_solver_registered=False,
                    real_recipe_inverse_design_allowed=False,
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
