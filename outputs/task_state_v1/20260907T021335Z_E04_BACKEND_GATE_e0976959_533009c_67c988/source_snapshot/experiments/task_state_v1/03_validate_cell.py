"""Candidate cell-solver review: no fitting and no test-label access."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dataclasses import asdict, replace
import json
import time
import subprocess
import numpy as np
import pandas as pd
import yaml
from src.task_state_learning.artifacts import ROOT,new_run,write_json,seal
from src.task_state_learning.scan_cell import CellConfig,simulate_sample
from src.task_state_learning.physics import PhysicsParameters


def main():
    path=ROOT/'config/task_state_v1/cell_validation.yaml'
    c=yaml.safe_load(path.read_text(encoding='utf-8'))
    out=new_run('E03_CELL_VALIDATION',path)
    suite=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests/task_state_v1',
                          '-p','test_scan_cell.py','-v'],cwd=ROOT,capture_output=True,encoding='utf-8')
    (out/'cell_tests.log').write_text(suite.stdout+suite.stderr,encoding='utf-8')
    cfg=CellConfig(window_over_w=c['window_over_w'],n_phi=16)
    rows=[]
    for waist in c['waist_um']:
        for switch in c['switches']:
            for index,case in enumerate(c['feedback_cases']):
                start=time.perf_counter()
                row={'waist_um':waist,'switch':switch,'feedback_case':index,'passed':False}
                p=PhysicsParameters(waist*1e-6,1030e-9,1.2,c['F1_reference_J_m2'],case['delta_m'],case['kappa'],case['rho'])
                kw=dict(tau_s=c['tau_s'],f_Hz=c['frequency_Hz'],v_m_s=c['frequency_Hz']*waist*1e-6*c['pitch_over_waist'],
                        h_m=c['hatch_um']*1e-6,passes=c['passes'],power_W=c['power_W'],switch=switch)
                try:
                    dense=simulate_sample(p,**kw,cfg=cfg)
                    exact=simulate_sample(p,**kw,cfg=replace(cfg,dense_dx_over_w0=0.),packet_cap=1)
                    for j,name in enumerate(['D','A']):
                        row[name+'_dense_m']=dense[j];row[name+'_pulse_m']=exact[j]
                        row[name+'_relative_error']=abs(dense[j]-exact[j])/max(abs(exact[j]),c['numerical_absolute_scale_m'])
                    row['accepted_log_threshold_drift']=dense[2]['max_span_log_threshold_drift']
                    row['accepted_depth_over_zr_drift']=dense[2]['max_span_dd_over_zr']
                    row['passed']=bool(row['D_relative_error']<=c['relative_tolerance'] and row['A_relative_error']<=c['relative_tolerance'])
                    row['failure_reason']='' if row['passed'] else 'CONTINUUM_LATTICE_DISCREPANCY'
                except Exception as exc:
                    row['failure_reason']=f'{type(exc).__name__}: {exc}'
                row['walltime_seconds']=time.perf_counter()-start
                rows.append(row)
                pd.DataFrame(rows).to_csv(out/'continuum_lattice_comparison.csv',index=False)
                print(f'case {len(rows)}/12: w={waist} {switch} feedback={index} PASS={row["passed"]} ({row["walltime_seconds"]:.2f}s)',flush=True)
    (out/'config_resolved.yaml').write_text(yaml.safe_dump({**c,'cell_config':asdict(cfg)},sort_keys=False),encoding='utf-8')
    gate={'experiment_id':'E03','gate_scope':'CELL_SOLVER_REVIEW_NOT_CALIBRATION',
          'status':'PASS' if suite.returncode==0 and all(r['passed'] for r in rows) else 'FAIL',
          'hypothesis_status':'NOT_TESTED','training_executed':False,'allowed_to_calibrate':False,
          'cases':len(rows),'passed_cases':sum(r['passed'] for r in rows),
          'failures_retained':True,'remaining':c['registration_additional_requirements']}
    write_json(out/'solver_registration_gate.json',gate);seal(out)
    print(json.dumps({'output':str(out),**gate},ensure_ascii=False,indent=2))
    return 0 if gate['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
