"""Finite raster validation on fixed ROI; no labels and no parameter fitting."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,time,subprocess,argparse
from dataclasses import replace
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import RegularGridInterpolator
import numba
from src.task_state_learning.artifacts import ROOT,new_run,write_json,seal
from src.task_state_learning.physics import PhysicsParameters
from src.task_state_learning.finite_scan import FiniteScanConfig,simulate_roi,simulate_points
from src.task_state_learning.observers import CanonicalObserver


def observe(h):
    values,valid=CanonicalObserver()(h[None],np.ones((1,160,160),bool))
    return values.iloc[0].to_dict(),valid


def comparison(a,b,scales,tolerance):
    rows=[]
    for key,scale in scales.items():
        x,y=a[key],b[key]
        if not np.isfinite([x,y]).all():
            rows.append({'descriptor':key,'error':np.nan,'passed':False,'reason':'UNDEFINED_DESCRIPTOR'})
        else:
            error=abs(x-y)/max(abs(y),scale)
            rows.append({'descriptor':key,'error':error,'passed':bool(error<=tolerance),'reason':''})
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config/task_state_v1/finite_validation.yaml')
    path=ROOT/parser.parse_args().config
    c=yaml.safe_load(path.read_text(encoding='utf-8'))
    out=new_run('E03_FINITE_ROI',path)
    numba.set_num_threads(c['threads'])
    cfg=FiniteScanConfig(scan_region_m=c['scan_region_m'],phase_x_pitch=c['phase_x_pitch'],phase_y_pitch=c['phase_y_pitch'])
    rows=[];validity=[];timings=[];case_id=0
    try:
        suite=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests/task_state_v1',
                              '-p','test_finite_scan.py','-v'],cwd=ROOT,capture_output=True,encoding='utf-8')
        (out/'finite_tests.log').write_text(suite.stdout+suite.stderr,encoding='utf-8')
        if suite.returncode:raise ValueError('Finite reference tests failed')
        for waist in c['waist_um']:
            for switch in c['switches']:
                for feedback in c['feedback_cases']:
                    start=time.perf_counter()
                    p=PhysicsParameters(waist*1e-6,1030e-9,1.2,c['F1_reference_J_m2'],feedback['delta_m'],feedback['kappa'],feedback['rho'])
                    recipe=dict(tau_s=c['tau_s'],f_Hz=c['frequency_Hz'],v_m_s=c['frequency_Hz']*waist*1e-6*c['pitch_over_waist'],
                                h_m=c['hatch_um']*1e-6,passes=c['passes'],power_W=c['power_W'],switch=switch)
                    heights=[];obs=[]
                    for tail in c['tail_waists']:
                        h,q,info=simulate_roi(p,cfg=replace(cfg,tail_waists=tail),**recipe)
                        heights.append(h)
                        value,valid=observe(h);obs.append(value)
                        valid['case']=case_id;valid['tail_waists']=tail;validity.append(valid)
                        write_json(out/f'case_{case_id}_tail_{tail}_diagnostics.json',info)
                    np.savez_compressed(out/f'case_{case_id}_fields.npz',height_base_um=heights[0],height_reference_um=heights[1])
                    for row in comparison(obs[0],obs[1],c['near_zero_scales'],c['relative_tolerance']):
                        rows.append({'case':case_id,'waist_um':waist,'switch':switch,'check':'tail',**row})
                    if case_id in c['grid_refinement_cases']:
                        coords=(np.arange(160)-79.5)*.5e-6
                        xx,yy=np.meshgrid(coords,coords)
                        mapped=[]
                        for n in c['grid_refinement_n']:
                            fine=(np.arange(n)-(n-1)/2)*80e-6/n
                            xf,yf=np.meshgrid(fine,fine)
                            d,_,_=simulate_points(p,xf,yf,cfg=replace(cfg,tail_waists=max(c['tail_waists'])),**recipe)
                            fixed=RegularGridInterpolator((fine,fine),-d*1e6)(np.c_[yy.ravel(),xx.ravel()]).reshape(160,160)
                            mapped.append(observe(fixed)[0])
                            np.save(out/f'case_{case_id}_grid_{n}_mapped_height_um.npy',fixed)
                            print(f'case {case_id}: refined grid {n} done',flush=True)
                        for row in comparison(mapped[-2],mapped[-1],c['near_zero_scales'],c['relative_tolerance']):
                            rows.append({'case':case_id,'waist_um':waist,'switch':switch,'check':'grid_finest',**row})
                        for row in comparison(mapped[-1],obs[-1],c['near_zero_scales'],c['relative_tolerance']):
                            rows.append({'case':case_id,'waist_um':waist,'switch':switch,'check':'grid_to_direct_points',**row})
                    timings.append({'case':case_id,'walltime_seconds':time.perf_counter()-start})
                    pd.DataFrame(rows).to_csv(out/'descriptor_checks.csv',index=False)
                    pd.DataFrame(timings).to_csv(out/'timing.csv',index=False)
                    pd.concat(validity).to_csv(out/'observer_validity.csv',index=False)
                    print(f'case {case_id+1}/12 complete ({timings[-1]["walltime_seconds"]:.2f}s)',flush=True)
                    case_id+=1
        passed=all(r['passed'] for r in rows)
        gate={'status':'PASS' if passed else 'FAIL','completed_cases':case_id,'checks':len(rows),
              'failed_checks':sum(not r['passed'] for r in rows)}
    except Exception as exc:
        gate={'status':'FAIL','error':f'{type(exc).__name__}: {exc}','completed_cases':case_id}
    gate.update(scope=c['scope'],allowed_to_calibrate=False,training_executed=False,remaining=c['remaining'])
    write_json(out/'finite_roi_gate.json',gate)
    write_json(out/'runtime.json',{'numba':numba.__version__,'threads':numba.get_num_threads()})
    (out/'config_resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
    seal(out)
    print(json.dumps({'output':str(out),**gate},ensure_ascii=False,indent=2))
    return 0 if gate['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
