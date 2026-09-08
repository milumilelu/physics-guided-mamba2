"""Independent analytic/reference checks, not full scan/gradient G1 acceptance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import subprocess
import numpy as np
import pandas as pd
import yaml
from src.task_state_learning.artifacts import ROOT,new_run,write_json,seal
from src.task_state_learning.grouping import require
from src.task_state_learning.physics import PhysicsParameters,step,repeat_identical_pulse


def main():
    config=ROOT/'config/task_state_v1/physics_numeric.yaml';out=new_run('E01_PHYS_UNIT',config)
    c=yaml.safe_load(config.read_text(encoding='utf-8'))
    gate={'experiment_id':'E01','execution_status':'FAIL','hypothesis_status':'NOT_TESTED',
          'gate_scope':'ANALYTIC_PHYSICS_REFERENCE','G1_complete':False,'training_executed':False}
    try:
        suite=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests/task_state_v1','-p','test_physics.py','-v'],cwd=ROOT,capture_output=True,encoding='utf-8')
        (out/'physics_unit_tests.log').write_text(suite.stdout+suite.stderr,encoding='utf-8')
        require(suite.returncode==0,'Physics unit tests failed')
        p=PhysicsParameters(**{k:c[k] for k in ['waist_m','wavelength_m','m_squared','F1_reference_J_m2','delta_reference_m','kappa','saturation_ratio']})
        rows=[]
        for ratio in [.5,1.,2.,10.]:
            for n in [1,2,10,100]:
                kw=dict(parameters=p,pulse_energy_J=ratio*p.F1_reference_J_m2*np.pi*p.waist_m**2/2,
                        duration_s=1e-12,beam_x_m=0.,beam_y_m=0.,switch='P00')
                d,_=repeat_identical_pulse(np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),multiplicity=n,**kw)
                reference=n*p.delta_reference_m*max(np.log(ratio),0.)
                err=abs(float(d[0])-reference)
                rows.append({'peak_ratio':ratio,'n_pulses':n,'depth_m':float(d[0]),'reference_m':reference,
                             'relative_error':err/reference if reference>0 else 0.,'passed':err<=c['analytic_rtol']*reference if reference>0 else err==0})
        analytic=pd.DataFrame(rows);analytic.to_csv(out/'analytic_checks.csv',index=False);require(analytic.passed.all(),'Analytic check failed')
        grids=[];reference=np.pi*p.delta_reference_m*p.waist_m**2/4*np.log(4.)**2
        for cells in [8,16,32,64]:
            dx=p.waist_m/cells;coords=np.arange(-2*cells,2*cells+1)*dx;x,y=np.meshgrid(coords,coords)
            d,_=step(np.zeros_like(x),np.zeros_like(x),x,y,parameters=p,
                     pulse_energy_J=4*p.F1_reference_J_m2*np.pi*p.waist_m**2/2,duration_s=1e-12,beam_x_m=0.,beam_y_m=0.,switch='P00')
            volume=float(d.sum()*dx**2)
            grids.append({'cells_per_waist':cells,'grid_spacing_m':dx,'volume_m3':volume,'analytic_volume_m3':reference,
                          'relative_error':abs(volume-reference)/reference})
        grid=pd.DataFrame(grids);grid.to_csv(out/'single_pulse_grid_refinement.csv',index=False)
        require(grid.relative_error.iloc[-1]<=c['gaussian_volume_grid_refinement_tolerance'],'Single-pulse grid failed')
        (out/'config_resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
        gate.update(execution_status='PASS',analytic_cases=len(rows),maximum_analytic_relative_error=float(analytic.relative_error.max()),
                    finest_grid_relative_error=float(grid.relative_error.iloc[-1]),
                    remaining=['scan_event_topology','real_parameter_calibration','scan_grid_and_packet_refinement','AD_FD','differentiable_backend'])
    except Exception as exc:gate['error']=f'{type(exc).__name__}: {exc}'
    gate['status']=gate['execution_status'];write_json(out/'gate_physics_unit.json',gate);seal(out)
    print(json.dumps({'output':str(out),**gate},ensure_ascii=False,indent=2));return 0 if gate['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
