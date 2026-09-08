"""E01 canonical observer acceptance only; differentiable observer is separate."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import subprocess
import numpy as np
import pandas as pd
import yaml
from src.task_state_learning.artifacts import ROOT, new_run, write_json, verify_seal, seal, sha256
from src.task_state_learning.contracts import load_contract, TARGETS
from src.task_state_learning.observers import CanonicalObserver
from src.task_state_learning.grouping import require


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-run',required=True,type=Path)
    ap.add_argument('--config',type=Path,default=ROOT/'config/task_state_v1/observer_parity.yaml')
    args=ap.parse_args(); audit=verify_seal(args.audit_run)
    require(json.loads((audit/'gate_G0.json').read_text(encoding='utf-8'))['status']=='PASS','G0 not PASS')
    out=new_run('E01_OBS',args.config)
    gate={'experiment_id':'E01','gate_scope':'CANONICAL_OBSERVER','execution_status':'FAIL',
          'hypothesis_status':'NOT_TESTED','G1_complete':False,'differentiable_observer':'NOT_RUN','training_executed':False}
    try:
        c=yaml.safe_load(args.config.read_text(encoding='utf-8'))
        suite=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests/task_state_v1','-p','test_observers.py','-v'],cwd=ROOT,capture_output=True,encoding='utf-8')
        (out/'observer_tests.log').write_text(suite.stdout+suite.stderr,encoding='utf-8')
        require(suite.returncode==0,'Observer tests failed')
        _,frozen,df,_,_=load_contract()
        expected=pd.read_csv(audit/'frozen_targets.csv')
        require(np.array_equal(df.dataset_index,expected.dataset_index),'Upstream identities changed')
        observer=CanonicalObserver(c['zero_threshold'],c['replacement_delta'])
        y,v=observer(frozen['H'],frozen['V'])
        targets=TARGETS+['A2_16_32','entropy_16_32']; rows=[]
        for target in targets:
            a,b=y[target].to_numpy(),expected[target].to_numpy()
            passed=np.isclose(a,b,atol=c['atol'],rtol=c['rtol'])
            for i in range(200):
                rows.append({'dataset_index':i,'target':target,'observed_canonical':float(a[i]),'frozen_target':float(b[i]),
                             'abs_error':float(abs(a[i]-b[i])),'passed':bool(passed[i])})
        parity=pd.DataFrame(rows); parity.to_csv(out/'observer_parity.csv',index=False)
        v.to_csv(out/'observer_validity.csv',index=False)
        require(parity.passed.all() and v.band_valid.all(),'Real observer parity/coverage failed')
        require(not y.composition_replacement_used.any(),'Unexpected replacement on real dataset')
        x=np.arange(160)*.5; a=np.broadcast_to(np.cos(2*np.pi*x/10),(160,160)).copy()
        fields=np.stack([np.full((160,160),-3.),a,a.T,a+a.T,np.random.default_rng(c['synthetic_seed']).normal(size=(160,160))])
        synthetic,validity=observer(fields,np.ones_like(fields,dtype=bool))
        synthetic['case']=['constant','cos_x','cos_y','orthogonal_cos','Gaussian']
        synthetic.to_csv(out/'synthetic_observations.csv',index=False)
        validity.to_csv(out/'observer_undefined_cases.csv',index=False)
        c.update(audit_run=str(audit.relative_to(ROOT)),audit_release_sha256=sha256(audit/'release_manifest.json'))
        (out/'config_resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
        gate.update(execution_status='PASS',gate_id='G1_OBS_CANONICAL',maximum_absolute_error=float(parity.abs_error.max()),
                    compared_values=len(parity),real_coverage=200,full_G1_remaining=['differentiable_observer','physics','AD_FD','backend'])
    except Exception as exc:
        gate['error']=f'{type(exc).__name__}: {exc}'
    gate['status']=gate['execution_status']; write_json(out/'gate_G1_OBS.json',gate);seal(out)
    print(json.dumps({'output':str(out),**gate},ensure_ascii=False,indent=2))
    return 0 if gate['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
