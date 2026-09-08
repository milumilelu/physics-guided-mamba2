"""Generate frozen B1 families and verify RK4 against refinement and SciPy."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,time,hashlib,subprocess
import numpy as np
import pandas as pd
import yaml
from scipy.integrate import solve_ivp
from src.task_state_learning.artifacts import ROOT,new_run,write_json,seal,input_manifest
from src.task_state_learning.benchmarks import b1_reference,b1_rhs,sample_b1_families,terminal_arrays,load_terminal_only
from src.task_state_learning.grouping import require


def independent(u,durations,coupling,c):
    state=np.zeros(4)
    for command,duration in zip(u,durations):
        sol=solve_ivp(lambda t,y:b1_rhs(y,command,coupling),(0.,duration),state,
                      method=c['independent_method'],rtol=c['independent_rtol'],atol=c['independent_atol'])
        require(sol.success,'Independent B1 solve failed')
        state=sol.y[:,-1]
    return state


def main():
    path=ROOT/'config/task_state_v1/b1_reference.yaml'
    c=yaml.safe_load(path.read_text(encoding='utf-8'))
    out=new_run('E04_B1_REFERENCE',path)
    (out/'terminal_only').mkdir();(out/'evaluator_only').mkdir()
    started=time.perf_counter();rows=[];identities=[];files=[]
    try:
        tests=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests/task_state_v1',
                              '-p','test_benchmarks.py','-v'],cwd=ROOT,capture_output=True,encoding='utf-8')
        (out/'tests.log').write_text(tests.stdout+tests.stderr,encoding='utf-8')
        require(tests.returncode==0,'B1 contract tests failed')
        streams=np.random.SeedSequence(c['seed']).spawn(len(c['families']))
        for (split,count),seed in zip(c['families'].items(),streams):
            f=sample_b1_families(count,seed,lengths=c['token_lengths'],dt_range=c['token_dt_range'])
            ids=[]
            for i in range(count):
                digest=hashlib.sha256(f['controls'][i].tobytes()+f['block_duration'][i].tobytes()).hexdigest()
                ids.append(digest)
                identities.append({'split':split,'family_index':i,'family_id':digest,
                                   'token_count':int(f['token_count'][i]),'token_dt':float(f['token_dt'][i]),
                                   'seed_spawn_key':str(seed.spawn_key)})
            for coupling in c['coupling']:
                label='coupled' if coupling else 'null'
                terminal,blocks=b1_reference(f['controls'],f['block_duration'],coupling=coupling,
                                             max_step=c['reference_max_step'],return_blocks=True)
                target=out/'terminal_only'/f'{split}_{label}.npz'
                np.savez_compressed(target,**terminal_arrays(f,terminal,ids))
                load_terminal_only(target);files.append(target)
                evaluation=out/'evaluator_only'/f'{split}_{label}.npz'
                np.savez_compressed(evaluation,**f,full_terminal_state=terminal,block_boundary_states=blocks,family_id=ids)
                files.append(evaluation)
                n=c['reference_points_per_split']
                refined=b1_reference(f['controls'][:n],f['block_duration'][:n],coupling=coupling,max_step=c['refined_max_step'])
                for i in range(n):
                    alt=independent(f['controls'][i],f['block_duration'][i],coupling,c)
                    for method,reference in [('half_step',refined[i]),('independent',alt)]:
                        error=np.abs(terminal[i]-reference)/np.maximum(np.abs(reference),c['reference_absolute_scale'])
                        rows.append({'split':split,'family_id':ids[i],'coupling':coupling,'method':method,
                                     'max_relative_error':float(error.max()),'passed':bool((error<=c['reference_relative_tolerance']).all())})
                pd.DataFrame(rows).to_csv(out/'reference_checks.csv',index=False)
                print(f'{split} {label}: {count} terminal histories generated and checked',flush=True)
        identity=pd.DataFrame(identities)
        require(not identity.family_id.duplicated().any(),'B1 family overlap between splits')
        identity.to_csv(out/'family_manifest.csv',index=False)
        files.append(out/'family_manifest.csv')
        hashes,digest=input_manifest(files)
        write_json(out/'dataset_hashes.json',{'sha256':digest,'files':hashes})
        require(all(r['passed'] for r in rows),'B1 reference tolerance failed')
        gate={'status':'PASS','family_counts':c['families'],'reference_checks':len(rows),
              'max_relative_error':max(r['max_relative_error'] for r in rows),'dataset_sha256':digest}
    except Exception as exc:
        gate={'status':'FAIL','error':f'{type(exc).__name__}: {exc}'}
    gate.update(gate_scope='B1_REFERENCE_AND_DATA_ISOLATION',training_executed=False,
                hypothesis_status='NOT_TESTED',remaining=['learner_backend_gate','B1_training_null_continuation_OOD'])
    write_json(out/'gate_G1_B1_REF.json',gate)
    write_json(out/'cost.json',{'walltime_seconds':time.perf_counter()-started})
    (out/'config_resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
    seal(out)
    print(json.dumps({'output':str(out),**gate},ensure_ascii=False,indent=2))
    return 0 if gate['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
