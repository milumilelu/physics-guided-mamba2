"""Train/save/reload/select smoke on separate synthetic families, never G2 data."""
import sys,importlib.util,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch,yaml
from src.task_state_learning.artifacts import ROOT,new_run,write_json,seal
from src.task_state_learning.benchmarks import sample_b1_families,b1_reference,terminal_arrays
from src.task_state_learning.losses import PairSampler,terminal_scale
from src.task_state_learning.training import fit,predict
from src.task_state_learning.grouping import require


def main():
    path=ROOT/'config/task_state_v1/repair_smoke.yaml'
    c=yaml.safe_load(path.read_text(encoding='utf-8'));out=new_run('REPAIR_TRAINING_SMOKE',path)
    spec=importlib.util.spec_from_file_location('b1_runner',ROOT/'experiments/task_state_v1/04_b1_terminal.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    torch.set_num_threads(2);rows=[]
    try:
        require(c['device']!='cuda' or torch.cuda.is_available(),'GPU smoke requires CUDA')
        require(set(c['tracks'])=={'coupled','null'},'Invalid track names')
        for track in c['tracks']:
            families=sample_b1_families(c['families'],np.random.SeedSequence(c['seed']),lengths=c['token_lengths'])
            truth=b1_reference(families['controls'],families['block_duration'],coupling=track=='coupled')
            arrays=terminal_arrays(families,truth,[f'smoke_{i}' for i in range(c['families'])])
            data=runner.tensors(arrays);scale=terminal_scale(data['terminal_x']);anchors=data['terminal_x'][:4]
            for name in c['models']:
                model=runner.make_model(name,c['seed'])
                dp={'anchors':anchors,'sampler':PairSampler(c['families'],8,c['seed'])} if name.endswith('_DP') else None
                result=fit(model,data,data,lr=1e-3,seed=c['seed'],device=c['device'],fixed_epochs=c['epochs'],
                           checkpoint_dir=out/'jobs'/f'{track}_{name}',dp=dp)
                prediction=predict(result['model'],data)
                restored=runner.make_model(name,c['seed'])
                restored.load_state_dict(torch.load(out/'jobs'/f'{track}_{name}'/'selected_state.pt',weights_only=True))
                replay=predict(restored,data)
                require(torch.equal(prediction,replay),'Checkpoint replay mismatch')
                regret,_=runner.sealed_design(restored,data,anchors,scale,None,out,f'{track}_{name}',c['seed'])
                require(np.isfinite(regret.regret).all() and (regret.regret>=-1e-12).all(),'Invalid sealed regret')
                rows.append({'track':track,'model':name,'finite':bool(torch.isfinite(prediction).all()),
                             'replay_max_error':float((prediction-replay).abs().max()),'design_pairs':len(regret)})
                pd.DataFrame(rows).to_csv(out/'checks.csv',index=False)
                print(f'{track}/{name}: train/save/reload/select PASS',flush=True)
        gate={'status':'PASS','completed_jobs':len(rows)}
    except Exception as exc:gate={'status':'FAIL','error':f'{type(exc).__name__}: {exc}','completed_jobs':len(rows)}
    gate.update(scope=c['scope'],training_executed=True,hypothesis_status='NOT_TESTED',formal_G2=False)
    write_json(out/'gate.json',gate);seal(out)
    print(json.dumps({'output':str(out),**gate},ensure_ascii=False,indent=2))
    return 0 if gate['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
