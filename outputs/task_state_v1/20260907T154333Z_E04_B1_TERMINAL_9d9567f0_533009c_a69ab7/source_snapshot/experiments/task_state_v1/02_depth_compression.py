"""E02: nested MAIN180 Ridge/GAM diagnostics with fold-local depth cross-fitting."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits
from src.task_state_learning.artifacts import ROOT,new_run,verify_seal,write_json,sha256,seal
from src.task_state_learning.diagnostics import Y,group_weights,target_statistics,per_row_risk,crossfit_depth,fit_predict,raw_features
from src.task_state_learning.grouping import require


def run(args,out,c,audit):
    targets=pd.read_csv(audit/'frozen_targets.csv')
    split=pd.read_csv(audit/'split_manifest.csv')
    inner=pd.read_csv(audit/'inner_split_manifest.csv')
    frame=targets.merge(split[['dataset_index','component_id','outer_fold']],on='dataset_index',validate='one_to_one')
    records=[]; risks=[]; tuning=[]; traces=[]; support=[]
    for fold in range(5):
        train=frame[frame.outer_fold.ne(fold)].copy(); test=frame[frame.outer_fold.eq(fold)].copy()
        inner_table=inner[inner.outer_fold.eq(fold)]
        partitions=[]
        for k in range(3):
            val_ids=inner_table[inner_table.inner_fold.eq(k)].dataset_index
            a=train[~train.dataset_index.isin(val_ids)].copy();b=train[train.dataset_index.isin(val_ids)].copy()
            da,db,trace=crossfit_depth(a,b,c['depth_crossfit_alpha'])
            traces.extend([{'outer_fold':fold,'inner_fold':k,**t} for t in trace])
            partitions.append((a,b,(da,db)))
        da,db,trace,depth_model=crossfit_depth(train,test,c['depth_crossfit_alpha'],return_model=True)
        traces.extend([{'outer_fold':fold,'inner_fold':-1,**t} for t in trace])
        null,scale=target_statistics(train)
        # Descriptive support geometry only; never removes rows from scoring.
        ux=raw_features(train,'M_U');ut=raw_features(test,'M_U')
        w=group_weights(train);mu=w@ux;sd=np.sqrt(np.maximum(w@((ux-mu)**2),1e-12))
        pair=np.linalg.norm(((ux[:,None]-ux[None,:])/sd),axis=2)
        pair[train.component_id.to_numpy()[:,None]==train.component_id.to_numpy()[None,:]]=np.inf
        threshold=float(np.quantile(pair.min(axis=1),.95))
        distance=np.linalg.norm((ut[:,None]-ux[None,:])/sd,axis=2).min(axis=1)
        support.extend([{'outer_fold':fold,'dataset_index':int(idx),'nearest_train_distance':float(d),
                         'train_threshold':threshold,'support_status':'IN_SUPPORT' if d<=threshold else 'OOS',
                         'scope':'unconditional_process_diagnostic_not_design_support_gate'}
                        for idx,d in zip(test.dataset_index,distance)])
        for family in c['families']:
            for variant in c['variants']:
                choices=[]
                for alpha in c['alpha_grid']:
                    component_losses=[]
                    for a,b,depth in partitions:
                        pred=fit_predict(a,b,variant,family,float(alpha),depth)
                        _,sc=target_statistics(a)
                        values=per_row_risk(b[Y],pred,sc)
                        component_losses.extend(pd.Series(values,index=b.component_id).groupby(level=0).mean().tolist())
                    score=float(np.mean(component_losses));choices.append((score,float(alpha)))
                    tuning.append({'outer_fold':fold,'model_id':family,'variant':variant,'alpha':alpha,'inner_risk':score})
                score,alpha=min(choices)
                pred,estimator=fit_predict(train,test,variant,family,alpha,(da,db),return_model=True)
                checkpoint=out/'checkpoints'/f'{family}_{variant}_fold{fold}.joblib'
                checkpoint.parent.mkdir(exist_ok=True)
                joblib.dump({'estimator':estimator,'depth_estimator':depth_model if variant=='M_UDhat' else None,
                             'variant':variant,'train_ids':train.dataset_index.tolist(),'targets':Y},checkpoint)
                model_hash=sha256(checkpoint)
                rowrisk=per_row_risk(test[Y],pred,scale)
                for pos,row in enumerate(test.itertuples(index=False)):
                    identity={'run_id':out.name,'experiment_id':'E02','model_id':family,'variant':variant,
                              'seed':17,'model_sha':model_hash,'outer_fold':fold,'dataset_index':int(row.dataset_index),'session_id':row.session_id,
                              'sample_id':int(row.sample_id),'component_id':row.component_id,'selected_alpha':alpha,
                              'information_track':'ORACLE_DIAGNOSTIC' if variant in ['M_D','M_Dh','M_UD'] else 'PROCESS_ONLY',
                              'physical_valid':'NOT_APPLICABLE','band_valid':True,
                              'support_status':'IN_SUPPORT' if distance[pos]<=threshold else 'OOS'}
                    risks.append({**identity,'risk':float(rowrisk[pos])})
                    for j,target in enumerate(Y):
                        records.append({**identity,'target':target,'observed':float(getattr(row,target)),
                                        'predicted':float(pred[pos,j]),'train_null':float(null[j]),'train_scale':float(scale[j])})
        print(f'E02 outer fold {fold+1}/5 completed',flush=True)
    oof=pd.DataFrame(records);risk=pd.DataFrame(risks)
    require(len(oof)==180*len(c['families'])*len(c['variants'])*len(Y),'OOF coverage mismatch')
    require(not oof.duplicated(['model_id','variant','dataset_index','target']).any(),'OOF duplicates')
    oof['split_sha']=sha256(audit/'split_manifest.csv');oof['data_sha']=sha256(audit/'frozen_targets.csv')
    oof.to_csv(out/'diagnostic_oof.csv',index=False)
    oof[oof.variant.str.startswith('M_')].to_csv(out/'depth_conditioning_oof.csv',index=False)
    oof[oof.variant.isin(['C1','C2','C3','C4','M_U'])].to_csv(out/'input_compression_oof.csv',index=False)
    risk.to_csv(out/'per_sample_risk.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'inner_tuning.csv',index=False)
    pd.DataFrame(support).to_csv(out/'support_diagnostics.csv',index=False)
    write_json(out/'depth_crossfit_trace.json',traces)
    metricrows=[]
    for (model,variant,target),g in oof.groupby(['model_id','variant','target']):
        w=group_weights(g);error=(g.predicted-g.observed).to_numpy();nullerr=(g.train_null-g.observed).to_numpy()
        denominator=float(w@(nullerr**2))
        metricrows.append({'model_id':model,'variant':variant,'target':target,'MAE':float(w@np.abs(error)),
                           'RMSE':float(np.sqrt(w@(error**2))),'group_balanced_Q2':1-float(w@(error**2))/denominator if denominator>0 else np.nan,
                           'coverage':len(g)})
    pd.DataFrame(metricrows).to_csv(out/'native_metrics.csv',index=False)
    summary=risk.groupby(['model_id','variant','component_id']).risk.mean().groupby(level=[0,1]).mean().reset_index()
    summary.to_csv(out/'risk_summary.csv',index=False)
    risk.groupby(['model_id','variant','outer_fold','session_id','component_id']).risk.mean().groupby(level=[0,1,2,3]).mean().reset_index().to_csv(out/'fold_session_risk.csv',index=False)
    contrastrows=[]
    contrasts=[('M_Dh','M_D'),('M_UD','M_U'),('M_UDhat','M_U')]+[('M_U',v) for v in ['C1','C2','C3','C4']]
    grouped=risk.groupby(['model_id','variant','outer_fold','component_id']).risk.mean()
    rng=np.random.default_rng(c['bootstrap_seed'])
    for model in c['families']:
        for first,second in contrasts:
            delta=(grouped.loc[(model,first)]-grouped.loc[(model,second)]).sort_index()
            # Fixed-fold component bootstrap preserves the fitted-model strata.
            sums=np.zeros(c['bootstrap_replicates']);count=0
            for fold in range(5):
                values=delta.loc[fold].to_numpy();count+=len(values)
                sums+=values[rng.integers(len(values),size=(c['bootstrap_replicates'],len(values)))].sum(axis=1)
            draws=sums/count
            contrastrows.append({'model_id':model,'contrast':first+' minus '+second,'mean_difference':float(delta.mean()),
                                 'ci95_low':float(np.quantile(draws,.025)),'ci95_high':float(np.quantile(draws,.975)),
                                 'scope':'descriptive_conditional_fixed_OOF_no_confirmatory_claim'})
    pd.DataFrame(contrastrows).to_csv(out/'paired_diagnostic_intervals.csv',index=False)
    return {'oof_rows':len(oof),'models':len(c['families'])*len(c['variants']),'outer_refits':5*len(c['families'])*len(c['variants']),
            'scope':'MAIN180 diagnostic OOF; HIST200 rerun and support-matched scientific interpretation not completed'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--audit-run',required=True,type=Path)
    ap.add_argument('--observer-run',required=True,type=Path)
    ap.add_argument('--config',type=Path,default=ROOT/'config/task_state_v1/depth_compression.yaml');args=ap.parse_args()
    audit=verify_seal(args.audit_run);obs=verify_seal(args.observer_run)
    require(json.loads((audit/'gate_G0.json').read_text(encoding='utf-8'))['status']=='PASS','G0 not PASS')
    require(json.loads((obs/'gate_G1_OBS.json').read_text(encoding='utf-8'))['status']=='PASS','Canonical target parity not PASS')
    oc=yaml.safe_load((obs/'config_resolved.yaml').read_text(encoding='utf-8'))
    require(oc['audit_release_sha256']==sha256(audit/'release_manifest.json'),'Observer is tied to another audit')
    c=yaml.safe_load(args.config.read_text(encoding='utf-8'));require(c['allowed_to_train'] and not c['deep_training_allowed'],'Invalid E02 permission scope')
    out=new_run('E02',args.config);t0=time.perf_counter()
    gate={'experiment_id':'E02','execution_status':'FAIL','hypothesis_status':'NOT_TESTED','training_executed':False,'deep_training_executed':False}
    try:
        suite=subprocess.run([sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests/task_state_v1','-v'],cwd=ROOT,capture_output=True,encoding='utf-8')
        (out/'tests.log').write_text(suite.stdout+suite.stderr,encoding='utf-8');require(suite.returncode==0,'Tests failed')
        c.update(audit_run=str(audit.relative_to(ROOT)),observer_run=str(obs.relative_to(ROOT)),audit_release_sha256=sha256(audit/'release_manifest.json'))
        (out/'config_resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
        with threadpool_limits(limits=1):
            gate.update(training_executed=True)
            result=run(args,out,c,audit)
        gate.update(execution_status='PASS',metric_status='VALID',**result)
    except Exception as exc: gate['error']=f'{type(exc).__name__}: {exc}'
    gate['status']=gate['execution_status'];gate['walltime_seconds']=time.perf_counter()-t0
    write_json(out/'E02_gate.json',gate);seal(out);print(json.dumps({'output':str(out),**gate},ensure_ascii=False,indent=2))
    return 0 if gate['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
