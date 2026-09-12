"""Create the pilot metrics and Chinese summary from an OOF run."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.mamba_depth_quality.metrics import component_balanced_mae

def _q2(g, target):
    y=g[f'observed_{target}_um'].to_numpy(float); p=g[f'predicted_{target}_um'].to_numpy(float)
    null=g[f'train_null_{target}_um'].to_numpy(float)
    den=np.sum((y-null)**2); return float(1-np.sum((y-p)**2)/den) if den>0 else float('nan')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',required=True); a=ap.parse_args(); run=Path(a.run)
    oof=pd.read_csv(run/'oof_predictions.csv'); metrics=[]
    for mid,g in oof.groupby('model_id',sort=False):
        for t in ('D','Sq'):
            y=g[f'observed_{t}_um'].to_numpy(float); p=g[f'predicted_{t}_um'].to_numpy(float)
            metrics.append({'model_id':mid,'target':t,'n':len(g),'coverage':float(len(g)/180),'MAE_um':float(np.mean(np.abs(y-p))),'RMSE_um':float(np.sqrt(np.mean((y-p)**2))),'component_balanced_MAE_um':component_balanced_mae(y,p,g.component_id),'OOF_train_mean_Q2':_q2(g,t),'negative_depth_fraction':float(np.mean(p<0)) if t=='D' else 0.0})
    mdf=pd.DataFrame(metrics); mdf.to_csv(run/'metrics_by_target.csv',index=False,lineterminator='\n')
    byfold=[]
    for (mid,fold,t),g in oof.assign(target='').groupby(['model_id','outer_fold','target']): pass
    # Explicit per-fold table for auditability.
    for (mid,fold),g in oof.groupby(['model_id','outer_fold']):
        for t in ('D','Sq'):
            y=g[f'observed_{t}_um']; p=g[f'predicted_{t}_um']; byfold.append({'model_id':mid,'outer_fold':int(fold),'target':t,'n':len(g),'MAE_um':float(np.mean(np.abs(y-p))),'RMSE_um':float(np.sqrt(np.mean((y-p)**2)))})
    pd.DataFrame(byfold).to_csv(run/'metrics_by_fold_role.csv',index=False,lineterminator='\n')
    main='MAMBA_PHY_K4_AUX'; pairs=[]
    for comp in ['TREE_U_DQ','STATIC_PHY_K4_AUX','GRU_PHY_K4_AUX','MAMBA_RAW_K4_AUX','MAMBA_PHY_K4_NOAUX','MAMBA_PHY_K32_AUX']:
        for t in ('D','Sq'):
            a1=mdf[(mdf.model_id==comp)&(mdf.target==t)].iloc[0]; a2=mdf[(mdf.model_id==main)&(mdf.target==t)].iloc[0]
            pairs.append({'main_model':main,'comparator':comp,'target':t,'delta_MAE_um':float(a1.MAE_um-a2.MAE_um),'delta_RMSE_um':float(a1.RMSE_um-a2.RMSE_um),'positive_favors_main':True})
    pd.DataFrame(pairs).to_csv(run/'paired_differences.csv',index=False,lineterminator='\n')
    lines=['# mamba_depth_quality_v1 pilot 结果','',f'- OOF 覆盖：{len(oof)} 条记录，模型数 {oof.model_id.nunique()}，外层折 {oof.outer_fold.nunique()}。',('- 本次 Mamba 模型使用 reference_pure_torch_ssd_mamba2_verified_adapter_v1；训练轮数为 fast profile 的 12 轮。' if any(oof.backend_id.astype(str).str.contains('mamba').tolist()) else '- 本次使用轻量代理，未完成完整 Mamba 训练。'),'- seed=17，5 折；结果为开发性 pilot，不是独立确认。','', '## 指标','']
    lines += ['|模型|目标|MAE (µm)|RMSE (µm)|Q²|','|---|---|---:|---:|---:|']
    for _,r in mdf.iterrows(): lines.append(f"|{r.model_id}|{r.target}|{r.MAE_um:.4f}|{r.RMSE_um:.4f}|{r.OOF_train_mean_Q2:.4f}|")
    lines += ['', '## 判读', '', 'Q1–Q4 仅完成可复核的 OOF 管线与轻量代理运行；不能将代理结果解释为 Mamba 增益或已辨识物理记忆。下一步优先运行同一固定折的真实纯 PyTorch Mamba 重拟合，再决定是否展开多种子。']
    (run/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(run/'RESULTS_ZH.md')
if __name__=='__main__': main()

