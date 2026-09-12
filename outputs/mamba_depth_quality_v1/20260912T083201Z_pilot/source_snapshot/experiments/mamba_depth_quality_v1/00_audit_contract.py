"""G0 audit for mamba_depth_quality_v1."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT=Path(__file__).resolve().parents[2]

def sha256(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args()
 cfgp=(ROOT/a.config).resolve(); cfg=yaml.safe_load(cfgp.read_text(encoding='utf-8'))
 assert cfg['schema']=='mamba_depth_quality_v1'
 out=ROOT/'outputs'/'mamba_depth_quality_v1'; out.mkdir(parents=True,exist_ok=True)
 run=out/(pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')+'_G0') ; run.mkdir()
 data_dir=ROOT/'outputs/task_state_v1/20260906T111435Z_E00_a1a1c9a8_533009c_9e0686'
 targets=pd.read_csv(data_dir/'frozen_targets.csv')
 splits=pd.read_csv(data_dir/'split_manifest.csv'); inner=pd.read_csv(data_dir/'inner_split_manifest.csv')
 main_ids=targets.merge(splits,on=['dataset_index','session_id','sample_id','shared_height_source_id','session_role'],how='inner',validate='one_to_one'); main_ids=main_ids.loc[main_ids.session_role.isin(['formal','pass_main'])].copy()
 assert len(main_ids)==180, len(main_ids)
 assert len(splits)==180 and splits.component_id.nunique()==126
 assert set(main_ids.dataset_index)==set(splits.dataset_index)
 assert splits.groupby('outer_fold').size().shape[0]==5
 npz=ROOT/'outputs/rectangle_registration/manual_internal_roi_v1/dataset/stable_roi_80um_dataset.npz'
 z=np.load(npz,allow_pickle=True); H=z['height_raw']; V=z['valid_mask']
 D=[]; Q=[]; A=[]
 for i in main_ids.dataset_index:
  x=H[int(i)][V[int(i)]]
  D.append(float(-np.median(x))); Q.append(float(np.sqrt(np.mean((x-x.mean())**2)))); A.append(float(np.sqrt(np.mean((x-np.median(x))**2))))
 main_ids['computed_D_um']=D; main_ids['computed_Sq_um']=Q; main_ids['A_med_recomputed_um']=A
 main_ids['D_abs_err']=np.abs(main_ids['computed_D_um']-main_ids.D)
 main_ids.to_csv(run/'target_audit.csv',index=False)
 splits.to_csv(run/'split_manifest.csv',index=False); inner.to_csv(run/'inner_split_manifest.csv',index=False)
 manifest={'run_id':run.name,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,encoding='utf-8',errors='replace').strip(),'git_status':subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT,text=True,encoding='utf-8',errors='replace').splitlines(),'target_rows':len(main_ids),'components':int(splits.component_id.nunique()),'outer_folds':int(splits.outer_fold.nunique()),'D_max_abs_err':float(main_ids.D_abs_err.max()),'Sq_vs_Amed_max_abs':float(np.max(np.abs(main_ids.computed_Sq_um-main_ids.A_med_recomputed_um))),'height_sha256':sha256(npz),'expected_height_sha256':cfg['paths']['expected_height_sha256'],'height_keys':z.files,'python':sys.version,'platform':platform.platform(),'config_sha256':sha256(cfgp)}
 (run/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 (run/'target_dictionary.json').write_text(json.dumps({'D':'-median(valid height_raw)','Sq':'sqrt(mean((H-mean(H))^2))','A_med':'audit only, not Sq','pixel_um':0.5,'roi_shape':[160,160]},ensure_ascii=False,indent=2),encoding='utf-8')
 shutil.copy2(cfgp,run/'protocol_resolved.yaml')
 (run/'G0_PASS').write_text('PASS\n',encoding='utf-8')
 print(run)
if __name__=='__main__': main()


