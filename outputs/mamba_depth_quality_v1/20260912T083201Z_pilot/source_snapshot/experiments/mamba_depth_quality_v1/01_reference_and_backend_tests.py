"""G1/G2 reference, token and Mamba numeric checks."""
from __future__ import annotations
import argparse,json,hashlib,sys,shutil
from pathlib import Path
import numpy as np,pandas as pd, torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.mamba_depth_quality.reference import fit_reference
from src.mamba_depth_quality.tokens import build_tokens
from src.mamba_depth_quality.mamba_adapter import MambaAdapter

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--contract-run',required=True); a=ap.parse_args()
 g0=Path(a.contract_run); out=ROOT/'outputs/mamba_depth_quality_v1'; run=out/(pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')+'_G1_G2'); run.mkdir()
 df=pd.read_csv(g0/'target_audit.csv'); ref=fit_reference(df); (run/'reference_parameters.json').write_text(json.dumps({'F1_J_per_m2':ref.F1_J_per_m2,'delta_m':ref.delta_m,'w0_m':ref.w0_m,'profile':ref.profile.tolist()},indent=2),encoding='utf-8')
 tokx,tokmask=build_tokens(df,ref); np.savez(run/'token_fixture.npz',x=tokx,mask=tokmask)
 m=MambaAdapter(d_model=tokx.shape[-1],d_state=8,chunk_size=32)
 x=torch.tensor(tokx[:2],dtype=torch.float32); mask=torch.tensor(tokmask[:2])
 with torch.no_grad(): full=m(x,mask); step=m.forward_stepwise(x,mask)
 maxerr=float((full-step).abs().max());
 assert np.isfinite(maxerr) and maxerr < 1e-4, maxerr
 # padding invariance: append masked zero tokens
 xp=torch.cat([x,torch.zeros(2,3,x.shape[-1])],1); mp=torch.cat([mask,torch.zeros(2,3,dtype=torch.bool)],1)
 with torch.no_grad(): yp=m(x,mask); ypp=m(xp,mp)
 paderr=float((yp-ypp[:,:yp.shape[1]]).abs().max()); assert paderr<1e-5,paderr
 # gradient check
 m.zero_grad(); loss=m(x,mask).sum(); loss.backward(); grad_ok=all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()); assert grad_ok
 report={'reference_id':'RLOG_EQEXP_v1','backend_id':'reference_pure_torch_ssd_mamba2_verified_adapter_v1','full_step_max_abs':maxerr,'padding_invariance_max_abs':paderr,'gradient_finite':bool(grad_ok),'token_shape':list(tokx.shape),'reference_profile_rows':len(ref.profile)}
 (run/'G1_G2_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); (run/'G1_G2_PASS').write_text('PASS\n',encoding='utf-8'); print(run)
if __name__=='__main__': main()





