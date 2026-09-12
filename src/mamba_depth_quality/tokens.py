import numpy as np
from .reference import reference_components, reference_depth

FEATURE_NAMES=["log_tau","log_f","log_v","log_h","pass_index","within_pass_progress",
"log1p_nu_start","log1p_nu_end","log1p_delta_nu","log_Ep_SI","log_F0_SI",
"log_dx_over_2w","log_h_over_2w","log_F0_over_F1","D_ref_start_um","D_ref_end_um"]

def build_tokens(frame, reference=None, points_per_pass=8, raw_ablation=False, normalize=None):
    if reference is None: raise ValueError("reference required")
    f,v,h,N,Ep,F0,nup=reference_components(frame,reference)
    out=[]; masks=[]
    for i in range(len(frame)):
        rows=[]; dprev=0.; total=nup[i]*N[i]
        for p in range(int(N[i])):
            for j in range(points_per_pass):
                a=(p+j/points_per_pass)*nup[i]; b=(p+(j+1)/points_per_pass)*nup[i]
                d0=reference.delta_m*a*max(np.log(F0[i]/reference.F1_J_per_m2),0)*1e6
                d1=reference.delta_m*b*max(np.log(F0[i]/reference.F1_J_per_m2),0)*1e6
                row=[np.log(max(frame.pulse_duration_fs.iloc[i]*1e-15,1e-30)),np.log(f[i]),np.log(v[i]),np.log(h[i]),p,j/points_per_pass,
                     np.log1p(a),np.log1p(b),np.log1p(b-a),np.log(max(Ep[i],1e-300)),np.log(max(F0[i],1e-300)),
                     np.log(max(v[i]/f[i]/(2*reference.w0_m),1e-300)),np.log(max(h[i]/(2*reference.w0_m),1e-300)),
                     np.log(max(F0[i]/reference.F1_J_per_m2,1e-300)),d0,d1]
                if raw_ablation: row[13:]=[0.,0.,0.]
                rows.append(row)
        out.append(rows); masks.append([True]*len(rows))
    max_t=max(len(r) for r in out)
    arr=np.zeros((len(out),max_t,len(FEATURE_NAMES)),float)
    msk=np.zeros((len(out),max_t),bool)
    for i,r in enumerate(out):
        arr[i,:len(r)]=r; msk[i,:len(r)]=True
    if normalize is not None: arr=(arr-normalize[0])/np.maximum(normalize[1],1e-12)
    return arr, msk

def fit_token_normalizer(tokens):
    return tokens.mean(axis=(0,1)), tokens.std(axis=(0,1))+1e-8
