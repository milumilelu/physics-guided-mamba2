import numpy as np
from src.mamba_depth_quality.data import load_main180

def test_main180_and_sq():
 f,h,v=load_main180(); assert len(f)==180 and f.component_id.nunique()==126
 x=h[0][v[0]]; assert abs(np.sqrt(np.mean((x-x.mean())**2))-f.iloc[0].Sq)<1e-8

def test_shift_invariance():
 f,h,v=load_main180(); x=h[0][v[0]]; q=np.sqrt(np.mean((x-x.mean())**2)); y=x+3.2; assert abs(q-np.sqrt(np.mean((y-y.mean())**2)))<1e-12
