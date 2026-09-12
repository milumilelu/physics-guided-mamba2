from dataclasses import dataclass
import numpy as np
from .data import component_weights

@dataclass
class FrozenReference:
    F1_J_per_m2: float
    delta_m: float
    w0_m: float = 0.874e-6
    power_W: float = 5.3333
    profile: object = None
    source_indices: object = None

def _exposure(frame, F1, w0_m, power_W):
    f = frame.frequency_kHz.to_numpy(float)*1e3; v=frame.velocity_mm_s.to_numpy(float)*1e-3
    h=frame.hatch_spacing_um.to_numpy(float)*1e-6; N=frame.pass_count.to_numpy(float)
    P = frame.measured_power_W.to_numpy(float) if "measured_power_W" in frame else np.full(len(frame), power_W)
    Ep=P/f; F0=2*Ep/(np.pi*w0_m*w0_m); A=np.pi*w0_m*w0_m/2
    nu=f*A/(v*h)*N
    c=nu*np.maximum(np.log(np.maximum(F0,1e-300)/F1),0)
    return F0,nu,c

def fit_reference(frame, w0_m=0.874e-6, power_W=5.3333, grid_size=64):
    # D observations are stored in micrometres; the closure parameter is SI
    # metres so reference_depth() can convert the prediction back to um.
    y=frame.D.to_numpy(float)*1e-6; w=component_weights(frame)
    P=frame.measured_power_W.to_numpy(float) if "measured_power_W" in frame else np.full(len(frame),power_W)
    f=frame.frequency_kHz.to_numpy(float)*1e3; Ep=P/f; F0=2*Ep/(np.pi*w0_m*w0_m)
    grid=np.geomspace(0.01*F0.min(),0.95*F0.max(),grid_size)
    prof=[]
    for F1 in grid:
        _,_,c=_exposure(frame,F1,w0_m,power_W)
        delta=max(0.,float(np.sum(w*c*y)/max(np.sum(w*c*c),1e-30)))
        prof.append((float(np.sum(w*(delta*c-y)**2)),float(F1),delta))
    prof.sort(key=lambda q:(q[0],q[1])); best=prof[0]
    return FrozenReference(best[1],best[2],w0_m,power_W,np.asarray(prof),frame.dataset_index.to_numpy())

def reference_depth(frame, ref):
    F0,nu,c=_exposure(frame,ref.F1_J_per_m2,ref.w0_m,ref.power_W)
    return ref.delta_m*c*1e6

def reference_components(frame, ref, points_per_pass=8):
    """Return per-sample exposure endpoints and feature scales in SI."""
    f=frame.frequency_kHz.to_numpy(float)*1e3; v=frame.velocity_mm_s.to_numpy(float)*1e-3
    h=frame.hatch_spacing_um.to_numpy(float)*1e-6; N=frame.pass_count.to_numpy(int)
    P=frame.measured_power_W.to_numpy(float); Ep=P/f; F0=2*Ep/(np.pi*ref.w0_m**2)
    A=np.pi*ref.w0_m**2/2; nu_pass=f*A/(v*h); return f,v,h,N,Ep,F0,nu_pass
