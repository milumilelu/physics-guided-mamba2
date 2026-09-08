"""SI-only NumPy reference ablation stepper; numerical model, not calibrated material truth."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .grouping import require


@dataclass(frozen=True)
class PhysicsParameters:
    waist_m: float
    wavelength_m: float
    m_squared: float
    F1_reference_J_m2: float
    delta_reference_m: float
    kappa: float
    saturation_ratio: float
    gamma_F: float = 0.
    gamma_delta: float = 0.
    reference_duration_s: float = 1e-12

    def __post_init__(self):
        require(all(np.isfinite(v) for v in vars(self).values()),'Nonfinite physical parameter')
        require(all(getattr(self,k)>0 for k in ['waist_m','wavelength_m','m_squared','F1_reference_J_m2',
                                               'delta_reference_m','reference_duration_s']),'Nonpositive physical parameter')
        require(self.kappa>=0 and 0<self.saturation_ratio<=1,'Invalid incubation parameter')

    @property
    def rayleigh_m(self):
        return np.pi*self.waist_m**2/(self.m_squared*self.wavelength_m)


def positive_part(value, epsilon=0.):
    value=np.asarray(value,dtype=np.float64)
    require(epsilon>=0,'Negative smoothing epsilon')
    if epsilon==0: return np.maximum(value,0.)
    positive=np.maximum(value,0.)
    return np.where(positive<epsilon,positive**2/(2*epsilon),positive-epsilon/2)


def step(depth,incubation,x_m,y_m,*,pulse_energy_J,duration_s,beam_x_m,beam_y_m,
         parameters,switch='P11',closure_log=0.,smoothing_epsilon=0.):
    require(switch in ['P00','P10','P01','P11'],'Unknown physical switch')
    d=np.asarray(depth,dtype=np.float64);q=np.asarray(incubation,dtype=np.float64)
    require(d.shape==q.shape,'Physical state shape mismatch')
    require(np.isfinite(d).all() and np.isfinite(q).all() and (d>=0).all() and (q>=0).all(),'Invalid physical state')
    require(np.isfinite(pulse_energy_J) and pulse_energy_J>=0 and np.isfinite(duration_s) and duration_s>0,'Invalid pulse')
    b=np.asarray(closure_log,dtype=float)
    require(np.isfinite(b).all() and (np.abs(b)<=np.log(2)+1e-15).all(),'Closure bound violation')
    require(np.isfinite(x_m).all() and np.isfinite(y_m).all() and np.isfinite([beam_x_m,beam_y_m]).all(),'Invalid position')
    if pulse_energy_J==0:
        return d.copy(),q.copy()
    p=parameters;defocus=switch[1]=='1';incubate=switch[2]=='1'
    waist=p.waist_m*np.sqrt(1+(d/p.rayleigh_m)**2) if defocus else p.waist_m
    fluence=(2*pulse_energy_J/(np.pi*waist**2))*np.exp(-2*((np.asarray(x_m)-beam_x_m)**2+(np.asarray(y_m)-beam_y_m)**2)/waist**2)
    F1=p.F1_reference_J_m2*(duration_s/p.reference_duration_s)**p.gamma_F
    delta=p.delta_reference_m*(duration_s/p.reference_duration_s)**p.gamma_delta
    threshold=F1*(p.saturation_ratio+(1-p.saturation_ratio)*np.exp(-q)) if incubate else F1
    threshold=threshold*np.exp(b)
    log_ratio=np.full(np.broadcast_shapes(np.shape(fluence),np.shape(threshold)),-np.inf)
    np.log(fluence/threshold,out=log_ratio,where=fluence>0)
    new_d=d+delta*positive_part(log_ratio,smoothing_epsilon)
    new_q=q+p.kappa*(fluence/(F1+fluence)) if incubate else q.copy()
    require(np.isfinite(new_d).all() and np.isfinite(new_q).all(),'Nonfinite solver output')
    return new_d,new_q


def repeat_identical_pulse(depth,incubation,x_m,y_m,*,multiplicity,**kwargs):
    """Exact multiplicity reference: never replace m pulses by one pulse of m*Ep."""
    require(isinstance(multiplicity,int) and multiplicity>=0,'Invalid pulse multiplicity')
    d=np.array(depth,dtype=np.float64,copy=True);q=np.array(incubation,dtype=np.float64,copy=True)
    for _ in range(multiplicity):
        d,q=step(d,q,x_m,y_m,**kwargs)
    return d,q
