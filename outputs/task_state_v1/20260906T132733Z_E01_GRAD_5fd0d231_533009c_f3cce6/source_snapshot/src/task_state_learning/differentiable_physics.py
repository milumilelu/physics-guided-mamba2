"""Functional differentiable SI stepper, with no in-place persistent-state updates."""
from __future__ import annotations
import math
import torch
from .physics import PhysicsParameters


def step(depth,incubation,x_m,y_m,*,pulse_energy_J,duration_s,beam_x_m,beam_y_m,
         parameters: PhysicsParameters,switch='P11',closure_log=0.,smoothing_epsilon=0.):
    if switch not in ('P00','P10','P01','P11'):
        raise ValueError('Unknown physical switch')
    if depth.shape != incubation.shape or depth.dtype != incubation.dtype:
        raise ValueError('Physical state shape/dtype mismatch')
    if depth.dtype not in (torch.float32,torch.float64):
        raise ValueError('Reference stepper only supports float32/float64')
    convert=lambda v: torch.as_tensor(v,dtype=depth.dtype,device=depth.device)
    energy=convert(pulse_energy_J);duration=convert(duration_s);b=convert(closure_log)
    if not bool(torch.isfinite(energy).all() and (energy>=0).all() and torch.isfinite(duration).all() and (duration>0).all()):
        raise ValueError('Invalid pulse')
    if not bool(torch.isfinite(depth).all() and torch.isfinite(incubation).all() and (depth>=0).all() and (incubation>=0).all()):
        raise ValueError('Invalid physical state')
    if not bool(torch.isfinite(b).all() and (b.abs()<=math.log(2)+1e-7).all()):
        raise ValueError('Closure bound violation')
    if smoothing_epsilon<0:
        raise ValueError('Negative smoothing epsilon')
    p=parameters
    waist=p.waist_m*torch.sqrt(1+(depth/p.rayleigh_m)**2) if switch[1]=='1' else convert(p.waist_m)
    positive_energy=energy>0
    safe_energy=torch.where(positive_energy,energy,torch.ones_like(energy))
    log_fluence=torch.log(2*safe_energy/(math.pi*waist**2))-2*((convert(x_m)-convert(beam_x_m))**2+(convert(y_m)-convert(beam_y_m))**2)/waist**2
    log_F1=math.log(p.F1_reference_J_m2)+p.gamma_F*torch.log(duration/p.reference_duration_s)
    log_threshold=log_F1+b
    if switch[2]=='1':
        log_threshold=log_threshold+torch.log(p.saturation_ratio+(1-p.saturation_ratio)*torch.exp(-incubation))
    log_ratio=log_fluence-log_threshold
    positive=torch.where(positive_energy,torch.clamp_min(log_ratio,0.),torch.zeros_like(log_ratio))
    if smoothing_epsilon:
        positive=torch.where(positive<smoothing_epsilon,positive**2/(2*smoothing_epsilon),positive-smoothing_epsilon/2)
    delta=p.delta_reference_m*(duration/p.reference_duration_s)**p.gamma_delta
    next_depth=depth+delta*positive
    next_q=incubation
    if switch[2]=='1':
        next_q=incubation+p.kappa*torch.where(positive_energy,torch.sigmoid(log_fluence-log_F1),torch.zeros_like(log_fluence))
    return next_depth,next_q
