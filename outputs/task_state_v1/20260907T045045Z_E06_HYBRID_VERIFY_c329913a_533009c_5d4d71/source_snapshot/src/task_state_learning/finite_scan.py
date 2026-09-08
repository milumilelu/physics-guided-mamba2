"""Finite raster, pointwise exact pulse recurrence with verified tail truncation.

The model has no lateral coupling: omitted non-ROI state cannot influence ROI
state. All physical pulses in the finite 200um scan are defined, with only far
Gaussian tails skipped numerically. No periodic boundary, continuous exposure,
energy aggregation, interpolation or learned closure is used here.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from numba import njit, prange
from .grouping import require
from .physics import PhysicsParameters


@dataclass(frozen=True)
class FiniteScanConfig:
    scan_region_m: float = 200e-6
    roi_region_m: float = 80e-6
    observation_n: int = 160
    phase_x_pitch: float = 0.
    phase_y_pitch: float = 0.
    tail_waists: float = 6.
    max_window_retries: int = 16
    max_updates_per_point: int = 2000000

    def __post_init__(self):
        require(np.isfinite(list(asdict(self).values())).all(), 'Nonfinite finite-scan setting')
        require(self.scan_region_m>=self.roi_region_m>0 and self.tail_waists>0, 'Invalid finite-scan extent')
        require(self.observation_n==160 and abs(self.roi_region_m-80e-6)<1e-16,
                'Canonical ROI must remain 160 pixels over 80um')
        require(self.max_window_retries>=1 and self.max_updates_per_point>=1, 'Invalid finite-scan budget')


@njit(parallel=True, cache=True)
def _points(xs,ys,nx,ny,dx,h,ox,oy,passes,ep,f1,delta,kappa,rho,w0,zr,
            defocus,incubate,tail,retries,update_budget):
    count=len(xs)
    depth=np.zeros(count); qout=np.zeros(count)
    updates=np.zeros(count,np.int64); attempts=np.zeros(count,np.int64)
    success=np.zeros(count,np.bool_)
    for i in prange(count):
        bound=w0
        for attempt in range(retries):
            attempts[i]=attempt+1
            radius=tail*bound
            jlo=max(0,int(math.ceil((ys[i]-radius-oy)/h)))
            jhi=min(ny-1,int(math.floor((ys[i]+radius-oy)/h)))
            klo=max(0,int(math.ceil((xs[i]-radius-ox)/dx)))
            khi=min(nx-1,int(math.floor((xs[i]+radius-ox)/dx)))
            d=0.;q=0.;used=0
            needed=passes*max(0,jhi-jlo+1)*max(0,khi-klo+1)
            if updates[i]+needed>update_budget:
                break
            for repeat in range(passes):
                for j in range(jlo,jhi+1):
                    ry=ys[i]-(oy+j*h)
                    for k in range(klo,khi+1):
                        rx=xs[i]-(ox+k*dx)
                        w=w0*math.sqrt(1+(d/zr)**2) if defocus else w0
                        logf=math.log(2*ep/(math.pi*w*w))-2*(rx*rx+ry*ry)/(w*w)
                        th=f1*(rho+(1-rho)*math.exp(-q)) if incubate else f1
                        inc=delta*max(0.,logf-math.log(th))
                        if incubate:
                            ratio=logf-math.log(f1)
                            # Stable logistic, preserving tiny tails without overflow.
                            frac=1/(1+math.exp(-ratio)) if ratio>=0 else math.exp(ratio)/(1+math.exp(ratio))
                            q+=kappa*frac
                        d+=inc
                        used+=1
            updates[i]+=used
            final_w=w0*math.sqrt(1+(d/zr)**2) if defocus else w0
            if math.isfinite(d) and math.isfinite(q) and final_w<=bound*(1+1e-12):
                depth[i]=d;qout[i]=q;success[i]=True
                break
            if not math.isfinite(final_w):
                break
            # Repeat the ENTIRE exposure from zero, including previous passes,
            # so newly included earlier pulses have the correct arrival order.
            bound=max(bound*1.5,final_w*1.1)
    return depth,qout,updates,attempts,success


def simulate_points(parameters: PhysicsParameters,x_m,y_m,*,tau_s,f_Hz,v_m_s,h_m,
                    passes,power_W,switch,cfg=FiniteScanConfig()):
    require(switch in ('P00','P10','P01','P11'),'Unknown switch')
    require(np.isfinite([tau_s,f_Hz,v_m_s,h_m,passes,power_W]).all()
            and min(tau_s,f_Hz,v_m_s,h_m,power_W)>0 and int(passes)==passes and passes>=1,'Invalid recipe')
    x,y=np.broadcast_arrays(np.asarray(x_m,dtype=float),np.asarray(y_m,dtype=float))
    require(np.isfinite(x).all() and np.isfinite(y).all(),'Nonfinite observation coordinates')
    dx=v_m_s/f_Hz
    nx=max(1,int(round(cfg.scan_region_m/dx)))
    ny=max(1,int(round(cfg.scan_region_m/h_m)))
    ox=-(nx-1)*dx/2+cfg.phase_x_pitch*dx
    oy=-(ny-1)*h_m/2+cfg.phase_y_pitch*h_m
    p=parameters
    f1=p.F1_reference_J_m2*(tau_s/p.reference_duration_s)**p.gamma_F
    delta=p.delta_reference_m*(tau_s/p.reference_duration_s)**p.gamma_delta
    require(np.isfinite([f1,delta]).all() and f1>0 and delta>0,'Nonfinite duration scales')
    d,q,updates,attempts,ok=_points(x.ravel(),y.ravel(),nx,ny,dx,h_m,ox,oy,int(passes),power_W/f_Hz,
                                  f1,delta,p.kappa,p.saturation_ratio,p.waist_m,p.rayleigh_m,
                                  switch[1]=='1',switch[2]=='1',cfg.tail_waists,
                                  cfg.max_window_retries,cfg.max_updates_per_point)
    require(ok.all(),f'Finite scan failed window/update budget at {int((~ok).sum())} points')
    info={'domain_model':'finite_unidirectional_raster','finite_region_simulated':True,
          'spatial_coupling':'none_pointwise_model','nominal_pulses_per_line':nx,
          'nominal_lines_per_pass':ny,'physical_pulses':nx*ny*int(passes),
          'evaluated_point_pulse_updates':int(updates.sum()),'max_window_attempts':int(attempts.max()),
          'config':asdict(cfg),'exposure_origin_xy_m':[ox,oy],
          'tail_convergence_validated':False,'scenario_status':'SCENARIO_ASSUMPTION',
          'scan_order':'increasing_y_tracks_increasing_x_pulses_same_order_each_pass',
          'turn_exposure':False,'decay_between_passes':False,'extra_exposure_guard_m':0.}
    return d.reshape(x.shape),q.reshape(x.shape),info


def simulate_roi(parameters,*,cfg=FiniteScanConfig(),**recipe):
    """Evaluate fixed observation pixel centers; returns H in um for canonical use.

    Grid refinement uses interpolation of finer *point samples* back to these
    same centers. Pixel-area averaging is a different unregistered measurement.
    """
    coords=(np.arange(cfg.observation_n)-(cfg.observation_n-1)/2)*cfg.roi_region_m/cfg.observation_n
    x,y=np.meshgrid(coords,coords)
    d,q,info=simulate_points(parameters,x,y,cfg=cfg,**recipe)
    return -d*1e6,q,info
