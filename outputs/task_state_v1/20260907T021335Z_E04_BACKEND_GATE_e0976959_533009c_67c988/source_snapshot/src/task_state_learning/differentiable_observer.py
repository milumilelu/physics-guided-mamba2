"""Autograd observer with frozen geometry and float64 accumulation.

Float32-input parity means the SAME float32 values are compared after promotion
to float64 on both paths. This is not a claim of native-float32 spectral parity.
The evaluator remains CanonicalObserver; no trainable observer constants exist.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from src.composition import ILR_A
from src.spectrum import dct_lambda_grid

TARGET_NAMES = ("D", "A_med", "ilr_z1", "ilr_z2", "ilr_z3", "ilr_z4",
                "A2_8_16", "entropy_8_16", "A2_16_32", "entropy_16_32")


@dataclass
class Observation:
    values: torch.Tensor
    valid: torch.Tensor
    composition_energy: torch.Tensor
    directional_energy: torch.Tensor
    replacement_used: torch.Tensor
    reasons: list[str]


class DifferentiableObserver(nn.Module):
    def __init__(self, theta_bins=36, zero_threshold=1e-10, replacement_delta=1e-6):
        super().__init__()
        if theta_bins not in (18, 36):
            raise ValueError("Only registered 36-main / 18-sensitivity bins are supported")
        self.theta_bins = theta_bins
        self.zero_threshold = zero_threshold
        self.replacement_delta = replacement_delta
        n = 160
        k = np.arange(n)[:, None]
        sample = np.arange(n)[None, :]
        dct = np.sqrt(2/n)*np.cos(np.pi*(sample+.5)*k/n)
        dct[0] = 1/np.sqrt(n)
        lam_dct = dct_lambda_grid((n, n), .5)
        non_dc = np.isfinite(lam_dct)
        masks = np.stack([non_dc & (lam_dct < 8)] +
                         [non_dc & (lam_dct >= lo) & (lam_dct < hi)
                          for lo, hi in [(8,16),(16,32),(32,64),(64,np.inf)]])
        # Last upper boundary must include all finite wavelengths >=64.
        masks[-1] = non_dc & (lam_dct >= 64)
        freq = np.fft.fftfreq(n, d=.5)
        fx, fy = np.meshgrid(freq, freq)
        radius = np.hypot(fx, fy)
        lam = np.full(radius.shape, np.inf)
        np.divide(1, radius, out=lam, where=radius > 0)
        theta = np.arctan2(fy, fx)
        edges = np.linspace(0, np.pi, theta_bins+1)
        bins = np.searchsorted(edges, np.mod(theta, np.pi), side="right")-1
        bins = np.clip(bins, 0, theta_bins-1)
        constants = {"dct_matrix": dct, "ilr_basis": ILR_A,
                     "composition_masks": masks.astype(float),
                     "window": np.outer(np.hanning(n),np.hanning(n)),
                     "cos2theta": np.cos(2*theta), "sin2theta": np.sin(2*theta)}
        for name, value in constants.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64))
        self.register_buffer("angle_bins", torch.as_tensor(bins.reshape(-1), dtype=torch.long))
        self.register_buffer("direction_masks", torch.as_tensor(np.stack([
            (lam >= lo) & (lam < hi) for lo, hi in [(8,16),(16,32)]]),dtype=torch.bool))

    def forward(self, height, valid_mask=None):
        if height.ndim != 3 or height.shape[1:] != (160,160):
            raise ValueError("Observer requires N x 160 x 160")
        h = height.to(dtype=torch.float64)
        if h.device != self.dct_matrix.device:
            raise ValueError("Move the observer buffers to the input device explicitly")
        v = torch.ones_like(h,dtype=torch.bool) if valid_mask is None else valid_mask.to(device=h.device,dtype=torch.bool)
        if v.shape != h.shape:
            raise ValueError("Mask shape mismatch")
        finite = torch.isfinite(h)
        usable = v.flatten(1).any(1) & (finite | ~v).flatten(1).all(1)
        safe = torch.where(finite,h,torch.zeros_like(h))
        medians=[]
        for i in range(len(h)):
            values=safe[i][v[i]]
            if values.numel()==0:
                medians.append(safe[i].sum()*0.)
            else:
                ordered=values.sort().values
                middle=len(ordered)//2
                medians.append((ordered[middle]+ordered[(len(ordered)-1)//2])/2)
        median=torch.stack(medians)
        residual=safe-median[:,None,None]
        counts=v.sum(dim=(1,2)).clamp_min(1)
        squared=(torch.where(v,residual,torch.zeros_like(residual))**2).sum(dim=(1,2))/counts
        # Zero amplitude is valid; choose the zero subgradient, without sqrt'(0).
        amplitude=torch.where(squared>0,torch.sqrt(torch.where(squared>0,squared,torch.ones_like(squared))),torch.zeros_like(squared))
        full=usable & v.flatten(1).all(1)
        c=self.dct_matrix @ residual @ self.dct_matrix.T
        energies=torch.einsum("nij,bij->nb",c**2,self.composition_masks)
        total=energies.sum(1)
        comp_valid=full & torch.isfinite(total) & (total>0)
        parts=energies/torch.where(total>0,total,torch.ones_like(total))[:,None]
        replace=parts<self.zero_threshold
        replaced=torch.where(replace,torch.full_like(parts,self.replacement_delta),parts)
        replaced=replaced/replaced.sum(1,keepdim=True)
        parts=torch.where(replace.any(1)[:,None],replaced,parts)
        z=torch.log(parts.clamp_min(1e-300)) @ self.ilr_basis.T
        centered=residual-residual.mean(dim=(1,2),keepdim=True)
        fft=torch.fft.fft2(centered*self.window)
        power=(fft.real**2+fft.imag**2)/(self.window**2).sum()
        directions=[];direction_energies=[];direction_valid=[]
        for mask in self.direction_masks:
            p=power*mask
            energy=p.sum(dim=(1,2))
            ok=full & torch.isfinite(energy) & (energy>0)
            denom=torch.where(energy>0,energy,torch.ones_like(energy))
            real=(p*self.cos2theta).sum(dim=(1,2));imag=(p*self.sin2theta).sum(dim=(1,2))
            moment=torch.complex(real,imag).abs()/denom
            histogram=torch.zeros((len(h),self.theta_bins),dtype=h.dtype,device=h.device).scatter_add(
                1,self.angle_bins[None,:].expand(len(h),-1),p.flatten(1))
            probabilities=histogram/denom[:,None]
            entropy=-(probabilities*torch.log(probabilities.clamp_min(1e-300))).sum(1)/np.log(self.theta_bins)
            directions.extend([moment,entropy]);direction_energies.append(energy);direction_valid.extend([ok,ok])
        values=torch.stack([-median,amplitude,*z.unbind(1),*directions],dim=1)
        valid=torch.stack([usable,usable,*([comp_valid]*4),*direction_valid],dim=1)
        values=torch.where(valid,values,torch.full_like(values,float("nan")))
        reasons=[]
        for i in range(len(h)):
            if not bool(usable[i]): reasons.append("EMPTY_OR_NONFINITE_VALID_HEIGHT")
            elif not bool(full[i]): reasons.append("INCOMPLETE_GRID_NO_REGISTERED_IMPUTATION")
            elif not bool(comp_valid[i]): reasons.append("ZERO_NON_DC_ENERGY")
            elif not bool(valid[i].all()): reasons.append("ZERO_DIRECTION_BAND_ENERGY")
            else: reasons.append("")
        return Observation(values,valid,total,torch.stack(direction_energies,1),replace.any(1)&comp_valid,reasons)
