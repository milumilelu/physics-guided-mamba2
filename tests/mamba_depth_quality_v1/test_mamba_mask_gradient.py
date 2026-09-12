import torch
from src.mamba_depth_quality.mamba_adapter import MambaAdapter

def test_mask_and_step():
 torch.manual_seed(3); m=MambaAdapter(); x=torch.randn(2,7,32); mask=torch.ones(2,7,dtype=torch.bool); y=m(x,mask); yp=m(torch.cat([x,torch.zeros(2,3,32)],1),torch.cat([mask,torch.zeros(2,3,dtype=torch.bool)],1)); assert torch.allclose(y,yp[:,:7],atol=1e-5,rtol=1e-5)
