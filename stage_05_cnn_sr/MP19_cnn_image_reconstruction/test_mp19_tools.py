from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp19_tools import UpsamplingCNN,Reconstruction

def test_zero_weights_expose_residual_and_direct_behavior():
    x=torch.rand(1,3,5,7)
    for mode in ['shuffle','transpose','interpolate']:
        model=UpsamplingCNN(mode)
        for p in model.parameters():p.data.zero_()
        torch.testing.assert_close(model(x),torch.nn.functional.interpolate(x,scale_factor=2,mode='bicubic',align_corners=False))
        model.residual=False;assert torch.count_nonzero(model(x))==0
        assert model(x).shape==(1,3,10,14)
    with pytest.raises(ValueError):UpsamplingCNN('unknown')

def test_identity_reconstruction_skip():
    model=Reconstruction(residual=True)
    for p in model.parameters():p.data.zero_()
    x=torch.rand(2,3,8,9);torch.testing.assert_close(model(x),x)
