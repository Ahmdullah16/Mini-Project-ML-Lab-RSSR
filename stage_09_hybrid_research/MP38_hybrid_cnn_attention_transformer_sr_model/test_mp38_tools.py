from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp38_tools import *

def test_hybrid_shape_gradients_and_invalid_bands():
    torch.manual_seed(38);model=build_hybrid();x=torch.rand(1,4,5,7,requires_grad=True)
    y=model(x);assert y.shape==(1,4,10,14) and torch.isfinite(y).all()
    y.mean().backward();assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):model(torch.ones(1,3,5,7))


@pytest.mark.parametrize('bands',[3,4])
def test_trace_zero_residual_and_state(bands):
    model=build_hybrid(bands=bands,zero_residual=True)
    x=torch.rand(1,bands,5,7)
    y,shapes=trace_hybrid(model,x)
    torch.testing.assert_close(y,torch.nn.functional.interpolate(x,scale_factor=2,mode='bicubic',align_corners=False),atol=0,rtol=0)
    assert model.training
    assert shapes['fuse']=={'input':[1,16,5,7],'output':[1,8,5,7]}
    assert shapes['gate']['output']==[1,bands,5,7]
    assert all(not m._forward_hooks for m in model.modules())


def test_trace_cleans_hooks_on_forward_error():
    model=build_hybrid().eval()
    with pytest.raises(ValueError):trace_hybrid(model,torch.ones(1,3,5,7))
    assert not model.training
    assert all(not m._forward_hooks for m in model.modules())


def test_gate_gradient_initially_zero_but_tail_can_learn():
    model=build_hybrid(zero_residual=True)
    model(torch.ones(1,4,5,7)).sum().backward()
    assert model.tail[0].bias.grad.abs().sum()>0
    assert model.gate[0].weight.grad.abs().sum()==0
