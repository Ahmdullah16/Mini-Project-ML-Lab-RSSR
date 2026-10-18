import pytest
import torch
from rssr.models import SRModel


def test_zero_gate_equals_bicubic_and_gate_exclusivity():
    x=torch.rand(1,4,9,13)
    model=SRModel(bands=4,width=8,constant_gate=0)
    torch.testing.assert_close(model(x),torch.nn.functional.interpolate(x,scale_factor=2,mode='bicubic',align_corners=False))
    with pytest.raises(ValueError): SRModel(gate=True,constant_gate=.5)


def test_hybrid_without_attention_has_both_branch_gradients():
    model=SRModel(width=8,use_attention=False)
    assert isinstance(model.attention,torch.nn.Identity)
    model(torch.rand(1,3,8,8)).square().mean().backward()
    assert model.local[0].weight.grad.abs().sum()>0
    assert model.context.attn.in_proj_weight.grad.abs().sum()>0
