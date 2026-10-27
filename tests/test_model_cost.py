import pytest
import torch
from torch import nn
from rssr.model_cost import measure, match
from rssr.architecture_registry import build


def test_convolution_hand_count_and_modes():
    model=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.ReLU())
    model.train(); model[1].eval()
    result=measure(model,(1,3,5,7))
    assert result['parameters'] == 112
    assert result['partial_macs'] == 4*5*7*3*9
    assert model.training and not model[1].training
    assert not model[0]._forward_hooks


def test_transformer_hand_count_with_window_padding():
    # Width4, window4: odd5x7 input pads to8x8 = four windows of16 tokens.
    model=build('cat-v1/M3',width=4)
    result=measure(model,(1,3,5,7))
    assert result['breakdown']['self_attention'] == 4*(4*16*4*4+2*16*16*4)
    assert result['breakdown']['linear'] == 4*16*(4*8+8*4)
    assert result['breakdown']['conv'] == 5*7*(3*4*9+4*12*9)
    assert result['output_shape'] == [1,3,10,14]


def test_failure_cleans_hooks_and_modes():
    model=build('cat-v1/M1',width=4).train()
    with pytest.raises(ValueError): measure(model,(1,4,5,7))
    assert all(m.training and not m._forward_hooks for m in model.modules())


def test_thresholds_and_bad_shape():
    ref={'parameters':100,'partial_macs':100}
    assert match({'parameters':105,'partial_macs':110},ref)['eligible_partial_budget']
    assert not match({'parameters':106,'partial_macs':100},ref)['eligible_partial_budget']
    with pytest.raises(ValueError): measure(build('cat-v1/M1'),(1,3,0,2))
