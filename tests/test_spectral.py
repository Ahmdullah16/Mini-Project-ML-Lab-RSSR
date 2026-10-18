import pytest
import torch
from rssr.spectral import ndvi,ergas,charbonnier,edge_l1

def test_reflectance_semantics():
    x=torch.tensor([[[.2,.0]],[[.6,.0]]])
    with pytest.raises(ValueError): ndvi(x,0,1)
    value,mask=ndvi(x,0,1,reflectance=True)
    assert value[0,0]==pytest.approx(.5) and not mask[0,1]
    assert ergas(x+.1,x+.1,2,compatible_units=True)==0
    with pytest.raises(ValueError): ergas(x,x,2)

def test_loss_gradients():
    x=torch.rand(1,3,8,8,requires_grad=True); ref=torch.zeros_like(x)
    loss=charbonnier(x,ref)+edge_l1(x,ref); loss.backward()
    assert torch.isfinite(x.grad).all()
