from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp37_tools import *

def test_fusion_values_and_contract():
    a=torch.ones(1,2,3,4);b=3*a
    torch.testing.assert_close(fuse_features(a,b),2*a)
    c=fuse_features(a,b,'concat');assert c.shape==(1,4,3,4)
    torch.testing.assert_close(c[:,:2],a);torch.testing.assert_close(c[:,2:],b)
    with pytest.raises(ValueError):fuse_features(a,b[:,:,:2])
    with pytest.raises(ValueError):fuse_features(a,b,'unknown')


def test_projection_represents_mean_and_gradients():
    a=torch.randn(1,4,3,5,requires_grad=True);b=torch.randn_like(a,requires_grad=True)
    layer=initialize_mean_projection(torch.nn.Conv2d(8,4,1))
    out=layer(fuse_features(a,b,'concat'))
    torch.testing.assert_close(out,fuse_features(a,b))
    out.sum().backward()
    torch.testing.assert_close(a.grad,torch.full_like(a,.5))
    torch.testing.assert_close(b.grad,torch.full_like(b,.5))


def test_cost_matches_actual_parameters():
    cost=projection_cost(8,5,7)
    assert cost=={'parameters':136,'macs':4480}
    assert cost['parameters']==sum(p.numel() for p in torch.nn.Conv2d(16,8,1).parameters())
    with pytest.raises(ValueError):projection_cost(0,5,7)
    with pytest.raises(ValueError):initialize_mean_projection(torch.nn.Conv2d(8,8,3))


@pytest.mark.parametrize('a,b',[(torch.ones(1,2,3,4,dtype=torch.int64),torch.ones(1,2,3,4,dtype=torch.int64)),
    (torch.empty(1,2,0,4),torch.empty(1,2,0,4)),
    (torch.ones(1,2,3,4),torch.ones(1,2,3,4,dtype=torch.float64)),
    (torch.full((1,2,3,4),float('nan')),torch.ones(1,2,3,4))])
def test_invalid_feature_contract(a,b):
    with pytest.raises(ValueError):fuse_features(a,b)
