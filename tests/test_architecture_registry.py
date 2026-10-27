import pytest
import torch
from torch import nn
from rssr.models import SRModel, Attention, WindowBlock
from rssr.architecture_registry import REGISTRY, build, describe


@pytest.mark.parametrize('key', list(REGISTRY))
def test_modules_shapes_and_gradients(key):
    torch.manual_seed(71)
    torch.set_num_threads(1)
    spec = REGISTRY[key]
    model = build(key, width=4)
    assert (model.local is not None) == spec.c
    assert isinstance(model.attention, Attention) == spec.a
    assert isinstance(model.context, WindowBlock) == spec.t
    for shape in [(1,3,1,1), (1,3,5,7), (1,3,8,8)]:
        x = torch.rand(shape)
        y = model(x)
        assert y.shape == (1,3,shape[2]*2,shape[3]*2)
        assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert describe(key, width=4)['checkpoint_sha256'] is None


@pytest.mark.parametrize('key,kind,attention', [
    ('M1','cnn',True), ('M3','transformer',True), ('M4','attention',True),
    ('M5','hybrid',False), ('M7','hybrid',True)])
def test_legacy_strict_state_dict_and_output(key,kind,attention):
    torch.manual_seed(72)
    old = SRModel(width=4, kind=kind, use_attention=attention)
    new = build('cat-v1/'+key, width=4)
    new.load_state_dict(old.state_dict(), strict=True)
    x = torch.rand(1,3,5,7)
    torch.testing.assert_close(old(x), new(x), rtol=0, atol=0)


def test_unique_structures_and_invalid_id():
    signatures = [tuple((n,tuple(p.shape)) for n,p in build(k).named_parameters()) for k in REGISTRY]
    assert len(set(signatures)) == 7
    with pytest.raises(ValueError): build('M8')


@pytest.mark.parametrize('key', list(REGISTRY))
def test_four_band_native_four_identity(key):
    x = torch.rand(1,4,3,5)
    model = build(key,bands=4,scale=4,width=4,zero_residual=True)
    torch.testing.assert_close(model(x), torch.nn.functional.interpolate(x,scale_factor=4,mode='bicubic',align_corners=False))
