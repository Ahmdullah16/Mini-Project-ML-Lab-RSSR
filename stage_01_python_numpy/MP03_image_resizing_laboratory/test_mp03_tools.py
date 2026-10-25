from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pytest
import torch
from mp03_tools import resize_hwc,compare_hwc


def test_half_pixel_hand_example():
    x=np.array([[[0.],[10.]]])
    np.testing.assert_allclose(resize_hwc(x,(1,4)).ravel(),[0,2.5,7.5,10])
    np.testing.assert_array_equal(resize_hwc(x,(1,4),'nearest').ravel(),[0,0,10,10])

@pytest.mark.parametrize('shape',[(1,1),(3,7),(9,2)])
def test_bilinear_reference_and_identity(shape):
    x=np.random.default_rng(17).random((5,4,4))
    actual=resize_hwc(x,shape)
    expected=torch.nn.functional.interpolate(torch.from_numpy(x.transpose(2,0,1))[None],size=shape,mode='bilinear',align_corners=False)[0].permute(1,2,0).numpy()
    np.testing.assert_allclose(actual,expected,atol=1e-14)
    np.testing.assert_array_equal(resize_hwc(x,x.shape[:2]),x)


def test_alias_and_invalid_input():
    x=(np.indices((8,8)).sum(0)%2)[...,None]
    assert np.unique(resize_hwc(x,(4,4),'nearest')).size==1
    np.testing.assert_allclose(resize_hwc(x,(4,4)),.5)
    for shape in [(0,2),(2.5,2),(True,2)]:
        with pytest.raises(ValueError): resize_hwc(x,shape)


def test_comparison_sign_overflow_masks_and_shapes():
    p=np.array([[[0,255],[255,0]]],np.uint8); r=np.array([[[255,0],[0,255]]],np.uint8)
    result=compare_hwc(p,r)
    np.testing.assert_array_equal(result['signed'][0,0],[-255,255])
    np.testing.assert_array_equal(result['mse'],[65025,65025])
    np.testing.assert_array_equal(compare_hwc(r,r)['mse'],0)
    mask=np.array([[True,False]])
    np.testing.assert_array_equal(compare_hwc(p,r,mask)['bias'],[-255,255])
    with pytest.raises(ValueError): compare_hwc(p,r,np.zeros((1,2),bool))
    with pytest.raises(ValueError): compare_hwc(p,r[:,:1])
