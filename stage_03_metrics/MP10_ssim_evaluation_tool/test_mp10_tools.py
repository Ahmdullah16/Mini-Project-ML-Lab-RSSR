from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pytest
import torch
from mp10_tools import mse_psnr, ssim_map
from rssr.core import metrics


def test_hand_metric_and_range():
    ref=np.zeros((2,2,1),dtype=np.uint8)
    pred=np.array([0,0,0,2],dtype=np.uint8).reshape(2,2,1)
    assert mse_psnr(pred,ref,2)==(1.,pytest.approx(6.020599913))
    assert np.isinf(mse_psnr(ref,ref)[1])
    assert mse_psnr(pred,ref,4)[1]-mse_psnr(pred,ref,2)[1]==pytest.approx(6.020599913)
    assert mse_psnr(pred/2.,ref/2.,1)[1]==mse_psnr(pred,ref,2)[1]


def test_ssim_closed_form_and_independent_implementation():
    x=np.full((9,10,2),.5); y=np.full_like(x,.6)
    expected=(2*.5*.6+.01**2)/(.5**2+.6**2+.01**2)
    np.testing.assert_allclose(ssim_map(x,y),expected)
    rng=np.random.default_rng(17); x=rng.random((12,13,3)); y=x+rng.normal(0,.03,x.shape)
    reference=metrics(torch.tensor(x.transpose(2,0,1)),torch.tensor(y.transpose(2,0,1)))
    assert ssim_map(x,y).mean()==pytest.approx(reference['ssim'],abs=1e-12)
    np.testing.assert_allclose(ssim_map(x,x),1.)


def test_metric_rejections():
    x=np.zeros((8,8,1))
    for bad in [0,-1,float('nan')]:
        with pytest.raises(ValueError): mse_psnr(x,x,bad)
    for bad in [0,2,9]:
        with pytest.raises(ValueError): ssim_map(x,x,window=bad)
    with pytest.raises(ValueError): mse_psnr(x,x[:2])
    with pytest.raises(ValueError): mse_psnr(x,np.full_like(x,np.nan))
