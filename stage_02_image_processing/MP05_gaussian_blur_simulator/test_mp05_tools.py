from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pytest
from mp05_tools import gaussian,filter_hwc,degrade_hwc,spectrum


def test_gaussian_normalization_symmetry_and_boundaries():
    k=gaussian(1,3); assert k.sum()==pytest.approx(1)
    np.testing.assert_array_equal(k,k[::-1,::-1])
    x=np.ones((9,9,2))
    np.testing.assert_allclose(filter_hwc(x,k),1)
    assert filter_hwc(x,k,'constant')[0,0,0]<1
    impulse=np.zeros((9,9,1)); impulse[4,4]=1
    np.testing.assert_allclose(filter_hwc(impulse,k)[1:8,1:8,0],k)

@pytest.mark.parametrize('scale',[2,4])
def test_degradation_shapes_noise_and_area(scale):
    x=np.arange(16*16*3,dtype=float).reshape(16,16,3)/1000
    clean=degrade_hwc(x,scale,0,0)
    assert clean.shape==(16//scale,16//scale,3)
    np.testing.assert_allclose(clean,x.reshape(16//scale,scale,16//scale,scale,3).mean((1,3)))
    a=degrade_hwc(x,scale,1,.01,17)
    np.testing.assert_array_equal(a,degrade_hwc(x,scale,1,.01,17))
    assert not np.array_equal(a,degrade_hwc(x,scale,1,.01,18))


def test_fft_inverse_parseval_and_invalid():
    x=np.random.default_rng(0).random((9,7)); f,_=spectrum(x)
    np.testing.assert_allclose(np.fft.ifft2(f).real,x,atol=1e-14)
    assert np.square(x).sum()==pytest.approx((np.abs(f)**2).sum()/x.size)
    with pytest.raises(ValueError): gaussian(float('nan'),2)
    with pytest.raises(ValueError): degrade_hwc(np.ones((7,7,1)),2)
