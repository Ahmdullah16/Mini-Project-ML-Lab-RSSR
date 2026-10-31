"""MP17 canonical numeric tests, no learned weights or data downloads."""
import numpy as np
import pytest
from rssr.convolution_lab import correlate, output_size, receptive_field


def test_impulse_average_kernel_geometry():
    x=np.zeros((9,9),dtype=float);x[4,4]=1
    y=correlate(x,np.ones((3,3))/9,stride=1,padding=1,dilation=1)
    assert y.shape==(9,9)
    assert np.count_nonzero(y)==9 and y.sum()==pytest.approx(1.)
    assert y[4,4]==pytest.approx(1/9)
    assert output_size(9,3,2,1)==5


def test_padding_and_receptive_field():
    y=correlate(np.ones((9,9)),np.ones((3,3))/9,padding=1)
    assert y[0,0]==pytest.approx(4/9)
    assert y[4,4]==pytest.approx(1.)
    assert receptive_field([(3,2,2)])=={"span":5,"jump":2}
    with pytest.raises(ValueError):output_size(2,5)
