import numpy as np
import pytest
import torch
from rssr.convolution_lab import correlate, output_size, receptive_field


def test_asymmetric_hand_example():
    x=np.arange(1,10).reshape(3,3); kernel=np.array([[1,0],[0,-1]])
    np.testing.assert_array_equal(correlate(x,kernel),np.full((2,2),-4))
    assert output_size(7,3,2,1,2)==3
    assert receptive_field([(3,2,1),(3,1,2)])=={'span':11,'jump':2}


def test_stride_dilation_against_torch():
    x=np.arange(63,dtype=float).reshape(7,9); k=np.array([[1,2,0],[-1,0,3]],dtype=float)
    for stride,padding,dilation in [(1,0,1),(2,1,1),(1,2,2),(2,3,3)]:
        expected=torch.nn.functional.conv2d(torch.tensor(x)[None,None],torch.tensor(k)[None,None],
                                            stride=stride,padding=padding,dilation=dilation)
        np.testing.assert_array_equal(correlate(x,k,stride,padding,dilation),expected[0,0].numpy())
    for kwargs in [{'stride':0},{'padding':-1},{'dilation':True}]:
        with pytest.raises(ValueError): correlate(x,k,**kwargs)
    with pytest.raises(ValueError): output_size(2,5)
    with pytest.raises(ValueError): correlate([[np.nan]],[[1]])
