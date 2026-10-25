from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pytest
from mp02_tools import brighten,masked_channel_stats


def test_overflow_and_channel_broadcast():
    a=np.array([[[250,5,10],[1,100,255]]],dtype=np.uint8)
    expected=np.array([[[255,0,12],[21,90,255]]],dtype=float)
    for loop in (False,True):
        np.testing.assert_array_equal(brighten(a,[20,-10,2],loop=loop),expected)
    assert a[0,0,0]==250


def test_mask_and_population_variance():
    a=np.array([[[1,2],[3,4]],[[5,6],[999,999]]])
    mask=np.array([[True,True],[True,False]])
    for loop in (False,True):
        mean,var=masked_channel_stats(a,mask,loop=loop)
        np.testing.assert_allclose(mean,[3,4]); np.testing.assert_allclose(var,[8/3,8/3])
    with pytest.raises(ValueError): masked_channel_stats(a,np.zeros((2,2),bool))
    with pytest.raises(ValueError): brighten(a,[1,2,3])
    with pytest.raises(ValueError): brighten(np.array([[[np.nan]]]),1)
