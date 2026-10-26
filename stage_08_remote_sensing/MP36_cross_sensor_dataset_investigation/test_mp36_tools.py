from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp36_tools import *

def test_ratios_do_not_assume_integer_scale():
    assert resolution_ratio(10,5)==2
    assert resolution_ratio(20,5)==4
    assert resolution_ratio(10,1.5)==pytest.approx(20/3)
    for value in (0,-1,float('nan')):
        with pytest.raises(ValueError):resolution_ratio(10,value)



def test_registration_units_and_invalid_values():
    assert registration_pixels(5,5)==1
    assert registration_pixels(2.5,5)==.5
    for offset,gsd in [(-1,5),(float('inf'),5),(0,0)]:
        with pytest.raises(ValueError):registration_pixels(offset,gsd)


def test_radiometric_error_is_not_sr_error():
    import numpy as np
    x=np.array([100,200],dtype=np.uint8)
    y=affine_radiometry(x,2,10)
    assert y.tolist()==[210,410] and y.dtype==np.float64
    assert x.tolist()==[100,200]
    reference=np.array([.2,.4]);prediction=affine_radiometry(reference,1.1,.01)
    assert float(np.mean((prediction-reference)**2))==pytest.approx(.0017)


@pytest.mark.parametrize('values,gain,offset',[([],1,0),([float('nan')],1,0),([1],float('inf'),0)])
def test_invalid_calibration(values,gain,offset):
    with pytest.raises(ValueError):affine_radiometry(values,gain,offset)


def test_response_mismatch_despite_same_band_name():
    assert band_response([.2,.8],[1,0])==.2
    assert band_response([.2,.8],[0,1])==.8
    assert band_response([.2,.8],[1,1])==pytest.approx(.5)
    for weights in ([0,0],[-1,2],[1],[float('nan'),1]):
        with pytest.raises(ValueError):band_response([.2,.8],weights)
