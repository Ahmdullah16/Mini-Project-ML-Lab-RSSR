from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp33_tools import *

def test_nodata_mask_and_empty_rejection():
    mean,mask=valid_mean([1,2,-9999,float('nan')])
    assert mean==1.5 and mask.tolist()==[True,True,False,False]
    with pytest.raises(ValueError):valid_mean([-9999,float('nan')])
