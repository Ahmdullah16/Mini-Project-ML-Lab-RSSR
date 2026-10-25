from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp34_tools import *

def test_pair_shape_constant_and_rejection():
    hr=torch.ones(4,16,20);old=hr.clone();lr=make_pair(hr)
    assert lr.shape==(4,8,10)
    torch.testing.assert_close(lr,torch.ones_like(lr));torch.testing.assert_close(hr,old)
    with pytest.raises(ValueError):make_pair(torch.ones(3,15,16))
    with pytest.raises(ValueError):make_pair(hr,1)
