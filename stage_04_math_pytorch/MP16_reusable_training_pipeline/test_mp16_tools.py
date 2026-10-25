from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp16_tools import *

def test_weighted_partial_batch_and_seed():
    x=torch.arange(5.)[:,None]
    a=list(make_loader(x,2*x+1));b=list(make_loader(x,2*x+1))
    assert [len(v[0]) for v in a]==[2,2,1]
    for left,right in zip(a,b):torch.testing.assert_close(left[0],right[0])
    assert weighted_mean([2.5,1],[2,1])==2
    with pytest.raises(ValueError):weighted_mean([1],[0])
    with pytest.raises(ValueError):make_loader(x,x[:2])
