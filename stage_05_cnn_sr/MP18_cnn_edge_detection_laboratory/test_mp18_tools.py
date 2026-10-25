from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp18_tools import *

def test_sobel_direction_and_boundary():
    step=torch.tensor([[0.,0,1],[0,0,1],[0,0,1]])
    out=sobel_components(step)
    assert out[0,1,1]==.5 and out[1,1,1]==0
    assert sobel_components(torch.ones(4,4)).abs().max()==0
    assert sobel_components(torch.ones(4,4),'constant').abs().max()==.5
    with pytest.raises(ValueError):sobel_components(torch.zeros(1,1))
