from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp38_tools import *

def test_hybrid_shape_gradients_and_invalid_bands():
    torch.manual_seed(38);model=build_hybrid();x=torch.rand(1,4,5,7,requires_grad=True)
    y=model(x);assert y.shape==(1,4,10,14) and torch.isfinite(y).all()
    y.mean().backward();assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):model(torch.ones(1,3,5,7))
