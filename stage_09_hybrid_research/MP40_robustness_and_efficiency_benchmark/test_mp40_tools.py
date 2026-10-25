from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp40_tools import *

def test_timer_counts_and_restores_mode():
    class Counter(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=0
        def forward(self,x):self.calls+=1;return x+1
    model=Counter();r=benchmark_cpu(model,torch.zeros(1),warmups=2,repeats=3)
    assert model.calls==5 and model.training and len(r['seconds'])==3 and r['median_seconds']>=0
    with pytest.raises(ValueError):benchmark_cpu(model,torch.zeros(1),repeats=0)
