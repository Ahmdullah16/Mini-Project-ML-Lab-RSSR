"""MP34 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import torch
from rssr.core import degrade

def make_pair(hr,scale=2):
    if scale not in (2,4) or isinstance(scale,bool):raise ValueError('Scale must be2 or4')
    if hr.ndim!=3 or min(hr.shape)==0 or not hr.is_floating_point() or not torch.isfinite(hr).all() or any(n%scale for n in hr.shape[-2:]):raise ValueError('Finite CHW tensor with divisible spatial shape required')
    return degrade(hr,scale)
