"""MP37 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import torch

def fuse_features(local,context,mode='mean'):
    if local.ndim!=4 or local.shape!=context.shape or not torch.isfinite(local).all() or not torch.isfinite(context).all():raise ValueError('Matching finite NCHW features required')
    if mode=='mean':return (local+context)/2
    if mode=='concat':return torch.cat([local,context],dim=1)
    raise ValueError('Choose mean or concat; channel projection is separate')
