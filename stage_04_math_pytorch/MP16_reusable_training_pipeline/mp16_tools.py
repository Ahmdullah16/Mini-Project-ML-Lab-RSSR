"""MP16 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader

def make_loader(x,y,batch_size=2,seed=16):
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not len(x) or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Expected paired finite nonempty sample matrices')
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<=0:raise ValueError('Positive batch size required')
    return DataLoader(TensorDataset(x,y),batch_size=batch_size,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))

def weighted_mean(means,sizes):
    means=np.asarray(means,dtype=float);sizes=np.asarray(sizes,dtype=float)
    if means.ndim!=1 or means.shape!=sizes.shape or not means.size or not np.isfinite(means).all() or not np.isfinite(sizes).all() or (sizes<=0).any():raise ValueError('Finite means and positive paired sample counts required')
    return float(np.dot(means,sizes)/sizes.sum())
