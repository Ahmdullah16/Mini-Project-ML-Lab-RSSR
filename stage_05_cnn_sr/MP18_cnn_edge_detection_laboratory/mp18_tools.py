"""MP18 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.nn import functional as F

def sobel_components(image,padding='replicate'):
    if image.ndim!=2 or not image.is_floating_point() or not torch.isfinite(image).all() or min(image.shape)<2:raise ValueError('Finite floating 2D image of at least2×2 required')
    if padding not in ('replicate','constant'):raise ValueError('Choose replicate or constant-zero padding')
    kernel=image.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/8
    padded=F.pad(image[None,None],(1,1,1,1),mode=padding)
    return F.conv2d(padded,torch.stack([kernel,kernel.T])[:,None])[0]
