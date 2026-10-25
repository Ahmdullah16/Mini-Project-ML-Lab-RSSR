"""MP32 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np

def pixel_to_map(transform,row,column,center=True):
    if not np.isfinite([row,column]).all():raise ValueError('Finite pixel coordinates required')
    offset=.5 if center else 0.
    return transform*(column+offset,row+offset)
