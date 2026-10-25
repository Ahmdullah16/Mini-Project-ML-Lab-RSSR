"""MP33 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np

def valid_mean(values,nodata=-9999.):
    values=np.asarray(values,dtype=float)
    valid=np.isfinite(values)&(values!=nodata)
    if not valid.any():raise ValueError('No valid pixels; no replacement mean invented')
    return float(values[valid].mean()),valid
