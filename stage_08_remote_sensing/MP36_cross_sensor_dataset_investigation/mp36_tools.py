"""MP36 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import math

def resolution_ratio(lr_metres,hr_metres):
    if not all(math.isfinite(v) and v>0 for v in (lr_metres,hr_metres)):raise ValueError('Positive finite ground sampling distances required')
    return lr_metres/hr_metres
