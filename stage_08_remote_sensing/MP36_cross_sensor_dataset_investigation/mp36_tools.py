"""MP36 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import math

def resolution_ratio(lr_metres,hr_metres):
    if not all(math.isfinite(v) and v>0 for v in (lr_metres,hr_metres)):raise ValueError('Positive finite ground sampling distances required')
    return lr_metres/hr_metres



def registration_pixels(offset_metres,hr_metres):
    """Magnitude of a geometric offset in HR pixels; not a quality score."""
    if not math.isfinite(offset_metres) or offset_metres<0:
        raise ValueError('Finite nonnegative registration offset required')
    resolution_ratio(1,hr_metres)
    return offset_metres/hr_metres


def affine_radiometry(values,gain,offset):
    """Float64 toy calibration; no clipping or fitting to reference data."""
    import numpy as np
    x=np.asarray(values,dtype=np.float64)
    if x.size==0 or not np.isfinite(x).all() or not all(math.isfinite(v) for v in (gain,offset)):
        raise ValueError('Nonempty finite values and calibration required')
    y=gain*x+offset
    if not np.isfinite(y).all():raise ValueError('Calibration overflow')
    return y


def band_response(spectrum,weights):
    """Toy equal-width spectral bins: sum(r_i*w_i)/sum(w_i).

    Real sensor use needs measured spectral response and wavelength integration.
    These discrete weights are teaching inputs, never a sensor calibration.
    """
    import numpy as np
    x=np.asarray(spectrum,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    if x.ndim!=1 or x.size==0 or x.shape!=w.shape or not np.isfinite(x).all() or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Equal nonempty finite vectors and nonnegative nonzero weights required')
    return float(np.dot(x,w/w.sum()))
