"""From-scratch HWC metrics with declared range and valid-window SSIM."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def paired(pred, ref, data_range):
    p, r = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
    if p.shape != r.shape or p.ndim != 3 or min(p.shape) < 1:
        raise ValueError('Matching nonempty HWC arrays required')
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('Nonfinite image samples')
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError('Positive finite declared data range required')
    return p, r


def mse_psnr(pred, ref, data_range=1.):
    p, r = paired(pred, ref, data_range)
    mse = float(np.mean((p-r)**2))
    return mse, float('inf') if mse == 0 else float(10*np.log10(data_range**2/mse))


def ssim_map(pred, ref, data_range=1., window=7):
    """Uniform population moments; no padding; output (H-w+1,W-w+1,C)."""
    p, r = paired(pred, ref, data_range)
    if not isinstance(window, int) or window < 1 or window % 2 != 1 or window > min(p.shape[:2]):
        raise ValueError('Odd positive window must fit image')
    a = sliding_window_view(p, (window, window), axis=(0, 1))
    b = sliding_window_view(r, (window, window), axis=(0, 1))
    ma, mb = a.mean((-1, -2)), b.mean((-1, -2))
    ca, cb = a-ma[..., None, None], b-mb[..., None, None]
    va, vb = (ca*ca).mean((-1, -2)), (cb*cb).mean((-1, -2))
    cov = (ca*cb).mean((-1, -2))
    c1, c2 = (.01*data_range)**2, (.03*data_range)**2
    return ((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))
