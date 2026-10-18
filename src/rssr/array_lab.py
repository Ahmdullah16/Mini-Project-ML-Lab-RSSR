"""MP02: explicit HWC arithmetic, valid-pixel statistics and equivalent loops."""
import numpy as np


def validate_hwc(image):
    a=np.asarray(image)
    if a.ndim!=3 or min(a.shape)<1 or not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all():
        raise ValueError('Expected nonempty finite numeric HWC array')
    return a.astype(np.float64)


def brighten(image,offset,maximum=255.,loop=False):
    """Clip each channel to [0, maximum]. Cast first to prevent integer overflow."""
    a=validate_hwc(image)
    delta=np.asarray(offset,dtype=np.float64)
    if delta.shape not in ((),(a.shape[2],)) or not np.isfinite(delta).all() or not np.isfinite(maximum) or maximum<=0:
        raise ValueError('Finite scalar or C-vector offset and positive maximum required')
    delta=np.broadcast_to(delta,(a.shape[2],))
    if not loop:
        return np.clip(a+delta,0,maximum)
    result=np.empty_like(a)
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            for band in range(a.shape[2]):
                result[row,col,band]=min(max(a[row,col,band]+delta[band],0),maximum)
    return result


def masked_channel_stats(image,mask,loop=False):
    """Population mean/variance per band over valid pixels; mask is HxW bool."""
    a=validate_hwc(image); mask=np.asarray(mask)
    if mask.dtype!=np.bool_ or mask.shape!=a.shape[:2] or not mask.any():
        raise ValueError('At least one valid pixel and boolean HxW mask required')
    if not loop:
        return a[mask].mean(axis=0),a[mask].var(axis=0)
    count=int(mask.sum()); mean=np.zeros(a.shape[2]); variance=np.zeros_like(mean)
    for row,col in np.argwhere(mask): mean+=a[row,col]
    mean/=count
    for row,col in np.argwhere(mask): variance+=(a[row,col]-mean)**2
    return mean,variance/count
