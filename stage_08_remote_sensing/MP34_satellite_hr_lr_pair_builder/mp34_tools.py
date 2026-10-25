"""MP34 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import torch
from rssr.core import degrade

def make_pair(hr,scale=2,*,sigma=.8,noise=0.,seed=0):
    if scale not in (2,4) or isinstance(scale,bool):raise ValueError('Scale must be2 or4')
    if hr.ndim!=3 or min(hr.shape)==0 or not hr.is_floating_point() or not torch.isfinite(hr).all() or any(n%scale for n in hr.shape[-2:]):raise ValueError('Finite CHW tensor with divisible spatial shape required')
    return degrade(hr,scale,sigma=sigma,noise=noise,seed=seed)

def georeferenced_pair(hr,transform,crs,*,scale=2,valid=None,sigma=.8,noise=0.,seed=0):
    """Aligned synthetic pair. Reject invalid HR rather than blur nodata into neighbors."""
    from affine import Affine
    import numpy as np
    if not crs or not np.isfinite(tuple(transform)[:6]).all() or abs(transform.determinant)==0:
        raise ValueError('Nondegenerate finite affine transform and declared CRS required')
    if valid is not None:
        mask=torch.as_tensor(valid,dtype=torch.bool)
        if tuple(mask.shape) not in (tuple(hr.shape),tuple(hr.shape[-2:])) or not mask.all():
            raise ValueError('This pair builder requires fully valid HR; do not blur nodata')
    lr=make_pair(hr,scale,sigma=sigma,noise=noise,seed=seed)
    low_transform=transform*Affine.scale(scale)
    h,w=hr.shape[-2:]
    corners=lambda t,height,width:[list(t*point) for point in [(0,0),(width,0),(width,height),(0,height)]]
    high_corners=corners(transform,h,w);low_corners=corners(low_transform,*lr.shape[-2:])
    if not np.allclose(high_corners,low_corners,rtol=0,atol=1e-8):raise ValueError('Pair geographic extents differ')
    return lr,{'crs':str(crs),'scale':scale,'hr_shape':list(hr.shape),'lr_shape':list(lr.shape),
               'hr_transform':list(transform)[:6],'lr_transform':list(low_transform)[:6],
               'hr_corners':high_corners,'lr_corners':low_corners,
               'operator':{'sigma_hr_pixels':sigma,'scale':scale,'noise_std_input_units':noise,'seed':seed,
                           'blur':'normalized Gaussian radius ceil(3sigma), replicate boundary','sampling':'nonoverlapping area mean; no clipping'},
               'validity':'fully valid HR required; no cloud or sensor-calibration claim'}
