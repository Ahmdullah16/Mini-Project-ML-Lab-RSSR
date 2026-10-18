"""Teaching implementations: HWC float64, half-pixel coordinates, edge replication."""
import numpy as np
from .array_lab import validate_hwc


def resize_hwc(image, shape, method='bilinear'):
    """No antialias filter. Nearest ties choose the larger index; no output clipping."""
    x=validate_hwc(image)
    if len(shape)!=2 or any(isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<1 for n in shape):
        raise ValueError('Output shape must contain two positive integers')
    if method not in ('nearest','bilinear'):
        raise ValueError('Use nearest or bilinear')
    h,w=shape
    y=np.clip((np.arange(h)+.5)*x.shape[0]/h-.5,0,x.shape[0]-1)
    z=np.clip((np.arange(w)+.5)*x.shape[1]/w-.5,0,x.shape[1]-1)
    if method=='nearest':
        return x[np.floor(y+.5).astype(int)[:,None],np.floor(z+.5).astype(int)[None,:]]
    y0=np.floor(y).astype(int); z0=np.floor(z).astype(int)
    y1=np.minimum(y0+1,x.shape[0]-1); z1=np.minimum(z0+1,x.shape[1]-1)
    a=(y-y0)[:,None,None]; b=(z-z0)[None,:,None]
    return (1-a)*((1-b)*x[y0[:,None],z0[None,:]]+b*x[y0[:,None],z1[None,:]])+a*((1-b)*x[y1[:,None],z0[None,:]]+b*x[y1[:,None],z1[None,:]])


def compare_hwc(prediction,reference,mask=None):
    """Signed error is prediction-reference; mask excludes whole pixels, not bands."""
    p,r=validate_hwc(prediction),validate_hwc(reference)
    if p.shape!=r.shape: raise ValueError('Matching HWC shapes required')
    if mask is None: mask=np.ones(p.shape[:2],dtype=bool)
    mask=np.asarray(mask)
    if mask.dtype!=np.bool_ or mask.shape!=p.shape[:2] or not mask.any():
        raise ValueError('Nonempty boolean HxW validity mask required')
    d=p-r; valid=d[mask]
    return {'signed':d,'absolute':np.abs(d),'valid_pixels':int(mask.sum()),'bias':valid.mean(0),'mae':np.abs(valid).mean(0),'mse':(valid**2).mean(0)}
