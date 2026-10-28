"""Bounded synthetic metric teaching; never receives an uploaded image."""
import numpy as np
import torch
from torch.nn import functional as F
from .core import metrics

KINDS=('Blur','Noise','Brightness','Contrast','Shift','Sharpening')

def reference():
    y,x=np.mgrid[:64,:64]
    return np.stack([x/63,y/63,((x//8+y//8)%2)*.6+.2]).astype(np.float32)

def demonstrate(kind, strength):
    if kind not in KINDS or not np.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('Known distortion and finite strength in [0,1] required')
    r=torch.from_numpy(reference());a=float(strength)
    blur=F.avg_pool2d(F.pad(r[None],(2,2,2,2),mode='replicate'),5,1)[0]
    if kind=='Blur':p=(1-a)*r+a*blur
    elif kind=='Noise':p=r+a*.2*torch.from_numpy(np.random.default_rng(17).standard_normal(r.shape).astype(np.float32))
    elif kind=='Brightness':p=r+.3*a
    elif kind=='Contrast':p=r+a*(r-.5)
    elif kind=='Sharpening':p=r+2*a*(r-blur)
    else:
        shift=round(4*a);p=F.pad(r,(shift,0,0,0),mode='replicate')[:,:,:64]
    clipped=float(((p<0)|(p>1)).float().mean());p=p.clamp(0,1)
    result=metrics(p,r,data_range=1.,border=0)
    result.pop('sam_rad') # Encoded RGB is not calibrated spectral reflectance.
    result['mae']=float((p-r).abs().double().mean())
    differences=[(torch.diff(p,dim=d)-torch.diff(r,dim=d)).abs().flatten() for d in (1,2)]
    result['gradient_mae']=float(torch.cat(differences).double().mean())
    return {'reference':r.numpy(),'distorted':p.numpy(),'difference':(p-r).abs().mean(0).numpy(),'metrics':result,'clipped_fraction':clipped,'strength':a,'kind':kind}
