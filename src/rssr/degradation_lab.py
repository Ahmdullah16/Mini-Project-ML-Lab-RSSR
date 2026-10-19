"""NumPy blur, sampling and frequency experiments with explicit conventions."""
import numpy as np
from .array_lab import validate_hwc


def gaussian(sigma,radius):
    if not np.isfinite(sigma) or sigma<=0 or isinstance(radius,bool) or not isinstance(radius,(int,np.integer)) or radius<0:
        raise ValueError('Finite positive sigma and nonnegative integer radius required')
    t=np.arange(-radius,radius+1,dtype=float)
    g=np.exp(-t*t/(2*sigma*sigma)); g/=g.sum()
    return g[:,None]*g[None,:]


def filter_hwc(image,kernel,boundary='edge'):
    x=validate_hwc(image); k=np.asarray(kernel,dtype=float)
    if k.ndim!=2 or min(k.shape)<1 or any(n%2==0 for n in k.shape) or not np.isfinite(k).all() or boundary not in ('edge','reflect','constant'):
        raise ValueError('Finite odd 2D kernel and edge/reflect/constant boundary required')
    padded=np.pad(x,((k.shape[0]//2,)*2,(k.shape[1]//2,)*2,(0,0)),mode=boundary)
    windows=np.lib.stride_tricks.sliding_window_view(padded,k.shape,axis=(0,1))
    return np.einsum('hwcij,ij->hwc',windows,k)


def degrade_hwc(image,scale=2,sigma=.8,noise=0.,seed=17,boundary='edge'):
    x=validate_hwc(image)
    if scale not in (2,4) or any(n%scale for n in x.shape[:2]): raise ValueError('Scale 2/4 must divide spatial dimensions')
    if not np.isfinite([sigma,noise]).all() or sigma<0 or noise<0: raise ValueError('Nonnegative finite sigma/noise required')
    blurred=filter_hwc(x,gaussian(sigma,int(np.ceil(3*sigma))),boundary) if sigma else x
    h,w,c=x.shape
    lr=blurred.reshape(h//scale,scale,w//scale,scale,c).mean((1,3))
    return lr+np.random.default_rng(seed).normal(0,noise,lr.shape)


def spectrum(image):
    x=np.asarray(image,dtype=float)
    if x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all(): raise ValueError('Finite nonempty 2D image required')
    f=np.fft.fft2(x)
    return f,np.fft.fftshift(np.log1p(np.abs(f)))
