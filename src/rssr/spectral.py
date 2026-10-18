"""Band-aware diagnostics; callers must establish reflectance compatibility."""
import torch
from .core import image_tensor


def ndvi(image, red, nir, reflectance=False):
    x=image_tensor(image)
    if not reflectance or red==nir or min(red,nir)<0 or max(red,nir)>=x.shape[0]:
        raise ValueError('Explicit compatible reflectance and distinct red/NIR bands required')
    denominator=x[nir]+x[red]
    valid=denominator.abs()>1e-8
    value=torch.full_like(denominator,float('nan'))
    value[valid]=(x[nir][valid]-x[red][valid])/denominator[valid]
    return value,valid


def ergas(pred,ref,scale,compatible_units=False):
    p,r=image_tensor(pred),image_tensor(ref)
    if not compatible_units or p.shape!=r.shape or scale not in (2,4):
        raise ValueError('Matching calibrated units, shapes and scale required')
    means=r.mean((1,2))
    if (means.abs()<1e-8).any():
        raise ValueError('ERGAS undefined for a zero-mean reference band')
    mse=(p-r).square().mean((1,2))
    return float(100/scale*torch.sqrt((mse/means.square()).mean()))


def charbonnier(pred,ref,epsilon=1e-3):
    if epsilon<=0: raise ValueError('Positive epsilon required')
    return ((pred-ref).square()+epsilon**2).sqrt().mean()


def edge_l1(pred,ref):
    if pred.shape!=ref.shape or min(pred.shape[-2:])<2:
        raise ValueError('Matching arrays with spatial dimensions >=2 required')
    horizontal=((pred[...,1:]-pred[...,:-1])-(ref[...,1:]-ref[...,:-1])).abs().mean()
    vertical=((pred[...,1:,:]-pred[...,:-1,:])-(ref[...,1:,:]-ref[...,:-1,:])).abs().mean()
    return (horizontal+vertical)/2
