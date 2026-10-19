"""Shared bounded RGB display inference for Gradio and Streamlit."""
import numpy as np
import torch
from .core import resize
from .models import tiled


def validate_rgb(image,max_side=1024):
    if image is None: raise ValueError('Choose an opaque RGB image')
    a=np.asarray(image)
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<1 or max(a.shape[:2])>max_side:
        raise ValueError(f'Expected uint8 RGB, at most {max_side} pixels per side')
    return a


def predict_rgb(image,model,state,checkpoint_sha256,max_side=1024):
    a=validate_rgb(image,max_side)
    if model.bands!=3: raise ValueError('RGB display requires a three-band checkpoint')
    x=torch.from_numpy(a.astype(np.float32)/255).permute(2,0,1)
    torch.set_num_threads(2)
    y=tiled(model,x[None])[0]
    baseline=resize(x,model.scale)
    display=lambda v:np.rint(v.permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    metadata={'run_id':state['run_id'],'checkpoint_sha256':checkpoint_sha256,
              'status':state['provenance'],'scale':model.scale,'sensor':'user supplied; unverified',
              'acquisition_time':'unknown','reference_metrics':'unavailable: no HR reference',
              'display_scaling':'fixed [0,1], clipped for RGB display only',
              'tiling':{'tile':128,'halo':32,'note':'approximate context for globally pooled models'}}
    return display(baseline),display(y),metadata
