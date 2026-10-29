"""Bounded, common-grid display comparison; never an accuracy reference."""
import numpy as np
import rasterio
import torch
from torch.nn import functional as F
from rasterio.enums import Resampling
from .scaling import HOST_PIXELS

def comparison_views(source,output,scale):
    with rasterio.io.MemoryFile(source) as mem:
        with mem.open() as ds:
            if ds.count!=3:raise ValueError('Display comparison requires RGB; no implicit band conversion')
            h,w=ds.height*scale,ds.width*scale
            if h*w>HOST_PIXELS:raise ValueError('Display baseline exceeds bounded host allocation')
            src=(ds.read(masked=True).astype(np.float32)/255.).astype(np.float32)
            valid=~np.ma.getmaskarray(src).any(0)
            tensor=torch.from_numpy(src.filled(0).copy())[None]
            expected_crs=ds.crs;expected_bounds=tuple(ds.bounds)
    ph=max(1,round(h*min(1,256/max(h,w))));pw=max(1,round(w*min(1,256/max(h,w))))
    baseline=F.interpolate(tensor,size=(h,w),mode='bicubic',align_corners=False)
    baseline=F.interpolate(baseline,size=(ph,pw),mode='area')[0].numpy().transpose(1,2,0)
    mask=F.interpolate(torch.from_numpy(valid.astype(np.float32))[None,None],size=(h,w),mode='nearest')
    mask=F.interpolate(mask,size=(ph,pw),mode='area')[0,0].numpy()==1
    with rasterio.io.MemoryFile(output) as mem:
        with mem.open() as ds:
            if ds.count!=3 or (ds.width,ds.height)!=(w,h) or ds.crs!=expected_crs or tuple(ds.bounds)!=expected_bounds:
                raise ValueError('Comparison grids, CRS or extent differ')
            out=ds.read(out_shape=(3,ph,pw),resampling=Resampling.average,masked=True)
            mask &= ~np.ma.getmaskarray(out).any(0)
            result=out.filled(0).transpose(1,2,0)
    difference=np.mean(abs(result-baseline),axis=2)
    baseline[~mask]=0;result[~mask]=0;difference[~mask]=0
    return {'baseline':np.clip(baseline,0,1),'model':np.clip(result,0,1),'disagreement':np.clip(difference,0,1),'valid':mask,'output_hw':[h,w]}

def crop_and_wipe(views,x,y,zoom,split):
    if not (0<=x<=1 and 0<=y<=1 and 0<=split<=1 and zoom in (1,2,4)):
        raise ValueError('Normalized centers/split and zoom1/2/4 required')
    h,w=views['valid'].shape;ch=max(1,h//zoom);cw=max(1,w//zoom)
    top=min(h-ch,max(0,round(y*(h-1))-ch//2));left=min(w-cw,max(0,round(x*(w-1))-cw//2))
    sl=(slice(top,top+ch),slice(left,left+cw));a=views['baseline'][sl];b=views['model'][sl]
    wipe=b.copy();cut=round(cw*split);wipe[:,:cut]=a[:,:cut]
    return {'baseline':a,'model':b,'wipe':wipe,'disagreement':views['disagreement'][sl],'preview_crop_xywh':[left,top,cw,ch]}
