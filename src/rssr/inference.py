"""Disk-backed georeferenced scientific inference, with explicit native x2 cascades."""
import json,math,shutil,time
from pathlib import Path
import numpy as np
import torch
from .experiment import load_model
from .core import digest
from .scaling import plan,TILE,HALO


def geotiff_plan(source,scale):
    import rasterio
    with rasterio.open(source) as ds:
        request=plan(ds.width,ds.height,scale)
        request['pipeline']='scientific-disk-cascade-v1'
        request.pop('output_rgb_bytes',None)
        request['output_float32_bytes']=request['output_pixels']*ds.count*4
        request['estimated_disk_bytes']=sum(ds.width*ds.height*4**i*ds.count*4 for i in range(1,request['stages']+1))+16*1024*1024
        request['method']='Repeated native x2 float32 stages; no display clipping or quantization. Invalid masks propagated; no recovered-detail guarantee.'
        return request


def geotiff(checkpoint,source,target,divisor,scale=None):
    import rasterio
    from rasterio.windows import Window
    from affine import Affine
    if not np.isfinite(divisor) or divisor<=0:raise ValueError('Positive finite input divisor required')
    model,state=load_model(checkpoint);scale=model.scale if scale is None else scale
    if model.scale!=2:raise ValueError('Scientific cascade currently requires a native x2 checkpoint')
    request=geotiff_plan(source,scale);target=Path(target)
    if target.exists():raise ValueError('Output already exists; choose a new output path')
    target.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(target.parent).free<request['estimated_disk_bytes']:raise ValueError('Insufficient free disk')
    stages=target.with_name(target.stem+'-stages')
    if stages.exists():raise ValueError('Stage directory already exists')
    stages.mkdir();previous=Path(source);records=[];started=time.perf_counter()
    grid=model.context.window if getattr(model,'context',None) else 1
    tile=max(grid,TILE//grid*grid);halo=math.ceil(HALO/grid)*grid
    torch.set_num_threads(2);model.eval()
    with rasterio.Env(GDAL_CACHEMAX=32*1024*1024),torch.inference_mode():
        for stage in range(1,request['stages']+1):
            output=stages/'final.partial.tif' if stage==request['stages'] else stages/f'stage-{stage}.tif'
            with rasterio.open(previous) as src:
                if src.count!=model.bands:raise ValueError('Band mismatch')
                h,w=src.height,src.width
                profile=dict(driver='GTiff',width=w*2,height=h*2,count=src.count,dtype='float32',
                             crs=src.crs,transform=src.transform*Affine.scale(.5),nodata=float('nan'),
                             tiled=True,blockxsize=256,blockysize=256,BIGTIFF='YES')
                with rasterio.open(output,'w',**profile) as dst:
                    for y in range(0,h,tile):
                        for x in range(0,w,tile):
                            y0=max(0,y-halo);x0=max(0,x-halo);y1=min(h,y+tile);x1=min(w,x+tile)
                            window=Window(x0,y0,min(w,x1+halo)-x0,min(h,y1+halo)-y0)
                            a=src.read(window=window).astype(np.float32)/(divisor if stage==1 else 1)
                            mask=(src.read_masks(window=window)>0).all(0)&np.isfinite(a).all(0);a[:,~mask]=0
                            out=model(torch.from_numpy(a)[None])[0].numpy()
                            if out.shape!=(model.bands,a.shape[1]*2,a.shape[2]*2) or not np.isfinite(out).all():raise ValueError('Invalid model output')
                            yy=(y-y0)*2;xx=(x-x0)*2
                            core=out[:,yy:yy+(y1-y)*2,xx:xx+(x1-x)*2].copy()
                            valid=mask[y-y0:y1-y0,x-x0:x1-x0].repeat(2,0).repeat(2,1)
                            core[:,~valid]=np.nan
                            dst.write(core,window=Window(x*2,y*2,(x1-x)*2,(y1-y)*2))
                    dst.descriptions=src.descriptions
                    dst.update_tags(description='Native x2 model cascade; unsupported detail possible',units='input units divided by divisor',checkpoint_sha256=digest(checkpoint))
            records.append({'stage':stage,'scale':2**stage,'path':str(target if stage==request['stages'] else output),'width':w*2,'height':h*2})
            previous=output
    previous.replace(target)
    target.with_suffix('.json').write_text(json.dumps({**request,'source_sha256':digest(source),'checkpoint_sha256':digest(checkpoint),
        'run_id':state['run_id'],'provenance':state['provenance'],'divisor':divisor,'scale':scale,'stages':records,
        'runtime_seconds':time.perf_counter()-started,'tiling':'128 LR pixels, 32 halo; globally pooled models are approximate',
        'status':'complete'},indent=2),encoding='utf-8')
    return target
