"""Exact power-of-two RGB cascades with bounded tile memory and disk-backed stages."""
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch

SCALES=(2,4,8,16,32,64)
HOST_PIXELS=1024*1024
TILE=128
HALO=32
PIPELINE='rgb-disk-cascade-v1'


def plan(width,height,scale):
    if any(type(v) is not int for v in (width,height,scale)) or not (1<=width<=1024 and 1<=height<=1024):
        raise ValueError('Input width and height must independently be integers from 1 to 1024')
    if scale not in SCALES:raise ValueError('Scale must be 2, 4, 8, 16, 32 or 64')
    stages=int(math.log2(scale));pixels=width*height*scale*scale
    stage_bytes=sum(width*height*4**i*3 for i in range(1,stages+1))
    return dict(pipeline=PIPELINE,input_width=width,input_height=height,scale=scale,
                output_width=width*scale,output_height=height*scale,output_pixels=pixels,
                stages=stages,output_rgb_bytes=pixels*3,
                estimated_disk_bytes=stage_bytes+16*1024*1024,
                estimated_working_memory_bytes=512*1024*1024,
                memory_note='Configured allowance for CPU weights, tile activations and GDAL cache; not a measured peak or guarantee for arbitrary models.',
                host_available=pixels<=HOST_PIXELS,host_output_pixel_limit=HOST_PIXELS,
                method='Repeated native learned x2; each stage clips to [0,1] and quantizes RGB uint8. Model includes its declared bicubic residual baseline. No extra scale interpolation.',
                limitation='Cascades are not separately trained x4-x64 models; accumulated artifacts and global-pooling tile-context differences are possible. More pixels do not prove recovered detail.')


def cascade(image,model,directory,scale,identity):
    """Write uncompressed tiled BigTIFF stages, never a full output tensor.

    Pixel-only display products: no inferred geographic CRS. Halo core trimming
    matches rssr.models.tiled; global attention is approximate, not seam-free.
    Existing directories are rejected to prevent replacing previous job output.
    """
    from .display import validate_rgb
    import rasterio
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    from PIL import Image
    image=validate_rgb(image,1024);h,w=image.shape[:2];request=plan(w,h,scale)
    if model.scale!=2 or model.bands!=3:raise ValueError('This cascade requires a native x2 RGB checkpoint')
    directory=Path(directory)
    if directory.exists():raise ValueError('Use a new output directory')
    directory.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(directory.parent).free<request['estimated_disk_bytes']:
        raise ValueError('Insufficient free disk for declared stage outputs and overhead')
    directory.mkdir();started=time.perf_counter();records=[];previous=None
    torch.set_num_threads(2);model.eval()
    grid=model.context.window if getattr(model,'context',None) else 1
    tile=max(grid,TILE//grid*grid);halo=math.ceil(HALO/grid)*grid
    try:
        with rasterio.Env(GDAL_CACHEMAX=32*1024*1024),torch.inference_mode():
            for stage in range(1,request['stages']+1):
                target=directory/f'stage-{stage}-x{2**stage}.tif';stage_start=time.perf_counter()
                src=rasterio.open(previous) if previous else None
                raw_min=float("inf");raw_max=-float("inf");clipped=0;samples=0
                try:
                    with rasterio.open(target,'w',driver='GTiff',width=w*2,height=h*2,count=3,dtype='uint8',
                                       tiled=True,blockxsize=256,blockysize=256,BIGTIFF='YES') as dst:
                        for y in range(0,h,tile):
                            for x in range(0,w,tile):
                                y0=max(0,y-halo);x0=max(0,x-halo)
                                y1=min(h,y+tile);x1=min(w,x+tile)
                                bottom=min(h,y1+halo);right=min(w,x1+halo)
                                a=src.read(window=Window(x0,y0,right-x0,bottom-y0)) if src else image[y0:bottom,x0:right].transpose(2,0,1)
                                t=torch.from_numpy(a.astype(np.float32)/255)[None]
                                prediction=model(t)[0]
                                if prediction.shape!=(3,a.shape[1]*2,a.shape[2]*2) or not torch.isfinite(prediction).all():
                                    raise ValueError('Model returned invalid shape or nonfinite values')
                                core=prediction[:,(y-y0)*2:(y1-y0)*2,(x-x0)*2:(x1-x0)*2]
                                raw_min=min(raw_min,float(core.min()));raw_max=max(raw_max,float(core.max()))
                                clipped+=int(((core<0)|(core>1)).sum());samples+=core.numel()
                                pixels=np.rint(core.numpy().clip(0,1)*255).astype(np.uint8)
                                dst.write(pixels,window=Window(x*2,y*2,(x1-x)*2,(y1-y)*2))
                finally:
                    if src:src.close()
                w*=2;h*=2;previous=target
                records.append(dict(stage=stage,scale=2**stage,width=w,height=h,file=target.name,bytes=target.stat().st_size,runtime_seconds=time.perf_counter()-stage_start,raw_range_normalized=[raw_min,raw_max],clipped_scalar_fraction=clipped/samples))
            with rasterio.open(previous) as src:
                ratio=min(1,512/max(w,h));pw=max(1,round(w*ratio));ph=max(1,round(h*ratio))
                preview=src.read(out_shape=(3,ph,pw),resampling=Resampling.average).transpose(1,2,0)
            Image.fromarray(preview).save(directory/'preview.png')
        metadata={**request,'identity':identity,'input_pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'input_shape':list(image.shape),
                  'runtime_seconds':time.perf_counter()-started,'stage_outputs':records,'output':previous.name,
                  'preview':'preview.png','preview_note':'Bounded area-resampled preview only; full TIFF dimensions are unchanged.',
                  'tile':tile,'halo':halo,'seam_note':'Overlapped context with core trimming. Global pooling is approximate; consult inference diagnostic and cascade seam tests.',
                  'reference_metrics':'unavailable without aligned genuine HR','status':'complete'}
        (directory/'provenance.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
        return metadata
    except Exception as error:
        (directory/'failure.json').write_text(json.dumps({'status':'incomplete','error':str(error),'completed_stages':records}),encoding='utf-8')
        raise


def job_bundle(image,specification,scale):
    """Small portable job: same decoded pixels, pinned asset spec and settings."""
    import io,zipfile
    from PIL import Image
    out=io.BytesIO();png=io.BytesIO();Image.fromarray(image).save(png,format='PNG')
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('run.py',"import json,sys\nfrom rssr.scaling import PIPELINE\nr=json.load(open('request.json'))\nassert r['pipeline']==PIPELINE, 'Use the matching RSSR pipeline version'\nsys.argv=['scale_job','input.png','model.json','output','--scale',str(r['scale'])]+sys.argv[1:]\nfrom rssr.scale_job import main\nmain()\n")
        z.writestr('input.png',png.getvalue());z.writestr('model.json',json.dumps(specification,indent=2))
        z.writestr('request.json',json.dumps(plan(image.shape[1],image.shape[0],scale),indent=2))
        z.writestr('run.txt',f'From the matching RSSR checkout, install app/requirements.txt, then run:\npython -m rssr.scale_job input.png model.json output --scale {scale}\nReview the plan; add --execute to run. Set PYTHONPATH to checkout/src if not installed.\nUse a CPU runtime locally or in Colab. No paid runtime required. Disk and runtime requirements may exceed your machine.\n')
    return out.getvalue()


def cache_key(image,specification,scale):
    """Content/shape/checkpoint/settings identity; independent of UI labels."""
    payload={'shape':list(image.shape),'pixels':hashlib.sha256(image.tobytes()).hexdigest(),
             'specification':specification,'request':plan(image.shape[1],image.shape[0],scale),
             'tile':TILE,'halo':HALO}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
