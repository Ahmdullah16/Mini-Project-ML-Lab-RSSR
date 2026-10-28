"""Local hash-pinned, windowed reflectance job. No network or implicit alignment."""
import argparse
import copy
import hashlib
import json
import shutil
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from .surface_reflectance import prepare, export

MAX_SOURCE_BYTES=512*1024*1024


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def inspect(request_path):
    request_path=Path(request_path).resolve()
    request=json.loads(request_path.read_text(encoding='utf-8'))
    window=request.get('window')
    if not isinstance(window,list) or len(window)!=4 or any(type(v) is not int for v in window):
        raise ValueError('Explicit integer [column,row,width,height] window required')
    col,row,width,height=window
    if min(col,row)<0 or not 1<=width<=1024 or not 1<=height<=1024:
        raise ValueError('Offsets must be nonnegative and dimensions1..1024')
    bands=request.get('band_assets',[])
    if not 1<=len(bands)<=13:raise ValueError('1..13 spectral assets required')
    assets=[*bands,request['scl_asset'],request['validity_asset'],request['saturation_asset']]
    paths=[];total=0
    for asset in assets:
        path=Path(asset['path'])
        if not path.is_absolute():path=request_path.parent/path
        path=path.resolve()
        if not path.is_file() or path.suffix.lower() not in ('.tif','.tiff'):
            raise ValueError('Existing local TIFF assets required; URLs and VRTs are not accepted')
        checksum=asset.get('sha256','')
        if len(checksum)!=64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ValueError('Lowercase SHA256 required for every asset')
        total+=path.stat().st_size;paths.append(path)
    if total>MAX_SOURCE_BYTES:raise ValueError('Source files exceed512MiB local verification cap')
    with ExitStack() as stack:
        readers=[stack.enter_context(rasterio.open(path,driver='GTiff')) for path in paths]
        first=readers[0]
        grid=(first.crs,first.transform,first.width,first.height)
        if not first.crs or col+width>first.width or row+height>first.height:
            raise ValueError('Window outside referenced raster; no implicit crop/resize')
        for i,ds in enumerate(readers):
            if len(ds.files)!=1:
                raise ValueError('External raster sidecars are not hash-pinned; use self-contained TIFFs')
            if ds.driver!='GTiff' or (ds.crs,ds.transform,ds.width,ds.height)!=grid:
                raise ValueError('All assets must be GeoTIFFs on the same grid; explicit alignment required')
            expected=1 if i<len(bands)+1 else len(bands)
            if ds.count!=expected:raise ValueError('Asset band count differs from role')
            if i>=len(bands) and any(t!='uint8' for t in ds.dtypes):
                raise ValueError('SCL and binary quality masks must be uint8')
            if i<len(bands) and ds.dtypes[0] not in ('uint16','int16','float32','float64'):
                raise ValueError('Display uint8 is not a scientific spectral source')
        transform=first.window_transform(Window(*window))
        output_grid={'crs':str(first.crs),'transform':list(transform)[:6],'shape':[height,width]}
    plan={'operation':'L2A surface-reflectance preparation; no super-resolution',
          'input_window':window,'output_shape':[len(bands),height,width],'output_grid':output_grid,
          'source_bytes_to_hash':total,'source_byte_cap':MAX_SOURCE_BYTES,
          'estimated_working_bytes':64*len(bands)*width*height+64*1024*1024,
          'memory_note':'Conservative array allowance plus64MiB; not a measured RSS peak or total GDAL guarantee',
          'raw_output_bytes':len(bands)*width*height*4,
          'checksums_verified':False,'execution':'requires --execute; no pixels read by this plan'}
    return request,assets,paths,plan


def run(request_path,output):
    request,assets,paths,plan=inspect(request_path)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    existing_parent=next(p for p in output.resolve().parents if p.exists())
    if shutil.disk_usage(existing_parent).free < plan['raw_output_bytes']+32*1024*1024:
        raise ValueError('Insufficient output disk space including32MiB reserve')
    for asset,path in zip(assets,paths):
        if sha256(path)!=asset['sha256']:raise ValueError('Source checksum mismatch: '+path.name)
    n=len(request['band_assets']);window=Window(*request['window'])
    with ExitStack() as stack:
        readers=[stack.enter_context(rasterio.open(path,driver='GTiff')) for path in paths]
        samples=[ds.read(1,window=window) for ds in readers[:n]]
        if len({a.dtype.str for a in samples})!=1:raise ValueError('Mixed spectral storage dtypes require explicit conversion')
        values=np.stack(samples)
        valid=readers[n+1].read(window=window);sat=readers[n+2].read(window=window)
        if not np.isin(valid,(0,1)).all() or not np.isin(sat,(0,1)).all():
            raise ValueError('Quality masks require explicit binary0/1 values, not255 or arbitrary flags')
        valid=valid.astype(bool)&np.stack([ds.read_masks(1,window=window)>0 for ds in readers[:n]])
        # Invalid quality-layer coverage cannot become an apparently valid observation.
        quality_valid=(readers[n].read_masks(1,window=window)>0)
        quality_valid &= (readers[n+1].read_masks(window=window)>0).all(0)
        quality_valid &= (readers[n+2].read_masks(window=window)>0).all(0)
        valid &= quality_valid[None]
        scl=readers[n].read(1,window=window)
        meta=copy.deepcopy(request['metadata']);grid=plan['output_grid']
        for key in ('grid','scl_grid','quality_grid'):
            if key in meta and meta[key]!=grid:raise ValueError('Declared crop grid disagrees with actual raster')
            meta[key]=copy.deepcopy(grid)
        if len(meta.get('bands',[]))!=n:raise ValueError('Metadata band count differs')
        for band,ds in zip(meta['bands'],readers[:n]):
            if 'grid' in band and band['grid']!=grid:raise ValueError('Declared band grid disagrees')
            if ds.nodata is None or ds.nodata!=band.get('nodata'):
                raise ValueError('Raster nodata must agree with explicit band metadata')
            band['grid']=copy.deepcopy(grid)
        meta['local_input_assets']=[{'role':'band' if i<n else ('scl','validity','saturation')[i-n],
            'file':paths[i].name,'sha256':asset['sha256']} for i,asset in enumerate(assets)]
        meta['requested_window']=request['window'];meta['request_sha256']=sha256(request_path)
        prepared,mask,provenance=prepare(values,scl,valid,sat.astype(bool),meta)
    export(output,prepared,mask,provenance)
    return {**plan,'checksums_verified':True,'execution':'completed local conversion',
            'output_sha256':sha256(output),'valid_fraction':float(mask.mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('request');p.add_argument('output');p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    _,_,_,plan=inspect(args.request)
    print(json.dumps(plan,indent=2),flush=True)
    if args.execute:print(json.dumps(run(args.request,args.output),indent=2))


if __name__=='__main__':main()
