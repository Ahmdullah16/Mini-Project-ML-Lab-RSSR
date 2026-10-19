"""Explicit bounded recent Sentinel RGB acquisition -> pilot SR, without HR metrics."""
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from .cog import bounded_cog
from .core import config_hash,digest
from .data import now
from .db import connect
from .experiment import load_model
from .inference import geotiff
from .ingest import request_json


def stored_path(root, value):
    """Resolve portable catalog paths; retain support for older absolute records."""
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def acquire_infer(config,checkpoint,root,fetch=request_json):
    model,state=load_model(checkpoint)
    if model.bands!=3: raise ValueError('Sentinel TCI requires a three-band RGB display checkpoint')
    size=config.get('crop_size',256)
    if not isinstance(size,int) or not 16<=size<=512: raise ValueError('Crop size must be 16..512')
    query={k:config[k] for k in ('bbox','datetime','query') if k in config}
    query['bbox']=[float(v) for v in query['bbox']]
    bbox = query['bbox']
    if len(bbox)!=4 or not np.isfinite(bbox).all() or not -180<=bbox[0]<bbox[2]<=180 or not -90<=bbox[1]<bbox[3]<=90:
        raise ValueError('Invalid geographic bounding box')
    if bbox[2]-bbox[0]>1 or bbox[3]-bbox[1]>1:
        raise ValueError('AOI edges must be at most one degree')
    query.update(collections=['sentinel-2-l2a'],limit=1,sortby=[{'field':'properties.datetime','direction':'desc'}])
    result=fetch('https://earth-search.aws.element84.com/v1/search',query)
    if not result['features']: raise ValueError('No observation matches the bounded query')
    item=result['features'][0]; asset=item['assets']['visual']; props=item['properties']
    if [b.get('common_name') for b in asset.get('eo:bands',[])]!=['red','green','blue']:
        raise ValueError('Provider band mapping is not RGB')
    if any(b.get('data_type')!='uint8' for b in asset.get('raster:bands',[])):
        raise ValueError('Only uint8 display imagery is compatible with this pilot')
    licenses=[l['href'] for l in item.get('links',[]) if l['rel']=='license']
    if not licenses: raise ValueError('Provider license link missing')
    root=Path(root); db=connect(root/'catalog.sqlite'); cp=digest(checkpoint)
    identity=config_hash({'item':item['id'],'asset':asset['href'],'checkpoint':cp,'bbox':query.get('bbox'),'size':size})
    folder=root/'fresh'/identity; folder.mkdir(parents=True,exist_ok=True)
    old=db.execute('SELECT status,output_path,output_sha256 FROM inference_jobs WHERE id=?',(identity,)).fetchone()
    if old and old[0]=='completed':
        if digest(stored_path(root,old[1]))!=old[2]: db.close(); raise ValueError('Existing SR output checksum changed')
        db.close(); cached=json.loads((folder/'provenance.json').read_text()); cached['cache_hit']=True; return cached
    provenance={'provider_item_id':item['id'],'query':query,'acquisition_time':props.get('datetime'),'availability_time':props.get('created'),'ingestion_time':now(),'asset_url':asset['href'],'license_links':licenses,'attribution':'Contains modified Copernicus Sentinel data; provider Element 84 Sentinel COG archive','checkpoint_sha256':cp,'checkpoint_run_id':state['run_id'],'domain_limitation':'NAIP-trained RGB display pilot applied to Sentinel TCI; cross-sensor quality unvalidated; not reflectance or scientific spectral inference','reference_metrics':None,'cloud_mask':'item-level cloud filter only; no pixel-level cloud mask'}
    with db:
        db.execute('INSERT OR REPLACE INTO inference_jobs(id,provider_item_id,acquisition_time,ingestion_time,status,checkpoint_sha256,provenance) VALUES(?,?,?,?,?,?,?)',(identity,item['id'],props.get('datetime'),provenance['ingestion_time'],'running',cp,json.dumps(provenance)))
    try:
        source=folder/'source.tif'; output=folder/'sr.tif'
        with bounded_cog(asset['href'],config.get('max_bytes',32_000_000)) as (url,transfer):
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',GDAL_HTTP_MULTIRANGE='SERIAL',GDAL_HTTP_MAX_RETRY='0'):
                with rasterio.open(url) as ds:
                    if ds.count!=3 or ds.dtypes!=('uint8',)*3 or not ds.crs: raise ValueError('Invalid RGB COG metadata')
                    a=transform_bounds('EPSG:4326',ds.crs,*query['bbox']); b=ds.bounds
                    left,bottom,right,top=max(a[0],b.left),max(a[1],b.bottom),min(a[2],b.right),min(a[3],b.top)
                    if left>=right or bottom>=top: raise ValueError('AOI does not intersect raster')
                    row,col=ds.index((left+right)/2,(bottom+top)/2)
                    window=Window(max(0,min(ds.width-size,col-size//2)),max(0,min(ds.height-size,row-size//2)),size,size)
                    values=ds.read(window=window); valid=(ds.read_masks(window=window)>0).all(0)
                    if valid.mean()<.8: raise ValueError('Crop has fewer than 80% valid display pixels')
                    profile=ds.profile.copy(); profile.update(width=size,height=size,transform=ds.window_transform(window),driver='GTiff')
                    with rasterio.open(source,'w',**profile) as out:
                        out.write(values); out.write_mask(valid.astype('uint8')*255)
                        out.descriptions=('red display','green display','blue display')
                    provenance.update(window=list(window.flatten()),crs=str(ds.crs),source_transform=list(profile['transform']),valid_fraction=float(valid.mean()))
        provenance['transfer']=transfer; provenance['source_sha256']=digest(source)
        geotiff(checkpoint,source,output,255.)
        provenance.update(inference_time=now(),output_sha256=digest(output),source_path=source.relative_to(root).as_posix(),output_path=output.relative_to(root).as_posix(),status='completed')
        (folder/'item.json').write_text(json.dumps(item,indent=2),encoding='utf-8')
        (folder/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        with db:
            db.execute("UPDATE inference_jobs SET inference_time=?,status='completed',source_path=?,source_sha256=?,output_path=?,output_sha256=?,provenance=? WHERE id=?",(provenance['inference_time'],provenance['source_path'],provenance['source_sha256'],provenance['output_path'],provenance['output_sha256'],json.dumps(provenance),identity))
        return provenance
    except Exception as error:
        with db: db.execute("UPDATE inference_jobs SET status='failed',failure=? WHERE id=?",(str(error),identity))
        raise
    finally: db.close()
