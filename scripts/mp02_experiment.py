"""Reproducible MP02 timing on a fixed training crop, never the held-out test."""
import json,time,statistics,platform,sys
from pathlib import Path
import numpy as np
import rasterio
from rssr.core import digest
from rssr.array_lab import brighten,masked_channel_stats
root=Path(__file__).resolve().parents[1]
manifest=root/'configs/naip-pilot-manifest.json'; source=json.loads(manifest.read_text())
scene=next(s for s in source['scenes'] if s['split']=='train'); path=root/'data'/scene['path']
assert digest(path)==scene['sha256']
with rasterio.open(path) as ds:
    a=ds.read(window=rasterio.windows.Window(0,0,64,64)).transpose(1,2,0)
    mask=(ds.read_masks(window=rasterio.windows.Window(0,0,64,64))>0).all(axis=0)
# Fixed numerical units from the pilot manifest, not physical reflectance.
a=a.astype(np.float64)/source['divisor']; offset=np.array([.01,.02,-.01,0])
np.testing.assert_array_equal(brighten(a,offset,1),brighten(a,offset,1,loop=True))
for left,right in zip(masked_channel_stats(a,mask),masked_channel_stats(a,mask,loop=True)):
    np.testing.assert_allclose(left,right,rtol=1e-12,atol=1e-12)
rows=[]
for operation,call in [('brightness',lambda loop:brighten(a,offset,1,loop)),('masked_stats',lambda loop:masked_channel_stats(a,mask,loop))]:
    for loop in (True,False):
        call(loop); timings=[]
        for _ in range(5):
            start=time.perf_counter(); call(loop); timings.append(time.perf_counter()-start)
        rows.append({'operation':operation,'implementation':'loop' if loop else 'vectorized','median_seconds':statistics.median(timings),'repetitions':5,'warmup':1})
mean,var=masked_channel_stats(a,mask)
record={'project':'MP02','source_sha256':scene['sha256'],'manifest_sha256':digest(manifest),'scene_id':scene['id'],'split':'train','crop_xywh':[0,0,64,64],'shape_hwc':list(a.shape),'valid_pixels':int(mask.sum()),'bands':source['bands'],'units':source['units'],'dtype':str(a.dtype),'equivalence':'passed','channel_mean':mean.tolist(),'population_variance':var.tolist(),'timings':rows,'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'platform':platform.platform()},'limitations':'One training crop; microbenchmark includes function validation and allocation, excludes IO. Timing varies with CPU load. No SR performance claim.'}
(root/'reports/mp02-array-lab.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
