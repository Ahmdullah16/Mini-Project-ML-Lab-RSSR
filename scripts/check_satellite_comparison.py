"""Offline comparison uses actual cached checkpoints; injected failure tests retry."""
import hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np,rasterio
from rasterio.transform import from_origin
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'app')]
from rssr.satellite_comparison import execute,identity
from inference_ui import CATALOG
with rasterio.io.MemoryFile() as mem:
 with mem.open(driver='GTiff',width=5,height=3,count=3,dtype='uint8',crs='EPSG:32632',transform=from_origin(100,200,10,10)) as ds:
  ds.write(np.full((3,3,5),80,np.uint8))
 source=mem.read()
app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=60).run()
app.radio(key='section').set_value('Satellite Explorer').run()
app.session_state['fresh_prepared']={'geotiff':source,'model':CATALOG[0],'metadata':{'fixture':True}}
app.radio(key='satellite_workflow').set_value('Compare models').run()
calls=[]
def fail_second(data,checkpoint,spec,scale,acquisition):
 calls.append(spec['label'])
 if spec==CATALOG[1]:raise ValueError('Injected retry test')
 return execute(data,checkpoint,spec,scale,acquisition)
with patch('satellite_compare_ui.execute',side_effect=fail_second):
 app.button(key='sat_compare_run').click().run(timeout=120)
assert not app.exception
cache=app.session_state['sat_compare_results'];assert len(cache)==1 and len(app.session_state['sat_compare_failures'])==1
first=identity(source,CATALOG[0],2);original_hash=hashlib.sha256(cache[first]['geotiff']).hexdigest()
second=identity(source,CATALOG[1],2)
app.button(key=second+'_retry').click().run(timeout=120)
assert not app.exception,[x.value for x in app.exception]
cache=app.session_state['sat_compare_results'];assert len(cache)==2 and not app.session_state['sat_compare_failures']
assert hashlib.sha256(cache[first]['geotiff']).hexdigest()==original_hash
for r in cache.values():
 assert r['metadata']['source_bytes_sha256']==hashlib.sha256(source).hexdigest()
 with rasterio.io.MemoryFile(r['geotiff']) as mem:
  with mem.open() as ds:
   assert (ds.width,ds.height,ds.crs.to_epsg())==(10,6,32632)
   assert tuple(ds.bounds)==(100,170,150,200)
saved={k:hashlib.sha256(v['geotiff']).hexdigest() for k,v in cache.items()}
app.selectbox(key='sat_detail_zoom').set_value(4).run()
app.slider(key='sat_detail_split').set_value(0.).run()
app.slider(key='sat_detail_x').set_value(1.).run()
assert not app.exception
assert saved=={k:hashlib.sha256(v['geotiff']).hexdigest() for k,v in app.session_state['sat_compare_results'].items()}
assert any('Shared preview crop' in x.value for x in app.caption)
app.multiselect(key='sat_compare_models').set_value([s['label'] for s in CATALOG]).run()
app.button(key='sat_compare_run').click().run(timeout=180)
assert not app.exception and not app.error
assert len(app.session_state['sat_compare_results'])==len(CATALOG)
app.selectbox(key='sat_compare_scale').set_value(4).run()
assert not app.session_state['sat_compare_results']
assert identity(source,CATALOG[0],2)!=identity(source,CATALOG[0],4)
assert identity(source,CATALOG[0],2)!=identity(source+b'changed',CATALOG[0],2)
(ROOT/'reports/seven-family-satellite-validation.json').write_text(json.dumps({'scope':'Offline synthetic5x3 georeferenced fixture, all catalog checkpoints across seven families; first two used for injected failure/retry; no provider query or generalization claim','verified':['same source SHA256 for both models','two exact10x6 outputs, CRS32632 and unchanged spatial extent','injected second-model failure preserves first output','targeted retry succeeds without changing first output','all catalog checkpoints executed on common source','shared zoom/divider controls preserve exact exports','scale change clears stale results','input and scale identities differ'],'not_verified':['provider scientific alignment','cloud masks','current public UI']},indent=2)+'\n',encoding='utf-8')
print('All catalog satellite model exports; source/CRS/extent, failure/retry and scale invalidation passed')
