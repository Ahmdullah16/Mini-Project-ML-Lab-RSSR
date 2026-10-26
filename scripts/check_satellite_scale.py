"""Offline satellite scale UI: exact-source job and actual cached model inference."""
import io,json,sys,zipfile,hashlib
from pathlib import Path
import numpy as np,rasterio
from rasterio.transform import from_origin
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'app'))
from satellite_scale_ui import scientific_job
spec=json.loads((ROOT/'configs/hosted-model.json').read_text(encoding='utf-8'))
def source(w,h):
    with rasterio.io.MemoryFile() as mem:
        with mem.open(driver='GTiff',width=w,height=h,count=3,dtype='uint8',crs='EPSG:32632',transform=from_origin(100,200,10,10)) as ds:ds.write(np.full((3,h,w),80,np.uint8))
        return mem.read()
data=source(5,3)
with zipfile.ZipFile(io.BytesIO(scientific_job(data,spec,64,{'fixture':True}))) as job:
    assert job.read('source.tif')==data
    assert json.loads(job.read('request.json'))['scale']==64
app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=60).run()
app.radio(key='section').set_value('Satellite Explorer').run()
app.session_state['fresh_prepared']={'geotiff':data,'model':spec,'metadata':{'fixture':True,'acquisition_time':'not a real acquisition'}}
app.run();selector=next(x for x in app.selectbox if x.label=='Satellite output scale');assert len(selector.options)==6
selector.set_value(4).run()
next(x for x in app.button if x.label=='Run prepared satellite scale').click().run(timeout=120)
assert not app.error and not app.exception,([x.value for x in app.error],[x.value for x in app.exception])
fresh=app.session_state['fresh_result']
with rasterio.io.MemoryFile(fresh['geotiff']) as mem:
    with mem.open() as ds:assert ds.width==20 and ds.height==12 and ds.crs.to_epsg()==32632
app.session_state['fresh_prepared']={'geotiff':source(1024,17),'model':spec,'metadata':{'fixture':True}}
app.run();next(x for x in app.selectbox if x.label=='Satellite output scale').set_value(64).run()
assert next(x for x in app.button if x.label=='Run prepared satellite scale').disabled
(ROOT/'reports/scale-satellite-validation.json').write_text(json.dumps({'scope':'Offline synthetic georeferenced source; actual cached pilot. No provider acquisition or cross-sensor validation.','source_job_bytes_equal':True,'actual_output':[20,12],'scale':4,'crs':'EPSG:32632','host_limit_rejection':True},indent=2),encoding='utf-8')
print('Satellite exact-source job, actual x4 output/CRS, and large-request rejection passed')
