"""Scale preflight on the exact acquired georeferenced source, before inference."""
import io,json,tempfile,zipfile
from pathlib import Path
import streamlit as st
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rssr.scaling import SCALES
from rssr.inference import geotiff,geotiff_plan


def scientific_job(source,spec,scale,acquisition):
    output=io.BytesIO()
    script='''import json,sys,tempfile
from pathlib import Path
from rssr.inference import geotiff,geotiff_plan
from rssr.hosted_assets import fetch_asset
r=json.load(open('request.json'));spec=json.load(open('model.json'))
plan=geotiff_plan('source.tif',r['scale'])
assert plan['pipeline']==r['pipeline'], 'Use the matching RSSR pipeline version'
print(json.dumps(plan,indent=2))
if '--execute' in sys.argv:
    checkpoint=fetch_asset(spec['checkpoint'],Path(tempfile.gettempdir())/'rssr-public-assets-v1')
    geotiff(checkpoint,'source.tif','output.tif',255.,r['scale'])
'''
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('source.tif',source);z.writestr('model.json',json.dumps(spec));z.writestr('request.json',json.dumps({'scale':scale,'divisor':255,'pipeline':'scientific-disk-cascade-v1'}))
        z.writestr('acquisition.json',json.dumps(acquisition,indent=2));z.writestr('run.py',script)
        z.writestr('README.txt','Use matching RSSR checkout, install app/requirements.txt and set PYTHONPATH to checkout/src. Run python run.py for preflight, then python run.py --execute. Same acquired source, not a new provider query. Output is float32 display-normalized data, not calibrated reflectance. Preserve acquisition.json with output. No paid service required.')
    return output.getvalue()


def render_satellite_scale():
    scale=st.selectbox('Satellite output scale',SCALES,format_func=lambda s:f'×{s}')
    st.caption('Acquire first, then inspect exact dimensions/resources and run the chosen scale. Larger scales are native ×2 float32 cascades; no display quantization between stages. All choices remain visible.')
    prepared=st.session_state.get('fresh_prepared')
    if prepared is None:return
    from inference_ui import resources
    with tempfile.TemporaryDirectory(prefix='rssr-geo-preflight-') as temp:
        source=Path(temp)/'source.tif';source.write_bytes(prepared['geotiff']);request=geotiff_plan(source,scale)
    st.write(f"Exact output {request['output_width']}×{request['output_height']} = {request['output_pixels']:,} pixels. Float32 final {request['output_float32_bytes']:,} bytes; estimated disk {request['estimated_disk_bytes']:,} bytes; working-memory allowance {request['estimated_working_memory_bytes']:,} bytes (not a measured guarantee).")
    st.caption('Configured host limit: 1,048,576 output pixels; local/Colab jobs use the identical acquired source and pinned model. Input and selected scale are never reduced. Prepared model: '+prepared['model']['label'])
    if not request['host_available']:st.warning('Requested execution is unavailable on the free host. Download the exact-source job.')
    st.download_button('Download exact satellite local/Colab job',scientific_job(prepared['geotiff'],prepared['model'],scale,prepared['metadata']),'satellite-job.zip','application/zip')
    if st.button('Run prepared satellite scale',disabled=not request['host_available']):
        try:
            model,state,checkpoint,lock=resources(prepared['model'])
            if not lock.acquire(blocking=False):raise ValueError('CPU busy; retry shortly')
            try:
                with tempfile.TemporaryDirectory(prefix='rssr-geo-run-') as temp:
                    source=Path(temp)/'source.tif';source.write_bytes(prepared['geotiff']);target=Path(temp)/'output.tif'
                    geotiff(checkpoint,source,target,255.,scale)
                    meta=json.loads(target.with_suffix('.json').read_text(encoding='utf-8'));meta['acquisition']=prepared['metadata'];meta['model']=prepared['model']
                    st.session_state['fresh_result']={'metadata':meta,'geotiff':target.read_bytes()}
            finally:lock.release()
        except Exception as error:st.error('Satellite inference did not complete: '+str(error))
    fresh=st.session_state.get('fresh_result')
    if fresh:
        with rasterio.io.MemoryFile(fresh['geotiff']) as memory:
            with memory.open() as src:
                ratio=min(1,512/max(src.width,src.height));preview=src.read(out_shape=(3,max(1,round(src.height*ratio)),max(1,round(src.width*ratio))),resampling=Resampling.average).transpose(1,2,0)
        lo,hi=np.nanpercentile(preview,[2,98])
        if hi>lo:st.image(np.clip((preview-lo)/(hi-lo),0,1),caption='Bounded contrast-stretched preview only; GeoTIFF preserves float32 values')
        st.caption('Completed scale ×'+str(fresh['metadata']['scale'])+'; settings above do not relabel this result. Finer sampling is not proven physical resolution. No aligned HR reference metrics.')
        st.json(fresh['metadata'])
        st.download_button('Full-resolution satellite GeoTIFF',fresh['geotiff'],'satellite-sr.tif','image/tiff')
        st.download_button('Satellite provenance',json.dumps(fresh['metadata'],indent=2),'satellite-provenance.json','application/json')
