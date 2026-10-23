"""Stateless Community Cloud entry point; persistent research storage is separate."""
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np
from PIL import Image
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rssr.hosted_assets import fetch_asset
from rssr.build_info import build_identity
from rssr.uploads import decode_image

BASE_CONFIG=json.loads((ROOT/'configs/hosted-model.json').read_text(encoding='utf-8'))
CATALOG=[BASE_CONFIG]+json.loads((ROOT/'configs/hosted-models.json').read_text(encoding='utf-8'))['models']
CACHE=Path(os.environ.get('RSSR_ASSET_CACHE',str(Path(tempfile.gettempdir())/'rssr-public-assets-v1')))


@st.cache_resource(show_spinner='Downloading and verifying the trained checkpoint…')
def resources(specification):
    from rssr.experiment import load_model
    path=fetch_asset(specification['checkpoint'],CACHE)
    model,state=load_model(path)
    if model.bands!=3 or model.scale!=2:
        raise ValueError('Hosted checkpoint must be compatible x2 RGB')
    return model,state,path,threading.Lock()


def png_bytes(image):
    output=io.BytesIO(); Image.fromarray(image).save(output,format='PNG'); return output.getvalue()


def model_choice():
    selected=st.selectbox('Model',[entry['label'] for entry in CATALOG],key='model_label')
    CONFIG=next(entry for entry in CATALOG if entry['label']==selected)
    st.info(CONFIG['limitations'])
    return CONFIG

def render_live():
    CONFIG=model_choice()
    choice=st.radio('Input',['Licensed example (cached acquisition)','Upload'],horizontal=True,key='input_choice')
    upload=st.file_uploader('PNG or JPEG, up to 512×512 pixels and 8 MB',type=['png','jpg','jpeg']) if choice=='Upload' else None
    if choice.startswith('Licensed'):
        st.caption(f"NAIP aerial training example acquired {CONFIG['sample']['acquisition_time']}; synthetic LR, not a fresh satellite query. {CONFIG['sample']['license']}")
    if st.button('Estimate super-resolution',type='primary'):
        try:
            if choice=='Upload' and upload is None: raise ValueError('Choose an image to upload')
            data=upload.getvalue() if choice=='Upload' else fetch_asset(CONFIG['sample'],CACHE).read_bytes()
            image=decode_image(data)
            model,state,path,lock=resources(CONFIG)
            if not lock.acquire(blocking=False): raise ValueError('The CPU is serving another request. Please retry shortly.')
            try:
                started=time.perf_counter()
                from rssr.display import predict_rgb
                baseline,sr,meta=predict_rgb(image,model,state,CONFIG['checkpoint']['sha256'],max_side=512)
                runtime=time.perf_counter()-started
            finally: lock.release()
            meta.update(model_name=CONFIG['label'],model_version=CONFIG['version'],runtime_seconds=runtime,inference_time=datetime.now(timezone.utc).isoformat(),storage='temporary session; not durable',application=build_identity(ROOT),input_shape=list(image.shape),output_shape=list(sr.shape))
            if choice.startswith('Licensed'):
                meta.update(sensor=CONFIG['sample']['sensor'],acquisition_time=CONFIG['sample']['acquisition_time'],source=CONFIG['sample']['source'],license=CONFIG['sample']['license'],input_kind='cached licensed training example')
            else: meta['input_kind']='user upload; provenance unverified'
            st.session_state['result']={'input':image,'baseline':baseline,'sr':sr,'metadata':meta}
        except Exception as error:
            st.error(f'Inference did not complete: {error}')
    if 'result' in st.session_state:
        result=st.session_state['result']
        st.subheader('Latest completed inference')
        st.caption('Result model: '+result['metadata'].get('model_name','Original CNN pilot'))
        enhance=st.checkbox('Enhance preview contrast (does not change inference or downloads)',value=True)
        low,high=np.percentile(result['input'],[2,98])
        zoom=st.slider('Synchronized zoom',1,4,1)
        xcenter=st.slider('Horizontal center',0.0,1.0,0.5)
        ycenter=st.slider('Vertical center',0.0,1.0,0.5)
        h,w=result['input'].shape[:2]; ch,cw=max(1,h//zoom),max(1,w//zoom)
        y0=int((h-ch)*ycenter); x0=int((w-cw)*xcenter)
        for column,key,label in zip(st.columns(3),['input','baseline','sr'],['Input LR','Bicubic baseline','SR estimate']):
            scale=1 if key=='input' else 2
            preview=result[key][y0*scale:(y0+ch)*scale,x0*scale:(x0+cw)*scale]
            if enhance and high>low:
                preview=np.rint(np.clip((preview.astype(float)-low)/(high-low),0,1)*255).astype(np.uint8)
            column.image(preview,caption=label,width='stretch')
        if enhance: st.caption('All previews use the same input-derived contrast range. Downloaded pixels keep the original fixed display scaling.')
        st.write(f"Inference runtime: {result['metadata']['runtime_seconds']:.3f} seconds (CPU)")
        with st.expander('Checkpoint and input provenance'):
            st.json(result['metadata'])
        st.download_button('Download SR PNG',png_bytes(result['sr']),'sr-estimate.png','image/png')
        st.download_button('Download provenance',json.dumps(result['metadata'],indent=2),'provenance.json','application/json')
        if st.button('Clear temporary result'):
            del st.session_state['result']; st.rerun()
    

def render_satellite():
    CONFIG=model_choice()
    st.caption('Explicit provider request; up to 32 MB, one 256×256 RGB display crop. Item-level cloud filtering only. Cross-sensor quality is unvalidated. No HR reference metrics are available. Results remain temporary.')
    bbox=st.text_input('AOI west,south,east,north',value='-114,43,-113.9,43.1')
    today=datetime.now(timezone.utc).date()
    start=st.date_input('Acquisition start',value=today-timedelta(days=30))
    end=st.date_input('Acquisition end',value=today)
    cloud=st.slider('Maximum item cloud cover (%)',0,100,20)
    if st.button('Acquire satellite crop and infer'):
        try:
            from rssr.fresh import acquire_infer,stored_path
            if start>end: raise ValueError('Start date must not be after end date')
            if (end-start).days>366: raise ValueError('Use an acquisition interval no longer than one year')
            bounds=[float(v.strip()) for v in bbox.split(',')]
            if len(bounds)!=4 or not np.isfinite(bounds).all() or not -180<=bounds[0]<bounds[2]<=180 or not -90<=bounds[1]<bounds[3]<=90:
                raise ValueError('Invalid geographic bounding box')
            if bounds[2]-bounds[0]>1 or bounds[3]-bounds[1]>1:
                raise ValueError('AOI edges must be at most one degree')
            model,state,path,lock=resources(CONFIG)
            if not lock.acquire(blocking=False): raise ValueError('The CPU is busy. Please retry shortly.')
            try:
                started=time.perf_counter()
                with tempfile.TemporaryDirectory(prefix='rssr-fresh-') as folder:
                    result=acquire_infer({'bbox':bounds,'datetime':f'{start}T00:00:00Z/{end}T23:59:59Z','query':{'eo:cloud_cover':{'lt':cloud}},'crop_size':256,'max_bytes':32_000_000},path,folder)
                    result.update(model_name=CONFIG['label'],model_version=CONFIG['version'],application=build_identity(ROOT),runtime_seconds=time.perf_counter()-started,refresh_time=datetime.now(timezone.utc).isoformat(),storage='temporary; local database removed after request')
                    st.session_state['fresh_result']={'metadata':result,'geotiff':stored_path(folder,result['output_path']).read_bytes()}
            finally: lock.release()
        except Exception as error:
            st.error(f'Satellite request did not complete: {error}')
    if 'fresh_result' in st.session_state:
        fresh=st.session_state['fresh_result']
        import rasterio.io
        with rasterio.io.MemoryFile(fresh['geotiff']) as memory:
            with memory.open() as raster:
                preview=raster.read([1,2,3]).transpose(1,2,0)
        lo,hi=np.nanpercentile(preview,[2,98])
        if hi>lo:
            st.image(np.clip((preview-lo)/(hi-lo),0,1),caption='Contrast-stretched SR preview; GeoTIFF preserves scientific values',width='stretch')
        st.success('Acquisition and inference completed. Downloads remain temporary.')
        with st.expander('Satellite timestamps and provenance'):
            st.json(fresh['metadata'])
        st.download_button('Download georeferenced SR',fresh['geotiff'],'satellite-sr.tif','image/tiff')
        st.download_button('Download satellite provenance',json.dumps(fresh['metadata'],indent=2),'satellite-provenance.json','application/json')
