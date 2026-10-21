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
from PIL import Image, ImageOps
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rssr.display import predict_rgb
from rssr.experiment import load_model
from rssr.hosted_assets import fetch_asset
from rssr.build_info import build_identity

CONFIG=json.loads((ROOT/'configs/hosted-model.json').read_text(encoding='utf-8'))
CACHE=Path(os.environ.get('RSSR_ASSET_CACHE',str(Path(tempfile.gettempdir())/'rssr-public-assets-v1')))


@st.cache_resource(show_spinner='Downloading and verifying the trained pilot checkpoint…')
def resources():
    path=fetch_asset(CONFIG['checkpoint'],CACHE)
    model,state=load_model(path)
    if model.bands!=3 or model.scale!=2:
        raise ValueError('Hosted checkpoint must be compatible x2 RGB')
    return model,state,path,threading.Lock()


def decode_image(data):
    if len(data)>8*1024*1024: raise ValueError('Image file must be at most 8 MB')
    with Image.open(io.BytesIO(data)) as image:
        if max(image.size)>512 or min(image.size)<1: raise ValueError('Choose an image at most 512 pixels per side')
        if image.mode not in ('RGB','L'): raise ValueError('Use opaque RGB or grayscale PNG/JPEG; remove transparency first')
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()


def png_bytes(image):
    output=io.BytesIO(); Image.fromarray(image).save(output,format='PNG'); return output.getvalue()


st.set_page_config(page_title='Remote sensing SR pilot',layout='wide')
st.title('Remote sensing super-resolution pilot')
st.warning(CONFIG['limitations'])
st.caption('Stateless demo: uploads and results are temporary. Cached model/example files can be rebuilt after a restart. Local SQLite and cached satellite files are not durable storage.')
with st.expander('Application version'):
    st.json(build_identity(ROOT))
st.selectbox('Model',[CONFIG['label']],key='model_label')
choice=st.radio('Input',['Licensed example (cached acquisition)','Upload'],horizontal=True)
upload=st.file_uploader('PNG or JPEG, up to 512×512 pixels and 8 MB',type=['png','jpg','jpeg']) if choice=='Upload' else None
if choice.startswith('Licensed'):
    st.caption(f"NAIP aerial training example acquired {CONFIG['sample']['acquisition_time']}; synthetic LR, not a fresh satellite query. {CONFIG['sample']['license']}")
if st.button('Estimate super-resolution',type='primary'):
    try:
        if choice=='Upload' and upload is None: raise ValueError('Choose an image to upload')
        data=upload.getvalue() if choice=='Upload' else fetch_asset(CONFIG['sample'],CACHE).read_bytes()
        image=decode_image(data)
        model,state,path,lock=resources()
        if not lock.acquire(blocking=False): raise ValueError('The CPU is serving another request. Please retry shortly.')
        try:
            started=time.perf_counter()
            baseline,sr,meta=predict_rgb(image,model,state,CONFIG['checkpoint']['sha256'],max_side=512)
            runtime=time.perf_counter()-started
        finally: lock.release()
        meta.update(model_version=CONFIG['version'],runtime_seconds=runtime,inference_time=datetime.now(timezone.utc).isoformat(),storage='temporary session; not durable',application=build_identity(ROOT),input_shape=list(image.shape),output_shape=list(sr.shape))
        if choice.startswith('Licensed'):
            meta.update(sensor=CONFIG['sample']['sensor'],acquisition_time=CONFIG['sample']['acquisition_time'],source=CONFIG['sample']['source'],license=CONFIG['sample']['license'],input_kind='cached licensed training example')
        else: meta['input_kind']='user upload; provenance unverified'
        st.session_state['result']={'input':image,'baseline':baseline,'sr':sr,'metadata':meta}
    except Exception as error:
        st.error(f'Inference did not complete: {error}')
if 'result' in st.session_state:
    result=st.session_state['result']
    st.subheader('Latest completed inference')
    enhance=st.checkbox('Enhance preview contrast (does not change inference or downloads)',value=True)
    low,high=np.percentile(result['input'],[2,98])
    for column,key,label in zip(st.columns(3),['input','baseline','sr'],['Input LR','Bicubic baseline','Pilot SR estimate']):
        preview=result[key]
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

with st.expander('On-demand Sentinel acquisition and georeferenced output'):
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
            model,state,path,lock=resources()
            if not lock.acquire(blocking=False): raise ValueError('The CPU is busy. Please retry shortly.')
            try:
                started=time.perf_counter()
                with tempfile.TemporaryDirectory(prefix='rssr-fresh-') as folder:
                    result=acquire_infer({'bbox':bounds,'datetime':f'{start}T00:00:00Z/{end}T23:59:59Z','query':{'eo:cloud_cover':{'lt':cloud}},'crop_size':256,'max_bytes':32_000_000},path,folder)
                    result.update(runtime_seconds=time.perf_counter()-started,refresh_time=datetime.now(timezone.utc).isoformat(),storage='temporary; local database removed after request')
                    st.session_state['fresh_result']={'metadata':result,'geotiff':stored_path(folder,result['output_path']).read_bytes()}
            finally: lock.release()
        except Exception as error:
            st.error(f'Satellite request did not complete: {error}')
    if 'fresh_result' in st.session_state:
        fresh=st.session_state['fresh_result']
        with st.expander('Satellite timestamps and provenance'):
            st.json(fresh['metadata'])
        st.download_button('Download georeferenced SR',fresh['geotiff'],'satellite-sr.tif','image/tiff')
        st.download_button('Download satellite provenance',json.dumps(fresh['metadata'],indent=2),'satellite-provenance.json','application/json')
