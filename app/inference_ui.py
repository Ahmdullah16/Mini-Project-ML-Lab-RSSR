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


@st.cache_resource
def cpu_lock():
    return threading.Lock()


@st.cache_resource(show_spinner='Downloading and verifying the trained checkpoint…')
def resources(specification):
    from rssr.experiment import load_model
    path=fetch_asset(specification['checkpoint'],CACHE)
    model,state=load_model(path)
    if model.bands!=3 or model.scale!=2:
        raise ValueError('Hosted checkpoint must be compatible x2 RGB')
    return model,state,path,cpu_lock()


def png_bytes(image):
    output=io.BytesIO(); Image.fromarray(image).save(output,format='PNG'); return output.getvalue()


def model_choice():
    selected=st.selectbox('Model',[entry['label'] for entry in CATALOG],key='model_label')
    CONFIG=next(entry for entry in CATALOG if entry['label']==selected)
    st.info(CONFIG['limitations'])
    return CONFIG

def render_live():
    from scale_ui import render_scaled
    return render_scaled(comparison=False)

def render_satellite():
    CONFIG=model_choice()
    st.caption('Explicit provider request; up to 32 MB, one 256×256 RGB display crop. Item-level cloud filtering only. Cross-sensor quality is unvalidated. No HR reference metrics are available. Results remain temporary.')
    bbox=st.text_input('AOI west,south,east,north',value='-114,43,-113.9,43.1')
    with st.expander('How to interpret satellite inputs and outputs'):
        st.write('AOI is a longitude/latitude bounding box in degrees. The raster CRS defines map coordinates; pixel size is measured in its CRS units. The output GeoTIFF preserves georeferencing with a finer sampling grid, not proven finer physical resolution.')
        st.write('Acquisition is when the sensor observed the scene; retrieval/refresh is when this request obtained it. A current request can return an older acquisition. Cached examples are labeled separately. The cloud limit is item-level filtering, not a per-pixel cloud mask; nodata identifies invalid samples, not all clouds.')
        st.write('The RGB display uses three bands and a preview stretch. Downloads preserve output values and provenance. Models were developed on synthetic NAIP aerial data; real satellite cross-sensor accuracy remains unvalidated.')
    today=datetime.now(timezone.utc).date()
    start=st.date_input('Acquisition start',value=today-timedelta(days=30))
    end=st.date_input('Acquisition end',value=today)
    cloud=st.slider('Maximum item cloud cover (%)',0,100,20)
    if st.button('Acquire satellite crop for preflight'):
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
                    result=acquire_infer({'bbox':bounds,'datetime':f'{start}T00:00:00Z/{end}T23:59:59Z','query':{'eo:cloud_cover':{'lt':cloud}},'crop_size':256,'max_bytes':32_000_000,'acquire_only':True},path,folder)
                    result.update(model_name=CONFIG['label'],model_version=CONFIG['version'],application=build_identity(ROOT),runtime_seconds=time.perf_counter()-started,refresh_time=datetime.now(timezone.utc).isoformat(),storage='temporary; local database removed after request')
                    st.session_state['fresh_prepared']={'metadata':result,'geotiff':stored_path(folder,result['source_path']).read_bytes(),'model':CONFIG}
            finally: lock.release()
        except Exception as error:
            st.error(f'Satellite request did not complete: {error}')
    workflow=st.radio('Satellite inference workflow',['Single model','Compare models'],key='satellite_workflow')
    if workflow=='Single model':
        from satellite_scale_ui import render_satellite_scale
        render_satellite_scale()
    else:
        from satellite_compare_ui import render
        render()
