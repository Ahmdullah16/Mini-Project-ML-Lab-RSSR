"""Sequential comparison of pinned RGB models on one prepared GeoTIFF."""
import json,tempfile
from pathlib import Path
import numpy as np
import rasterio
import streamlit as st
from rasterio.enums import Resampling
from rssr.scaling import SCALES
from rssr.inference import geotiff_plan
from rssr.satellite_comparison import identity,execute

def preview(data,divisor=1.):
    with rasterio.io.MemoryFile(data) as mem:
        with mem.open() as ds:
            ratio=min(1,256/max(ds.width,ds.height));h=max(1,round(ds.height*ratio));w=max(1,round(ds.width*ratio))
            a=ds.read(out_shape=(3,h,w),resampling=Resampling.average,masked=True).filled(0).transpose(1,2,0)/divisor
            return np.clip(a,0,1)

def render():
    from inference_ui import CATALOG,resources
    from satellite_scale_ui import scientific_job
    prepared=st.session_state.get('fresh_prepared')
    st.caption('Acquire once above. Every selected model uses exactly those source bytes; changing AOI/date fields does not replace the prepared acquisition until Acquire is submitted.')
    selected=st.multiselect('Satellite models to compare',[s['label'] for s in CATALOG],default=[s['label'] for s in CATALOG[:2]],key='sat_compare_models')
    scale=st.selectbox('Satellite comparison scale',SCALES,key='sat_compare_scale')
    if prepared is None:return
    source=prepared['geotiff']
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp)/'source.tif';p.write_bytes(source);request=geotiff_plan(p,scale)
    st.write(f"Exact output {request['output_width']}×{request['output_height']}; {request['output_pixels']:,} pixels per model; estimated disk {request['estimated_disk_bytes']:,} bytes per model. Sequential execution; host cap1,048,576pixels. All stages use native×2 float32 cascades. Source is display RGB, not calibrated multispectral reflectance.")
    st.image(preview(source,255.),caption='Common acquired input · fixed intensity range0–255; masked pixels shown black')
    specs=[s for s in CATALOG if s['label'] in selected]
    keys={identity(source,s,scale) for s in specs}
    # Drop stale results when source, selection or scale changes, bounding session memory.
    cache={k:v for k,v in st.session_state.get('sat_compare_results',{}).items() if k in keys}
    failures={k:v for k,v in st.session_state.get('sat_compare_failures',{}).items() if k in keys}
    st.session_state['sat_compare_results']=cache;st.session_state['sat_compare_failures']=failures
    def run_one(spec):
        key=identity(source,spec,scale)
        try:
            _,_,checkpoint,lock=resources(spec)
            if not lock.acquire(blocking=False):raise ValueError('CPU busy; retry this model')
            try:cache[key]=execute(source,checkpoint,spec,scale,prepared['metadata'])
            finally:lock.release()
            failures.pop(key,None)
        except Exception as error:failures[key]=str(error)
    if st.button('Run selected satellite comparisons',key='sat_compare_run',disabled=not specs or not request['host_available']):
        for spec in specs:
            if identity(source,spec,scale) not in cache:run_one(spec)
    if cache or failures:
        st.subheader('Shared detail view · same crop and fixed display contrast')
        zoom=st.selectbox('Detail zoom',[1,2,4],key='sat_detail_zoom')
        cx=st.slider('Detail horizontal center',0.,1.,.5,key='sat_detail_x')
        cy=st.slider('Detail vertical center',0.,1.,.5,key='sat_detail_y')
        split=st.slider('Before/after divider · bicubic left, model right',0.,1.,.5,key='sat_detail_split')
        st.caption('Common controls apply to every model below. Previews are bounded to256pixels per side; zoom enlarges a shared preview crop, not a new inference. Bicubic is calculated on the exact output grid before area preview reduction; it is an interpolation baseline, not HR ground truth. Fixed0–1 contrast; invalid preview pixels shown black. Disagreement is measured before display clipping, then bounded for visualization.')
    rows=[]
    for spec in specs:
        key=identity(source,spec,scale)
        if not request['host_available']:
            st.download_button('Local job · '+spec['label'],scientific_job(source,spec,scale,prepared['metadata']),key+'.zip','application/zip',key=key+'_job')
        if key in failures:
            st.error(spec['label']+': '+failures[key])
            if st.button('Retry only '+spec['label'],key=key+'_retry',disabled=not request['host_available']):run_one(spec)
        if key in cache:
            result=cache[key];meta=result['metadata']
            st.image(preview(result['geotiff']),caption=spec['label']+' · identical fixed0–1 display, no per-model stretch')
            from rssr.comparison_views import comparison_views,crop_and_wipe
            views=comparison_views(source,result['geotiff'],scale)
            detail=crop_and_wipe(views,cx,cy,zoom,split)
            cols=st.columns(3)
            cols[0].image(detail['baseline'],caption='Bicubic detail · interpolation only',width='stretch')
            cols[1].image(detail['wipe'],caption='Bicubic/model divider · '+spec['label'],width='stretch')
            cols[2].image(detail['disagreement'],caption='Absolute disagreement from bicubic, not error to HR',clamp=True,width='stretch')
            st.caption('Shared preview crop x,y,width,height: '+str(detail['preview_crop_xywh']))
            st.download_button('GeoTIFF · '+spec['label'],result['geotiff'],key+'.tif','image/tiff',key=key+'_tiff')
            st.download_button('Provenance · '+spec['label'],json.dumps(meta,indent=2),key+'.json','application/json',key=key+'_meta')
            rows.append({'model':spec['label'],'scale':scale,'width':request['output_width'],'height':request['output_height'],'seconds':meta['runtime_seconds'],'checkpoint':spec['checkpoint']['sha256'],'reference_accuracy':'unavailable: no aligned HR'})
    if rows:
        st.dataframe(rows,hide_index=True)
        st.bar_chart({r['model']:r['seconds'] for r in rows},y_label='Measured runtime (seconds)')
        st.download_button('Comparison results JSON',json.dumps(rows,indent=2),'satellite-comparison.json','application/json')
    st.info('This is a shared-source execution comparison, not validated reconstruction accuracy. Source CRS/extent/masks remain in GeoTIFFs. Shared preview zoom/bicubic/disagreement are display aids; no genuine HR is available here and these are not reconstruction-error maps.')
