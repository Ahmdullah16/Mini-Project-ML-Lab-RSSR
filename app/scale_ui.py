"""Preflight-first exact-scale RGB interface; bounded host, portable large jobs."""
import io,json,tempfile,time,hashlib,zipfile
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from rssr.scaling import SCALES,plan,cascade,job_bundle,cache_key


def render_scaled(comparison=False):
    from inference_ui import CATALOG,CACHE,BASE_CONFIG,fetch_asset,resources,decode_image
    prefix='scaled_compare' if comparison else 'scaled_live'
    st.subheader('Compare models' if comparison else 'Exact-scale super-resolution')
    st.caption('Scientific reconstruction workflow. Native ×2 or explicitly labeled repeated ×2 cascades; no generative enhancement. Cascades are unvalidated for true high-scale detail recovery.')
    labels=[s['label'] for s in CATALOG]
    diagnostic=Path(__file__).resolve().parents[1]/'reports/inference-diagnostic-v1.json'
    if diagnostic.exists():
        with st.expander('Previous inference diagnostic evidence'):
            st.caption('Recorded synthetic/training full-frame versus tiled checks, not frozen-test quality. Global pooling can vary with tile context.')
            st.json(json.loads(diagnostic.read_text(encoding='utf-8')))
    selected=st.multiselect('Models to compute',labels,default=labels if comparison else labels[:1],key=prefix+'models')
    scale=st.selectbox('Output scale',SCALES,format_func=lambda s:f'×{s}',key=prefix+'scale')
    st.write('Wout = s × Win; Hout = s × Hin. Pixel count increases by s², without guaranteeing recovery of missing detail. Repeated stages clip and quantize RGB; the model includes bicubic interpolation in its residual baseline.')
    choice=st.radio('Input',['Licensed cached training example','Upload'],key=prefix+'choice')
    upload=st.file_uploader('Opaque RGB/grayscale PNG/JPEG; width and height each ≤1024; 8 MB',type=['png','jpg','jpeg'],key=prefix+'upload') if choice=='Upload' else None
    if st.button('Prepare input and show execution plan',key=prefix+'prepare'):
        try:
            if choice=='Upload' and upload is None:raise ValueError('Choose an image')
            data=upload.getvalue() if upload else fetch_asset(BASE_CONFIG['sample'],CACHE).read_bytes()
            st.session_state[prefix+'prepared']={'image':decode_image(data),'source':{'kind':'user upload; unverified'} if upload else {'kind':'cached training example',**BASE_CONFIG['sample']}}
        except Exception as error:st.error(str(error))
    prepared=st.session_state.get(prefix+'prepared')
    if prepared is None:return
    image=prepared['image'];request=plan(image.shape[1],image.shape[0],scale)
    st.caption('Plan applies to this prepared input. After changing the upload/source, press Prepare again. Execution uses these exact prepared pixels.')
    st.image(Image.fromarray(image).resize((max(1,round(image.shape[1]*min(1,256/max(image.shape[:2])))),max(1,round(image.shape[0]*min(1,256/max(image.shape[:2])))))),caption=f"Prepared {image.shape[1]}×{image.shape[0]} input preview")
    st.write(f"Output: {request['output_width']}×{request['output_height']} · {request['output_pixels']:,} pixels · {request['stages']} native ×2 stage(s)")
    st.write(f"Per model: raw output {request['output_rgb_bytes']:,} bytes; estimated disk for retained stages {request['estimated_disk_bytes']:,} bytes; configured working-memory allowance {request['estimated_working_memory_bytes']:,} bytes. Actual model/runtime overhead varies.")
    st.caption(f"Free-host configured execution limit: {request['host_output_pixel_limit']:,} output pixels per model, sequential CPU execution. This is a policy limit, not a measured peak-memory guarantee. Files and caches are temporary.")
    if not request['host_available']:st.warning('This exact request is unavailable on the free host. No input resizing or scale reduction will be performed. Download the matching local/Colab job below.')
    for label in selected:
        spec=next(s for s in CATALOG if s['label']==label)
        st.download_button('Download local/Colab job · '+label,job_bundle(image,spec,scale),f'rssr-x{scale}-job.zip','application/zip',key=prefix+'job'+label)
    if st.button('Run prepared request',disabled=not request['host_available'] or not selected,key=prefix+'run'):
        outputs=[];errors=[]
        old=st.session_state.get(prefix+'results',{})
        cached={r['key']:r for r in old.get('outputs',[])}
        for label in selected:
            spec=next(s for s in CATALOG if s['label']==label)
            try:
                key=cache_key(image,spec,scale)+hashlib.sha256(json.dumps(prepared["source"],sort_keys=True).encode()).hexdigest()
                if key in cached:
                    outputs.append(cached[key]);continue
                model,state,path,lock=resources(spec)
                if not lock.acquire(blocking=False):raise ValueError('CPU busy; retry shortly')
                try:
                    with tempfile.TemporaryDirectory(prefix='rssr-scale-') as temp:
                        folder=Path(temp)/'job'
                        meta=cascade(image,model,folder,scale,{'model':spec,'run_id':state['run_id'],'source':prepared['source']})
                        import rasterio
                        with rasterio.open(folder/meta['output']) as src:pixels=src.read().transpose(1,2,0)
                        outputs.append({'key':key,'label':label,'pixels':pixels,'metadata':meta,'tiff':(folder/meta['output']).read_bytes(),'preview':(folder/'preview.png').read_bytes()})
                finally:lock.release()
            except Exception as error:errors.append(label+': '+str(error))
        st.session_state[prefix+'results']={'input':image.copy(),'outputs':outputs,'errors':errors,'scale':scale}
    batch=st.session_state.get(prefix+'results')
    if not batch:return
    for error in batch['errors']:st.error(error)
    if not batch['outputs']:return
    st.caption('Completed results retain their original model and scale labels. Changing controls does not relabel these outputs.')
    zoom=st.slider('Synchronized zoom',1,4,1,key=prefix+'zoom')
    cx=st.slider('Horizontal center',0.0,1.0,.5,key=prefix+'x');cy=st.slider('Vertical center',0.0,1.0,.5,key=prefix+'y')
    original=batch['input'];s=batch['scale'];h,w=original.shape[:2];ch=max(1,h//zoom);cw=max(1,w//zoom);y=int((h-ch)*cy);x=int((w-cw)*cx)
    baseline=np.asarray(Image.fromarray(original).resize((w*s,h*s),Image.Resampling.BICUBIC))
    panels=[('LR',original,1),('Bicubic interpolation',baseline,s)]+[(r['label'],r['pixels'],s) for r in batch['outputs']]
    for start in range(0,len(panels),2):
        for col,(label,a,factor) in zip(st.columns(2),panels[start:start+2]):
            crop=Image.fromarray(a[y*factor:(y+ch)*factor,x*factor:(x+cw)*factor]);crop.thumbnail((512,512))
            col.image(crop,caption=f'{label}: full {a.shape[1]}×{a.shape[0]}, bounded crop preview')
    st.caption('Bicubic here uses Pillow RGB bicubic interpolation; it is not a learned model. HR-reference metrics are unavailable. Visual differences and extra pixels do not demonstrate fidelity; cascade artifacts can accumulate.')
    for r in batch['outputs']:
        meta=r['metadata'];disagreement=float(np.abs(r['pixels'].astype(np.float32)-baseline.astype(np.float32)).mean())
        st.caption(f'Mean absolute disagreement from Pillow bicubic = sum(|SR - bicubic|)/N = {disagreement:.4f} uint8 levels, where N is the number of RGB channel samples. This is not HR error; smaller does not establish better reconstruction.')
        st.write(f"{r['label']} · ×{meta['scale']} · {meta['runtime_seconds']:.3f} s pipeline runtime")
        st.download_button('Full-resolution BigTIFF · '+r['label'],r['tiff'],f'sr-x{meta["scale"]}.tif','image/tiff',key=prefix+'tiff'+r['label'])
        st.download_button('Provenance · '+r['label'],json.dumps(meta,indent=2),'provenance.json','application/json',key=prefix+'meta'+r['label'])
        with st.expander('Stages and provenance · '+r['label']):st.json(meta)

    bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        input_png=io.BytesIO();Image.fromarray(batch['input']).save(input_png,format='PNG')
        archive.writestr('input.png',input_png.getvalue())
        for i,r in enumerate(batch['outputs']):
            archive.writestr(f'model-{i+1}.tif',r['tiff'])
            archive.writestr(f'model-{i+1}-provenance.json',json.dumps(r['metadata'],indent=2))
    st.download_button('Download completed comparison bundle',bundle.getvalue(),'comparison.zip','application/zip',key=prefix+'bundle')
