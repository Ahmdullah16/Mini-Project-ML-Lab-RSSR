"""Explicit, sequential comparisons using the existing published RGB checkpoints."""
import json
import time
from datetime import datetime,timezone
import numpy as np
import streamlit as st
from rssr.comparison import SETTINGS,input_digest,result_key,validate_result,comparison_bundle,png_bytes

def render_compare():
    from inference_ui import CATALOG,BASE_CONFIG,CACHE,resources,fetch_asset,decode_image,build_identity,ROOT
    st.subheader('Compare models on one input')
    st.caption('Scientific reconstruction workflow: native ×2 RGB outputs, no hidden sharpening. PNGs are clipped/quantized display exports, not calibrated multispectral products. Perceptual enhancement is not yet available.')
    selected=st.multiselect('Models to compute',[s['label'] for s in CATALOG],default=[s['label'] for s in CATALOG],key='compare_models')
    st.info('Available: published CNN pilot, CNN v2 and hybrid v2. Standalone attention/Transformer research checkpoints are not currently portable compatible RGB options; no output is substituted for them. Genuine HR is unavailable for this input workflow.')
    choice=st.radio('Comparison input',['Licensed cached training example','Upload'],key='compare_input')
    upload=st.file_uploader('Comparison RGB PNG/JPEG, ≤512×512 and 8 MB',type=['png','jpg','jpeg'],key='compare_upload') if choice=='Upload' else None
    if st.button('Run selected models',type='primary'):
        try:
            if not selected:raise ValueError('Select at least one model')
            if choice=='Upload' and upload is None:raise ValueError('Choose an image to upload')
            data=upload.getvalue() if upload else fetch_asset(BASE_CONFIG['sample'],CACHE).read_bytes()
            image=decode_image(data);digest=input_digest(image)
            old=st.session_state.get('comparison',{})
            cached={r['key']:r for r in old.get('results',[])} if old.get('input_sha256')==digest else {}
            source={'kind':'user upload; provenance unverified'} if upload else {'kind':'cached training example',**BASE_CONFIG['sample']}
            batch={'input':image,'input_sha256':digest,'source':source,'results':[],'errors':[]}
            progress=st.progress(0,text='Preparing selected models')
            for i,label in enumerate(selected):
                spec=next(s for s in CATALOG if s['label']==label);key=result_key(image,spec)
                progress.progress(i/len(selected),text=f'{i+1}/{len(selected)}: {label}')
                try:
                    if key in cached:result=cached[key]
                    else:
                        model,state,path,lock=resources(spec)
                        if not lock.acquire(blocking=False):raise ValueError('Model CPU lock busy; retry this comparison')
                        try:
                            from rssr.display import predict_rgb
                            started=time.perf_counter()
                            baseline,sr,meta=predict_rgb(image,model,state,spec['checkpoint']['sha256'],max_side=512)
                            elapsed=time.perf_counter()-started
                        finally:lock.release()
                        validate_result(image,baseline,sr,meta,spec)
                        meta.update(model_name=label,model_version=spec['version'],checkpoint_url=spec['checkpoint']['url'],
                                    input_sha256=digest,input_shape=list(image.shape),output_shape=list(sr.shape),
                                    runtime_seconds=elapsed,runtime_scope='CPU inference plus display conversion; excludes download/model load',
                                    computed_at=datetime.now(timezone.utc).isoformat(),settings=SETTINGS,application=build_identity(ROOT))
                        result={'key':key,'baseline':baseline,'sr':sr,'metadata':meta}
                    batch['results'].append(result)
                except Exception as error:batch['errors'].append({'model':label,'error':str(error)})
            st.session_state['comparison']=batch
            progress.progress(1.0,text='Comparison finished; see successful outputs and any failures below')
        except Exception as error:st.error(f'Comparison not started: {error}')
    if 'comparison' not in st.session_state:return
    batch=st.session_state['comparison']
    st.caption('Latest completed batch · input SHA-256 '+batch['input_sha256']+'. Controls above apply only on Run; the labels below belong to these stored outputs.')
    for failure in batch['errors']:st.warning(f"{failure['model']}: {failure['error']}. Retry with that model selected.")
    if not batch['results']:return
    zoom=st.slider('Comparison zoom',1,4,1,key='compare_zoom')
    cx=st.slider('Comparison horizontal center',0.0,1.0,.5,key='compare_x')
    cy=st.slider('Comparison vertical center',0.0,1.0,.5,key='compare_y')
    enhance=st.checkbox('Shared preview contrast only',value=False,key='compare_contrast')
    image=batch['input'];h,w=image.shape[:2];ch,cw=max(1,h//zoom),max(1,w//zoom);y=int((h-ch)*cy);x=int((w-cw)*cx)
    low,high=np.percentile(image,[2,98]) if enhance else (0,255)
    panels=[('Original LR',image,1),('Bicubic',batch['results'][0]['baseline'],2)]+[(v['metadata']['model_name'],v['sr'],2) for v in batch['results']]
    for start in range(0,len(panels),2):
        for col,(label,pixels,scale) in zip(st.columns(2),panels[start:start+2]):
            crop=pixels[y*scale:(y+ch)*scale,x*scale:(x+cw)*scale]
            if enhance and high>low:crop=np.rint(np.clip((crop.astype(float)-low)/(high-low),0,1)*255).astype(np.uint8)
            col.image(crop,caption=f'{label} · full dimensions {pixels.shape[1]}×{pixels.shape[0]} RGB',width='stretch')
    st.caption('Same crop and one LR-derived contrast range for every panel; full outputs retain fixed scaling. ×2 gives four times the pixels, not twice the physical resolving power. No aligned HR: PSNR/SSIM/MAE/RMSE against HR are unavailable. Visual sharpness alone does not establish fidelity.')
    for result in batch['results']:
        meta=result['metadata'];disagreement=float(np.abs(result['sr'].astype(float)-result['baseline'].astype(float)).mean())
        st.write(f"{meta['model_name']}: {meta['runtime_seconds']:.3f} s CPU. Mean absolute difference from bicubic: {disagreement:.3f} uint8 levels; this is disagreement, not reference error or accuracy.")
        st.download_button('Download '+meta['model_name'],png_bytes(result['sr']),result['key'][:12]+'.png','image/png',key='png-'+result['key'])
        with st.expander('Provenance · '+meta['model_name']):st.json(meta)
    st.download_button('Download comparison bundle',comparison_bundle(batch),'rssr-comparison.zip','application/zip')
    if st.button('Clear comparison results'):
        del st.session_state['comparison'];st.rerun()
