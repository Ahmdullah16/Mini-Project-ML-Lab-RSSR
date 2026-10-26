"""Local pilot demo. Checkpoints are operator-controlled, never uploaded pickles."""
import os
from pathlib import Path
import numpy as np
import torch
from rssr.core import resize,digest
from rssr.experiment import load_model
from rssr.models import tiled


def predict(image,checkpoint):
    from rssr.display import predict_rgb,validate_rgb
    validate_rgb(image)
    model,state=load_model(checkpoint)
    return predict_rgb(image,model,state,digest(checkpoint))


def build():
    import gradio as gr
    from assets import models,export_result
    available=models(); checkpoint=next(iter(available.values()))
    with gr.Blocks(delete_cache=(60,60)) as demo:
        gr.Markdown('# Remote sensing SR laboratory\nPilot model estimates; generated detail is not verified ground truth. RGB uploads are display images, not calibrated reflectance. No HR reference metrics are fabricated. Temporary upload/output cache expires after 60 seconds; no research retention.')
        image=gr.Image(type='numpy',image_mode='RGB',label='LR RGB upload')
        selection=gr.Dropdown(choices=list(available),value=next(iter(available)),label='Operator-provided compatible model')
        sample_manifest=os.environ.get('RSSR_SAMPLES')
        if sample_manifest:
            import json
            samples=json.loads(Path(sample_manifest).read_text())
            gr.Examples(examples=[[str(Path(sample_manifest).resolve().parent/s['path'])] for s in samples],inputs=image,label='Licensed samples (metadata shown below)')
            gr.JSON(value=samples,label='Sample acquisition and license metadata')
        from rssr.scaling import SCALES,plan,cascade
        scale_choice=gr.Dropdown(choices=list(SCALES),value=2,label='Scale: native x2 or repeated x2 cascade')
        planning=gr.JSON(label='Exact dimensions and configured resource estimate')
        def preflight(a,scale):
            from rssr.display import validate_rgb
            a=validate_rgb(a)
            return plan(a.shape[1],a.shape[0],int(scale))
        image.change(preflight,inputs=[image,scale_choice],outputs=planning)
        scale_choice.change(preflight,inputs=[image,scale_choice],outputs=planning)
        gr.Markdown('Input width and height each <=1024. BigTIFF output is written incrementally; preview is bounded. Review resource estimates before running locally. Repeated x2 stages clip/quantize RGB and can accumulate artifacts. More pixels do not prove recovered detail.')
        button=gr.Button('Execute exact-scale local job')
        with gr.Row():
            baseline=gr.Image(label='Bicubic baseline'); sr=gr.Image(label='Pilot model estimate')
        metadata=gr.JSON(label='Provenance')
        image_export=gr.File(label='Download full-resolution RGB BigTIFF')
        metadata_export=gr.File(label='Download provenance')
        def selected(a,name,scale):
            import tempfile
            from PIL import Image
            if name not in available:raise ValueError('Unknown model selection')
            preflight(a,scale)
            model,state=load_model(available[name])
            folder=Path(tempfile.mkdtemp(prefix='rssr-gradio-scale-'))/'job'
            result=cascade(a,model,folder,int(scale),{'checkpoint_sha256':digest(available[name]),'run_id':state['run_id'],'model_label':name})
            result['retention']='Local job directory persists until operator removes it: '+str(folder)
            preview=np.asarray(Image.open(folder/'preview.png')).copy()
            baseline_preview=None
            if result['output_pixels']<=1024**2:
                baseline_preview=Image.fromarray(a).resize((result['output_width'],result['output_height']),Image.Resampling.BICUBIC)
                baseline_preview.thumbnail((512,512))
            return baseline_preview,preview,result,str(folder/result['output']),str(folder/'provenance.json')
        button.click(selected,inputs=[image,selection,scale_choice],outputs=[baseline,sr,metadata,image_export,metadata_export],api_name='predict_selected',concurrency_limit=1,concurrency_id='inference')
        legacy=gr.Button(visible=False)
        legacy.click(lambda a:predict(a,checkpoint),inputs=image,outputs=[baseline,sr,metadata],api_name='predict',concurrency_limit=1,concurrency_id='inference')
        with gr.Accordion('Acquire recent Sentinel RGB and export georeferenced estimates',open=False):
            gr.Markdown('Explicit bounded provider query and download, up to 32 MB. RGB display pilot only; cross-sensor quality is unvalidated. No HR metrics. Satellite outputs are saved to the operator data directory; uploaded user images are not added there.')
            bbox=gr.Textbox(value='-114,43,-113.9,43.1',label='AOI west,south,east,north (degrees)')
            dates=gr.Textbox(value='2026-10-01T00:00:00Z/2026-10-19T00:00:00Z',label='Acquisition interval (ISO 8601 start/end)')
            cloud=gr.Slider(0,100,value=20,step=1,label='Maximum item-level cloud cover (%)')
            acquire=gr.Button('Query, acquire crop and estimate SR')
            fresh_metadata=gr.JSON(label='Acquisition, inference and transfer provenance')
            fresh_files=gr.Files(label='Download georeferenced SR and provenance')
            def fresh(bounds,interval,cloud_limit,name,scale):
                from rssr.fresh import acquire_infer,stored_path
                values=[float(v.strip()) for v in bounds.split(',')]
                if len(values)!=4 or not np.isfinite(values).all() or not -180<=values[0]<values[2]<=180 or not -90<=values[1]<values[3]<=90:
                    raise ValueError('Invalid geographic bounding box')
                if values[2]-values[0]>1 or values[3]-values[1]>1: raise ValueError('AOI edges must be at most one degree')
                if name not in available: raise ValueError('Unknown model')
                result=acquire_infer({'bbox':values,'datetime':interval,'query':{'eo:cloud_cover':{'lt':cloud_limit}},'crop_size':256,'max_bytes':32_000_000,'scale':int(scale)},available[name],os.environ.get('RSSR_DATA_ROOT','data'))
                output=stored_path(os.environ.get('RSSR_DATA_ROOT','data'),result['output_path'])
                return result,[str(output),str(output.parent/'provenance.json')]
            acquire.click(fresh,inputs=[bbox,dates,cloud,selection,scale_choice],outputs=[fresh_metadata,fresh_files],api_name='fresh',concurrency_limit=1,concurrency_id='inference')
        with gr.Accordion('Cached satellite catalog and data age',open=False):
            gr.Markdown('Read-only metadata snapshot. Partial refreshes are shown separately from the last complete refresh. This does not download imagery or retrain the model.')
            catalog=gr.JSON(label='Catalog freshness and recent observations')
            read_catalog=gr.Button('Read local catalog')
            def catalog_status():
                from rssr.catalog import snapshot
                path=os.environ.get('RSSR_CATALOG')
                return snapshot(path) if path else {'status':'No catalog configured; set RSSR_CATALOG on the host'}
            read_catalog.click(catalog_status,outputs=catalog,api_name='catalog',concurrency_limit=1)
    return demo


if __name__=='__main__':
    build().queue(max_size=2).launch(server_name=os.environ.get('RSSR_HOST','127.0.0.1'),server_port=int(os.environ.get('RSSR_PORT','7860')),max_file_size='8mb',share=False)
