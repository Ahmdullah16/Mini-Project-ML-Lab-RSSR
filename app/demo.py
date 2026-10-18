"""Local pilot demo. Checkpoints are operator-controlled, never uploaded pickles."""
import os
from pathlib import Path
import numpy as np
import torch
from rssr.core import resize,digest
from rssr.experiment import load_model
from rssr.models import tiled


def predict(image,checkpoint):
    if image is None:
        raise ValueError("Choose an opaque RGB image")
    a=np.asarray(image)
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<1 or max(a.shape[:2])>1024:
        raise ValueError("Expected uint8 RGB, at most 1024 pixels per side")
    x=torch.from_numpy(a.astype(np.float32)/255).permute(2,0,1)
    model,state=load_model(checkpoint)
    if model.bands!=3:
        raise ValueError("RGB upload requires a three-band checkpoint; use CLI for multispectral GeoTIFF")
    torch.set_num_threads(2)
    y=tiled(model,x[None])[0]
    baseline=resize(x,model.scale)
    display=lambda v:np.rint(v.permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    return display(baseline),display(y),{'run_id':state['run_id'],'checkpoint_sha256':digest(checkpoint),'status':state['provenance'],'scale':model.scale,'sensor':'user supplied; unverified','acquisition_time':'unknown','reference_metrics':'unavailable: no HR reference','display_scaling':'fixed [0,1], clipped for RGB display only'}


def build():
    import gradio as gr
    checkpoint=os.environ.get('RSSR_CHECKPOINT')
    if not checkpoint or not Path(checkpoint).is_file():
        raise ValueError('Set RSSR_CHECKPOINT to an existing three-band pilot checkpoint')
    with gr.Blocks(delete_cache=(60,60)) as demo:
        gr.Markdown('# Remote sensing SR laboratory\nPilot model estimates; generated detail is not verified ground truth. RGB uploads are display images, not calibrated reflectance. No HR reference metrics are fabricated. Temporary upload/output cache expires after 60 seconds; no research retention.')
        image=gr.Image(type='numpy',image_mode='RGB',label='LR RGB upload')
        button=gr.Button('Estimate SR')
        with gr.Row():
            baseline=gr.Image(label='Bicubic baseline'); sr=gr.Image(label='Pilot model estimate')
        metadata=gr.JSON(label='Provenance')
        button.click(lambda a:predict(a,checkpoint),inputs=image,outputs=[baseline,sr,metadata],api_name='predict',concurrency_limit=1)
    return demo


if __name__=='__main__':
    build().queue(max_size=2).launch(server_name='127.0.0.1',max_file_size='8mb',share=False)
