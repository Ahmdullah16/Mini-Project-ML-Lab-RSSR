"""Bounded real-model scale/export checks; no training or frozen test evaluation."""
import json,sys,tempfile,time,threading,platform,hashlib
from pathlib import Path
import numpy as np
import torch,rasterio,psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rssr.scaling import cascade,plan,SCALES
from rssr.hosted_assets import fetch_asset
from rssr.experiment import load_model
from rssr.models import tiled
catalog=[json.loads((ROOT/'configs/hosted-model.json').read_text(encoding='utf-8'))]+json.loads((ROOT/'configs/hosted-models.json').read_text(encoding='utf-8'))['models']
records=[];process=psutil.Process();peak=[process.memory_info().rss];done=threading.Event()
def sample():
    while not done.wait(.05):peak[0]=max(peak[0],process.memory_info().rss)
thread=threading.Thread(target=sample);thread.start()
try:
    with tempfile.TemporaryDirectory(prefix='rssr-scale-validation-') as temp:
        for mi,spec in enumerate(catalog):
            path=fetch_asset(spec['checkpoint'],Path(tempfile.gettempdir())/'rssr-public-assets-v1')
            model,state=load_model(path)
            for scale in SCALES:
                a=np.random.default_rng(7).integers(0,256,(3,5,3),dtype=np.uint8)
                folder=Path(temp)/f'm{mi}-s{scale}'
                result=cascade(a,model,folder,scale,{'spec':spec})
                with rasterio.open(folder/result['output']) as src:assert (src.width,src.height)==(5*scale,3*scale)
                records.append({'model':spec['label'],'input_shape':list(a.shape),'scale':scale,'output':[5*scale,3*scale],'runtime_seconds':result['runtime_seconds']})
            a=np.random.default_rng(11).integers(0,256,(129,131,3),dtype=np.uint8)
            folder=Path(temp)/f'm{mi}-seam';result=cascade(a,model,folder,2,{'spec':spec})
            with rasterio.open(folder/result['output']) as src:actual=src.read()
            x=torch.from_numpy(a.transpose(2,0,1).astype(np.float32)/255)[None]
            with torch.inference_mode():
                full=np.rint(model(x)[0].numpy().clip(0,1)*255).astype(np.uint8)
                reference=np.rint(tiled(model,x)[0].numpy().clip(0,1)*255).astype(np.uint8)
            assert np.array_equal(actual,reference)
            delta=np.abs(actual.astype(float)-full.astype(float));mask=np.zeros(delta.shape[1:],bool);mask[254:258,:]=True;mask[:,254:258]=True
            records.append({'model':spec['label'],'input_shape':list(a.shape),'scale':2,'existing_tiled_equal':True,
                            'whole_frame_uint8_max_difference':float(delta.max()),'seam_strip_mae_uint8':float(delta[:,mask].mean()),'other_mae_uint8':float(delta[:,~mask].mean())})
            if mi==0:
                a=np.zeros((1024,1024,3),np.uint8);folder=Path(temp)/'maximum-input'
                result=cascade(a,model,folder,2,{'spec':spec})
                with rasterio.open(folder/result['output']) as src:assert (src.width,src.height)==(2048,2048)
                records.append({'model':spec['label'],'input_shape':list(a.shape),'scale':2,'output':[2048,2048],'runtime_seconds':result['runtime_seconds'],'full_export_bytes':(folder/result['output']).stat().st_size})
finally:done.set();thread.join()
report={'recorded_date':'2026-10-26','scope':'Synthetic inputs, actual pinned published checkpoints; no scientific quality claim or frozen evaluation.',
        'rows':records,'sampled_process_peak_rss_bytes':peak[0],'memory_sampling_seconds':.05,'memory_limitations':'Sampled process RSS includes imports and reference computations; not exact allocator peak or large-job guarantee.',
        'maximum_scale_plan_only':plan(1024,1024,64),'environment':{'python':platform.python_version(),'torch':torch.__version__,'rasterio':rasterio.__version__}}
(ROOT/'reports/scale-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(len(records),'runs/checks; sampled RSS',peak[0],flush=True)
