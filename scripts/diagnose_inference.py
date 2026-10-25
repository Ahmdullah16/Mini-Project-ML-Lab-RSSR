"""Run protocol inference-diagnostic-v1 using existing cached assets only."""
import json,sys,tempfile,platform,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rssr.core import degrade,digest,resize
from rssr.models import tiled
from rssr.experiment import load_model
from rssr.display import predict_rgb
from rssr.comparison import input_digest
from rssr.inference_diagnostics import difference_summary

def main():
    protocol=ROOT/'configs/inference-diagnostic-v1.json'
    catalog=[json.loads((ROOT/'configs/hosted-model.json').read_text(encoding='utf-8'))]+json.loads((ROOT/'configs/hosted-models.json').read_text(encoding='utf-8'))['models']
    cache=Path(tempfile.gettempdir())/'rssr-public-assets-v1'
    paths=[cache/s['checkpoint']['sha256'] for s in catalog]
    if not all(p.is_file() and digest(p)==s['checkpoint']['sha256'] for p,s in zip(paths,catalog)):
        raise FileNotFoundError('Verified catalog checkpoints must already be cached; no download occurs in this diagnostic')
    torch.set_num_threads(2)
    yy,xx=np.mgrid[:129,:131]
    fixture=np.stack([np.rint(xx/130*255),np.rint(yy/128*255),(xx>=65)*255],axis=-1).astype(np.uint8)
    assert fixture[0,0].tolist()==[0,0,0] and fixture[-1,-1].tolist()==[255,255,255]
    manifest=json.loads((ROOT/'configs/naip-pilot-manifest.json').read_text(encoding='utf-8'))
    scene=next(s for s in manifest['scenes'] if s['split']=='train');source=ROOT/'data'/scene['path']
    assert digest(source)==scene['sha256']
    with rasterio.open(source) as raster:
        hr=raster.read([1,2,3],window=Window(100,100,264,260),masked=True)
        assert not np.ma.getmaskarray(hr).any()
    lr=degrade(torch.tensor(hr.data.astype(np.float32)/manifest['divisor']),scale=2,sigma=.8)
    image=np.rint(lr.permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    quantization=np.abs(image.astype(float)/255-lr.permute(1,2,0).numpy())
    records=[]
    for name,input_image in [('synthetic_rgb_edges',fixture),('pinned_training_crop',image)]:
        x=torch.from_numpy(input_image.astype(np.float32)/255).permute(2,0,1)[None]
        for specification,path in zip(catalog,paths):
            model,state=load_model(path);assert model.bands==3 and model.scale==2
            with torch.inference_mode():
                started=time.perf_counter();full=model(x)[0];full_seconds=time.perf_counter()-started
                started=time.perf_counter();patches=tiled(model,x,tile=128,halo=32)[0];tile_seconds=time.perf_counter()-started
            base,display,metadata=predict_rgb(input_image,model,state,specification['checkpoint']['sha256'],max_side=512)
            quantize=lambda t:np.rint(t.permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
            assert np.array_equal(display,quantize(patches)) and np.array_equal(base,quantize(resize(x[0],2)))
            records.append({'input':name,'input_sha256':input_digest(input_image),'model':specification['label'],
                            'checkpoint_sha256':specification['checkpoint']['sha256'],'run_id':state['run_id'],
                            'model_config':state['config']['model'],'training_steps':state['config'].get('steps'),
                            'full_seconds':full_seconds,'tiled_seconds':tile_seconds,'timing_scope':'single diagnostic forwards, not warmed efficiency benchmark',
                            'shared_display_exact':True,**difference_summary(full.numpy(),patches.numpy())})
    result={'protocol':'inference-diagnostic-v1','protocol_sha256':digest(protocol),'recorded_at':datetime.now(timezone.utc).isoformat(),
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'platform':platform.platform(),'device':'cpu','threads':2},
            'source':scene,'source_window':[100,100,264,260],'license':manifest['license'],'normalization':manifest['units'],
            'input_quantization':{'mae_normalized':float(quantization.mean()),'max_abs_normalized':float(quantization.max())},
            'rows':records,'limits':'Implementation diagnostics only; no new reference-accuracy/quality gain, training, validation selection or test access. Synthetic/NAIP aerial domain cannot establish uploaded-satellite fidelity.'}
    (ROOT/'reports/inference-diagnostic-v1.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps([{'input':r['input'],'model':r['model'],'mae':r['mae_normalized'],'max':r['max_abs_normalized'],'clip':r['tiled_out_of_range_fraction'],'uint8_difference':r['uint8_disagreement_fraction']} for r in records],indent=2))

if __name__=='__main__':main()
