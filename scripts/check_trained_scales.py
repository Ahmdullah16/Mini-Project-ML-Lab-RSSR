"""Measured trained RGB cascade capability; incompatible bands stay unavailable."""
import json
from pathlib import Path
import tempfile
import threading
import time
import numpy as np
import psutil
import rasterio
import torch
from rssr.core import digest
from rssr.experiment import load_model
from rssr.models import tiled
from rssr.scaling import SCALES, cascade, plan
ROOT=Path(__file__).resolve().parents[1]

def main():
    historical=json.loads((ROOT/'reports/historical-architecture-bindings.json').read_text())['rows']
    pilots=json.loads((ROOT/'reports/missing-model-pilot.json').read_text())['runs']
    rows=[];seams=[];torch.set_num_threads(2)
    image=np.random.default_rng(17).integers(0,256,(3,5,3),dtype=np.uint8)
    for number in range(1,8):
        key=f'cat-v1/M{number}'
        candidates=[r for r in historical if r['status']=='verified historical binding' and r['registry_config']['architecture_id']==key and r['registry_config']['bands']==3 and r['registry_config']['scale']==2]
        recent=[r for r in pilots if r['architecture_id']==key]
        if not candidates and not recent:
            rows.extend({'architecture_id':key,'scale':s,'status':'unavailable: verified historical weights are four-band; RGB cascade requires three bands','preflight':plan(5,3,s)} for s in SCALES)
            continue
        record=candidates[0] if candidates else recent[0]
        checkpoint=ROOT/'data/runs'/record['run_id']/'best.pt' if candidates else ROOT/record['checkpoint']
        assert digest(checkpoint)==record['checkpoint_sha256']
        model,_=load_model(checkpoint)
        tensor=torch.from_numpy(np.random.default_rng(19).random((1,3,65,133),dtype=np.float32))
        with torch.inference_mode():
            difference=(model(tensor)-tiled(model,tensor)).abs()
        seams.append({'architecture_id':key,'input_hw':[65,133],'tile':128,'halo':32,'max_abs_normalized':float(difference.max()),'seam_band_max_abs_normalized':float(difference[:,:,:,254:258].max()),'interpretation':'Measured deviation only; global attention is context dependent, no universal seam-free claim.'})
        for scale in SCALES:
            request=plan(5,3,scale)
            print(key,scale,request['output_width'],request['output_height'],flush=True)
            samples=[];stop=threading.Event();process=psutil.Process()
            def sample():
                while not stop.is_set():
                    samples.append(process.memory_info().rss);stop.wait(.01)
            sampler=threading.Thread(target=sample);sampler.start()
            try:
                with tempfile.TemporaryDirectory(prefix='trained-scale-',dir=ROOT.parent.parent/'work') as temp:
                    folder=Path(temp)/'output'
                    metadata=cascade(image,model,folder,scale,{'architecture_id':key,'checkpoint_sha256':digest(checkpoint)})
                    with rasterio.open(folder/metadata['output']) as ds:
                        assert (ds.width,ds.height,ds.count)==(5*scale,3*scale,3)
                        pixels=ds.read()
                        assert pixels.dtype==np.uint8
                    rows.append({'architecture_id':key,'scale':scale,'status':'executed synthetic geometry/export; not quality evidence','checkpoint_sha256':digest(checkpoint),'run_id':record['run_id'],'preflight':request,'provenance':metadata,'export_sha256':digest(folder/metadata['output']),'sampled_process_rss_max_bytes':max(samples) if samples else process.memory_info().rss})
            finally:
                stop.set();sampler.join()
    report={'schema':'trained-rgb-capability-v1','source_hashes':{p:digest(ROOT/p) for p in ['scripts/check_trained_scales.py','src/rssr/scaling.py','reports/missing-model-pilot.json','reports/historical-architecture-bindings.json']},'scope':'First listed compatible checkpoint per model, fixed synthetic odd rectangle 5x3; no test or satellite pixels. Higher scales are repeated learned native2 with bicubic residual and per-stage uint8 quantization. Temporary exports removed after verification.','memory_method':'Process RSS sampled every10ms, includes loaded libraries/prior allocator state; sampled maximum is not guaranteed peak or per-job allocation. No maximum-size execution claim.','rows':rows,'native2_seams':seams,'maximum_request_preflight_only':plan(1024,1024,64)}
    (ROOT/'reports/trained-rgb-capability.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Executed',sum(r['status'].startswith('executed') for r in rows),'of',len(rows),'entries')
if __name__=='__main__':main()
