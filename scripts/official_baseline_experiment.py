"""Evaluate pretrained official RGB SwinIR on development scenes only."""
import json
import statistics
import time
from pathlib import Path
import torch
from rssr.core import metrics,resize
from rssr.experiment import load_tiles
from rssr.official_baseline import load_swinir

root=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
study=json.loads((root/'reports/submission-study.json').read_text())
entry=next(r for r in study if r['bands']==3)
manifest=root/entry['manifest_path'];data=json.loads(manifest.read_text())
model,provenance=load_swinir(root)
rows=[];started=time.perf_counter()
with torch.inference_mode():
    for item,x,hr in load_tiles(data,'validation',manifest.parent):
        y=model(x[None])[0];assert y.shape==hr.shape and torch.isfinite(y).all()
        for method,pred in [('official_swinir',y),('bicubic',resize(x,2))]:
            rows.append({'tile_id':item['id'],'scene_id':item['scene_id'],'method':method,**metrics(pred,hr,border=2)})
scene=[]
for sid in sorted({r['scene_id'] for r in rows}):
    means={method:statistics.mean(r['psnr'] for r in rows if r['scene_id']==sid and r['method']==method) for method in ['official_swinir','bicubic']}
    scene.append({'scene_id':sid,**means,'delta_db':means['official_swinir']-means['bicubic']})
report={'provenance':provenance,'parameters':sum(p.numel() for p in model.parameters()),
        'seconds':time.perf_counter()-started,'device':'CPU, 2 threads, float32','rows':rows,
        'scenes':scene,'mean_delta_db':statistics.mean(s['delta_db'] for s in scene),
        'protocol':'same RGB validation patches/range=1/border=2, no clipping; synthetic Gaussian+area x2',
        'limitations':'Official natural-image pretrained model; different data/training budget/degradation. Not an original benchmark reproduction. No NIR input or held-out scene evaluated.'}
(root/'reports/official-swinir-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('Official SwinIR mean validation delta',report['mean_delta_db'],flush=True)
