"""Validation-only robustness, spectral error, paired scene uncertainty and CPU cost."""
import json
import platform
import statistics
import threading
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from rssr.core import degrade,metrics,resize,digest
from rssr.experiment import load_model,load_tiles

root=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
records=json.loads((root/'reports/submission-study.json').read_text())
summaries=[];rng=np.random.default_rng(2026)
for bands in [4,3]:
    for variant in sorted({r['variant'] for r in records if r['bands']==bands}):
        group=[r for r in records if r['bands']==bands and r['variant']==variant]
        assert len(group)==3
        scene_ids=sorted(s['scene_id'] for s in group[0]['validation_scenes'])
        differences=np.array([statistics.mean(next(s['delta_psnr'] for s in r['validation_scenes'] if s['scene_id']==sid) for r in group) for sid in scene_ids])
        bootstrap=differences[rng.integers(0,len(differences),(4000,len(differences)))].mean(1)
        summaries.append({'bands':bands,'variant':variant,'mean_delta_db':float(differences.mean()),
                          'seed_sd_db':statistics.stdev(r['scene_mean_delta_db'] for r in group),
                          'paired_scene_bootstrap_95':np.quantile(bootstrap,[.025,.975]).tolist(),
                          'scenes':4,'seeds':3,'bootstrap_replicates':4000})
selected=[max((r for r in records if r['bands']==b and r['variant']==v),key=lambda r:r['scene_mean_delta_db']) for b,v in [(4,'cnn'),(4,'attention'),(4,'hybrid'),(3,'cnn'),(3,'hybrid')]]
robustness=[];efficiency=[]
for record in selected:
    checkpoint=root/'data/runs'/record['run_id']/'best.pt'
    assert digest(checkpoint)==record['checkpoint_sha256']
    model,state=load_model(checkpoint)
    manifest=root/record['manifest_path'].replace('\\','/');m=json.loads(manifest.read_text())
    tiles=load_tiles(m,'validation',manifest.parent)
    with torch.inference_mode():
        for sigma,noise in [(.4,0),(.8,0),(1.2,0),(1.6,0),(.8,.005),(.8,.01)]:
            scene_values={}
            for i,(item,_,hr) in enumerate(tiles):
                x=degrade(hr,2,sigma,noise,seed=2026+i)
                for method,pred in [('model',model(x[None])[0]),('bicubic',resize(x,2))]:
                    value=metrics(pred,hr,border=2)
                    scene_values.setdefault((item['scene_id'],method),[]).append(value)
            scenes=[]
            for sid in sorted({k[0] for k in scene_values}):
                scores={method:{metric:statistics.mean(r[metric] for r in scene_values[(sid,method)]) for metric in ['psnr','ssim','sam_rad',*[f'rmse_band_{i}' for i in range(record['bands'])]]} for method in ['model','bicubic']}
                scenes.append({'scene_id':sid,'scores':scores})
            robustness.append({'bands':record['bands'],'variant':record['variant'],'seed':record['seed'],
                               'sigma_hr_pixels':sigma,'noise_std':noise,'scenes':scenes})
        x=torch.zeros((1,record['bands'],64,64));macs={'conv_linear_mha':0};handles=[]
        def count(module,inputs,output):
            if isinstance(module,torch.nn.Conv2d):
                macs['conv_linear_mha']+=output.numel()*(module.in_channels//module.groups)*np.prod(module.kernel_size)
            elif isinstance(module,torch.nn.MultiheadAttention):
                q=inputs[0];batch,tokens,dim=q.shape
                macs['conv_linear_mha']+=batch*(4*tokens*dim*dim+2*tokens*tokens*dim)
            elif isinstance(module,torch.nn.Linear):
                macs['conv_linear_mha']+=output.numel()*module.in_features
        for module in model.modules():
            if isinstance(module,(torch.nn.Conv2d,torch.nn.Linear,torch.nn.MultiheadAttention)):
                handles.append(module.register_forward_hook(count))
        model(x)
        for h in handles:h.remove()
        for _ in range(5):model(x)
        memory=[];stop=threading.Event();process=psutil.Process()
        def sample():
            while not stop.wait(.001):memory.append(process.memory_info().rss)
        monitor=threading.Thread(target=sample,daemon=True);monitor.start();times=[]
        try:
            for _ in range(20):
                started=time.perf_counter();model(x);times.append(time.perf_counter()-started)
        finally:stop.set();monitor.join()
        efficiency.append({'bands':record['bands'],'variant':record['variant'],
                           'parameters':sum(p.numel() for p in model.parameters()),
                           'input_shape':list(x.shape),'warmup':5,'repetitions':20,
                           'median_seconds':statistics.median(times),'p95_seconds':float(np.quantile(times,.95)),
                           'sampled_process_peak_rss_bytes':max(memory,default=process.memory_info().rss),
                           'conv_linear_attention_macs':int(macs['conv_linear_mha']),
                           'counted_flops_at_2_per_mac':int(2*macs['conv_linear_mha']),
                           'excluded_ops':'normalization, activations, pooling, softmax, interpolation and elementwise operations; not total FLOPs'})
    print('Benchmarked',record['bands'],record['variant'],flush=True)
report={'environment':{'platform':platform.platform(),'torch':torch.__version__,'device':'cpu','threads':2,'dtype':'float32','gpu_memory':'not applicable; no GPU used'},
        'selection':'best validation scene-mean PSNR seed within each specified model; no test access',
        'selected':selected,'summary':summaries,'robustness':robustness,'efficiency':efficiency,
        'limitations':'Four development scenes; bootstrap intervals are descriptive and unstable at n=4. Seed selection is optimistic. SAM/per-band RMSE use normalized DN, not reflectance. Synthetic blur/noise is not real cross-sensor validation. RSS includes interpreter and cached allocations; sampled peak is not allocator peak.'}
(root/'reports/development-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Development benchmark','',report['limitations'],'','| Bands | Variant | Mean PSNR delta dB | Seed SD | Scene bootstrap 95% |','|---|---|---:|---:|---|']
for r in summaries:lines.append(f"| {r['bands']} | {r['variant']} | {r['mean_delta_db']:.4f} | {r['seed_sd_db']:.4f} | {r['paired_scene_bootstrap_95']} |")
(root/'reports/development-benchmark.md').write_text('\n'.join(lines)+'\n')
