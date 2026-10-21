"""Bounded, resumable development study. Never evaluates the held-out split."""
import copy
import json
import os
import statistics
from pathlib import Path
from rssr.core import digest
from rssr.data import prepare
from rssr.experiment import train, evaluate

root=Path(__file__).resolve().parents[1]; os.chdir(root)
variants={
    'cnn':{'kind':'cnn'}, 'attention':{'kind':'attention'},
    'transformer':{'kind':'transformer'}, 'hybrid':{'kind':'hybrid'},
    'hybrid_no_attention':{'kind':'hybrid','use_attention':False},
    'hybrid_add':{'kind':'hybrid','fusion':'add'},
    'hybrid_half':{'kind':'hybrid','constant_gate':.5},
    'hybrid_gate':{'kind':'hybrid','gate':True},
    'hybrid_consistency':{'kind':'hybrid'},
    'srcnn_educational':{'kind':'srcnn'},
}
plan={'seeds':[17,23,41],'steps':1000,'validation_interval':100,
      'variants':variants,'test_access':False,'training':'Adam .001, batch 4, L1, zero residual where applicable',
      'selection':'checkpoint lowest validation L1; compare mean scene PSNR; never select by test',
      'limits':'CPU only, existing data, 1000 steps per model/seed, no new imagery',
      'rgb_variants':['cnn','hybrid'],'consistency_weight':.1}
plan_path=root/'configs/submission-study.json'
if plan_path.exists() and json.loads(plan_path.read_text())!=plan: raise ValueError('Study plan changed')
plan_path.write_text(json.dumps(plan,indent=2)+'\n')
ledger=root/'reports/submission-study.json'
records=json.loads(ledger.read_text()) if ledger.exists() else []
old=json.loads((root/'reports/geographic-study.json').read_text())
for bands in [4,3]:
    source=root/'configs/naip-development-manifest.json'
    if bands==3:
        m=json.loads(source.read_text());m['bands']=m['bands'][:3];m['band_indices']=[0,1,2]
        source=root/'configs/naip-rgb-development-manifest.json'
        if source.exists() and json.loads(source.read_text())!=m: raise ValueError('RGB manifest changed')
        source.write_text(json.dumps(m,indent=2)+'\n')
    prepared=prepare(source,root/'data',limit=16); mh=digest(prepared)
    for label,changes in variants.items():
        if bands==3 and label not in plan['rgb_variants']: continue
        for seed in plan['seeds']:
            config=json.loads((root/'configs/pilot.json').read_text())
            config.update(seed=seed,steps=1000,validate_every=100)
            config['model'].update(bands=bands,zero_residual=True,**changes)
            if label=='hybrid_consistency':config['consistency_weight']=.1
            existing=next((r for r in records if r['variant']==label and r['bands']==bands and r['seed']==seed),None)
            if existing:
                assert existing['config']==config and existing['manifest_sha256']==mh
                assert digest(root/'data/runs'/existing['run_id']/'best.pt')==existing['checkpoint_sha256']
                continue
            prior=next((r for r in old if bands==4 and r['model']==label and r['seed']==seed and r['config']==config and r['manifest_sha256']==mh),None)
            if prior:
                run=root/'data/runs'/prior['run_id']; assert digest(run/'best.pt')==prior['checkpoint_sha256']
            else: run=train(config,prepared,root/'data')
            values=json.loads(evaluate(run/'best.pt',prepared,root/'data','validation').read_text())
            scenes=[]
            for sid in sorted({r['scene_id'] for r in values}):
                scores={method:{metric:statistics.mean(r[metric] for r in values if r['scene_id']==sid and r['method']==method and r[metric] is not None) for metric in ['psnr','ssim','sam_rad']} for method in ['model','bicubic']}
                scenes.append({'scene_id':sid,'scores':scores,'delta_psnr':scores['model']['psnr']-scores['bicubic']['psnr']})
            training=json.loads((run/'training.json').read_text())
            records.append({'variant':label,'bands':bands,'seed':seed,'config':config,
                            'run_id':run.name,'checkpoint_sha256':digest(run/'best.pt'),
                            'manifest_sha256':mh,'manifest_path':str(prepared.relative_to(root)),
                            'seconds':training['seconds'],'parameters':training['parameters'],
                            'reused_prior':prior is not None,'validation_scenes':scenes,
                            'scene_mean_delta_db':statistics.mean(s['delta_psnr'] for s in scenes)})
            temporary=ledger.with_suffix('.tmp');temporary.write_text(json.dumps(records,indent=2)+'\n');temporary.replace(ledger)
            print(bands,label,seed,records[-1]['scene_mean_delta_db'],flush=True)
