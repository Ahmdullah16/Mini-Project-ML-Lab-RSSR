"""One frozen held-out evaluation, never a training/model-selection workflow."""
import argparse
import json
import subprocess
import statistics
from pathlib import Path
import torch
from rssr.core import digest,metrics,resize,config_hash
from rssr.experiment import load_model,load_tiles
from rssr.official_baseline import load_swinir

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
plan_path=root/'configs/final-evaluation.json';report_path=root/'reports/final-heldout.json'
if args.freeze:
    if plan_path.exists():raise ValueError('Selection already frozen; do not overwrite')
    benchmark=json.loads((root/'reports/development-benchmark.json').read_text())
    selected=benchmark['selected']
    plan={'protocol':'final-synthetic-naip-v1','selection':'best validation scene-mean PSNR seed within CNN/attention/hybrid RGBNIR and CNN/hybrid RGB; attention is best four-band mean architecture. No hybrid superiority claimed.',
          'selected':[{k:r[k] for k in ['bands','variant','seed','run_id','checkpoint_sha256','manifest_path','manifest_sha256']} for r in selected],
          'official_rgb':json.loads((root/'configs/official-swinir.json').read_text()),
          'metric_protocol':{'data_range':1,'border':2,'clipping':False,'ssim':'uniform 7x7 population','sam':'radians excluding zero-norm spectra','aggregation':'per-tile metrics then per-scene means; one test scene, no geographic CI'},
          'heldout':'original test scene only, 16 deterministic nonoverlapping 32x32 HR patches; same synthetic Gaussian .8 + area x2',
          'decision':'No training, hyperparameter, checkpoint or protocol changes based on test results. Report all selected models and baselines including losses.',
          'limitations':'One test scene cannot establish geographic or cross-sensor generalization. Official SwinIR has different pretraining and degradation.'}
    plan_path.write_text(json.dumps(plan,indent=2)+'\n');print('Wrote freeze; commit this protocol before running evaluation.');raise SystemExit(0)
plan=json.loads(plan_path.read_text());ph=config_hash(plan)
committed=json.loads(subprocess.check_output(['git','show','HEAD:configs/final-evaluation.json'],cwd=root,text=True))
if committed!=plan:raise ValueError('Protocol must be committed unchanged before test access')
if report_path.exists():
    existing=json.loads(report_path.read_text())
    if existing['protocol_hash']!=ph:raise ValueError('Existing held-out protocol differs')
    print('Existing held-out result preserved; no repeat evaluation.');raise SystemExit(0)
torch.set_num_threads(2);rows=[]
with torch.inference_mode():
    for bands in [4,3]:
        entries=[r for r in plan['selected'] if r['bands']==bands]
        models={}
        for entry in entries:
            checkpoint=root/'data/runs'/entry['run_id']/'best.pt'
            if digest(checkpoint)!=entry['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
            models[entry['variant']]=load_model(checkpoint)[0]
        if bands==3:
            models['official_swinir'],official=load_swinir(root)
            if official!=plan['official_rgb']:raise ValueError('Official model configuration changed')
        path=root/entries[0]['manifest_path'].replace('\\','/')
        if digest(path)!=entries[0]['manifest_sha256']:raise ValueError('Frozen manifest changed')
        manifest=json.loads(path.read_text())
        for item,x,hr in load_tiles(manifest,'test',path.parent):
            predictions={name:model(x[None])[0] for name,model in models.items()}
            predictions.update({name:resize(x,2,name) for name in ['nearest','bilinear','bicubic']})
            for name,pred in predictions.items():
                rows.append({'bands':bands,'model':name,'tile_id':item['id'],'scene_id':item['scene_id'],**metrics(pred,hr,border=2)})
summary=[]
for bands in [4,3]:
    for name in sorted({r['model'] for r in rows if r['bands']==bands}):
        group=[r for r in rows if r['bands']==bands and r['model']==name]
        summary.append({'bands':bands,'model':name,'tiles':len(group),'scenes':len({r['scene_id'] for r in group}),
                        **{k:statistics.mean(r[k] for r in group) for k in ['psnr','ssim','sam_rad',*[f'rmse_band_{i}' for i in range(bands)]]}})
report={'protocol_hash':ph,'freeze_commit':subprocess.check_output(['git','log','-1','--format=%H','--','configs/final-evaluation.json'],cwd=root,text=True).strip(),
        'environment':{'torch':torch.__version__,'device':'cpu','threads':2},'summary':summary,'rows':rows,
        'limitations':plan['limitations'],'post_test_tuning':False}
temp=report_path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(report_path)
print(json.dumps(summary,indent=2))
