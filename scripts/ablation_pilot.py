"""Bounded three-seed ablation pilot. Validation only; no scene-level CI possible."""
import argparse,copy,json,os,statistics
from pathlib import Path
from rssr.data import prepare
from rssr.experiment import train,evaluate
from rssr.core import digest
root=Path(__file__).resolve().parents[1]; os.chdir(root)
parser=argparse.ArgumentParser(); parser.add_argument('--resume',action='store_true')
args=parser.parse_args()
ledger=root/'reports/ablation-pilot-runs.json'
if ledger.exists() and not args.resume:
    parser.error('Existing run ledger preserved. Use --resume to verify/continue it.')
manifest=prepare(root/'configs/naip-pilot-manifest.json',root/'data',limit=16)
base=json.loads((root/'configs/pilot.json').read_text())
variants={
 'cnn':{'kind':'cnn'},'cnn_attention':{'kind':'attention'},
 'cnn_transformer':{'kind':'hybrid','use_attention':False},
 'hybrid':{'kind':'hybrid'},'hybrid_add':{'kind':'hybrid','fusion':'add'},
 'hybrid_gate':{'kind':'hybrid','gate':True},
 'hybrid_constant_gate':{'kind':'hybrid','constant_gate':.5},
 'hybrid_consistency':{'kind':'hybrid'},
}
plan={'seeds':[17,23,41],'steps':100,'variants':variants,'consistency_weight':.1,'selection':'lowest validation L1 at steps 20,40,60,80,100','evaluation':'validation only; one scene, 16 nonoverlapping tiles; average tile metrics','hypothesis':'Exploratory controls; no final selection, novelty or geographic generalization claim','budget':'24 CPU runs, no new downloads, no GPU'}
(root/'configs/ablation-pilot.json').write_text(json.dumps(plan,indent=2))
records=json.loads(ledger.read_text()) if ledger.exists() else []
for name,model in variants.items():
 for seed in plan['seeds']:
    cfg=copy.deepcopy(base); cfg['seed']=seed; cfg['model'].update(model)
    if name=='hybrid_consistency': cfg['consistency_weight']=.1
    existing=next((r for r in records if r['variant']==name and r['seed']==seed),None)
    if existing:
        if existing['config']!=cfg or existing['manifest_sha256']!=digest(manifest):
            raise ValueError('Existing run configuration or manifest differs')
        checkpoint=root/'data/runs'/existing['run_id']/'best.pt'
        if digest(checkpoint)!=existing['checkpoint_sha256']:
            raise ValueError('Existing checkpoint checksum differs')
        print(name,seed,'verified existing run',flush=True)
        continue
    run=train(cfg,manifest,root/'data'); ev=evaluate(run/'best.pt',manifest,root/'data')
    rows=json.loads(ev.read_text()); measured=[r for r in rows if r['method']=='model']; classical=[r for r in rows if r['method']=='bicubic']
    training=json.loads((run/'training.json').read_text())
    record={'variant':name,'seed':seed,'run_id':run.name,'checkpoint_sha256':digest(run/'best.pt'),'manifest_sha256':digest(manifest),'git_commit':training['git_commit'],'dirty':training['dirty'],'parameters':training['parameters'],'training_seconds':training['seconds'],'psnr':statistics.mean(r['psnr'] for r in measured),'bicubic_psnr':statistics.mean(r['psnr'] for r in classical),'sam_rad':statistics.mean(r['sam_rad'] for r in measured),'ssim':statistics.mean(r['ssim'] for r in measured),'evaluation_sha256':digest(ev),'config':cfg}
    records.append(record)
    (root/'reports/ablation-pilot-runs.json').write_text(json.dumps(records,indent=2))
    print(name,seed,round(record['psnr'],4),flush=True)
lines=['# Executed ablation pilot','','Validation only: one NAIP scene / 16 tiles; 100 steps per model. Mean ± sample SD across three seeds describes initialization variability, not geographic uncertainty. No independent-scene confidence interval can be estimated. All settings were fixed before these runs.','','| Variant | Parameters | PSNR mean ± seed SD (dB) | SAM mean (rad) | Training seconds total |','|---|---:|---:|---:|---:|']
for name in variants:
 r=[x for x in records if x['variant']==name]
 lines.append(f"| {name} | {r[0]['parameters']} | {statistics.mean(x['psnr'] for x in r):.4f} ± {statistics.stdev(x['psnr'] for x in r):.4f} | {statistics.mean(x['sam_rad'] for x in r):.6f} | {sum(x['training_seconds'] for x in r):.2f} |")
lines+=['',f"Bicubic: {records[0]['bicubic_psnr']:.4f} dB. Compare within this fixed numerical scaling only; physical reflectance calibration is unverified.",'','Gate inputs contain LR only; the constant 0.5 control separates attenuation from learned conditioning. Consistency weight 0.1 applies the synthetic-naip-v1 forward operator, not a cross-sensor physics model. CNN+Transformer omits attention; additive fusion averages branch features. Parameter counts differ and are disclosed. The test split remains unused. See ablation-pilot-runs.json for measured run IDs, hashes and configurations.']
(root/'reports/ablation-pilot.md').write_text('\n'.join(lines),encoding='utf-8')
