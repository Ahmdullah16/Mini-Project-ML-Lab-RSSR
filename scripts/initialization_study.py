"""Predeclared initialization diagnostic. Validation only, equal 1000-step budget."""
import copy,json,os,statistics
from pathlib import Path
from rssr.core import digest
from rssr.data import prepare
from rssr.experiment import train,evaluate
root=Path(__file__).resolve().parents[1]; os.chdir(root)
ledger=root/'reports/initialization-study.json'
if ledger.exists(): raise SystemExit('Existing diagnostic ledger preserved; inspect records instead of overwriting')
manifest=prepare(root/'configs/naip-pilot-manifest.json',root/'data',limit=16)
base=json.loads((root/'configs/pilot.json').read_text()); base.update(steps=1000,validate_every=100)
plan={'hypothesis':'Random residual initialization may dominate the weak 100-step pilot. Compare random versus zero residual at a common 1000-step budget.','seeds':[17,23,41],'models':['cnn','hybrid'],'steps':1000,'selection':'lowest validation L1 at 100-step intervals','test_access':False,'limitations':'Single validation scene; no geographic inference or model finalization.'}
(root/'configs/initialization-study.json').write_text(json.dumps(plan,indent=2))
records=[]
for kind in plan['models']:
 for zero in [False,True]:
  for seed in plan['seeds']:
    cfg=copy.deepcopy(base); cfg['seed']=seed; cfg['model'].update(kind=kind,zero_residual=zero)
    run=train(cfg,manifest,root/'data'); ev=evaluate(run/'best.pt',manifest,root/'data')
    rows=json.loads(ev.read_text()); report=json.loads((run/'training.json').read_text())
    record={'model':kind,'zero_residual':zero,'seed':seed,'run_id':run.name,'checkpoint_sha256':digest(run/'best.pt'),'manifest_sha256':digest(manifest),'config':cfg,'seconds':report['seconds'],'git_commit':report['git_commit'],'dirty':report['dirty'],'psnr':statistics.mean(r['psnr'] for r in rows if r['method']=='model'),'bicubic_psnr':statistics.mean(r['psnr'] for r in rows if r['method']=='bicubic')}
    records.append(record); ledger.write_text(json.dumps(records,indent=2)); print(kind,zero,seed,round(record['psnr'],4),flush=True)
