"""Bounded validation-only geographic diagnostic; completed runs are resumable."""
import copy
import json
import os
import statistics
from pathlib import Path
from rssr.core import digest
from rssr.data import prepare
from rssr.experiment import train, evaluate

root = Path(__file__).resolve().parents[1]
os.chdir(root)
ledger = root / 'reports/geographic-study.json'
plan = {
    'models': ['cnn', 'hybrid'], 'seeds': [17, 23, 41], 'steps': 1000,
    'zero_residual': True, 'validation_interval': 100,
    'selection': 'lowest mean validation L1; equal patches per scene',
    'test_access': False,
    'limitations': 'Four validation scenes, synthetic NAIP degradation; not satellite generalization.'
}
plan_path = root / 'configs/geographic-study.json'
if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
    raise ValueError('Existing study plan differs')
plan_path.write_text(json.dumps(plan, indent=2))
manifest = prepare(root / 'configs/naip-development-manifest.json', root / 'data', limit=16)
mh = digest(manifest)
base = json.loads((root / 'configs/pilot.json').read_text())
base.update(steps=plan['steps'], validate_every=plan['validation_interval'])
records = json.loads(ledger.read_text()) if ledger.exists() else []
for kind in plan['models']:
    for seed in plan['seeds']:
        cfg = copy.deepcopy(base)
        cfg['seed'] = seed
        cfg['model'].update(kind=kind, zero_residual=True)
        existing = [r for r in records if r['model'] == kind and r['seed'] == seed]
        if existing:
            r = existing[0]
            if r['config'] != cfg or r['manifest_sha256'] != mh or digest(root / 'data/runs' / r['run_id'] / 'best.pt') != r['checkpoint_sha256']:
                raise ValueError('Resume evidence mismatch')
            continue
        run = train(cfg, manifest, root / 'data')
        rows = json.loads(evaluate(run / 'best.pt', manifest, root / 'data').read_text())
        scenes = []
        for sid in sorted({r['scene_id'] for r in rows}):
            means = {m: statistics.mean(r['psnr'] for r in rows if r['scene_id'] == sid and r['method'] == m) for m in ['model', 'bicubic']}
            scenes.append({'scene_id': sid, **means, 'delta_db': means['model'] - means['bicubic']})
        training = json.loads((run / 'training.json').read_text())
        records.append({'model': kind, 'seed': seed, 'config': cfg, 'run_id': run.name,
                        'checkpoint_sha256': digest(run / 'best.pt'), 'manifest_sha256': mh,
                        'seconds': training['seconds'], 'git_commit': training['git_commit'],
                        'dirty': training['dirty'], 'validation_scenes': scenes,
                        'scene_mean_delta_db': statistics.mean(s['delta_db'] for s in scenes)})
        ledger.write_text(json.dumps(records, indent=2))
        print(kind, seed, records[-1]['scene_mean_delta_db'], flush=True)
