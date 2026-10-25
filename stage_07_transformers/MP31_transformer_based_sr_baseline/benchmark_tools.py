"""Build a teaching comparison from executed development evidence, never retest."""
import hashlib,json,statistics
from pathlib import Path


def comparison(root):
    root=Path(root);study_path=root/'reports/submission-study.json';bench_path=root/'reports/development-benchmark.json'
    study=json.loads(study_path.read_text());benchmark=json.loads(bench_path.read_text())
    runs=[r for r in study if r['bands']==4 and r['variant'] in ('cnn','transformer')]
    if len(runs)!=6 or {(r['variant'],r['seed']) for r in runs}!={(m,s) for m in ('cnn','transformer') for s in (17,23,41)}:
        raise ValueError('Expected six fixed four-band development runs')
    rows=[]
    for r in runs:
        config=r['config'];scenes=r['validation_scenes']
        if (config['steps'],config['batch_size'],config['lr'],config['validate_every'])!=(1000,4,.001,100) or len(scenes)!=4:
            raise ValueError('Comparison budget or scene count changed')
        model_psnr=statistics.mean(s['scores']['model']['psnr'] for s in scenes)
        bicubic_psnr=statistics.mean(s['scores']['bicubic']['psnr'] for s in scenes)
        if abs((model_psnr-bicubic_psnr)-r['scene_mean_delta_db'])>1e-10:raise ValueError('Recorded delta does not reconcile')
        rows.append({'model':r['variant'],'seed':r['seed'],'run_id':r['run_id'],
                     'checkpoint_sha256':r['checkpoint_sha256'],'manifest_sha256':r['manifest_sha256'],
                     'model_psnr':model_psnr,'bicubic_psnr':bicubic_psnr,'delta_psnr':r['scene_mean_delta_db'],
                     'parameters':r['parameters'],'training_seconds':r['seconds'],'config':config})
    summary=[]
    for model in ('cnn','transformer'):
        group=[r for r in rows if r['model']==model]
        summary.append({'model':model,'mean_delta_db':statistics.mean(r['delta_psnr'] for r in group),
                        'seed_sd_delta_db':statistics.stdev(r['delta_psnr'] for r in group),
                        'parameters':group[0]['parameters']})
    efficiency=[r for r in benchmark['efficiency'] if r['bands']==4 and r['variant'] in ('cnn','transformer')]
    return {'source_hashes':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [study_path,bench_path]},
            'rows':rows,'summary':summary,'efficiency':efficiency,'benchmark_environment':benchmark['environment'],
            'protocol':'Existing four-train/four-validation NAIP scenes;16 patches/scene; RGBNIR; synthetic x2 sigma.8 area LR; range1 border2; per-scene aggregation; Adam .001 batch4 L1;1000 updates; min validation L1 checkpoint every100; seeds17/23/41',
            'scope':'Read-only teaching export of previously executed validation/efficiency; no new training, timing, checkpoint inference or held-out evaluation',
            'limitations':'Unshifted educational window Transformer, not faithful SwinIR. Equal updates are not equal parameters/time. Seed SD is not geographic confidence. Four validation scenes and no calibrated reflectance. Partial MACs/RSS limitations retained in efficiency rows.'}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];report=comparison(root)
    (root/'reports/mp31-transformer-sr.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))
