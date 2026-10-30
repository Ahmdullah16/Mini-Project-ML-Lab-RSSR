"""Execute the committed bounded M2/M6 pilot; resume by verified completed record."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from rssr.core import digest
from rssr.experiment import train, evaluate, load_model, load_tiles
from rssr.db import connect

ROOT=Path(__file__).resolve().parents[1]


def subset_manifest(source, target):
    manifest=json.loads(source.read_text(encoding='utf-8'))
    manifest['tiles']=[dict(r,path=os.path.relpath(source.parent/r['path'],target.parent).replace('\\','/'))
                       for r in manifest['tiles'] if r['split'] in ('train','validation')]
    manifest['scenes']=[r for r in manifest['scenes'] if r['split'] in ('train','validation')]
    if {r['split'] for r in manifest['tiles']}!={'train','validation'}:raise ValueError('Need both development splits')
    content=json.dumps(manifest,indent=2)+'\n'
    if target.exists() and target.read_text(encoding='utf-8')!=content:raise ValueError('Existing subset differs')
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.write_text(content,encoding='utf-8')
    return manifest


def run(limit=None, plan_path=None):
    plan_path=ROOT/(plan_path or 'configs/missing-model-pilot.json');plan=json.loads(plan_path.read_text(encoding='utf-8'))
    source=ROOT/plan['source_manifest']
    if digest(source)!=plan['source_manifest_sha256']:raise ValueError('Source manifest changed')
    work=ROOT/'data'/plan['id'];manifest_path=work/'manifest.json'
    manifest=subset_manifest(source,manifest_path)
    db=connect(work/'catalog.sqlite')
    with db:
        for row in manifest['tiles']:
            path=(manifest_path.parent/row['path']).resolve()
            if digest(path)!=row['sha256']:raise ValueError('Development tile changed')
            db.execute('INSERT OR IGNORE INTO pairs_or_tiles VALUES(?,?,?,?,?,?,?,?,?)',
                       (row['id'],row['split'],str(path),str(path),manifest['scale'],None,
                        'existing synthetic paired tile',manifest['protocol'],row['sha256']))
    db.close()
    report_path=ROOT/'reports'/plan.get('report_file','missing-model-pilot.json')
    report=json.loads(report_path.read_text()) if report_path.exists() else {
        'schema':plan['id'],'plan_sha256':digest(plan_path),
        'source_manifest_sha256':digest(source),'subset_manifest_sha256':digest(manifest_path),
        'scope':plan['limits'],'runs':[],'test_pixels_accessed':False}
    if report['plan_sha256']!=digest(plan_path) or report['subset_manifest_sha256']!=digest(manifest_path):
        raise ValueError('Refuse reuse with changed plan/data')
    started=time.perf_counter();executed=0
    for architecture in plan['architectures']:
        for seed in plan['seeds']:
            prior=next((r for r in report['runs'] if r['architecture_id']==architecture and r['seed']==seed),None)
            if prior:
                if digest(ROOT/prior['checkpoint'])!=prior['checkpoint_sha256']:raise ValueError('Recorded checkpoint changed')
                continue
            if (limit is not None and executed>=limit) or time.perf_counter()-started>plan['max_wall_seconds_between_runs']:
                return report
            config={k:plan[k] for k in ['steps','batch_size','lr','validate_every','consistency_weight']}
            config.update(seed=seed,model=dict(plan['model'],architecture_id=architecture))
            out=train(config,manifest_path,work)
            checkpoint=out/'best.pt';model,state=load_model(checkpoint)
            evaluation=evaluate(checkpoint,manifest_path,work,split='validation')
            rows=json.loads(evaluation.read_text());groups=defaultdict(list)
            for row in rows:groups[(row['method'],row['scene_id'])].append(row['psnr'])
            means=[{'method':method,'scene_id':scene,'mean_tile_psnr_db':float(np.mean(values))}
                   for (method,scene),values in sorted(groups.items())]
            _,lr,_=load_tiles(manifest,'validation',manifest_path.parent)[0]
            with torch.inference_mode():pred=model(lr[None])[0]
            if pred.shape!=(3,32,32) or not torch.isfinite(pred).all():raise ValueError('Invalid reload inference')
            training=json.loads((out/'training.json').read_text())
            record={'architecture_id':architecture,'seed':seed,'run_id':state['run_id'],
                    'config':config,'checkpoint':checkpoint.relative_to(ROOT).as_posix(),
                    'checkpoint_sha256':digest(checkpoint),'selected_step':state['selected_step'],
                    'within_run_selection':'minimum validation L1','seconds_training':training['seconds'],
                    'parameters':training['parameters'],'history':training['history'],
                    'training_record_sha256':digest(out/'training.json'),
                    'environment':training['environment'],'hardware':training['hardware'],
                    'git_commit':training['git_commit'],'dirty':training['dirty'],
                    'validation_record_sha256':digest(evaluation),'validation_scene_means':means,
                    'reload_input_shape':list(lr.shape),'reload_output_shape':list(pred.shape),
                    'reload_output_sha256':hashlib.sha256(pred.numpy().tobytes()).hexdigest(),
                    'status':'trained bounded RGB native2 pilot; not research accepted or remotely deployable'}
            report['runs'].append(record)
            report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            executed+=1
            print(architecture,seed,round(training['seconds'],3),'seconds; selected',state['selected_step'],flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--plan');args=parser.parse_args()
    report=run(args.limit,args.plan);print('Verified completed pilot runs:',len(report['runs']))
