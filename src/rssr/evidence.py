"""Read-only evidence index. No model import, network access or experiment execution."""
import hashlib
import json
import math
import statistics
from pathlib import Path

UNITS = {'psnr': 'dB', 'delta_psnr': 'dB over bicubic', 'ssim': 'unitless',
         'sam_rad': 'radians', 'parameters': 'count', 'median_seconds': 'seconds',
         'p95_seconds': 'seconds', 'sampled_process_peak_rss_bytes': 'bytes',
         'conv_linear_attention_macs': 'partial MACs', 'validation_l1': 'normalized DN',
         'train_l1': 'normalized DN', 'train_mse': 'normalized DN squared'}


def build_index(root):
    root = Path(root)
    rows, artifacts, normalized = [], [], set()

    def read(name):
        return json.loads((root/'reports'/name).read_text(encoding='utf-8'))

    def add(source, pointer, group, model, bands, split, values, *, scenes=None,
            tiles=None, seed=None, run=None, condition='sigma=.8, noise=0, x2', protocol='synthetic-naip-v1', task='SR'):
        normalized.add(source)
        for metric, value in values.items():
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f'Invalid metric {source}:{pointer}/{metric}')
            rows.append({'study': group, 'model': model, 'bands': bands, 'split': split,
                         'task': task, 'dataset': 'NAIP aerial', 'data_kind': 'synthetic' if task == 'SR' else 'same-grid reconstruction',
                         'metric': metric, 'value': value, 'unit': UNITS.get(metric, 'normalized DN' if metric.startswith('rmse') else 'declared in source'),
                         'scenes': scenes, 'tiles': tiles, 'seed': seed, 'run_id': run,
                         'condition': condition, 'protocol': protocol,
                         'source': 'reports/'+source, 'source_pointer': pointer})

    for i, r in enumerate(read('ablation-pilot-runs.json')):
        add('ablation-pilot-runs.json', f'/{i}', 'Historical 100-step ablations', r['variant'], 4, 'validation',
            {k:r[k] for k in ('psnr','ssim','sam_rad')}, scenes=1, tiles=16, seed=r['seed'], run=r['run_id'])
        add('ablation-pilot-runs.json', f'/{i}', 'Historical 100-step ablations', 'bicubic', 4, 'validation',
            {'psnr':r['bicubic_psnr']}, scenes=1, tiles=16, seed=r['seed'], run=r['run_id'])
        add('ablation-pilot-runs.json', f'/{i}', 'Historical 100-step ablations', r['variant'], 4, 'validation',
            {'delta_psnr':r['psnr']-r['bicubic_psnr']}, scenes=1, tiles=16, seed=r['seed'], run=r['run_id'])
    for i, r in enumerate(read('submission-study.json')):
        for j, scene in enumerate(r['validation_scenes']):
            for method, scores in scene['scores'].items():
                add('submission-study.json', f'/{i}/validation_scenes/{j}/scores/{method}', 'Development 1000-step study',
                    r['variant'] if method=='model' else 'bicubic', r['bands'], 'validation', scores,
                    scenes=1, tiles=16, seed=r['seed'], run=r['run_id'])
        add('submission-study.json', f'/{i}', 'Development seed summaries', r['variant'], r['bands'], 'validation',
            {'delta_psnr':r['scene_mean_delta_db']}, scenes=4, tiles=64, seed=r['seed'], run=r['run_id'])
    frozen = read('final-heldout.json')
    plan = json.loads((root/'configs/final-evaluation.json').read_text())
    for i, r in enumerate(frozen['summary']):
        selected = next((s for s in plan['selected'] if s['bands']==r['bands'] and s['variant']==r['model']), {})
        add('final-heldout.json', f'/summary/{i}', 'Frozen held-out evaluation', r['model'], r['bands'], 'test',
            {k:v for k,v in r.items() if k in ('psnr','ssim','sam_rad') or k.startswith('rmse_band_')},
            scenes=r['scenes'], tiles=r['tiles'], seed=selected.get('seed'), run=selected.get('run_id'), protocol=frozen['protocol_hash'])
    benchmark = read('development-benchmark.json')
    for i, r in enumerate(benchmark['robustness']):
        for j, scene in enumerate(r['scenes']):
            for method, scores in scene['scores'].items():
                add('development-benchmark.json', f'/robustness/{i}/scenes/{j}/scores/{method}', 'Robustness / spectral',
                    r['variant'] if method=='model' else 'bicubic', r['bands'], 'validation', scores,
                    scenes=1, tiles=16, seed=r['seed'], condition=f"sigma={r['sigma_hr_pixels']}, noise={r['noise_std']}, x2")
    for i, r in enumerate(benchmark['efficiency']):
        add('development-benchmark.json', f'/efficiency/{i}', 'CPU efficiency', r['variant'], r['bands'], 'benchmark',
            {k:r[k] for k in ('parameters','median_seconds','p95_seconds','sampled_process_peak_rss_bytes','conv_linear_attention_macs')},
            condition='CPU float32; 64×64 LR; 5 warmups, 20 repetitions', protocol='CPU benchmark; partial MACs and sampled process RSS')
    official = read('official-swinir-validation.json')
    for method in sorted({r['method'] for r in official['rows']}):
        group = [r for r in official['rows'] if r['method']==method]
        add('official-swinir-validation.json', '/rows', 'Official pretrained baseline', method, 3, 'validation',
            {k:statistics.mean(r[k] for r in group) for k in ('psnr','ssim','sam_rad')}, scenes=len({r['scene_id'] for r in group}),
            tiles=len(group), protocol='Matched metrics; different DIV2K pretraining/degradation')
    for filename in ['mp19-mp22-cnn.json','mp23-mp25-attention.json']:
        report=read(filename)
        for i,r in enumerate(report['results']):
            model=r.get('model',r.get('mode'))
            task='reconstruction' if filename.startswith('mp23') or model=='identity' else 'SR'
            add(filename,f'/results/{i}','Teaching experiments',model,3,'validation',{'psnr':r['validation_psnr']},
                scenes=4,tiles=64,seed=report['seed'],protocol=filename,task=task)
    curves=[]
    for i,r in enumerate(read('mp26-attention-sr.json')['curves']):
        for point in r['history']:
            curves.append(dict(point,model=r['model'],seed=r['seed'],run_id=r['run_id'],source='reports/mp26-attention-sr.json'))
    for path in sorted((root/'reports').glob('*')):
        if path.suffix not in ('.json','.md','.png'):
            continue
        artifacts.append({'path':path.relative_to(root).as_posix(), 'bytes':path.stat().st_size,
                          'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                          'coverage':'normalized metrics' if path.name in normalized else 'source artifact; not a comparable metric table'})
    projects=[]
    folders={p['id']:p for p in json.loads((root/'docs/project_folders.json').read_text(encoding='utf-8'))}
    for project in json.loads((root/'docs/syllabus.json').read_text(encoding='utf-8')):
        ident=project['id'];number=int(ident[2:]);brief=Path(project['brief'])
        notebook=Path(folders[ident]['notebook'])
        complete = '**Status: Complete (implementation and teaching deliverables)**' in (root/brief).read_text(encoding='utf-8')
        projects.append({'id':ident,'title':project['title'],'objective':project['objective'],
                         'stage':brief.parts[0], 'brief':brief.as_posix(),'notebook':notebook.as_posix(),
                         'status':'Complete implementation and teaching deliverables; learner mastery unassessed' if complete else ('Expanded / execution recorded; criteria not all assessed' if number<=26 else 'Introductory / full expansion pending'),
                         'exception':'Independent held-out SRCNN table pending' if number==20 else None})
    if len(projects)!=40 or len({p['stage'] for p in projects})!=9:
        raise ValueError('Syllabus identity/stage mismatch')
    return {'schema_version':1,'rows':rows,'curves':curves,'artifacts':artifacts,'projects':projects,
            'limits':'One held-out scene; synthetic NAIP, not cross-sensor validation. Normalized DN, not reflectance. No hybrid superiority. Historical and teaching protocols are separate.',
            'pending_artifacts':['Final report','Presentation','Complete submission package']}
