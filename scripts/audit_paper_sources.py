"""Audit recorded paper evidence, without opening datasets or model weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def read(root,path):
    return (root/path).read_text(encoding='utf-8').replace('\r\n','\n')


def audit(root):
    files=['paper/main.tex','paper/generated/results.tex','paper/generated/interpretation.tex',
           'paper/generated/contribution-audit.tex','paper/references.bib','references.bib',
           'reports/paper-evidence.json','reports/final-heldout.json','configs/final-evaluation.json',
           'reports/historical-architecture-bindings.json','src/rssr/models.py',
           'src/rssr/architecture_registry.py','src/rssr/experiment.py',
           'reports/architecture-budget-preflight.json','reports/reflectance-job-validation.json',
           'reports/public-access-20261028.json']
    sources={p:hashlib.sha256(read(root,p).encode()).hexdigest() for p in files}
    manuscript='\n'.join(read(root,p) for p in files if p.endswith('.tex'))
    cited=sorted({key.strip() for group in re.findall(r'\\cite\{([^}]+)\}',manuscript) for key in group.split(',')})
    bibliography=read(root,'paper/references.bib')
    keys=set(re.findall(r'@\w+\s*\{\s*([^,]+),',bibliography))
    if set(cited)-keys:raise ValueError('Unresolved citation keys: '+str(set(cited)-keys))
    if bibliography!=read(root,'references.bib'):raise ValueError('Paper bibliography differs from authoritative copy')
    for target in re.findall(r'\\input\{([^}]+)\}',read(root,'paper/main.tex')):
        if not (root/'paper'/target).is_file():raise ValueError('Missing TeX input: '+target)
    final=json.loads(read(root,'reports/final-heldout.json'))
    derived=json.loads(read(root,'reports/paper-evidence.json'))
    if final['protocol_hash']!=derived['frozen_protocol_hash']:raise ValueError('Frozen protocol mismatch')
    measured=[]
    for i,row in enumerate(final['summary']):
        if row['scenes']!=1:raise ValueError('Revisit single-scene interpretation')
        measured.append({'category':'executed_measured','source':'reports/final-heldout.json',
            'pointer':f'/summary/{i}','bands':row['bands'],'model':row['model'],
            'scenes':row['scenes'],'tiles':row['tiles'],
            'metrics':{k:v for k,v in row.items() if k in ('psnr','ssim','sam_rad') or k.startswith('rmse_band_')},
            'limitation':'Mean tile scores in one held-out scene; no geographic CI; normalized DN, not calibrated reflectance'})
    bindings=json.loads(read(root,'reports/historical-architecture-bindings.json'))['rows']
    counts=Counter(r['registry_config']['architecture_id'] for r in bindings if r['status']=='verified historical binding')
    models=[{'id':f'cat-v1/M{i}','historical_bindings':counts[f'cat-v1/M{i}'],
             'category':'implemented_tested','limitation':'Binding compatibility does not imply all configurations/scales trained or scientifically evaluated'} for i in range(1,8)]
    return {'schema':'ieee-source-audit-v1','scope':'Recorded-source and citation integrity, not full scientific or bibliography verification',
            'source_sha256_utf8_lf':sources,'citation_keys_resolved':cited,
            'frozen_protocol_hash':final['protocol_hash'],'measured':measured,'models':models,
            'metric_units':{'psnr':'dB, fixed range1','ssim':'dimensionless',
                            'sam_rad':'radians; normalized-DN descriptor, not physical spectral validation',
                            'rmse_band_*':'normalized DN; per-band tile-score scene mean'},
            'categories':{'executed_measured':'Historical observations with protocol/units',
                          'implemented_tested':'Software checks, not quality evidence',
                          'planned':'Unexecuted experiments/formatting tasks',
                          'unavailable':'Missing reference files, authors or scientific evidence'},
            'gaps':['Supplied conference template and HOWTO comparison unavailable',
                    'M2/M6 training and seven-model matched-budget geographic evaluation absent',
                    'New manuscript reconstruction paused at supplied-reference gate',
                    'Citation-key resolution is not metadata/full-text verification',
                    'Latest corrected PDF visual reinspection pending'],
            'training_selection_note':'Within-run best checkpoint uses minimum validation L1 (experiment.py); final-evaluation.json selects among seeds/configurations by development PSNR. These are distinct selection stages.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    output=json.dumps(audit(ROOT),indent=2,allow_nan=False)+'\n'
    target=ROOT/'reports/ieee-paper/evidence/source-audit.json'
    if args.check:
        if not target.exists() or target.read_text(encoding='utf-8')!=output:raise SystemExit('Stale paper source audit')
    else:target.write_text(output,encoding='utf-8')
    print('Paper source audit '+('verified' if args.check else 'generated')+'; no inference or training')
