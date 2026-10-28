"""Generate a shared-prose report source, not new experiments or a final PDF."""
import argparse
import hashlib
import json
from pathlib import Path
from rssr.learning_content import load_lesson

ROOT=Path(__file__).resolve().parents[1]


def tex(text):
    escapes={'&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$','{':r'\{','}':r'\}'}
    return ''.join(escapes.get(c,c) for c in text)


def build(root):
    parts=['# Detailed project supplement — working source\n\nExisting teaching prose and recorded validation; not final scientific acceptance. Code cells are neither rendered nor executed. Complete parameter audits, new figures, individual mathematical acceptance and final typesetting remain pending. Author details are unavailable; supplied IEEE templates were reviewed (see reports/ieee-paper/evidence/supplied-template-review.json). No venue page limit is assumed.']
    parts.append((root/'docs/learning-reference.md').read_text(encoding='utf-8'))
    parts.append((root/'docs/model-instructor.md').read_text(encoding='utf-8'))
    parts.append((root/'docs/metric-instructor.md').read_text(encoding='utf-8-sig'))
    parts.append((root/'docs/website-validation-update.md').read_text(encoding='utf-8'))
    audit=json.loads((root/'docs/contribution-audit.json').read_text(encoding='utf-8'))
    parts.append('# Contribution audit\n\n'+audit['scope'])
    for item in audit['items']:
        parts.append('## '+item['proposed'])
        for key in ['prior_work','difference','needed','observed','supported_claim']:
            parts.append('**'+key.replace('_',' ').title()+':** '+item[key])
        parts.append('References: '+', '.join(item['references']))
        parts.append('Versioned evidence: '+', '.join(item['evidence']))
    manifest={'kind':'derived teaching prose, not executed experiments','lessons':[]}
    for i in range(1,41):
        lesson=load_lesson(root,f'MP{i:02d}')
        parts.extend(['# '+lesson['id']+' · '+lesson['scope'],lesson['coverage'],lesson['overview'],lesson['explanations'],'## Recorded validation',lesson['validation']])
        manifest['lessons'].append({k:lesson[k] for k in ['id','scope','source_hashes','coverage']})
    text='\n\n'.join(parts)+'\n'
    manifest['supplement_sha256_utf8_lf']=hashlib.sha256(text.encode()).hexdigest()
    for f in ['docs/learning-reference.md','docs/model-instructor.md','docs/contribution-audit.json','docs/metric-instructor.md','docs/website-validation-update.md']:
        manifest.setdefault('shared_sources',{})[f]=hashlib.sha256((root/f).read_text(encoding='utf-8').encode()).hexdigest()
    audit_tex='\n\n'.join(r'\paragraph{'+tex(item['proposed'])+'} '+
        ' '.join(tex(item[k].rstrip('.'))+'.' for k in ['prior_work','difference','needed','observed','supported_claim'])
        for item in audit['items'])+'\n'
    return text,json.dumps(manifest,indent=2)+'\n',audit_tex


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    outputs=build(ROOT)
    for path,text in zip(['paper/supplement.md','reports/learning-content-manifest.json','paper/generated/contribution-audit.tex'],outputs):
        target=ROOT/path
        if args.check:
            if not target.exists() or target.read_text(encoding='utf-8')!=text:raise SystemExit('Stale: '+path)
        else:target.write_text(text,encoding='utf-8')
    print('40 lesson sources and shared supplement '+('verified' if args.check else 'generated'))
