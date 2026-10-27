"""No-data, no-training budget preflight against fixed M7 width16 reference."""
import hashlib,json,platform,time
from pathlib import Path
import torch
from rssr.architecture_registry import REGISTRY,build
from rssr.model_cost import measure,match

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    started=time.perf_counter(); groups=[]
    for bands in (3,4):
        shape=(1,bands,64,64)
        with torch.random.fork_rng():
            torch.manual_seed(75)
            ref=measure(build('cat-v1/M7',bands=bands,width=16),shape)
            rows=[]
            for key in REGISTRY:
                for width in range(2,65,2):
                    cost=measure(build(key,bands=bands,width=width),shape)
                    rows.append({'architecture_id':key,'config':{'bands':bands,'scale':2,'width':width,'window':4},**cost,**match(cost,ref)})
        choices=[]
        for key in REGISTRY:
            candidates=[r for r in rows if r['architecture_id']==key]
            eligible=[r for r in candidates if r['eligible_partial_budget']]
            closest=min(eligible or candidates,key=lambda r:(max(r['parameter_fraction']/.05,r['partial_mac_fraction']/.10),r['config']['width']))
            choices.append({'architecture_id':key,'eligible_count':len(eligible),'candidate':closest,
                            'decision':'eligible for partial-budget comparison' if eligible else 'no eligible width in searched range; do not call matched'})
        groups.append({'bands':bands,'reference':{'architecture_id':'cat-v1/M7','width':16,**ref},'choices':choices,'all_candidates':[{k:r[k] for k in ('architecture_id','config','parameters','partial_macs','parameter_fraction','partial_mac_fraction','eligible_partial_budget')} for r in rows]})
    paths=['src/rssr/models.py','src/rssr/architecture_registry.py','src/rssr/model_cost.py','scripts/preflight_architecture_budgets.py']
    report={'schema':'cat-partial-budget-v1','seed':75,'command':'python scripts/preflight_architecture_budgets.py','environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','threads':2},'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},'groups':groups,'duration_seconds':time.perf_counter()-started,'exit_code':0,'limitations':['No imagery, optimizer steps, checkpoint loading or quality metrics.','Search: even widths2..64, window4, native2, reference M7 width16; choice uses costs only.','Partial Conv/Linear/MHA count excludes normalization, activations, pooling, interpolation, softmax, bias and elementwise work; not full FLOPs or matched runtime.','Training schedule, independent-region manifest and acquisition/resource gates remain open.']}
    (ROOT/'reports/architecture-budget-preflight.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for group in groups:
        for row in group['choices']:
            c=row['candidate'];print(group['bands'],row['architecture_id'],c['config']['width'],row['eligible_count'],c['parameters'],c['partial_macs'])
    print('seconds',report['duration_seconds'])


if __name__=='__main__':main()
