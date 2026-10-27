"""Bind existing local artifacts by checksum/config/provenance; no research reruns."""
import hashlib,json,platform,time
from pathlib import Path
import torch
from rssr.core import digest
from rssr.legacy_architecture import registry_config
from rssr.model_factory import from_config
from rssr.models import SRModel

ROOT=Path(__file__).resolve().parents[1]


def main():
    started=time.perf_counter();torch.set_num_threads(2);rows=[]
    sources=['reports/submission-study.json','reports/ablation-pilot-runs.json']
    for source in sources:
        for index,record in enumerate(json.loads((ROOT/source).read_text())):
            row={'source':source,'source_row':index,'run_id':record['run_id'],'historical_variant':record['variant'],'checkpoint_sha256':record['checkpoint_sha256']}
            try:config=registry_config(record['config']['model'])
            except ValueError as exc:
                rows.append({**row,'status':'outside registry','reason':str(exc)});continue
            row['registry_config']=config
            checkpoint=ROOT/'data/runs'/record['run_id']/'best.pt'
            training=checkpoint.parent/'training.json'
            if not checkpoint.exists() or not training.exists():
                rows.append({**row,'status':'unavailable locally','reason':'Checkpoint or training record missing; no download attempted'});continue
            assert digest(checkpoint)==record['checkpoint_sha256'],record['run_id']
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            trace=json.loads(training.read_text())
            assert state['run_id']==trace['run_id']==record['run_id']
            assert state['config']==trace['config']==record['config']
            assert state['manifest_hash']==trace['manifest_hash']
            if 'manifest_sha256' in record:
                assert state['manifest_hash']==record['manifest_sha256']
            assert trace['history'] and state.get('selected_step',0)>0
            new=from_config(config).eval();new.load_state_dict(state['model'],strict=True)
            old=SRModel(**state['config']['model']).eval();old.load_state_dict(state['model'],strict=True)
            with torch.random.fork_rng(),torch.inference_mode():
                torch.manual_seed(77);x=torch.rand(1,config['bands'],5,7)
                a=old(x);b=new(x)
            assert torch.equal(a,b),record['run_id']
            rows.append({**row,'status':'verified historical binding','training_record_sha256':digest(training),
                         'manifest_sha256':state['manifest_hash'],'selected_step':state['selected_step'],
                         'full_training_config':state['config'],'synthetic_output_shape':list(a.shape),
                         'synthetic_output_sha256':hashlib.sha256(a.numpy().tobytes()).hexdigest(),
                         'maximum_relabel_difference':float((a-b).abs().max()),'seed_for_compatibility_check':77,
                         'parameters':sum(p.numel() for p in new.parameters()),'deployed':False})
    paths=sources+['src/rssr/models.py','src/rssr/legacy_architecture.py','src/rssr/architecture_registry.py','scripts/bind_legacy_architectures.py']
    report={'schema':'historical-cat-binding-v1','command':'python scripts/bind_legacy_architectures.py','source_sha256':{p:digest(ROOT/p) for p in paths},'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'CPU','threads':2},'rows':rows,'seconds':time.perf_counter()-started,'exit_code':0,'limitations':['Existing pilot training only; not new matched-budget/protocol-v2 evidence.','Synthetic5x7 output equality tests compatibility only; no held-out pixels or quality rerun.','No checkpoint is rewritten, uploaded or added to deployment catalog; dataset manifests/pixels are not reopened, so manifest hashes are cross-checked declarations only.','Historical binding applies to each full configuration, never every width/band/scale of that model ID.','No equivalent historical M2/M6 checkpoint expected; status remains untrained for those variants.']}
    (ROOT/'reports/historical-architecture-bindings.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    from collections import Counter
    print(dict(Counter(r['status'] for r in rows)))
    print(dict(Counter(r['registry_config']['architecture_id'] for r in rows if r['status']=='verified historical binding')))


if __name__=='__main__':main()
