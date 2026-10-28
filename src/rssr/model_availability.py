"""Shared seven-family evidence/availability view; no network or model loading."""
import json
from pathlib import Path
NAMES=('CNN','Attention','Transformer','CNN + Attention','CNN + Transformer','Attention + Transformer','CNN + Attention + Transformer')

def availability(root):
    history=json.loads((root/'reports/historical-architecture-bindings.json').read_text(encoding='utf-8'))['rows']
    pilots=json.loads((root/'reports/missing-model-pilot.json').read_text(encoding='utf-8'))['runs']
    hosted=json.loads((root/'configs/hosted-models.json').read_text(encoding='utf-8'))['models']
    result=[]
    for i,name in enumerate(NAMES,1):
        key=f'cat-v1/M{i}'
        records=[r for r in history if r['status']=='verified historical binding' and r['registry_config']['architecture_id']==key]
        local=[r for r in pilots if r['architecture_id']==key]
        training_hashes={r['checkpoint_sha256'] for r in records}
        deployments=[r for r in hosted if r.get('training_checkpoint_sha256') in training_hashes]
        result.append({'model':f'M{i} · {name}','bands':sorted({r['registry_config']['bands'] for r in records}|{r['config']['model']['bands'] for r in local}),
                       'training':f'{len(records)} historical bindings; {len(local)} bounded pilots',
                       'rgb_execution':bool(deployments),'checkpoint_sha256':[r['checkpoint']['sha256'] for r in deployments] or [r['checkpoint_sha256'] for r in local+records],
                       'status':'Pinned hosted RGB checkpoint available; native ×2, higher scales cascaded' if deployments else 'RGB pilot verified locally; pinned remote checkpoint distribution missing' if local else 'Verified four-band weights; compatible RGB training/checkpoint missing'})
    return result
