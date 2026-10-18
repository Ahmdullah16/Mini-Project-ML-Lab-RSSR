"""Run explicit bounded pilot; does not touch final test data."""
import json
from pathlib import Path
from rssr.data import prepare
from rssr.experiment import train,evaluate
from rssr.db import connect,export

root=Path(__file__).resolve().parents[1]
data=root/'data'
manifest=prepare(root/'configs/naip-pilot-manifest.json',data,limit=16)
config=json.loads((root/'configs/pilot.json').read_text())
records=[]
for kind in ['cnn','attention','transformer','hybrid','srcnn']:
    cfg=json.loads(json.dumps(config)); cfg['model']['kind']=kind
    run=train(cfg,manifest,data)
    evaluation=evaluate(run/'best.pt',manifest,data)
    records.append({'model':kind,'run':str(run.relative_to(root)),'evaluation':str(evaluation.relative_to(root))})
    print(kind,run.name,flush=True)
(root/'reports/pilot-runs.json').write_text(json.dumps(records,indent=2))
db=connect(data/'catalog.sqlite'); export(db,data/'exports'); db.close()
print('Pilot complete: validation only, held-out test untouched')
