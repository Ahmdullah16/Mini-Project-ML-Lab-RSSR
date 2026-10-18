"""Train a separate explicit RGB pilot for the display-only upload demo."""
import json
from pathlib import Path
from rssr.data import prepare
from rssr.experiment import train
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'configs/naip-pilot-manifest.json').read_text())
m['name']+=' RGB subset'; m['bands']=['red','green','blue']; m['band_indices']=[0,1,2]
path=root/'data'/'rgb-manifest.json'; path.write_text(json.dumps(m,indent=2))
manifest=prepare(path,root/'data',limit=16)
cfg=json.loads((root/'configs/pilot.json').read_text()); cfg['model']['bands']=3; cfg['model']['kind']='cnn'
run=train(cfg,manifest,root/'data')
(root/'reports/demo-checkpoint.json').write_text(json.dumps({'checkpoint':str((run/'best.pt').relative_to(root)),'prepared_manifest':str(manifest.relative_to(root))},indent=2))
print(run/'best.pt')
