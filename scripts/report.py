"""Generate scene-aggregated tables from actual JSON run records only."""
import csv,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
runs=json.loads((root/'reports/pilot-runs.json').read_text())
rows=[]
for run in runs:
    evaluation=json.loads((root/run['evaluation']).read_text())
    training=json.loads((root/run['run']/'training.json').read_text())
    for method in sorted({r['method'] for r in evaluation}):
        selected=[r for r in evaluation if r['method']==method]
        scene_means={scene:np.mean([r['psnr'] for r in selected if r['scene_id']==scene]) for scene in {r['scene_id'] for r in selected}}
        rows.append({'run_id':training['run_id'],'trained_model':run['model'],'method':method,'scene_mean_psnr_db':float(np.mean(list(scene_means.values()))),'scene_count':len(scene_means),'parameters':training['parameters'],'train_seconds':training['seconds']})
with (root/'reports/pilot-summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
lines=['# Measured pilot validation results','', 'One validation scene. Synthetic degradation on real NAIP aerial imagery; no final test or geographic confidence interval.','', '| Model | Method | PSNR dB | Parameters | Training seconds |','|---|---|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['trained_model']} | {r['method']} | {r['scene_mean_psnr_db']:.4f} | {r['parameters']} | {r['train_seconds']:.2f} |")
(root/'reports/pilot-summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
