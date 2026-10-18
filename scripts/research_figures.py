"""Figures from executed validation runs; no fabricated values or test access."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rssr.experiment import load_model,load_tiles
from rssr.core import resize
root=Path(__file__).resolve().parents[1]
runs=json.loads((root/'reports/ablation-pilot-runs.json').read_text())
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
names=list(dict.fromkeys(r['variant'] for r in runs))
vals=[[r['psnr'] for r in runs if r['variant']==name] for name in names]
ax.bar(range(len(names)),[np.mean(v) for v in vals],yerr=[np.std(v,ddof=1) for v in vals],capsize=4)
ax.axhline(runs[0]['bicubic_psnr'],color='darkred',linestyle='--',label='Bicubic')
ax.set_xticks(range(len(names)),names,rotation=30,ha='right'); ax.set_ylabel('Validation PSNR (dB)')
ax.set_title('100-step pilot: mean and seed SD, one validation scene'); ax.legend()
fig.savefig(root/'reports/ablation-pilot.png',dpi=140); plt.close(fig)
record=next(r for r in runs if r['variant']=='hybrid_gate' and r['seed']==17)
model,state=load_model(root/'data/runs'/record['run_id']/'best.pt')
manifest=next(p for p in (root/'data/prepared').glob('*/manifest.json') if __import__('rssr.core',fromlist=['digest']).digest(p)==state['manifest_hash'])
row,lr,hr=load_tiles(json.loads(manifest.read_text()),'validation',manifest.parent)[0]
with torch.inference_mode(): sr=model(lr[None])[0]
base=resize(lr,2)
fig,axes=plt.subplots(1,5,figsize=(14,3),layout='constrained')
for ax,data,label in zip(axes[:4],[lr,base,sr,hr],['Synthetic LR','Bicubic','Gated hybrid (seed 17)','HR reference']):
 ax.imshow((data[:3].permute(1,2,0).numpy()/.3).clip(0,1),interpolation='nearest'); ax.set_title(label); ax.axis('off')
error=(sr-hr).abs().mean(0).numpy(); im=axes[4].imshow(error,vmin=0,vmax=.05,cmap='magma'); axes[4].set_title('Mean absolute band error'); axes[4].axis('off'); fig.colorbar(im,ax=axes[4],shrink=.6)
fig.suptitle('First validation tile; common RGB display [0,0.3] scaled units, clipping for display only')
fig.savefig(root/'reports/validation-example.png',dpi=140); plt.close(fig)
(root/'reports/figure-provenance.json').write_text(json.dumps({'run_id':record['run_id'],'tile_id':row['id'],'selection':'first validation tile; seed 17 fixed representative, not selected best','rgb_display_range':[0,.3],'error_range':[0,.05],'physical_calibration':'unverified'},indent=2))
print('Generated two figures from recorded runs')
