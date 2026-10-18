"""MP03/04 real training-crop experiments; no downloads or held-out evaluation."""
import json,platform
from pathlib import Path
import numpy as np
import torch
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rssr.core import digest
from rssr.image_foundations import resize_hwc,compare_hwc
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'configs/naip-pilot-manifest.json').read_text())
s=next(s for s in m['scenes'] if s['split']=='train'); path=root/'data'/s['path']
assert digest(path)==s['sha256']
with rasterio.open(path) as ds:
    window=rasterio.windows.Window(0,0,64,64)
    hr=ds.read(window=window).transpose(1,2,0).astype(float)/m['divisor']
    mask=(ds.read_masks(window=window)>0).all(0)
assert mask.all(), 'This resize experiment requires a fully valid crop'
# Explicit area averaging is the common LR-generation operator for this comparison.
lr=hr.reshape(32,2,32,2,4).mean((1,3))
outputs={name:resize_hwc(lr,(64,64),name) for name in ('nearest','bilinear')}
outputs['bicubic']=torch.nn.functional.interpolate(torch.from_numpy(lr.transpose(2,0,1))[None],size=(64,64),mode='bicubic',align_corners=False,antialias=False)[0].permute(1,2,0).numpy()
rows={name:{key:value.tolist() for key,value in compare_hwc(p,hr).items() if key in ('bias','mae','mse')} for name,p in outputs.items()}
checker=(np.indices((64,64)).sum(0)%2)[...,None]
alias={name:resize_hwc(checker,(32,32),name) for name in ('nearest','bilinear')}
fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
for ax,(name,p) in zip(axes[0],{'HR':hr,**outputs}.items()):
    ax.imshow((p[:,:,:3]/.3).clip(0,1)); ax.set_title(name); ax.axis('off')
for ax,(name,p) in zip(axes[1],{'checkerboard':checker,**alias,'area average':checker.reshape(32,2,32,2,1).mean((1,3))}.items()):
    ax.imshow(p[:,:,0],vmin=0,vmax=1,cmap='gray',interpolation='nearest'); ax.set_title(name); ax.axis('off')
fig.suptitle('MP03: common RGB display [0,0.3]; unfiltered sampling can alias')
fig.savefig(root/'reports/mp03-resizing.png',dpi=130); plt.close(fig)
pairs={'identical':hr.copy(),'brightness +0.02':hr+.02,'one-pixel shift':np.roll(hr,1,axis=1)}
# Roll wraps the first column: exclude it for every case to keep comparison support equal.
valid=mask.copy(); valid[:,0]=False
comparison={name:compare_hwc(p,hr,valid) for name,p in pairs.items()}
fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
for col,(name,result) in enumerate(comparison.items()):
    signed=result['signed'].mean(2); absolute=result['absolute'].mean(2)
    signed[~valid]=np.nan; absolute[~valid]=np.nan
    im=axes[0,col].imshow(signed,cmap='coolwarm',vmin=-.04,vmax=.04)
    ab=axes[1,col].imshow(absolute,cmap='magma',vmin=0,vmax=.04)
    axes[0,col].set_title(name)
    for ax in axes[:,col]: ax.axis('off')
fig.colorbar(im,ax=axes[0],label='Signed mean band error'); fig.colorbar(ab,ax=axes[1],label='Mean absolute band error')
fig.savefig(root/'reports/mp04-differences.png',dpi=130); plt.close(fig)
record={'source_scene':s['id'],'source_sha256':s['sha256'],'split':'train','crop_xywh':[0,0,64,64],'bands':m['bands'],'units':m['units'],'numpy':np.__version__,'torch':torch.__version__,'platform':platform.platform(),'mp03':{'lr_operator':'nonoverlapping 2x2 area average','coordinates':'half-pixel, replicated boundaries; nearest ties upwards','bicubic':'PyTorch align_corners=False, antialias=False; no clipping before metrics','errors_by_band':rows,'checkerboard_means':{name:float(x.mean()) for name,x in alias.items()}},'mp04':{name:{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in result.items() if k not in ('signed','absolute')} for name,result in comparison.items()},'limitations':'Single training crop, no generalization claims; numerical units are not verified reflectance. Identical and altered pairs share a known grid; shift is a controlled misalignment, not a registration algorithm.'}
(root/'reports/mp03-mp04-experiments.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({'resizing_mse':{k:v['mse'] for k,v in rows.items()},'comparison':record['mp04']},indent=2))
