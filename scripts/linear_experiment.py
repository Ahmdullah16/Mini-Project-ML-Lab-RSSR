"""Real training-crop SVD rank/error/storage experiment for MP12."""
import json
from pathlib import Path
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rssr.core import digest
from rssr.linear_lab import approximate

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'configs/naip-pilot-manifest.json').read_text())
scene=next(s for s in manifest['scenes'] if s['split']=='train')
source=root/'data'/scene['path']; assert digest(source)==scene['sha256']
with rasterio.open(source) as ds:
    window=rasterio.windows.Window(0,0,64,64)
    x=ds.read(1,window=window).astype(float)/manifest['divisor']
    assert (ds.read_masks(1,window=window)>0).all()
rows=[]; images=[]
for rank in [0,1,2,4,8,16,32,64]:
    y,meta=approximate(x,rank)
    assert np.isclose(meta['frobenius_error']**2,meta['tail_energy'],atol=1e-12)
    rows.append({'rank':rank,'mse':float(((x-y)**2).mean()),**meta})
    if rank in [1,4,16,64]: images.append((rank,y))
assert all(a['mse']>=b['mse']-1e-15 for a,b in zip(rows,rows[1:]))
fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for ax,(rank,y) in zip(axes.flat,images):
    ax.imshow(y,cmap='gray',vmin=x.min(),vmax=x.max()); ax.set_title(f'Rank {rank}, fixed reference display range')
axes[1,1].semilogy([q['rank'] for q in rows[:-1]],[q['mse'] for q in rows[:-1]],marker='o')
axes[1,1].set(xlabel='Rank',ylabel='MSE (normalized numerical units squared)',title='Error; full rank is numerical zero')
axes[1,2].plot([q['rank'] for q in rows],[q['factor_bytes'] for q in rows],marker='o',label='U, singular values, V transpose')
axes[1,2].axhline(x.nbytes,label='Dense float64 matrix',color='orange')
axes[1,2].set(xlabel='Rank',ylabel='Raw numeric bytes',title='Storage excludes metadata/compression'); axes[1,2].legend(fontsize=8)
fig.savefig(root/'reports/mp12-svd.png',dpi=120); plt.close(fig)
record={'source':scene,'band':'red (single-band grayscale display)','shape':list(x.shape),
        'license':manifest['license'],'citation':manifest['citation'],'units':manifest['units'],
        'divisor':manifest['divisor'],'dtype':'float64','protocol':'No centering, clipping, rescaling or test data; economy SVD',
        'rows':rows,'limitations':'Low-rank approximation of known HR is not super-resolution or a learned compression codec; file-format overhead excluded.'}
(root/'reports/mp12-svd.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Eight ranks executed; tail energy identity and decreasing MSE verified')
