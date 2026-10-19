"""MP09–11: controlled distortions, SSIM maps and static evaluation dashboard."""
import json
from pathlib import Path
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rssr.core import digest
from rssr.degradation_lab import gaussian, filter_hwc
from rssr.metric_lab import mse_psnr, ssim_map

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'configs/naip-pilot-manifest.json').read_text())
scene=next(s for s in manifest['scenes'] if s['split']=='train')
source=root/'data'/scene['path']
assert digest(source)==scene['sha256']
with rasterio.open(source) as ds:
    window=rasterio.windows.Window(0,0,64,64)
    x=ds.read(window=window).transpose(1,2,0).astype(float)/manifest['divisor']
    assert (ds.read_masks(window=window)>0).all()
rng=np.random.default_rng(17)
cases={'identical':x.copy(),'blur':filter_hwc(x,gaussian(1.,3)),
       'contrast':(x-x.mean((0,1)))*.8+x.mean((0,1)),
       'noise 0.01':x+rng.normal(0,.01,x.shape)}
noise=rng.normal(size=x.shape)
noise*=np.sqrt(np.mean((cases['blur']-x)**2)/np.mean(noise**2))
cases['matched-MSE noise']=x+noise
rows=[]
fig,axes=plt.subplots(3,len(cases),figsize=(16,8),layout='constrained')
error_max=max(np.abs(y-x).mean(2).max() for y in cases.values())
for col,(name,y) in enumerate(cases.items()):
    mse,psnr=mse_psnr(y,x)
    smap=ssim_map(y,x)
    rows.append({'case':name,'mse':mse,'psnr_db':psnr if np.isfinite(psnr) else 'positive infinity',
                 'ssim':float(smap.mean()),'rmse_by_band':np.sqrt(((y-x)**2).mean((0,1))).tolist()})
    # Fixed crop-specific display range only; metrics always use unmodified values.
    axes[0,col].imshow(y[:,:,0],cmap='gray',vmin=float(x[:,:,0].min()),vmax=float(x[:,:,0].max()))
    axes[0,col].set_title(name+' (red band)')
    axes[1,col].imshow(np.abs(y-x).mean(2),vmin=0,vmax=error_max,cmap='magma')
    axes[1,col].set_title('Mean absolute band error')
    axes[2,col].imshow(smap.mean(2),vmin=-1,vmax=1,cmap='viridis')
    axes[2,col].set_title(f'SSIM {smap.mean():.4f} (valid windows)')
fig.savefig(root/'reports/mp09-mp11-dashboard.png',dpi=120)
plt.close(fig)
range_table=[]
for std in [.005,.01,.02]:
    y=x+np.random.default_rng(17).normal(0,std,x.shape)
    range_table.append({'noise_std':std,**{f'psnr_range_{v}':mse_psnr(y,x,v)[1] for v in [1.,255.]}})
record={'source':scene,'bands':manifest['bands'],'units':manifest['units'],'window':[0,0,64,64],
        'protocol':{'range':1.,'border':0,'ssim_window':7,'ssim_moments':'population','padding':'none; valid windows only','aggregation':'equal weight to all bands and valid windows','clipping':False,'seed':17},
        'rows':rows,'range_sensitivity':range_table,
        'aggregation_example':{'per_image_mse':[.01,.0001],'mean_image_psnr_db':30.,'psnr_of_mean_mse_db':float(-10*np.log10(.00505))},
        'limitations':'One training crop and controlled distortions; no model ranking or sensor-quality claim. Incorrect range 255 is included only as an error demonstration.'}
(root/'reports/mp09-mp11-experiments.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2))
