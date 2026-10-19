"""MP05–08 fixed training-crop experiments; all predictions trace to one source."""
import json
from pathlib import Path
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rssr.core import digest
from rssr.degradation_lab import gaussian,filter_hwc,degrade_hwc,spectrum
root=Path(__file__).resolve().parents[1]; m=json.loads((root/'configs/naip-pilot-manifest.json').read_text()); s=next(s for s in m['scenes'] if s['split']=='train'); p=root/'data'/s['path']; assert digest(p)==s['sha256']
with rasterio.open(p) as ds:
    window=rasterio.windows.Window(0,0,64,64); x=ds.read(window=window).transpose(1,2,0).astype(float)/m['divisor']; assert (ds.read_masks(window=window)>0).all()
impulse=np.zeros((31,31,1)); impulse[15,15]=1
edge=np.zeros_like(impulse); edge[:,15:]=1
rows=[]; fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
for col,(sigma,radius) in enumerate([(0.5,2),(1.,3),(2.,6)]):
    k=gaussian(sigma,radius); y=filter_hwc(x,k)
    axes[0,col].imshow(k); axes[0,col].set_title(f'sigma={sigma}, radius={radius}')
    axes[1,col].plot(filter_hwc(edge,k)[15,:,0],label='edge response'); axes[1,col].plot(filter_hwc(impulse,k)[15,:,0],label='impulse'); axes[1,col].legend()
    rows.append({'sigma':sigma,'radius':radius,'kernel_sum':float(k.sum()),'training_crop_mse_by_band':((y-x)**2).mean((0,1)).tolist(),'boundary_constant_corner':float(filter_hwc(np.ones_like(edge),k,'constant')[0,0,0]),'boundary_edge_corner':float(filter_hwc(np.ones_like(edge),k,'edge')[0,0,0])})
fig.savefig(root/'reports/mp05-kernels.png',dpi=120); plt.close(fig)
pairs=[]; out=root/'data/teaching-pairs'; out.mkdir(exist_ok=True)
for scale in (2,4):
 y=x[::scale,::scale]; target=out/f'x{scale}-decimation.npz'; np.savez(target,hr=x,lr=y)
 pairs.append({'scale':scale,'sampling':'unfiltered decimation, phase zero','shape':list(y.shape),'path':str(target.relative_to(root)).replace('\\','/'),'sha256':digest(target)})
 for sigma in (0.,.8):
  for noise in (0.,.01):
    y=degrade_hwc(x,scale,sigma,noise,17); target=out/f'x{scale}-sigma{sigma}-noise{noise}.npz'; np.savez(target,hr=x,lr=y)
    pairs.append({'scale':scale,'sampling':'area average after optional Gaussian','sigma':sigma,'noise_std':noise,'seed':17,'shape':list(y.shape),'path':str(target.relative_to(root)).replace('\\','/'),'sha256':digest(target),'finite':bool(np.isfinite(y).all())})
fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
images=[('HR',x[:,:,0]),('blurred',filter_hwc(x,gaussian(2,6))[:,:,0]),('noisy',x[:,:,0]+np.random.default_rng(17).normal(0,.01,x.shape[:2]))]
frequency=[]
for col,(name,image) in enumerate(images):
    f,log=spectrum(image); error=float(np.max(np.abs(np.fft.ifft2(f).real-image)))
    frequency.append({'name':name,'inverse_max_error':error,'spatial_energy':float((image**2).sum()),'frequency_energy_normalized':float((np.abs(f)**2).sum()/image.size)})
    axes[0,col].imshow(image,vmin=0,vmax=.3,cmap='gray'); axes[0,col].set_title(name)
    axes[1,col].imshow(log,vmin=0,vmax=3,cmap='magma'); axes[1,col].set_title('log(1+|DFT|), shifted')
fig.savefig(root/'reports/mp08-frequency.png',dpi=120); plt.close(fig)
record={'source_sha256':s['sha256'],'source_scene':s['id'],'split':'train','crop_xywh':[0,0,64,64],'units':m['units'],'mp05':rows,'mp06_mp07_pairs':pairs,'mp08':frequency,'limitations':'One crop; discrete synthetic forward operator, not a sensor-calibrated PSF. Gaussian noise is unbounded and not clipped; spectrum uses periodic boundary assumption.'}
(root/'reports/mp05-mp08-experiments.json').write_text(json.dumps(record,indent=2))
print('Executed kernel sweep, ten deterministic degradation pairs and Fourier checks')
