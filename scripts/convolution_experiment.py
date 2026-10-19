"""MP17 impulse geometry and licensed real-crop filtering, no training."""
import json
from pathlib import Path
import numpy as np
import torch
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rssr.core import digest
from rssr.convolution_lab import correlate, output_size, receptive_field

root=Path(__file__).resolve().parents[1]
impulse=np.zeros((9,9)); impulse[4,4]=1
kernel=np.arange(1,10,dtype=float).reshape(3,3)
settings=[(1,0,1),(1,1,1),(2,1,1),(1,2,2)]
rows=[]; fig,axes=plt.subplots(2,4,figsize=(13,7),layout='constrained')
for ax,(stride,padding,dilation) in zip(axes[0],settings):
    y=correlate(impulse,kernel,stride,padding,dilation)
    expected=torch.nn.functional.conv2d(torch.tensor(impulse)[None,None],torch.tensor(kernel)[None,None],stride=stride,padding=padding,dilation=dilation)[0,0].numpy()
    assert np.array_equal(y,expected)
    rows.append({'stride':stride,'padding':padding,'dilation':dilation,'output_shape':list(y.shape),'nonzero_pixels':int(np.count_nonzero(y)),'torch_max_error':float(np.abs(y-expected).max())})
    ax.imshow(y,cmap='viridis',vmin=0,vmax=9); ax.set_title(f's={stride}, p={padding}, d={dilation}\nshape={y.shape}')
manifest=json.loads((root/'configs/naip-pilot-manifest.json').read_text())
scene=next(s for s in manifest['scenes'] if s['split']=='train')
source=root/'data'/scene['path']; assert digest(source)==scene['sha256']
with rasterio.open(source) as ds:
    window=rasterio.windows.Window(0,0,64,64)
    x=ds.read(1,window=window).astype(float)/manifest['divisor']
    assert (ds.read_masks(1,window=window)>0).all()
kernels={'identity':np.array([[1.]]),'mean':np.ones((3,3))/9,
         'horizontal derivative':np.array([[-1,0,1],[-2,0,2],[-1,0,1]])/8}
real=[]
for ax,(name,k) in zip(axes[1],kernels.items()):
    y=correlate(x,k,padding=k.shape[0]//2)
    real.append({'kernel':name,'minimum':float(y.min()),'maximum':float(y.max()),'mse_vs_input':float(((y-x)**2).mean())})
    if name=='horizontal derivative': ax.imshow(y,cmap='coolwarm',vmin=-abs(y).max(),vmax=abs(y).max())
    else: ax.imshow(y,cmap='gray',vmin=x.min(),vmax=x.max())
    ax.set_title(name+'; zero boundary')
# Gradients reveal exactly which inputs influence one interior output.
z=torch.zeros((1,1,15,15),dtype=torch.float64,requires_grad=True)
a=torch.nn.functional.conv2d(z,torch.ones((1,1,3,3),dtype=z.dtype),stride=2)
b=torch.nn.functional.conv2d(a,torch.ones((1,1,3,3),dtype=z.dtype),dilation=2)
b[0,0,1,1].backward(); mask=z.grad[0,0].numpy()!=0
coords=np.argwhere(mask); span=(coords.max(0)-coords.min(0)+1).tolist()
geometry=receptive_field([(3,2,1),(3,1,2)]); assert span==[geometry['span']]*2
axes[1,3].imshow(mask,cmap='gray',vmin=0,vmax=1); axes[1,3].set_title(f'Receptive field: span {span}\n{mask.sum()} contributing pixels')
fig.savefig(root/'reports/mp17-convolution.png',dpi=120); plt.close(fig)
report={'convention':'HW float64 cross-correlation; kernels not flipped; zero padding; CPU',
        'impulses':rows,'receptive_field':dict(geometry,measured_span=span,contributing_pixels=int(mask.sum())),
        'real_source':scene,'license':'NAIP public domain; USDA/OpenSR attribution',
        'window':[0,0,64,64],'band':'red','scaling':'stored DN / 10000; not calibrated reflectance',
        'real_filters':real,'limitations':'Fixed filtering demonstration only; edge padding biases border pixels. No learned model, SR quality claim or held-out data.'}
(root/'reports/mp17-convolution.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
