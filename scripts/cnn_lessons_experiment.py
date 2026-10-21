"""MP19–22 fixed teaching experiments; never alter frozen research selections."""
import json
import time
import statistics
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rssr.cnn_labs import Reconstruction,UpsamplingCNN
from rssr.models import SRModel
from rssr.experiment import load_tiles
from rssr.core import metrics,resize

root=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
records=json.loads((root/'reports/submission-study.json').read_text())
entry=next(r for r in records if r['bands']==3);path=root/entry['manifest_path'].replace('\\','/');manifest=json.loads(path.read_text())
training=load_tiles(manifest,'train',path.parent);validation=load_tiles(manifest,'validation',path.parent)
specs={'identity':lambda:Reconstruction(), 'srcnn_mse':lambda:SRModel(kind='srcnn',bands=3),
       'direct_shuffle':lambda:UpsamplingCNN(residual=False),
       'residual_shuffle':lambda:UpsamplingCNN(),
       'residual_transpose':lambda:UpsamplingCNN('transpose'),
       'residual_interpolate':lambda:UpsamplingCNN('interpolate')}
rows=[];curves={};images={};kernels={}
for name,factory in specs.items():
    torch.manual_seed(19);model=factory();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    generator=torch.Generator().manual_seed(19);history=[];start=time.perf_counter()
    # Small synthetic forward/backward sanity fixture before real training.
    synthetic=torch.rand((1,3,16,16),generator=generator)
    expected=(1,3,16,16) if name=='identity' else (1,3,32,32)
    assert tuple(model(synthetic).shape)==expected
    for step in range(300):
        ids=torch.randint(len(training),(4,),generator=generator).tolist()
        x=torch.stack([training[i][2 if name=='identity' else 1] for i in ids]);y=torch.stack([training[i][2] for i in ids])
        model.train();optimizer.zero_grad();loss=F.mse_loss(model(x),y);loss.backward();optimizer.step()
        if step in [0,49,99,199,299]:history.append({'step':step+1,'train_mse':loss.item()})
    model.eval();values=[];constant=torch.stack([r[2] for r in training]).mean((0,2,3))[:,None,None]
    with torch.inference_mode():
        for item,lr,hr in validation:
            x=hr if name=='identity' else lr;pred=model(x[None])[0]
            baseline=constant.expand_as(hr) if name=='identity' else resize(lr,2)
            values.append({'scene_id':item['scene_id'],'model':metrics(pred,hr,border=2),'baseline':metrics(baseline,hr,border=2)})
        inp=validation[0][2 if name=='identity' else 1][None]
        for _ in range(3):model(inp)
        timings=[]
        for _ in range(10):
            t=time.perf_counter();model(inp);timings.append(time.perf_counter()-t)
        images[name]=model(inp)[0].permute(1,2,0).numpy()
        # Constant response exposes uneven overlap/subpixel phase without claiming a quality metric.
        if name.startswith('residual_'):kernels[name]=model(torch.ones((1,3,8,8)))[0,0].numpy()
    rows.append({'model':name,'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':time.perf_counter()-start,
                 'median_inference_seconds':statistics.median(timings),'validation_psnr':statistics.mean(v['model']['psnr'] for v in values),
                 'baseline_psnr':statistics.mean(v['baseline']['psnr'] for v in values),'per_tile':values})
    curves[name]=history;print(name,rows[-1]['validation_psnr'],flush=True)
fig,axes=plt.subplots(2,4,figsize=(13,7),layout='constrained')
reference=validation[0][2].permute(1,2,0).numpy();lo,hi=np.percentile(reference,[2,98])
for ax,(name,array) in zip(axes.flat,[('Reference',reference),*images.items()]):
    ax.imshow(np.clip((array-lo)/(hi-lo),0,1));ax.set_title(name);ax.axis('off')
for name,history in curves.items():axes[1,3].semilogy([h['step'] for h in history],[h['train_mse'] for h in history],label=name)
axes[1,3].set(xlabel='Training step',ylabel='MSE');axes[1,3].legend(fontsize=6)
fig.savefig(root/'reports/mp19-mp22-cnn.png',dpi=120);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
for ax,(name,array) in zip(axes,kernels.items()):
    ax.imshow(array,cmap='viridis');ax.set_title(name,fontsize=9)
fig.savefig(root/'reports/mp22-phase-response.png',dpi=120);plt.close(fig)
report={'seed':19,'device':'CPU two threads float32','torch':torch.__version__,'optimizer':'Adam','lr':.001,'steps':300,'batch_size':4,
        'loss':'MSE','checkpoint_selection':'final fixed step; not tuned on validation; teaching weights not retained',
        'manifest_path':entry['manifest_path'],'manifest_sha256':entry['manifest_sha256'],'training_scenes':4,'validation_scenes':4,
        'source_license':'NAIP public domain USDA/OpenSR','range':'stored RGB DN/10000; not calibrated reflectance',
        'results':rows,'curves':curves,'limitations':'One teaching seed, unequal parameter counts despite equal update budgets. SRCNN is compact adaptation. Validation table is not a held-out research test; original final checkpoints and evaluation are unchanged. Constant-response panels have independent scales and are qualitative.'}
(root/'reports/mp19-mp22-cnn.json').write_text(json.dumps(report,indent=2)+'\n')
