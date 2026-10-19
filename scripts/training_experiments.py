"""MP13–15 regression, chain-rule checks and PyTorch equivalence experiments."""
import json
from pathlib import Path
import numpy as np
import torch
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rssr.core import digest
from rssr.training_lab import linear_loss,initialize,mlp_loss,forward,finite_difference_error

root=Path(__file__).resolve().parents[1]; torch.set_num_threads(2)
x=np.linspace(-1,1,32)[:,None]; y=2*x+.5
linear=[]
for rate in [.001,.1,1.1]:
    w=np.zeros((1,1)); b=np.zeros(1); history=[]
    for step in range(60):
        loss,g=linear_loss(x,y,w,b); history.append(loss)
        w-=rate*g['w']; b-=rate*g['b']
    linear.append({'learning_rate':rate,'loss_before_updates':history,'final_loss':linear_loss(x,y,w,b)[0],
                   'weight':w.item(),'bias':b.item()})

toy_x=np.linspace(-1,1,64)[:,None]; toy_y=np.sin(3*toy_x)
train_ids=np.arange(0,64,2); val_ids=np.arange(1,64,2)
p=initialize(hidden=8); q={k:torch.tensor(v,requires_grad=True) for k,v in p.items()}
grad_error=finite_difference_error(toy_x[train_ids],toy_y[train_ids],p)
optimizer=torch.optim.SGD(list(q.values()),lr=.05)
history=[]
for step in range(1000):
    loss,g=mlp_loss(toy_x[train_ids],toy_y[train_ids],p)
    for key in p: p[key]-=.05*g[key]
    optimizer.zero_grad(set_to_none=True)
    tx=torch.tensor(toy_x[train_ids]); ty=torch.tensor(toy_y[train_ids])
    pred=torch.tanh(tx@q['w1']+q['b1'])@q['w2']+q['b2']
    objective=((pred-ty)**2).mean(); objective.backward(); optimizer.step()
    if step%20==0 or step==999:
        history.append({'step':step+1,'training_mse':mlp_loss(toy_x[train_ids],toy_y[train_ids],p)[0],
                        'validation_mse':mlp_loss(toy_x[val_ids],toy_y[val_ids],p)[0]})
parity=max(float(np.abs(p[k]-q[k].detach().numpy()).max()) for k in p)
assert grad_error<1e-8 and parity<1e-10
torch.manual_seed(17); dropout=torch.nn.Dropout(.5); inputs=torch.ones(64)
dropout.train(); training=dropout(inputs)
dropout.eval(); evaluation=dropout(inputs)
assert torch.equal(evaluation,inputs) and not torch.equal(training,inputs)

# A real-data exercise: red-to-NIR numerical regression, explicitly not SR.
manifest=json.loads((root/'configs/naip-pilot-manifest.json').read_text())
real={}; provenance=[]
for split in ['train','validation']:
    scene=next(s for s in manifest['scenes'] if s['split']==split)
    path=root/'data'/scene['path']; assert digest(path)==scene['sha256']
    with rasterio.open(path) as ds:
        window=rasterio.windows.Window(0,0,64,64)
        a=ds.read([1,4],window=window).astype(float)/manifest['divisor']
        valid=(ds.read_masks([1,4],window=window)>0).all(0)
    features=a[0][valid][::16,None]; targets=a[1][valid][::16,None]
    real[split]=(features,targets); provenance.append(scene)
params=initialize(); before=mlp_loss(*real['train'],params)[0]
for _ in range(1000):
    _,g=mlp_loss(*real['train'],params)
    for key in params: params[key]-=.05*g[key]
mean_target=float(real['train'][1].mean())
real_report={'source_scenes':provenance,'sampling':'first 64x64 crop, valid pixels, every sixteenth sample',
             'license':manifest['license'],'citation':manifest['citation'],'divisor':manifest['divisor'],'units':manifest['units'],
             'task':'red-to-NIR regression in fixed numerical units; not super-resolution or calibrated spectral prediction',
             'initial_training_mse':before,'training_mse':mlp_loss(*real['train'],params)[0],
             'validation_mse':mlp_loss(*real['validation'],params)[0],
             'training_mean_baseline_validation_mse':float(np.mean((real['validation'][1]-mean_target)**2))}
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for row in linear: axes[0].semilogy(row['loss_before_updates'],label=str(row['learning_rate']))
axes[0].set(xlabel='Update count',ylabel='MSE',title='Linear learning-rate comparison'); axes[0].legend(title='Learning rate')
axes[1].plot(toy_x,toy_y,label='target sin(3x)'); axes[1].plot(toy_x,forward(toy_x,p),label='tanh network'); axes[1].legend(); axes[1].set(title='Toy regression',xlabel='x',ylabel='y')
for key in ['training_mse','validation_mse']: axes[2].semilogy([h['step'] for h in history],[h[key] for h in history],label=key)
axes[2].set(title='Fixed 1000-step training',xlabel='Step',ylabel='MSE'); axes[2].legend()
fig.savefig(root/'reports/mp13-mp15-training.png',dpi=120); plt.close(fig)
report={'seed':17,'dtype':'float64','device':'cpu','optimizer':'full-batch SGD, constant rate .05, no weight decay',
        'linear':linear,'toy_split':{'train_indices':train_ids.tolist(),'validation_indices':val_ids.tolist(),'test_used':False},
        'finite_difference_max_absolute_error':grad_error,'numpy_torch_final_parameter_max_difference':parity,
        'toy_history':history,'dropout':{'training_zero_count':int((training==0).sum()),'evaluation_identity':True},
        'real_regression':real_report,'limitations':'Toy interpolation split is not geographic independence. Real exercise uses distinct declared train/validation scenes. No held-out test or SR claim.'}
(root/'reports/mp13-mp15-training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'gradient_error':grad_error,'numpy_torch_difference':parity,'real':real_report},indent=2))
