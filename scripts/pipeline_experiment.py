"""MP16: seeded DataLoader regression and exact production checkpoint recovery."""
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from rssr.core import digest, degrade
from rssr.experiment import train

root=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
def regression(seed):
    torch.manual_seed(seed)
    x=torch.linspace(-1,1,16)[:,None]; y=2*x+1
    loader=DataLoader(TensorDataset(x,y),batch_size=4,shuffle=True,
                      generator=torch.Generator().manual_seed(seed),num_workers=0)
    model=torch.nn.Linear(1,1); optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    losses=[]
    for epoch in range(100):
        model.train(); total=0.; count=0
        for bx,by in loader:
            optimizer.zero_grad(); loss=((model(bx)-by)**2).mean()
            loss.backward(); optimizer.step(); total+=loss.item()*len(bx); count+=len(bx)
        losses.append(total/count)
    model.eval()
    with torch.inference_mode():
        valx=torch.linspace(-.95,.95,15)[:,None]
        val=float(((model(valx)-(2*valx+1))**2).mean())
    return model.state_dict(),losses,val

a,losses,val=regression(16); b,_,_=regression(16)
assert all(torch.equal(a[k],b[k]) for k in a)
assert losses[-1]<1e-10 and val<1e-10
with tempfile.TemporaryDirectory(prefix='rssr-mp16-') as folder:
    temp=Path(folder); tiles=[]
    for i,split in enumerate(['train','validation']):
        hr=torch.linspace(0,1,16).repeat(3,16,1); p=temp/f'{i}.npz'
        np.savez(p,hr=hr.numpy(),lr=degrade(hr).numpy())
        tiles.append({'id':str(i),'split':split,'path':p.name,'sha256':digest(p)})
    manifest=temp/'manifest.json'
    manifest.write_text(json.dumps({'bands':['r','g','b'],'scale':2,'tiles':tiles}))
    config={'seed':16,'steps':8,'batch_size':1,'lr':.001,'validate_every':2,
            'model':{'kind':'cnn','width':8,'bands':3,'scale':2}}
    full=train(config,manifest,temp); half=train(dict(config,steps=4),manifest,temp)
    resumed=train(config,manifest,temp,half/'last.pt')
    first=torch.load(full/'last.pt',weights_only=True)
    second=torch.load(resumed/'last.pt',weights_only=True)
    exact=all(torch.equal(first['model'][k],second['model'][k]) for k in first['model'])
    assert exact and first['best']==second['best']
    rejected=False
    try: train(config,manifest,temp,half/'best.pt')
    except ValueError: rejected=True
    assert rejected
report={'seed':16,'device':'cpu','torch':torch.__version__,'regression':{
    'training_examples':16,'validation_examples':15,'epochs':100,'batch_size':4,
    'optimizer':'SGD','learning_rate':.1,'num_workers':0,
    'initial_epoch_mse':losses[0],'final_epoch_mse':losses[-1],
    'validation_mse':val,'seeded_parameters_equal':True},
    'production_resume':{'steps':8,'interruption_step':4,'parameters_exact':exact,
                         'validation_best_equal':True,'inference_checkpoint_rejected':rejected},
    'real_data_evidence':'geographic-study.json',
    'limitations':'Synthetic resume fixture shares a ramp across train/validation and tests mechanics only, not generalization. Separate real geographic train/validation evidence is linked; no new real training or held-out evaluation here.'}
(root/'reports/mp16-pipeline.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
