import json
import numpy as np
import torch
import pytest
from rssr.core import digest,degrade
from rssr.experiment import train,load_model


def test_resume_matches_uninterrupted(tmp_path):
    tiles=[]
    for i,split in enumerate(['train','validation']):
        hr=torch.linspace(0,1,16).repeat(3,16,1)
        p=tmp_path/f'{i}.npz'; np.savez(p,hr=hr.numpy(),lr=degrade(hr).numpy())
        tiles.append({'id':str(i),'split':split,'path':str(p),'sha256':digest(p)})
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps({'bands':['r','g','b'],'scale':2,'tiles':tiles}))
    config={'seed':3,'steps':4,'batch_size':1,'lr':.001,'validate_every':1,'model':{'kind':'cnn','width':8,'bands':3,'scale':2}}
    full=train(config,manifest,tmp_path)
    half=train(dict(config,steps=2),manifest,tmp_path)
    resumed=train(config,manifest,tmp_path,half/'last.pt')
    a=torch.load(full/'last.pt',weights_only=True); b=torch.load(resumed/'last.pt',weights_only=True)
    for k in a['model']:
        assert torch.equal(a['model'][k],b['model'][k]),k
    assert a['best']==b['best']
    assert load_model(resumed/'best.pt')[1]['selected_step']==a['best_step']
    with pytest.raises(ValueError): train(config,manifest,tmp_path,half/'best.pt')
