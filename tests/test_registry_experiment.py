"""One optimizer step on a synthetic ramp: integration, never quality evidence."""
import json
import sqlite3
import numpy as np
import pytest
import torch
from rssr.core import digest, degrade
from rssr.experiment import train, load_model, evaluate
from rssr.model_factory import from_config
from rssr.architecture_registry import REGISTRY


@pytest.mark.parametrize('key', list(REGISTRY))
def test_registry_training_reload_evaluation_provenance(tmp_path,key):
    tiles=[]
    for i, split in enumerate(['train','validation']):
        hr=torch.linspace(0,1,16).repeat(3,16,1)
        path=tmp_path/f'{i}.npz'
        np.savez(path,hr=hr.numpy(),lr=degrade(hr).numpy())
        tiles.append({'id':str(i),'scene_id':'synthetic-ramp-not-geography','split':split,
                      'path':str(path),'sha256':digest(path)})
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'bands':['r','g','b'],'scale':2,'tiles':tiles}))
    config={'seed':74,'steps':1,'batch_size':1,'lr':.001,'validate_every':1,
            'model':{'architecture_id':key,'width':4,'bands':3,'scale':2}}
    output=train(config,manifest,tmp_path)
    model,state=load_model(output/'best.pt')
    assert model.architecture_id == key and state['selected_step'] == 1
    assert state['config'] == config
    report=json.loads((output/'training.json').read_text())
    assert np.isfinite(report['history'][0]['validation_l1'])
    with sqlite3.connect(tmp_path/'catalog.sqlite') as database:
        for tile in tiles:
            database.execute('INSERT INTO pairs_or_tiles(id,split,lr_path,hr_path,scale) VALUES(?,?,?,?,?)',
                             (tile['id'],tile['split'],tile['path'],tile['path'],2))
    result=json.loads(evaluate(output/'best.pt',manifest,tmp_path).read_text())
    assert {r['method'] for r in result} == {'nearest','bilinear','bicubic','model'}
    with sqlite3.connect(tmp_path/'catalog.sqlite') as database:
        row=database.execute('SELECT model,status,checkpoint_hash FROM runs').fetchone()
    assert row == (key,'completed',digest(output/'best.pt'))


def test_ambiguous_config_rejected():
    with pytest.raises(ValueError,match='not both'):
        from_config({'architecture_id':'cat-v1/M1','kind':'cnn'})
