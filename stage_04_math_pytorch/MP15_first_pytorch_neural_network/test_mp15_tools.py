from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pytest
import torch
from mp15_tools import linear_loss,initialize,mlp_loss,finite_difference_error


def test_linear_hand_derivative():
    loss,g=linear_loss(np.array([[1.],[2.]]),np.array([[3.],[5.]]),np.zeros((1,1)),np.zeros(1))
    assert loss==17 and g['w'][0,0]==-13 and g['b'][0]==-8
    assert linear_loss(np.array([[1.],[2.]]),np.array([[3.],[5.]]),np.array([[1.3]]),np.array([.8]))[0]==pytest.approx(1.685)


def test_chain_rule_matches_numerical_and_autograd():
    x=np.array([[-.7],[.2],[.8]]); y=np.sin(x)
    p=initialize(); saved={k:v.copy() for k,v in p.items()}
    assert finite_difference_error(x,y,p)<1e-9
    for key in p: np.testing.assert_array_equal(p[key],saved[key])
    loss,g=mlp_loss(x,y,p)
    t={k:torch.tensor(v,requires_grad=True) for k,v in p.items()}
    pred=torch.tanh(torch.tensor(x)@t['w1']+t['b1'])@t['w2']+t['b2']
    objective=((pred-torch.tensor(y))**2).mean(); objective.backward()
    assert objective.item()==pytest.approx(loss)
    for key in g: np.testing.assert_allclose(g[key],t[key].grad.numpy(),atol=1e-12)


def test_shapes_and_nonfinite_rejected():
    with pytest.raises(ValueError): mlp_loss(np.ones((3,1)),np.ones((3,2)),initialize())
    with pytest.raises(ValueError): mlp_loss(np.ones((3,1)),np.full((3,1),np.nan),initialize())
