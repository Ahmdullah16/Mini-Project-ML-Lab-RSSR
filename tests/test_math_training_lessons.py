"""Canonical math/reproducibility batch; no real satellite data or held-out study."""
import numpy as np
import pytest
import torch
from rssr.linear_lab import approximate
from rssr.training_lab import linear_loss,initialize,mlp_loss,finite_difference_error
from rssr.pipeline_lab import make_loader,weighted_mean,tiny_regression


def test_mp12_svd_tail_energy_rank_boundaries():
    original=np.diag([3.,2.,1.])
    reconstruction,record=approximate(original,1)
    np.testing.assert_allclose(reconstruction,np.diag([3.,0.,0.]))
    assert record["tail_energy"]==5
    assert record["frobenius_error"]==pytest.approx(np.sqrt(5))
    for rank in range(4):
        pred,meta=approximate(original,rank)
        assert np.linalg.norm(original-pred)**2==pytest.approx(meta["tail_energy"])
    assert approximate(original,3)[1]["tail_energy"]==pytest.approx(0,abs=1e-20)
    with pytest.raises(ValueError):approximate(original,4)


def test_mp13_one_step_hand_gradient():
    x=np.array([[1.],[2.]])
    y=np.array([[3.],[5.]])
    w=np.zeros((1,1));b=np.zeros(1)
    loss,g=linear_loss(x,y,w,b)
    assert loss==17
    assert g["w"].item()==-13 and g["b"].item()==-8
    w-=0.1*g["w"];b-=0.1*g["b"]
    assert linear_loss(x,y,w,b)[0]==pytest.approx(1.685)


def test_mp14_finite_difference_and_immutable_inputs():
    x=np.array([[-.7],[.2],[.8]]);y=np.sin(x)
    p=initialize(hidden=4,seed=17)
    frozen={k:v.copy() for k,v in p.items()}
    error=finite_difference_error(x,y,p)
    assert error<1e-7
    for key in p:np.testing.assert_array_equal(p[key],frozen[key])


def test_mp15_autograd_exact_same_weights():
    x=np.array([[-.7],[.2],[.8]])
    y=np.sin(x);p=initialize(hidden=4,seed=17)
    loss,grad=mlp_loss(x,y,p)
    t={k:torch.tensor(v,requires_grad=True) for k,v in p.items()}
    pred=torch.tanh(torch.tensor(x)@t["w1"]+t["b1"])@t["w2"]+t["b2"]
    err=((pred-torch.tensor(y))**2).mean();err.backward()
    assert err.item()==pytest.approx(loss)
    for key in grad:np.testing.assert_allclose(t[key].grad.numpy(),grad[key],atol=1e-12)


def test_mp16_partial_batch_weight_and_replay():
    x=torch.arange(5.,dtype=torch.float64)[:,None]
    first=list(make_loader(x,2*x+1,2,16))
    second=list(make_loader(x,2*x+1,2,16))
    assert [len(x) for x,_ in first]==[2,2,1]
    for a,b in zip(first,second):torch.testing.assert_close(a[0],b[0])
    assert weighted_mean([2.5,1],[2,1])==2
    report=tiny_regression(epochs=3,batch_size=2,seed=16)
    assert report["repeat_identical"] is True
    assert report["train_count"]==5 and report["validation_count"]==2
    assert len(report["history"])==3
    assert report["last_batch_sizes"]==[2,2,1]
    with pytest.raises(ValueError):weighted_mean([1],[0])
    with pytest.raises(ValueError):make_loader(x,x[:2])
    with pytest.raises(ValueError):tiny_regression(epochs=51)
