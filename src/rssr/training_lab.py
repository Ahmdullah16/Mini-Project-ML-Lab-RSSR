"""Small NumPy regression primitives with explicit sample-by-feature shapes."""
import numpy as np


def regression_arrays(x,y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or min(*x.shape,*y.shape)<1:
        raise ValueError('Nonempty N-by-D inputs and N-by-O targets required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite regression arrays required')
    return x,y


def linear_loss(x,y,w,b):
    x,y=regression_arrays(x,y)
    pred=x@w+b
    if pred.shape!=y.shape: raise ValueError('Prediction and target shapes differ')
    residual=pred-y; delta=2*residual/residual.size
    return float(np.mean(residual**2)),{'w':x.T@delta,'b':delta.sum(0)}


def initialize(inputs=1,hidden=4,outputs=1,seed=17):
    rng=np.random.default_rng(seed)
    return {'w1':rng.normal(0,.3,(inputs,hidden)),'b1':np.zeros(hidden),
            'w2':rng.normal(0,.3,(hidden,outputs)),'b2':np.zeros(outputs)}


def forward(x,p):
    return np.tanh(x@p['w1']+p['b1'])@p['w2']+p['b2']


def mlp_loss(x,y,p):
    x,y=regression_arrays(x,y)
    hidden=np.tanh(x@p['w1']+p['b1'])
    pred=hidden@p['w2']+p['b2']
    if pred.shape!=y.shape: raise ValueError('Prediction and target shapes differ')
    residual=pred-y; delta=2*residual/residual.size
    dh=(delta@p['w2'].T)*(1-hidden**2)
    return float(np.mean(residual**2)),{'w1':x.T@dh,'b1':dh.sum(0),
                                      'w2':hidden.T@delta,'b2':delta.sum(0)}


def finite_difference_error(x,y,p,epsilon=1e-6):
    """Central differences for every parameter; restores the parameter afterward."""
    if not np.isfinite(epsilon) or epsilon<=0: raise ValueError('Positive finite epsilon required')
    _,grad=mlp_loss(x,y,p); maximum=0.
    for name,value in p.items():
        for index in np.ndindex(value.shape):
            old=value[index]
            try:
                value[index]=old+epsilon; plus=mlp_loss(x,y,p)[0]
                value[index]=old-epsilon; minus=mlp_loss(x,y,p)[0]
            finally: value[index]=old
            maximum=max(maximum,abs((plus-minus)/(2*epsilon)-grad[name][index]))
    return maximum
