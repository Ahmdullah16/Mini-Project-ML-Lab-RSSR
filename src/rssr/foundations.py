"""Small from-scratch numerical examples for the first four learning stages."""
import numpy as np


def normalize_u8(x):
    if np.asarray(x).dtype != np.uint8:
        raise ValueError('Expected uint8')
    return np.asarray(x,dtype=np.float32)/255


def gaussian_kernel(sigma=1.,radius=2):
    if sigma<=0 or radius<0:
        raise ValueError('Positive sigma, nonnegative radius required')
    t=np.arange(-radius,radius+1); k=np.exp(-(t[:,None]**2+t[None,:]**2)/(2*sigma*sigma))
    return k/k.sum()


def correlate(image,kernel):
    image=np.asarray(image,dtype=float); kernel=np.asarray(kernel,dtype=float)
    if image.ndim!=2 or kernel.ndim!=2 or any(n%2!=1 for n in kernel.shape):
        raise ValueError('2D image and odd kernel dimensions required')
    pad=np.pad(image,[(kernel.shape[0]//2,)*2,(kernel.shape[1]//2,)*2],mode='edge')
    windows=np.lib.stride_tricks.sliding_window_view(pad,kernel.shape)
    return np.einsum('ijab,ab->ij',windows,kernel)


def softmax(x):
    x=np.asarray(x,dtype=float); y=np.exp(x-x.max(axis=-1,keepdims=True))
    return y/y.sum(axis=-1,keepdims=True)


def attention(q,k,v):
    q,k,v=map(lambda a:np.asarray(a,dtype=float),(q,k,v))
    if q.ndim!=2 or k.ndim!=2 or v.ndim!=2 or q.shape[1]!=k.shape[1] or k.shape[0]!=v.shape[0]:
        raise ValueError('Expected Q:Nxd K:Mxd V:Mxe')
    weights=softmax(q@k.T/np.sqrt(q.shape[1]))
    return weights@v,weights


def gradient_descent(start=0.,target=3.,rate=.1,steps=20):
    w=float(start); history=[w]
    for _ in range(steps):
        w-=rate*2*(w-target); history.append(w)
    return np.array(history)


def paired_scene_ci(a,b,seed=0,repeats=1000):
    """Bootstrap independent scene means, never pixels. Return mean(a-b),95% CI."""
    a,b=np.asarray(a),np.asarray(b)
    if a.ndim!=1 or a.shape!=b.shape or len(a)<2 or not np.isfinite(a-b).all():
        raise ValueError('At least two paired finite scene means required')
    d=a-b; ids=np.random.default_rng(seed).integers(len(d),size=(repeats,len(d)))
    return float(d.mean()),np.quantile(d[ids].mean(1),[.025,.975])
