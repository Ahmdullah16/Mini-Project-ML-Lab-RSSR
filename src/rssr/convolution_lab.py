"""Single-channel cross-correlation and receptive-field geometry for MP17."""
import numpy as np


def integer(value, minimum, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def output_size(size, kernel, stride=1, padding=0, dilation=1):
    size=integer(size,1,'size'); kernel=integer(kernel,1,'kernel')
    stride=integer(stride,1,'stride'); padding=integer(padding,0,'padding')
    dilation=integer(dilation,1,'dilation')
    result=(size+2*padding-dilation*(kernel-1)-1)//stride+1
    if result<1: raise ValueError('Effective kernel exceeds padded input')
    return result


def correlate(image, kernel, stride=1, padding=0, dilation=1):
    """HW float64, zero padding, no kernel reversal (PyTorch convention)."""
    x=np.asarray(image,dtype=np.float64); k=np.asarray(kernel,dtype=np.float64)
    if x.ndim!=2 or k.ndim!=2 or min(*x.shape,*k.shape)<1:
        raise ValueError('Nonempty 2D input and kernel required')
    if not np.isfinite(x).all() or not np.isfinite(k).all():
        raise ValueError('Finite values required')
    h=output_size(x.shape[0],k.shape[0],stride,padding,dilation)
    w=output_size(x.shape[1],k.shape[1],stride,padding,dilation)
    padded=np.pad(x,padding)
    result=np.empty((h,w))
    for row in range(h):
        for col in range(w):
            patch=padded[row*stride:row*stride+dilation*(k.shape[0]-1)+1:dilation,
                         col*stride:col*stride+dilation*(k.shape[1]-1)+1:dilation]
            result[row,col]=np.sum(patch*k)
    return result


def receptive_field(layers):
    """Square layer tuples (kernel, stride, dilation); bounding span and jump.

    Span is not the count of sampled pixels: dilation can leave holes.
    Padding changes alignment/boundary overlap, not this interior span.
    """
    field,jump=1,1
    for kernel,stride,dilation in layers:
        kernel=integer(kernel,1,'kernel'); stride=integer(stride,1,'stride')
        dilation=integer(dilation,1,'dilation')
        field+=(kernel-1)*dilation*jump
        jump*=stride
    return {'span':field,'jump':jump}
