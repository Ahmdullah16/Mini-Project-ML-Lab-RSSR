"""Bias-free educational multi-head attention using the checked MP27 primitive."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'MP27_self_attention_from_scratch'))
from attention_tools import attention, matrix


def multihead(tokens, projections, heads):
    """X[N,D], four [D,D] matrices -> Y[N,D], attention[heads,N,N]."""
    x=matrix(tokens,'tokens'); width=x.shape[1]
    if isinstance(heads,bool) or not isinstance(heads,int) or heads<=0 or width%heads:
        raise ValueError('Heads must be a positive integer divisor of feature width')
    if len(projections)!=4:
        raise ValueError('Provide Q, K, V and output matrices')
    weights=[matrix(w,'projection') for w in projections]
    if any(w.shape!=(width,width) for w in weights):
        raise ValueError('Every projection must have shape [D,D]')
    with np.errstate(over='ignore',invalid='ignore'):
        q,k,v=(x@w for w in weights[:3])
    size=width//heads; outputs=[]; maps=[]
    for head in range(heads):
        sl=slice(head*size,(head+1)*size)
        out,probability=attention(q[:,sl],k[:,sl],v[:,sl])
        outputs.append(out);maps.append(probability)
    with np.errstate(over='ignore',invalid='ignore'):
        result=np.concatenate(outputs,axis=1)@weights[3]
    if not np.isfinite(result).all():raise ValueError('Nonfinite output projection')
    return result,np.stack(maps)


def projection_parameters(width):
    if isinstance(width,bool) or not isinstance(width,int) or width<=0:
        raise ValueError('Width must be a positive integer')
    return 4*width*width
