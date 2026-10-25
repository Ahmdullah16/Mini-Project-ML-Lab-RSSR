"""Pre-normalized educational encoder; sinusoidal positions are optional inputs."""
import math
import torch
from torch import nn


def positions(tokens, width):
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in (tokens,width)) or width%2:
        raise ValueError('Positive token count and even positive width required')
    frequency=torch.exp(torch.arange(0,width,2,dtype=torch.float64)*(-math.log(10000.)/width))
    angles=torch.arange(tokens,dtype=torch.float64)[:,None]*frequency
    result=torch.empty(tokens,width,dtype=torch.float64)
    result[:,0::2]=angles.sin();result[:,1::2]=angles.cos()
    return result


class Encoder(nn.Module):
    """X[B,N,D] -> X+MHA(LN(X)) -> Y+FFN(LN(Y)); dropout0."""
    def __init__(self,width=4,heads=2):
        super().__init__()
        if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in (width,heads)) or width%heads:
            raise ValueError('Positive width divisible by positive head count required')
        self.width=width
        self.norm1=nn.LayerNorm(width,eps=1e-5)
        self.attn=nn.MultiheadAttention(width,heads,batch_first=True,dropout=0.)
        self.norm2=nn.LayerNorm(width,eps=1e-5)
        self.expand=nn.Linear(width,2*width)
        self.activation=nn.GELU()
        self.reduce=nn.Linear(2*width,width)

    def forward(self,x):
        if x.ndim!=3 or x.shape[-1]!=self.width or min(x.shape)==0 or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('Expected finite nonempty floating [batch,tokens,width]')
        normalized=self.norm1(x)
        attended,_=self.attn(normalized,normalized,normalized,need_weights=False)
        residual=x+attended
        hidden=self.activation(self.expand(self.norm2(residual)))
        output=residual+self.reduce(hidden)
        return output,{'input':list(x.shape),'normalized':list(normalized.shape),
                       'attention':list(attended.shape),'residual':list(residual.shape),
                       'ffn_hidden':list(hidden.shape),'output':list(output.shape)}
