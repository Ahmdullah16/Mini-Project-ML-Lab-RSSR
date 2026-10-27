"""MP37 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import torch

def fuse_features(local,context,mode='mean'):
    if (local.ndim!=4 or local.shape!=context.shape or min(local.shape)<1
        or not local.is_floating_point() or local.dtype!=context.dtype or local.device!=context.device
        or not torch.isfinite(local).all() or not torch.isfinite(context).all()):
        raise ValueError('Matching nonempty finite floating NCHW features on same device required')
    if mode=='mean':return (local+context)/2
    if mode=='sum':return local+context
    if mode=='concat':return torch.cat([local,context],dim=1)
    raise ValueError('Choose sum, mean or concat; channel projection is separate')


def projection_cost(channels,height,width,batch=1):
    """Dense biased1x1,2C->C: parameters and multiply-accumulate pairs only.

    Excludes bias additions, concatenation copies, branch costs and activations.
    A MAC is one multiply plus accumulate; it is not reported as a FLOP.
    """
    if any(type(v) is not int or v<1 for v in (channels,height,width,batch)):
        raise ValueError('Positive integer dimensions required')
    return {'parameters':2*channels*channels+channels,
            'macs':batch*height*width*2*channels*channels}


def initialize_mean_projection(layer):
    """Initialize a biased1x1 projection to [I/2,I/2], bias0, in place."""
    if (not isinstance(layer,torch.nn.Conv2d) or layer.kernel_size!=(1,1)
        or layer.in_channels!=2*layer.out_channels or layer.groups!=1
        or layer.stride!=(1,1) or layer.padding!=(0,0)):
        raise ValueError('Expected dense2C->C1x1 unit-stride projection')
    with torch.no_grad():
        layer.weight.zero_()
        if layer.bias is not None:layer.bias.zero_()
        c=layer.out_channels
        indices=torch.arange(c,device=layer.weight.device)
        layer.weight[indices,indices,0,0]=.5
        layer.weight[indices,indices+c,0,0]=.5
    return layer
