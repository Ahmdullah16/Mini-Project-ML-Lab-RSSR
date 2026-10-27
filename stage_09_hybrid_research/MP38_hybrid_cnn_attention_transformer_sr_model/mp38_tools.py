"""MP38 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from rssr.models import SRModel
import torch

def build_hybrid(bands=4,width=8,gate=True,zero_residual=False):
    return SRModel(bands=bands,width=width,kind='hybrid',gate=gate,zero_residual=zero_residual)


def trace_hybrid(model,x):
    """Record actual module shapes without retaining tensors or changing weights.

    Runs in inference mode, restores model train/eval state and removes hooks even
    if forward fails. Input is finite floating NCHW; outputs are not quality scores.
    """
    if (not isinstance(model,SRModel) or model.kind!='hybrid' or x.ndim!=4
        or min(x.shape)<1 or not x.is_floating_point() or not torch.isfinite(x).all()):
        raise ValueError('Hybrid model and nonempty finite floating NCHW input required')
    shapes={};handles=[];training=model.training
    def hook(name):
        def record(module,inputs,output):
            shapes[name]={'input':list(inputs[0].shape),'output':list(output.shape)}
        return record
    try:
        for name in ('head','local','context','fuse','attention','tail','gate'):
            module=getattr(model,name)
            if module is not None:handles.append(module.register_forward_hook(hook(name)))
        model.eval()
        with torch.inference_mode():output=model(x)
        if not torch.isfinite(output).all():raise ValueError('Nonfinite reconstruction')
        return output,shapes
    finally:
        for handle in handles:handle.remove()
        model.train(training)
