"""MP40 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import time
import numpy as np
import torch

def benchmark_cpu(model,image,warmups=2,repeats=5):
    if any(isinstance(v,bool) or not isinstance(v,int) for v in (warmups,repeats)) or warmups<0 or repeats<1:raise ValueError('Nonnegative warmups and positive repetitions required')
    if image.device.type!='cpu' or any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('This timer is CPU only')
    was_training=model.training
    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(warmups):model(image)
            times=[]
            for _ in range(repeats):
                start=time.perf_counter();model(image);times.append(time.perf_counter()-start)
        return {'seconds':times,'median_seconds':float(np.median(times)),'warmups':warmups,'repeats':repeats,'scope':'CPU forward only; no IO; not peak memory'}
    finally:model.train(was_training)
