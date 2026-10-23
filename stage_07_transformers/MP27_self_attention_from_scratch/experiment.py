"""Bounded MP27 experiment. Reads one existing training scene; no downloads."""
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import rasterio
from rasterio.windows import Window
from attention_tools import attention, entropy, score_bytes


def run(root):
    root = Path(root); started = time.perf_counter()
    manifest = json.loads((root/'configs/naip-rgb-development-manifest.json').read_text())
    scene = next(s for s in manifest['scenes'] if s['split'] == 'train')
    source = root/'data'/scene['path']
    if not source.is_file():
        raise FileNotFoundError('Rebuild frozen data using docs/frozen-reproduction.md; no automatic download')
    if hashlib.sha256(source.read_bytes()).hexdigest() != scene['sha256']:
        raise ValueError('Training scene checksum mismatch')
    with rasterio.open(source) as ds:
        window = Window(ds.width//2-4, ds.height//2-4, 8, 8)
        pixels = ds.read([1, 2, 3], window=window).astype(np.float64)/manifest['divisor']
        if not (ds.read_masks([1, 2, 3], window=window)>0).all():
            raise ValueError('Chosen training crop contains invalid pixels')
    tokens = pixels.reshape(3, -1).T
    rng = np.random.default_rng(27)
    fixtures = {'synthetic_normal':rng.normal(size=(64, 3)), 'naip_train_rgb':tokens}
    rows = []
    for name, x in fixtures.items():
        for scale in [.1, 1., 10., 100.]:
            for scaled in [False, True]:
                out, weights = attention(x*scale, x*scale, x, scaled=scaled)
                rows.append({'input':name, 'feature_scale':scale, 'divide_by_sqrt_d':scaled,
                             'mean_entropy_nats':float(entropy(weights).mean()),
                             'mean_largest_weight':float(weights.max(axis=1).mean()),
                             'max_row_sum_error':float(np.abs(weights.sum(axis=1)-1).max()),
                             'output_finite':bool(np.isfinite(out).all())})
    out, weights = attention([[1, 0]], [[1, 0], [0, 1]], [[10], [20]])
    return {'seed':27, 'device':'CPU', 'python':platform.python_version(), 'numpy':np.__version__,
            'task':'Untrained attention demonstration, not SR accuracy',
            'source':{'scene':scene['id'], 'sha256':scene['sha256'], 'split':'train',
                      'license':manifest['license'], 'window':list(window.flatten()),
                      'shape':[64, 3], 'units':manifest['units']},
            'hand_example':{'weights':weights.tolist(), 'output':out.tolist()}, 'rows':rows,
            'memory':[{'tokens':n, 'float32_score_bytes':score_bytes(n)} for n in [16, 64, 128, 256, 512, 1024, 4096]],
            'memory_scope':'Analytical one-head score matrix only, not measured peak RAM; NumPy demo uses float64',
            'seconds':time.perf_counter()-started}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    result = run(root)
    (root/'reports/mp27-self-attention.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
