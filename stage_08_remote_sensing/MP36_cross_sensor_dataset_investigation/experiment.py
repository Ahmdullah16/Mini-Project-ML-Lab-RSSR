"""Synthetic analytical examples only; no sensor imagery or metric evaluation."""
import json,platform
from pathlib import Path
import numpy as np
from mp36_tools import resolution_ratio,registration_pixels,affine_radiometry,band_response
ROOT=Path(__file__).resolve().parents[2]
x=np.array([.2,.4]);y=affine_radiometry(x,1.1,.01)
record={'scope':'Synthetic analytical teaching values, not measured sensor calibration or SR quality',
        'scale_ratios':{'10m_to_5m':resolution_ratio(10,5),'20m_to_5m':resolution_ratio(20,5),'10m_to_1_5m':resolution_ratio(10,1.5)},
        '5m_offset_in_5m_pixels':registration_pixels(5,5),
        'radiometry':{'reference':x.tolist(),'gain':1.1,'offset':.01,'prediction':y.tolist(),'mse':float(np.mean((y-x)**2))},
        'toy_spectral_responses':[band_response([.2,.8],w) for w in ([1,0],[0,1],[1,1])],
        'environment':{'python':platform.python_version(),'numpy':np.__version__}}
(ROOT/'reports/mp36-numerical-examples.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
