"""Inspect the pinned training raster; no downloads, training or test access."""
import hashlib
import json
import platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mp32_tools import inspect_raster,masked_statistics

ROOT=Path(__file__).resolve().parents[2]

def main():
    fixture=np.array([[[1,3],[99,5]],[[2,4],[99,6]]])
    stats=masked_statistics(fixture,np.array([[True,True],[False,True]]))
    assert stats[0]['mean']==3 and stats[1]['mean']==4
    manifest=json.loads((ROOT/'configs/naip-pilot-manifest.json').read_text())
    source=next(s for s in manifest['scenes'] if s['split']=='train')
    path=ROOT/'data'/source['path']
    if not path.is_file():raise FileNotFoundError('Pinned training raster missing; follow docs/frozen-reproduction.md. This script never downloads it.')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
    pixels,mask,metadata=inspect_raster(path,Window(238,238,128,128))
    record={'recorded_at':datetime.now(timezone.utc).isoformat(),'question':'How do raster coordinates, nodata and band choice affect interpretation?',
            'source':source,'license':manifest['license'],'sensor':manifest['sensor'],'bands':manifest['bands'],
            'units':manifest['units'],'metadata':metadata,'synthetic_statistics':stats,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'rasterio':rasterio.__version__},
            'display':{'divisor':manifest['divisor'],'range':[0,1],'rgb_bands':[0,1,2],'false_colour_bands':[3,0,1]},
            'limits':'One cached aerial training crop, not fresh satellite imagery. Nodata is not a cloud mask. DN scaling does not establish reflectance. No model or HR-reference accuracy experiment.'}
    (ROOT/'reports/mp32-raster-explorer.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,order,title in zip(axes[:2],[[0,1,2],[3,0,1]],['RGB: red / green / blue','False colour: NIR / red / green']):
        rgb=np.clip(pixels[order].transpose(1,2,0)/manifest['divisor'],0,1)
        alpha=mask[order].all(0).astype(float)
        ax.imshow(np.dstack([rgb,alpha]));ax.set(title=title,xlabel='Crop column (pixels)',ylabel='Crop row (pixels)')
    for i,band in enumerate(manifest['bands']):
        axes[2].hist(pixels[i][mask[i]]/manifest['divisor'],bins=30,histtype='step',label=band)
    axes[2].set(title='Valid band distributions',xlabel='Stored DN / 10000',ylabel='Pixel count');axes[2].legend()
    fig.suptitle('Pinned NAIP training crop · fixed display range [0,1] · no reflectance claim')
    fig.tight_layout();fig.savefig(ROOT/'reports/mp32-band-views.png',dpi=130);plt.close(fig)
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
