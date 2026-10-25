"""MP32 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np

def pixel_to_map(transform,row,column,center=True):
    if not np.isfinite([row,column]).all():raise ValueError('Finite pixel coordinates required')
    offset=.5 if center else 0.
    return transform*(column+offset,row+offset)

def masked_statistics(image,valid):
    """CHW numeric bands, HW or CHW validity; nonfinite pixels excluded per band."""
    image=np.asarray(image,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
    if image.ndim!=3 or not all(image.shape) or valid.shape not in (image.shape,image.shape[1:]):
        raise ValueError('Expected nonempty CHW bands and HW or CHW mask')
    valid=np.broadcast_to(valid,image.shape)&np.isfinite(image)
    rows=[]
    for band,mask in zip(image,valid):
        values=band[mask]
        rows.append({'valid_pixels':int(values.size),'minimum':float(values.min()) if values.size else None,
                     'maximum':float(values.max()) if values.size else None,'mean':float(values.mean()) if values.size else None})
    return rows

def inspect_raster(path,window=None):
    """Read at most 256×256 pixels; retain CRS/transform and per-band nodata."""
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(path) as dataset:
        window=window or Window(0,0,min(256,dataset.width),min(256,dataset.height))
        if min(window.width,window.height)<=0 or max(window.width,window.height)>256 or min(window.col_off,window.row_off)<0 or window.col_off+window.width>dataset.width or window.row_off+window.height>dataset.height:
            raise ValueError('Window must lie inside raster and be at most 256×256')
        array=dataset.read(window=window,masked=True)
        valid=~np.ma.getmaskarray(array)
        transform=dataset.window_transform(window)
        metadata={'width':dataset.width,'height':dataset.height,'bands':dataset.count,'dtype':dataset.dtypes[0],
                  'crs':str(dataset.crs) if dataset.crs else None,'crs_units':dataset.crs.linear_units if dataset.crs and dataset.crs.is_projected else 'angular or unknown',
                  'nodata':dataset.nodata,'pixel_size':list(dataset.res),'transform':list(transform)[:6],
                  'window':[window.col_off,window.row_off,window.width,window.height],
                  'first_pixel_center':list(pixel_to_map(transform,0,0)),
                  'statistics':masked_statistics(array.data,valid)}
    return array.data,valid,metadata
