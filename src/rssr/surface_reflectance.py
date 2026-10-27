"""Bounded, metadata-explicit L2A surface reflectance, separate from display RGB.

This local admission/conversion layer does not assert provider metadata is correct,
fetch imagery, align mismatched grids, perform atmospheric correction or run SR.
"""
import copy
import json
import math
import os
import tempfile
from pathlib import Path
import numpy as np


def prepare(values, scl, valid_masks, saturation_masks, metadata):
    """Convert CHW values once using reviewed per-band scale/offset declarations.

    Keep SCL4/5/6 (vegetation/bare soil/water) only, intersect all band masks and
    exclude saturation. Negative or >1 finite reflectances are NOT clipped.
    """
    meta=copy.deepcopy(metadata)
    for key in ('product_id','acquisition_time','processing_baseline','processor',
                'provider','license','radiometry_source','radiometry_review','mask_source'):
        if not isinstance(meta.get(key),str) or not meta[key].strip():
            raise ValueError(f'Missing explicit {key}')
    if meta.get('processing_level')!='L2A':
        raise ValueError('Only already atmospherically corrected L2A is admitted')
    if meta.get('harmonization_status') not in ('provider_applied','not_applied'):
        raise ValueError('Resolve provider harmonization status before conversion')
    if meta.get('conversion_applied',False):
        raise ValueError('Surface-reflectance conversion already applied')
    x=np.asarray(values);scl=np.asarray(scl)
    if x.ndim!=3 or not 1<=x.shape[0]<=13 or not all(1<=v<=1024 for v in x.shape[1:]):
        raise ValueError('Expected 1..13 bands with each spatial side1..1024')
    bands=meta.get('bands',[])
    if len(bands)!=x.shape[0] or len({b.get('name') for b in bands})!=len(bands):
        raise ValueError('Distinct metadata required for every band')
    grid=meta.get('grid',{})
    from affine import Affine
    from rasterio.crs import CRS
    try:
        CRS.from_user_input(grid['crs'])
        coefficients=grid['transform']
        if len(coefficients)!=6 or not np.isfinite(coefficients).all():raise ValueError()
        transform=Affine(*coefficients)
        if transform.determinant==0:raise ValueError()
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('Valid explicit CRS and six affine coefficients required') from exc
    if grid.get('shape')!=list(x.shape[1:]):raise ValueError('Declared grid shape differs')
    if meta.get('scl_grid')!=grid or meta.get('quality_grid')!=grid:
        raise ValueError('Quality layers must be explicitly aligned to the same grid')
    if scl.shape!=x.shape[1:] or not np.issubdtype(scl.dtype,np.integer) or not np.isin(scl,range(12)).all():
        raise ValueError('Aligned integer SCL classes0..11 required')
    for mask in (valid_masks,saturation_masks):
        if np.asarray(mask).dtype!=np.bool_ or np.shape(mask)!=x.shape:
            raise ValueError('Boolean per-band valid/saturation masks required')
    encoding=meta.get('input_encoding')
    if encoding=='encoded_dn':
        if not np.issubdtype(x.dtype,np.integer):raise ValueError('Encoded DN requires integer source samples')
    elif encoding=='surface_reflectance':
        if not np.issubdtype(x.dtype,np.floating):raise ValueError('Decoded reflectance requires floating source samples')
    else:raise ValueError('Explicit input encoding required; display RGB is not reflectance')
    valid=np.isin(scl,(4,5,6)) & np.asarray(valid_masks).all(0) & ~np.asarray(saturation_masks).any(0)
    result=np.empty(x.shape,dtype=np.float32)
    for i,band in enumerate(bands):
        if not isinstance(band.get('name'),str) or not band['name'] or band.get('units')!='1':
            raise ValueError('Named dimensionless reflectance band required')
        if band.get('grid')!=grid:raise ValueError('Band grid mismatch; no implicit resampling')
        if not band.get('resampling') or not band.get('source_asset'):
            raise ValueError('Source asset and resampling declaration required')
        for name in ('native_resolution_m','center_wavelength_nm','scale','offset','nodata'):
            value=band.get(name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'Explicit finite band {name} required')
        if min(band['native_resolution_m'],band['center_wavelength_nm'],band['scale'])<=0:
            raise ValueError('Positive native resolution, wavelength and scale required')
        if encoding=='surface_reflectance' and (band['scale']!=1 or band['offset']!=0):
            raise ValueError('Already decoded reflectance requires identity conversion')
        valid &= np.isfinite(x[i]) & (x[i]!=band['nodata'])
        converted=x[i].astype(np.float64)*band['scale']+band['offset']
        if np.any(np.isfinite(x[i]) & (~np.isfinite(converted) | (np.abs(converted)>np.finfo(np.float32).max))):
            raise ValueError('Conversion overflows float32')
        result[i]=converted.astype(np.float32)
    if not valid.any():raise ValueError('No valid pixels after conservative mask intersection')
    result[:,~valid]=np.nan
    meta.update(conversion_applied=True,output_encoding='surface_reflectance',output_dtype='float32',
                atmospheric_correction='provider L2A retained; none applied here',
                conversion_formula='reflectance = input * scale + offset; no clipping or stretching',
                valid_policy='SCL4/5/6 intersect all band validity masks and exclude any band saturation/nodata/nonfinite',
                valid_fraction=float(valid.mean()),metadata_status='caller declarations; provider review required',
                reference_metrics=None,super_resolution_applied=False)
    return result,valid,meta


def export(path, values, valid, metadata):
    """Publish one float32 GeoTIFF atomically without overwriting existing files."""
    import rasterio
    from affine import Affine
    if values.dtype!=np.float32 or values.ndim!=3 or valid.shape!=values.shape[1:] or valid.dtype!=np.bool_:
        raise ValueError('Prepared float32 CHW array and boolean mask required')
    if metadata.get('conversion_applied') is not True or not np.isfinite(values[:,valid]).all():
        raise ValueError('Prepared provenance and finite valid values required')
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.reflectance-',suffix='.tif',dir=path.parent);os.close(fd)
    temp=Path(name)
    try:
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(temp,'w',driver='GTiff',height=values.shape[1],width=values.shape[2],
                               count=values.shape[0],dtype='float32',nodata=np.nan,
                               crs=metadata['grid']['crs'],transform=Affine(*metadata['grid']['transform'])) as out:
                out.write(values);out.write_mask(valid.astype('uint8')*255)
                out.descriptions=tuple(b['name']+' surface reflectance' for b in metadata['bands'])
                out.update_tags(provenance=json.dumps(metadata,allow_nan=False),quantity='surface_reflectance',units='1')
        # Atomic creation, unlike replace this never overwrites a concurrent file.
        os.link(temp,path)
    finally:
        temp.unlink(missing_ok=True)
    return path
