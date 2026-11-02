"""Synthetic Stage-08 fixtures using canonical tools; no raster/data acquisition."""
import sys
from pathlib import Path

import numpy as np
import torch
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from stage_08_remote_sensing.MP32_satellite_image_explorer.mp32_tools import pixel_to_map, masked_statistics
from stage_08_remote_sensing.MP33_satellite_preprocessing_pipeline.mp33_tools import fit_standardization, standardize, extract_patches
from stage_08_remote_sensing.MP34_satellite_hr_lr_pair_builder.mp34_tools import georeferenced_pair
from stage_08_remote_sensing.MP35_geographic_dataset_splitting.mp35_tools import audit_geography
from stage_08_remote_sensing.MP36_cross_sensor_dataset_investigation.mp36_tools import resolution_ratio, registration_pixels, affine_radiometry, band_response

TRANSFORM = Affine(10,0,100,0,-10,200)


def raster(mask_mode='exclude sentinel'):
    image = np.array([[[1,3],[99,5]], [[2,4],[99,6]]],dtype=np.float64)
    valid = np.ones_like(image,dtype=bool)
    if mask_mode == 'exclude sentinel':
        valid[:,1,0] = False
    elif mask_mode == 'all invalid':
        valid[:] = False
    elif mask_mode != 'include all':
        raise ValueError('Unknown mask setting')
    return dict(image=image.tolist(), mask=valid.tolist(), statistics=masked_statistics(image,valid),
                center=list(pixel_to_map(TRANSFORM,2,3)), corner=list(pixel_to_map(TRANSFORM,2,3,False)),
                transform=list(TRANSFORM)[:6], crs='EPSG:32611 (synthetic projected grid, metres)',
                coordinate_example='row2,column3 is a separate analytical grid example, outside the 2x2 statistics fixture')


def normalization(split='train', threshold=1.):
    image = np.array([[[1,3],[99,5]], [[2,2],[99,2]]],dtype=np.float64)
    mask = np.ones_like(image,dtype=bool); mask[:,1,0] = False
    stats = fit_standardization(image,mask,split=split)
    z, valid = standardize(image,mask,stats)
    # Reuse frozen TRAIN statistics on a synthetic shifted validation fixture.
    validation = image+2
    val_z, val_mask = standardize(validation,mask,stats)
    patches, margins = extract_patches(image,mask,TRANSFORM,size=2,min_valid_fraction=threshold)
    return dict(input=image.tolist(),mask=mask.tolist(),training_statistics=stats,
                normalized=z.tolist(), normalized_mask=valid.tolist(), validation_input=validation.tolist(),
                validation_output=val_z.tolist(),validation_mask=val_mask.tolist(),
                accepted_patches=[dict(row=p['row'],column=p['column'],valid_fraction=p['valid_fraction'],
                                       transform=list(p['transform'])[:6]) for p in patches],margins=margins,
                translated_patch_corner=list((TRANSFORM*Affine.translation(2,0))*(0,0)))


def pairing(scale=2, sigma=0., invalid=False):
    hr = torch.arange(16,dtype=torch.float32).reshape(1,4,4)
    valid = torch.ones_like(hr,dtype=torch.bool)
    if invalid:
        valid[0,0,0] = False
    lr, metadata = georeferenced_pair(hr,TRANSFORM,'EPSG:32611',scale=scale,valid=valid,sigma=sigma,noise=0.,seed=34)
    return dict(hr=hr.tolist(),lr=lr.tolist(),valid=valid.tolist(),metadata=metadata,
                hr_center=list(pixel_to_map(TRANSFORM,0,0)),
                lr_center=list(pixel_to_map(TRANSFORM*Affine.scale(scale),0,0)))


def geography(buffer=.02, repeated_region=False):
    scenes = [dict(id='a',region_id='a',split='train',wgs84_bounds=[0,0,1,1]),
              dict(id='b',region_id='a' if repeated_region else 'b',split='test',wgs84_bounds=[2,0,3,1])]
    # These are toy boxes, not frozen scene/test metadata.
    return dict(scenes=scenes,audit=audit_geography(scenes,buffer_degrees=buffer))


def feasibility(offset=5., gain=1.1):
    x = np.array([.2,.4]); y = affine_radiometry(x,gain,.01)
    return dict(ratios=[dict(lr_m=lr,hr_m=hr,ratio=resolution_ratio(lr,hr)) for lr,hr in ((10,5),(20,5),(10,1.5))],
                offset_metres=offset,hr_metres=5,offset_hr_pixels=registration_pixels(offset,5),
                reference=x.tolist(),prediction=y.tolist(),residual=(y-x).tolist(),
                toy_mse=float(np.mean((y-x)**2)),gain=gain,bias=.01,
                spectrum=[.2,.8],weights=[[1,0],[0,1],[1,1]],
                toy_responses=[band_response([.2,.8],w) for w in ([1,0],[0,1],[1,1])])


def current_mp36_prose(text):
    """Update site-only historical wording from the already reviewed feasibility report."""
    replacements = {
        'registration/correction methodology and complete feasibility acceptance remain open':
            'original registration/correction methodology was reviewed on 2026-10-27; version-specific file applicability and complete feasibility acceptance remain open',
        'Source review remains incomplete while the official paper/correction detail is unavailable.':
            'The original paper/correction review was resolved on 2026-10-27; v2 archive applicability remains unverified.',
        'Next: complete official registration/correction review and freeze a bounded inspection plan before asking for archive transfer.':
            'Next: verify version-specific processing lineage and freeze a bounded inspection plan before authorized archive transfer.',
        'Next acceptance gate: verify official registration methodology and published correction, complete the source-specific band/timestamp/mask matrix, and prepare a concrete bounded resource proposal. The paper endpoint returned429; do not infer a registration tolerance.':
            'The original method and URL correction were already reviewed. Next acceptance gate: verify v2 file lineage, complete the source-specific band/timestamp/mask matrix and prepare a bounded resource proposal. The original 15 m SIFT match-rejection threshold is not a residual-registration bound.',
        'Paper/correction retrieval returned429, leaving registration-method review open.':
            'This historical retrieval failure was resolved by the 2026-10-27 methodology review. Version-specific archive inspection and measured residual alignment remain open.'}
    for old,new in replacements.items():
        text = text.replace(old,new)
    return text
