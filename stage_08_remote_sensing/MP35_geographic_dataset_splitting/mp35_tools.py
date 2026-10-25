"""Validate declared scene splits without assigning or modifying them."""
from pathlib import Path
import sys
import math
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from rssr.data import check_splits


def audit_geography(scenes, buffer_degrees=0.02):
    """WGS84 axis-aligned envelope audit; degrees are not metric distances.

    Antimeridian-crossing envelopes are deliberately unsupported. Equal-to-buffer
    gaps fail, following the frozen shared pipeline's strict separation rule.
    Same-split overlaps are allowed; cross-split regions must differ.
    """
    if not math.isfinite(buffer_degrees) or buffer_degrees < 0:
        raise ValueError('Buffer must be finite and nonnegative')
    if not scenes:
        raise ValueError('At least one scene is required')
    ids=set()
    for scene in scenes:
        ident=scene.get('id')
        if not isinstance(ident,str) or not ident or ident in ids:
            raise ValueError('Scene IDs must be unique nonempty strings')
        ids.add(ident)
        if not isinstance(scene.get('region_id'),str) or not scene['region_id']:
            raise ValueError('Region ID is required')
        b=scene['wgs84_bounds']
        if len(b)!=4 or not all(math.isfinite(x) for x in b):
            raise ValueError('Four finite bounds required')
        if not (-180<=b[0]<b[2]<=180 and -90<=b[1]<b[3]<=90):
            raise ValueError('Invalid WGS84 envelope or antimeridian crossing')
    check_splits(scenes,buffer_degrees=buffer_degrees)
    pairs=[]
    for i,a in enumerate(scenes):
        for b in scenes[i+1:]:
            if a['split']==b['split']:continue
            A=a['wgs84_bounds']; B=b['wgs84_bounds']
            clearance=max(B[0]-A[2],A[0]-B[2],B[1]-A[3],A[1]-B[3])
            pairs.append({'a':a['id'],'b':b['id'],'max_axis_clearance_degrees':clearance})
    return {'buffer_degrees':buffer_degrees,'cross_split_pairs':pairs,
            'counts':{s:sum(x['split']==s for x in scenes) for s in ('train','validation','test')},
            'rule':'At least one axis must have gap strictly greater than buffer; not a geodesic distance.'}
