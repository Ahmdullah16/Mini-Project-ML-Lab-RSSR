"""MP39 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from rssr.foundations import paired_scene_ci
import math
import statistics


def summarize_pilot(rows):
    """Summarize the pinned one-scene/three-seed pilot, not a new study.

    Reject duplicate runs/seeds and mixed manifests; seed spread is never a
    substitute for scene uncertainty. Values are mean per-run tile PSNR in dB.
    """
    if not rows:raise ValueError('Nonempty pilot records required')
    if len({r['manifest_sha256'] for r in rows})!=1:
        raise ValueError('Mixed data manifests cannot be pooled')
    if len({r['run_id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate run IDs')
    groups={}
    for r in rows:
        if not math.isfinite(r['psnr']) or not math.isfinite(r['bicubic_psnr']):
            raise ValueError('Finite PSNR values required')
        groups.setdefault(r['variant'],[]).append(r)
    output=[]
    for variant,group in sorted(groups.items()):
        if sorted(r['seed'] for r in group)!=[17,23,41]:
            raise ValueError('Expected exactly one run for each declared seed17/23/41')
        if len({r['parameters'] for r in group})!=1:
            raise ValueError('Parameter count changed across seeds')
        values=[r['psnr'] for r in group]
        deltas=[r['psnr']-r['bicubic_psnr'] for r in group]
        output.append({'variant':variant,'seeds':[17,23,41],'parameters':group[0]['parameters'],
            'mean_psnr_db':statistics.mean(values),'sample_seed_sd_db':statistics.stdev(values),
            'mean_delta_bicubic_db':statistics.mean(deltas),
            'geographic_ci':None,'ci_reason':'Only one validation scene; seeds are not independent regions',
            'run_ids':[r['run_id'] for r in sorted(group,key=lambda r:r['seed'])]})
    return output
