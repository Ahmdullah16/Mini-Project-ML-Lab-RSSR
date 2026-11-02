"""Bounded Stage-09 teaching; historical development sources are read-only."""
import hashlib
import json
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from stage_09_hybrid_research.MP37_feature_fusion_laboratory.mp37_tools import fuse_features,projection_cost,initialize_mean_projection
from stage_09_hybrid_research.MP38_hybrid_cnn_attention_transformer_sr_model.mp38_tools import build_hybrid,trace_hybrid
from stage_09_hybrid_research.MP39_controlled_ablation_study.mp39_tools import summarize_pilot,paired_scene_ci
from stage_09_hybrid_research.MP40_robustness_and_efficiency_benchmark.mp40_tools import summarize_robustness


def read_record(root,name):
    path=Path(root)/'reports'/name
    raw=path.read_bytes()
    return json.loads(raw),{'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()}


def fusion(mode='mean'):
    a=torch.tensor([1.,2.]).reshape(1,2,1,1).requires_grad_()
    b=torch.tensor([3.,4.]).reshape(1,2,1,1).requires_grad_()
    result=fuse_features(a,b,mode)
    result.sum().backward()
    with torch.random.fork_rng(devices=[]):
        layer=initialize_mean_projection(torch.nn.Conv2d(4,2,1))
        with torch.inference_mode():projected=layer(fuse_features(a,b,'concat'))
    return dict(a=a.detach().flatten().tolist(),b=b.detach().flatten().tolist(),
                output=result.detach().flatten().tolist(),output_shape=list(result.shape),
                gradient_a=a.grad.flatten().tolist(),gradient_b=b.grad.flatten().tolist(),
                mean_projection=projected.flatten().tolist(),
                cost=projection_cost(4,3,5,batch=2),cost_geometry='N=2,C=4,H=3,W=5 (separate analytical example)')


def hybrid(bands=3,gate=True):
    if bands not in (3,4):raise ValueError('Teaching band counts are3 or4')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(38)
        x=torch.linspace(0,1,bands*5*7).reshape(1,bands,5,7)
        model=build_hybrid(bands=bands,gate=gate,zero_residual=True)
        out,trace=trace_hybrid(model,x)
        skip=F.interpolate(x,scale_factor=2,mode='bicubic',align_corners=False)
    return dict(input=x[0].tolist(),output=out[0].tolist(),input_shape=list(x.shape),output_shape=list(out.shape),
                trace=trace,parameters=sum(p.numel() for p in model.parameters()),
                difference_from_bicubic=float((out-skip).abs().max()),seed=38,zero_residual=True,
                scalar_gate_example=.4+.5*.1)


def ablation(root=ROOT,units=3):
    if units not in (1,3):raise ValueError('Choose one or three synthetic units')
    rows,source=read_record(root,'ablation-pilot-runs.json')
    if len(rows)!=24:raise ValueError('Expected the historical24-run pilot, not a substituted study')
    table=summarize_pilot(rows)
    a,b=[2.,3.,4.][:units],[1.,2.,3.][:units]
    # The one-unit option deliberately rejects a geographic confidence interval.
    mean,interval=paired_scene_ci(a,b,seed=39,repeats=1000)
    return dict(historical=table,source=source,source_runs=len(rows),toy_a=a,toy_b=b,
                toy_mean=mean,toy_interval95=interval.tolist(),seed=39,repeats=1000)


def robustness(root=ROOT,bands=3,axis='blur'):
    if bands not in (3,4) or axis not in ('blur','noise'):raise ValueError('Choose bands3/4 and blur/noise')
    source,provenance=read_record(root,'development-benchmark.json')
    rows=summarize_robustness(source['robustness'])
    selected=[r for r in rows if r['bands']==bands and
              (r['noise_std']==0 if axis=='blur' else r['sigma_hr_pixels']==.8)]
    key='sigma_hr_pixels' if axis=='blur' else 'noise_std'
    selected.sort(key=lambda r:(r['variant'],r[key]))
    if not selected:raise ValueError('No recorded conditions for this protocol')
    return dict(rows=selected,axis=key,source=provenance,environment=source['environment'],
                selection=source['selection'],limitations=source['limitations'],
                efficiency=[r for r in source['efficiency'] if r['bands']==bands],
                toy_model_db=[30,25],toy_bicubic_db=[28,26],toy_effect_db=[30-28,25-26])
