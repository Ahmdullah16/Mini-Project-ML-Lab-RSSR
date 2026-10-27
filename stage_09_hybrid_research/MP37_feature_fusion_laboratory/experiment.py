"""Deterministic CPU fusion experiment; reuse historical scores, no training."""
import hashlib
import json
from pathlib import Path
import statistics
import torch
from mp37_tools import ROOT, fuse_features, initialize_mean_projection, projection_cost
from rssr.models import SRModel


def run():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    a=torch.tensor([1.,2.]).reshape(1,2,1,1).requires_grad_()
    b=torch.tensor([3.,4.]).reshape(1,2,1,1).requires_grad_()
    mean=fuse_features(a,b)
    mean.sum().backward()
    projection=initialize_mean_projection(torch.nn.Conv2d(4,2,1))
    projected=projection(fuse_features(a.detach(),b.detach(),'concat'))
    torch.testing.assert_close(projected,mean.detach())
    concat=SRModel(bands=3,width=8,kind='hybrid',fusion='concat').eval()
    addition=SRModel(bands=3,width=8,kind='hybrid',fusion='add').eval()
    common={k:v for k,v in concat.state_dict().items() if not k.startswith('fuse.')}
    addition.load_state_dict(common,strict=True)
    initialize_mean_projection(concat.fuse)
    x=torch.linspace(0,1,3*5*7).reshape(1,3,5,7)
    with torch.inference_mode():
        yc,ya=concat(x),addition(x)
    torch.testing.assert_close(yc,ya,atol=1e-6,rtol=1e-6)
    source=ROOT/'reports/ablation-pilot-runs.json'
    rows=json.loads(source.read_text(encoding='utf-8'))
    historical={}
    for variant in ('hybrid','hybrid_add'):
        selected=[r for r in rows if r['variant']==variant]
        assert sorted(r['seed'] for r in selected)==[17,23,41]
        historical[variant]={'psnr_mean_db':statistics.mean(r['psnr'] for r in selected),
            'sample_seed_sd_db':statistics.stdev(r['psnr'] for r in selected),
            'parameters':sorted({r['parameters'] for r in selected}),
            'run_ids':[r['run_id'] for r in selected]}
    return {'scope':'Synthetic algebra/shape experiment plus unchanged historical validation scores',
        'environment':{'torch':torch.__version__,'device':'cpu','threads':2,'seed':17},
        'worked':{'sum':fuse_features(a,b,'sum').detach().flatten().tolist(),
                  'mean':mean.detach().flatten().tolist(),
                  'projected':projected.detach().flatten().tolist(),
                  'gradient_a':a.grad.flatten().tolist(),'gradient_b':b.grad.flatten().tolist()},
        'model_fixture':{'input_shape':list(x.shape),'output_shape':list(yc.shape),
            'max_abs_difference':float((yc-ya).abs().max()),
            'concat_parameters':sum(p.numel() for p in concat.parameters()),
            'mean_parameters':sum(p.numel() for p in addition.parameters()),
            'projection_cost':projection_cost(8,5,7)},
        'historical_source':'reports/ablation-pilot-runs.json',
        'historical_sha256_utf8_lf':hashlib.sha256(source.read_text(encoding='utf-8').encode('utf-8')).hexdigest(),
        'historical':historical,
        'limitations':['No new training or test evaluation','Historical one-scene seed SD is not geographic uncertainty',
                        'Different budgets confound historical architecture comparison','Equal fusion output is not an SR quality claim']}


if __name__=='__main__':
    report=run()
    (ROOT/'reports/mp37-fusion.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
