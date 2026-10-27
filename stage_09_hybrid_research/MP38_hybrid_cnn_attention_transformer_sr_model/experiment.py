"""Trace shared hybrid architecture on synthetic tensors; no training or metrics."""
import json
import torch
from torch.nn.functional import interpolate
from mp38_tools import ROOT,build_hybrid,trace_hybrid


def run():
    torch.manual_seed(38);torch.set_num_threads(2)
    cases=[]
    for bands in (3,4):
        x=torch.linspace(0,1,bands*5*7).reshape(1,bands,5,7)
        model=build_hybrid(bands=bands,zero_residual=True)
        output,shapes=trace_hybrid(model,x)
        bicubic=interpolate(x,scale_factor=2,mode='bicubic',align_corners=False)
        torch.testing.assert_close(output,bicubic,atol=0,rtol=0)
        # Gradient existence only; no optimizer step or fitted weights.
        model(x).sum().backward()
        cases.append({'bands':bands,'input_shape':list(x.shape),'output_shape':list(output.shape),
            'module_shapes':shapes,'parameters':sum(p.numel() for p in model.parameters()),
            'zero_residual_max_abs_bicubic_difference':float((output-bicubic).abs().max()),
            'tail_weight_gradient_finite':bool(torch.isfinite(model.tail[0].weight.grad).all()),
            'tail_weight_gradient_nonzero':bool(model.tail[0].weight.grad.abs().sum()>0),
            'gate_gradient_max_abs':float(model.gate[0].weight.grad.abs().max())})
    return {'scope':'Synthetic architecture/initialization checks only',
            'environment':{'torch':torch.__version__,'device':'cpu','threads':2,'seed':38},
            'cases':cases,'optimizer_steps':0,
            'limitations':['Zero residual reproduces bicubic by construction, not learned quality',
                           'No matched-budget quality comparison or real error atlas',
                           'Gate gradient is zero initially when multiplied by a zero residual']}


if __name__=='__main__':
    report=run()
    (ROOT/'reports/mp38-architecture.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
