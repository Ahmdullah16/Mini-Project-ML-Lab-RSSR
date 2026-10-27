"""Derive teaching ablation table from existing24 runs; never retrain."""
import hashlib
import json
from mp39_tools import ROOT,paired_scene_ci,summarize_pilot


def run():
    source=ROOT/'reports/ablation-pilot-runs.json'
    text=source.read_text(encoding='utf-8')
    rows=json.loads(text)
    table=summarize_pilot(rows)
    mean,ci=paired_scene_ci([2.,3.,4.],[1.,2.,3.],seed=39,repeats=1000)
    lookup={r['variant']:r for r in table}
    return {'scope':'Derived historical one-scene table and separate synthetic bootstrap example',
        'source':'reports/ablation-pilot-runs.json','sha256_utf8_lf':hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'source_runs':len(rows),'table':table,
        'descriptive_contrasts_db':{
            'learned_minus_constant_gate':lookup['hybrid_gate']['mean_psnr_db']-lookup['hybrid_constant_gate']['mean_psnr_db'],
            'consistency_minus_hybrid':lookup['hybrid_consistency']['mean_psnr_db']-lookup['hybrid']['mean_psnr_db']},
        'synthetic_bootstrap_only':{'a':[2,3,4],'b':[1,2,3],'seed':39,'repeats':1000,
            'mean_difference':mean,'percentile95':ci.tolist()},
        'limitations':['No new training, validation inference or test access',
            'Parameter budgets differ','No geographic CI from one scene',
            'Synthetic bootstrap interval is not an experimental confidence interval']}


if __name__=='__main__':
    report=run()
    (ROOT/'reports/mp39-ablation-analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
