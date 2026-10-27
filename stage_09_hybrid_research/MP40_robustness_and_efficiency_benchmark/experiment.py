"""Derive curves/tables from recorded development benchmark, no new inference."""
import hashlib
import json
from mp40_tools import ROOT,summarize_robustness


def run():
    path=ROOT/'reports/development-benchmark.json'
    text=path.read_text(encoding='utf-8');source=json.loads(text)
    return {'scope':'Derived historical development evidence; no new benchmark run',
        'source':'reports/development-benchmark.json','sha256_utf8_lf':hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'environment':source['environment'],'selection':source['selection'],
        'robustness':summarize_robustness(source['robustness']),
        'efficiency':source['efficiency'],'limitations':source['limitations'],
        'pending':['Real cross-sensor validation','Complete FLOP accounting','Allocator peak memory',
                   'GPU results unavailable: CPU study only']}


def plot(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='rssr-mp40-historical'
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for i,bands in enumerate((3,4)):
        records=[r for r in report['robustness'] if r['bands']==bands]
        for j,condition in enumerate(('blur','noise')):
            ax=axes[i,j]
            for variant in sorted({r['variant'] for r in records}):
                selected=[r for r in records if r['variant']==variant and
                    (r['noise_std']==0 if condition=='blur' else r['sigma_hr_pixels']==.8)]
                key='sigma_hr_pixels' if condition=='blur' else 'noise_std'
                selected.sort(key=lambda r:r[key])
                ax.plot([r[key] for r in selected],[r['mean_delta_bicubic_db'] for r in selected],
                        marker='o',label=variant,color={'cnn':'tab:blue','hybrid':'tab:orange','attention':'tab:green'}[variant])
            ax.axhline(0,color='black',lw=.8,ls='--')
            ax.set_title(f'{bands} bands: {condition} sweep')
            ax.set_xlabel('Gaussian blur sigma (HR pixels)' if condition=='blur' else 'Noise SD (normalized DN)')
            ax.set_ylabel('Mean paired PSNR - bicubic (dB)')
            ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Historical development results: 4 scenes; selected seed per model\nSynthetic shifts, not real cross-sensor validation',fontsize=12)
    fig.savefig(path,metadata={'Date':None} if str(path).endswith('.svg') else {})
    plt.close(fig)


if __name__=='__main__':
    report=run()
    (ROOT/'reports/mp40-benchmark-analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    target=ROOT/'reports/figures/mp40-robustness.svg';target.parent.mkdir(parents=True,exist_ok=True)
    plot(report,target)
    print('Derived30 condition rows,5 efficiency rows and robustness figure from existing evidence')
