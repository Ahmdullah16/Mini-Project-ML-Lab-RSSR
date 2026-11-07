"""Actual seven-family RGB display panels from pinned assets; no HR scoring."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rssr.hosted_assets import fetch_asset
from rssr.experiment import load_model
from rssr.model_availability import availability
from rssr.scaling import cascade


def generate(output,cache):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Choose a new output directory to preserve earlier evidence')
    catalog=json.loads((ROOT/'configs/hosted-models.json').read_text(encoding='utf-8'))['models']
    selected=[]
    for row in availability(ROOT):
        candidates=[s for s in catalog if s['checkpoint']['sha256'] in row['checkpoint_sha256']]
        if len(candidates)!=1:raise ValueError('Expected one pinned hosted model for '+row['model'])
        selected.append((row['model'].split(' · ')[0],candidates[0]))
    sample=catalog[0]['sample'];input_path=fetch_asset(sample,cache)
    image=np.asarray(Image.open(input_path).convert('RGB'))
    # Preserve decoded dimensions and pixels; no reference image is available.
    output.mkdir(parents=True);Image.fromarray(image).save(output/'input-lr.png')
    records=[]
    import torch,rasterio
    torch.set_num_threads(2)
    for key,spec in selected:
        checkpoint=fetch_asset(spec['checkpoint'],cache);model,state=load_model(checkpoint)
        if model.bands!=3 or model.scale!=2:raise ValueError('Expected native x2 RGB')
        for scale in (2,4):
            meta=cascade(image,model,output/f'{key}-x{scale}',scale,{'model':spec,'run_id':state['run_id'],'source':sample})
            with rasterio.open(output/f'{key}-x{scale}'/meta['output']) as src:sr=src.read().transpose(1,2,0)
            assert sr.shape==(image.shape[0]*scale,image.shape[1]*scale,3)
            baseline=np.asarray(Image.fromarray(image).resize((sr.shape[1],sr.shape[0]),Image.Resampling.BICUBIC))
            target=output/f'{key}-x{scale}.png';Image.fromarray(sr).save(target)
            records.append({'model':key,'scale':scale,'label':spec['label'],'model_specification':spec,'run_id':state['run_id'],'checkpoint_sha256':spec['checkpoint']['sha256'],'output_shape':list(sr.shape),'runtime_seconds':meta['runtime_seconds'],'pillow_bicubic_disagreement_mae_uint8':float(np.abs(sr.astype(float)-baseline.astype(float)).mean()),'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'pipeline_metadata':meta})
            print(key,scale,sr.shape,flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for scale in (2,4):
        rows=[r for r in records if r['scale']==scale]
        shape=rows[0]['output_shape'];base=np.asarray(Image.fromarray(image).resize((shape[1],shape[0]),Image.Resampling.BICUBIC))
        fig,axes=plt.subplots(3,3,figsize=(12,12))
        panels=[('LR input — shown with nearest-neighbour display',image),('Pillow bicubic interpolation',base)]
        panels += [(r['model']+' | '+('native ×2' if scale==2 else '×2 → ×2 cascade')+f"\n{r['runtime_seconds']:.3f} s | ΔMAE {r['pillow_bicubic_disagreement_mae_uint8']:.3f}",np.asarray(Image.open(output/f"{r['model']}-x{scale}.png"))) for r in rows]
        for ax,(title,pixels) in zip(axes.flat,panels):ax.imshow(pixels,interpolation='nearest',vmin=0,vmax=255);ax.set_title(title,fontsize=9);ax.axis('off')
        fig.suptitle(f'Executed seven-model display comparison ×{scale} | {image.shape[1]}×{image.shape[0]} → {shape[1]}×{shape[0]}',fontsize=14)
        fig.text(.05,.02,'Licensed NAIP training example; not independent evaluation. ΔMAE is disagreement with bicubic, not HR error.\nM1/M7: 1000 steps; M2–M6: 100 steps. Unequal budgets prohibit an architectural quality ranking. No HR reference.',fontsize=10)
        fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(output/f'comparison-x{scale}.png',dpi=180);fig.savefig(output/f'comparison-x{scale}.pdf');plt.close(fig)
        differences=[np.abs(np.asarray(Image.open(output/f"{r['model']}-x{scale}.png")).astype(float)-base.astype(float)).mean(axis=2) for r in rows]
        vmax=max(1.,max(float(d.max()) for d in differences));fig,axes=plt.subplots(2,4,figsize=(14,7))
        for ax,row,delta in zip(axes.flat,rows,differences):im=ax.imshow(delta,cmap='magma',vmin=0,vmax=vmax,interpolation='nearest');ax.set_title(row['model']);ax.axis('off')
        axes.flat[-1].axis('off');fig.colorbar(im,ax=axes.ravel().tolist(),label='Mean |SR − Pillow bicubic| across RGB (uint8 levels)',shrink=.7)
        fig.suptitle(f'×{scale} shared-range disagreement maps — no HR-reference accuracy');fig.savefig(output/f'disagreement-x{scale}.png',dpi=180);plt.close(fig)
    evidence={'scope':'Executed training-example RGB display demonstration, not a scientific evaluation','source':sample,'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),'input_shape':list(image.shape),'results':records,'reference_metrics':'unavailable — no aligned HR supplied','settings':{'scales':[2,4],'tile':128,'halo':32,'display_interpolation':'nearest','heatmap_range':'one shared range per scale'},'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'configs/hosted-models.json',ROOT/'src/rssr/scaling.py',ROOT/'src/rssr/models.py']}}
    (output/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'README.txt').write_text('Executed seven-model demonstration on a public-domain NAIP training example. Input and checkpoints are checksum-pinned in evidence.json. Images preserve exact scale/dimensions. The original training input is small; nearest-neighbour display avoids smoothing for presentation.\n\nNative x2 and repeated x2+x2 outputs are observed results. Differences from bicubic describe disagreement only. No HR, spectral, geographic or satellite transfer metrics were computed. Model budgets differ: do not rank architectures. Cascade clipping/quantization and tiled context restrictions are described in per-job provenance.\n',encoding='utf-8')
    print('Completed 14 actual model-scale executions; no HR scoring or frozen-test access')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--cache',default=os.environ.get('RSSR_ASSET_CACHE',str(Path(os.environ.get('TEMP','.'))/'rssr-public-assets-v1')));generate(**vars(p.parse_args()))
