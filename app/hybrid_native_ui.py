"""Native MP37–40: controlled tensors and explicit historical evidence."""
import hashlib
import json
import time
import altair as alt
import pandas as pd
import streamlit as st
import torch
from hybrid_labs import ROOT,fusion,hybrid,ablation,robustness


def render(ident,root=ROOT):
    torch.set_num_threads(2)
    st.subheader(ident+' · hybrid research teaching laboratory')
    st.info('Synthetic controls and historical development evidence are labeled separately. No training, '
            'checkpoint inference, frozen-test access or new hardware benchmark. Features are not calibrated reflectance.')
    settings={'project':ident,'device':'CPU','torch':torch.__version__,'threads':2}
    if ident=='MP37':
        mode=st.selectbox('Fusion operation',['mean','sum','concat'],key='MP37_mode')
        settings['mode']=mode
        st.latex(r'S=A+B,\quad M=(A+B)/2,\quad Z=[A;B],\quad Y=P_AA+P_BB+b')
        st.latex(r'P_A=P_B=I/2,\ b=0\ \Rightarrow\ Y=M;\quad n_{param}=2C^2+C,\quad MACs=NHW\,2C^2')
        st.write('A and B are aligned NCHW features: batch N, channels C, height H, width W. '
                 'Canonical A=[1,2],B=[3,4] gives mean[2,3], sum[4,6], concat[1,2,3,4]. '
                 'The initialized1×1 projection exactly represents the mean; it is not trained. '
                 'Derivatives of the output sum are1/2 for mean and1 for sum/concat. '
                 'Separate cost example N=2,C=4,H=3,W=5 gives36 parameters and960 MACs. '
                 'A MAC is a multiply-accumulate pair; branch work, copies, bias additions and activations are excluded.')
        st.graphviz_chart('digraph {rankdir=LR; Local -> Fusion; Context -> Fusion; Fusion -> "Mean / Sum"; Fusion -> "Concat 2C" -> "1x1 projection C";}')
        st.caption('Mismatched shapes, devices, dtypes or nonfinite values are rejected. A1×1 projection cannot repair a spatial shift; opposite features can cancel in a mean.')
        compute=lambda:fusion(mode)
    elif ident=='MP38':
        count=st.selectbox('Synthetic input bands',[3,4],key='MP38_bands')
        gate=st.checkbox('Include sigmoid residual gate',value=True,key='MP38_gate')
        settings.update(bands=count,gate=gate,seed=38,zero_residual=True)
        st.latex(r'\widehat x=B_{bicubic}(y)+g\odot R(y),\quad \partial\widehat x/\partial g=R(y)')
        st.write('y is the synthetic1×C×5×7 input; R is the residual and g the sigmoid gate (or1 if disabled). '
                 'CNN/local features and unshifted window context fuse before attention and a zeroed residual tail. '
                 'PixelShuffle×2 produces1×C×10×14. Before shuffle there are4C channels:12 for RGB,16 for four bands. '
                 'A scalar baseline.4 plus gate.5 times residual.1 gives.45. A zero residual gives exactly bicubic '
                 'and zero sensitivity to the gate, not learned improvement. The gate is not calibrated confidence.')
        st.graphviz_chart('digraph {rankdir=LR; Input -> Head -> CNN; Head -> "Window context"; CNN -> Fusion; "Window context" -> Fusion; Fusion -> Attention -> "Zero tail / shuffle" -> Multiply; Input -> Bicubic -> Add; Input -> Gate; Gate -> Multiply; Multiply -> Add;}')
        st.caption('Actual module shapes are captured with temporary hooks; the canonical tracer removes hooks and restores model state. No RGB image conversion or physical reflectance claim.')
        compute=lambda:hybrid(count,gate)
    elif ident=='MP39':
        units=st.selectbox('Synthetic independent units for bootstrap',[3,1],key='MP39_units')
        settings.update(units=units,seed=39,repeats=1000)
        st.latex(r'\Delta_i=PSNR_{A,i}-PSNR_{B,i},\quad \bar\Delta=n^{-1}\sum_i\Delta_i,\quad s_{seed}=\sqrt{\sum_j(p_j-\bar p)^2/(k-1)}')
        st.write('The historical24-run pilot uses one validation scene and seeds17,23,41. '
                 'Mean PSNR and sample seed SD describe optimization variability, not independent geographic uncertainty. '
                 'Parameter budgets differ, so contrasts do not isolate architectural causation. '
                 'Separate toy independent units A=[2,3,4],B=[1,2,3] have constant effect1. '
                 'Bootstrap1000 paired resamples with seed39 gives [1,1]. This toy interval is NOT an experimental confidence interval.')
        st.graphviz_chart('digraph {rankdir=LR; "Historical run IDs / manifest" -> "Validate seeds and counts" -> "Descriptive mean / seed SD"; "Separate synthetic units" -> "Paired resampling" -> "Toy percentile interval";}')
        st.caption('One independent unit cannot support the canonical geographic interval and is rejected. Duplicate runs, missing seeds and mixed manifests fail rather than being pooled.')
        compute=lambda:ablation(root,units)
    else:
        count=st.selectbox('Historical band protocol',[3,4],key='MP40_bands')
        axis=st.selectbox('Historical degradation sweep',['blur','noise'],key='MP40_axis')
        settings.update(bands=count,axis=axis)
        st.latex(r'\Delta_{scene}=PSNR_{model}-PSNR_{bicubic},\quad \bar\Delta=N^{-1}\sum_{scene}\Delta_{scene}')
        st.write('Keep band count, model and degradation condition separate. Blur sweep fixes noise0; noise sweep '
                 'fixes Gaussian sigma.8 HR pixels. Positive means model exceeds bicubic in the recorded condition; '
                 'negative findings are retained. Toy model30→25 dB and baseline28→26 dB gives relative effect+2→−1 dB. '
                 'dB differences are not percentage scores. Curves connect sampled settings only, not a continuous robustness guarantee.')
        st.graphviz_chart('digraph {rankdir=LR; "Recorded development benchmark" -> "Select band / condition" -> "Paired scene differences" -> "Means and curves"; "Recorded CPU efficiency" -> "Hardware / shape / scope";}')
        st.caption('Efficiency is PRECOMPUTED on the recorded machine. Sampled process RSS is not allocator peak; partial MACs are not complete FLOPs. GPU results, real cross-sensor validity and hosted huge-job feasibility remain unavailable.')
        compute=lambda:robustness(root,count,axis)
    st.markdown('Primary references: [Residual learning](https://arxiv.org/abs/1512.03385), '
                '[SwinIR](https://arxiv.org/abs/2108.10257), '
                '[Bootstrap methods](https://doi.org/10.1214/aos/1176344552). '
                'Our compact models and toy calculations are not full paper reproductions.')
    if not st.button('Run '+ident,key=ident+'_run'):return
    started=time.perf_counter()
    try:result=compute()
    except (ValueError,OSError,KeyError,TypeError) as exc:
        st.error('Teaching request rejected: '+str(exc));return
    if ident=='MP37':
        table=pd.DataFrame({'A':result['a'],'B':result['b'],'mean_projection':result['mean_projection'],
                            'gradient_A':result['gradient_a'],'gradient_B':result['gradient_b']})
        st.dataframe(table);st.write('Selected output:',result['output'],'; shape:',result['output_shape'])
        st.write(result['cost_geometry'],result['cost'])
    elif ident=='MP38':
        st.write('Input/output:',result['input_shape'],result['output_shape'],'; parameters:',result['parameters'])
        st.dataframe(pd.DataFrame(result['input'][0]));st.caption('Synthetic input band0; fixed numeric range, no display stretch.')
        st.dataframe(pd.DataFrame(result['output'][0]));st.caption('Synthetic output band0; no clipping of bicubic overshoot.')
        table=pd.DataFrame([dict(module=k,**v) for k,v in result['trace'].items()]);st.dataframe(table)
        st.write('Maximum absolute difference from bicubic:',result['difference_from_bicubic'],'; scalar example:',result['scalar_gate_example'])
    elif ident=='MP39':
        table=pd.DataFrame(result['historical']);st.dataframe(table)
        st.write('Historical source:',result['source'],'; runs:',result['source_runs'])
        st.write('Separate toy inputs:',result['toy_a'],result['toy_b'],'; mean:',result['toy_mean'],'; percentile95:',result['toy_interval95'])
        st.warning('No geographic confidence interval is available from the one-scene pilot. Seed spread is not scene uncertainty; do not treat the toy interval as research evidence.')
    else:
        table=pd.DataFrame(result['rows']);st.dataframe(table)
        st.altair_chart(alt.Chart(table).mark_line(point=True).encode(
            x=alt.X(result['axis']+':Q',title='Sigma (HR pixels)' if axis=='blur' else 'Noise SD (normalized DN)'),
            y=alt.Y('mean_delta_bicubic_db:Q',title='Mean scene-paired PSNR difference (dB)',scale=alt.Scale(zero=True)),
            color='variant:N',tooltip=['variant',result['axis'],'mean_delta_bicubic_db','selected_seed']),width='stretch')
        st.dataframe(pd.DataFrame(result['efficiency']));st.write('Recorded environment:',result['environment'])
        st.write('Seed selection:',result['selection']);st.write('Source:',result['source']);st.write('Limits:',result['limitations'])
        st.write('Toy relative effects (dB):',result['toy_effect_db'])
    sources=list((root/'stage_09_hybrid_research').glob('MP*/mp*_tools.py'))+[root/'app/hybrid_labs.py',root/'src/rssr/models.py',root/'src/rssr/foundations.py']
    settings.update(seconds=time.perf_counter()-started,scope='Teaching controls / historical data only; not newly measured SR quality',
                    source_sha256={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    st.download_button('Download result and provenance',json.dumps({'settings':settings,'result':result},indent=2,allow_nan=False),ident+'-lesson.json','application/json',key=ident+'_json')
    st.download_button('Download numerical table',table.to_csv(index=False),ident+'-table.csv','text/csv',key=ident+'_csv')
    st.success(f'{ident} teaching computation finished in {settings["seconds"]:.4f} s. Research gates remain open.')
