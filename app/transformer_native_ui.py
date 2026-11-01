"""MP27–31 on-site controls, kept separate from historical scientific evidence."""
import hashlib
import json
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from transformer_labs import ROOT, single, heads, encoder, windows, baseline


def matrix(label, values):
    st.write(label)
    st.dataframe(pd.DataFrame(values))


def heatmap(label, values):
    a = np.asarray(values)
    rows, cols = np.indices(a.shape)
    data = pd.DataFrame({'query':rows.ravel(), 'key':cols.ravel(), 'weight':a.ravel()})
    st.write(label)
    st.altair_chart(alt.Chart(data).mark_rect().encode(
        x=alt.X('key:O', title='Key token (column)'),
        y=alt.Y('query:O', title='Query token (row)'),
        color=alt.Color('weight:Q', scale=alt.Scale(domain=[0,1]), title='Weight / allowed path'),
        tooltip=['query','key','weight']), width='stretch')
    st.caption('Shared range 0–1; exact numerical values are provided in the table and JSON export.')
    matrix(label+' · exact values', a)


def render(ident, root=ROOT):
    torch.set_num_threads(2)
    st.subheader(ident+' · native Transformer laboratory')
    st.info('Bounded CPU teaching computation. No training, checkpoint download or held-out evaluation. '
            'Synthetic feature values are not calibrated reflectance. Change a control to clear the previous output, then Run.')
    settings = {'project':ident}
    if ident == 'MP27':
        scale = st.selectbox('Feature multiplier c', [.1,1.,10.,100.], index=1, key='MP27_scale')
        masked = st.checkbox('Allow only the first key', key='MP27_mask')
        settings.update(scale=scale, masked=masked, seed=27, dtype='float64')
        st.latex(r'A_{ij}=\frac{\exp(L_{ij}-m_i)}{\sum_{k\in S_i}\exp(L_{ik}-m_i)},\quad L=QK^T/\sqrt d,\quad O=AV')
        st.write('Q[N,d] asks queries; K[M,d] contains keys; V[M,dv] contains values. '
                 'S is the set of allowed keys and m is its row maximum. Masked entries get zero. '
                 'The canonical example Q=[1,0], K=identity, V=[10,20] gives O≈13.302385 at c=1. '
                 'Multiplying Q and K by c multiplies logits by c²; the √d divisor does not cancel this. '
                 'Entropy −Σp log p measures concentration in nats, not accuracy. Stable softmax cannot repair an already overflowing dot product.')
        st.graphviz_chart('digraph {rankdir=LR; Q -> Scores; K -> Scores; Scores -> Mask -> "Subtract row max" -> Softmax -> "Multiply V" -> Output; V -> "Multiply V";}')
        st.caption('Failure case: a mask with no allowed key in a row is rejected, not normalized to NaN.')
        compute = lambda: single(scale, masked)
    elif ident == 'MP28':
        count = st.selectbox('Heads h (fixed total width D=4)', [1,2,4], index=1, key='MP28_heads')
        settings.update(heads=count, seed=28, dtype='float64', biases=False)
        st.latex(r'd=D/h,\quad O=\mathrm{Concat}(A_1V_1,\ldots,A_hV_h)W_O,\quad P=4D^2')
        st.write('Each Q/K/V/O projection is D×D. Split projected feature columns into h equal heads; '
                 'each head uses its own √d divisor. Concatenation restores width D, never token positions. '
                 'Seed28 tokens and all four matrices stay fixed across choices: only the head partition changes. '
                 'For D=4, projections have 64 entries for every h. A head map has N×N probabilities; '
                 'float64 storage is 8hN² bytes, excluding other arrays. More heads do not prove semantic specialization.')
        st.graphviz_chart('digraph {rankdir=LR; X -> "Fixed Q K V projections" -> "Split features" -> "Per-head attention" -> Concatenate -> "Output projection";}')
        st.caption('Canonical hand example: X=I₂, all projections=I₂, h=2. Output [[p,0.5],[0.5,p]], p=e/(e+1). Invalid h=3 for D=4 is rejected.')
        compute = lambda: heads(count)
    elif ident == 'MP29':
        count = st.selectbox('Encoder heads (width4)', [1,2,4], index=1, key='MP29_heads')
        positional = st.checkbox('Add sinusoidal positions', key='MP29_positions')
        settings.update(heads=count, positions=positional, seed=29, dtype='float64', epsilon=1e-5, dropout=0)
        st.latex(r'Z=X+PE,\quad Y=Z+\mathrm{MHA}(\mathrm{LN}_1(Z)),\quad O=Y+W_2\mathrm{GELU}(W_1\mathrm{LN}_2(Y)+b_1)+b_2')
        st.latex(r'\mathrm{LN}(x)=\gamma(x-\mu)/\sqrt{\mathrm{mean}((x-\mu)^2)+10^{-5}}+\beta')
        st.latex(r'PE_{p,2i}=\sin(p/10000^{2i/D}),\quad PE_{p,2i+1}=\cos(p/10000^{2i/D})')
        st.write('B=batch, N=tokens, D=features. LN uses population variance over D per token, with gain γ and bias β. '
                 'Canonical LN([1,3])≈[−0.999995,0.999995]; PE at position0 is [0,1,0,1]. '
                 'FFN expands D→2D→D; both residual additions preserve B×N×D. Fixed seeded weights are untrained. '
                 'Positions attach coordinates to slots; the displayed RMS change is not reconstruction accuracy.')
        st.graphviz_chart('digraph {rankdir=LR; X -> Z; PE -> Z; Z -> LN1 -> MHA -> Add1; Z -> Add1; Add1 -> LN2 -> "Linear 4 to 8" -> GELU -> "Linear 8 to 4" -> Add2; Add1 -> Add2;}')
        st.caption('Failure cases: odd positional width, incompatible heads or nonfinite tensors are rejected. Epsilon keeps constant-token normalization finite. This pre-LN ordering differs from the original post-LN Transformer.')
        compute = lambda: encoder(count, positional)
    elif ident == 'MP30':
        size = st.selectbox('Patch side P', [1,2,4], index=1, key='MP30_patch')
        shift = st.selectbox('Window boundary shift', [0,1], index=1, key='MP30_shift')
        settings.update(patch=size, shift=shift, window=2, dtype='integer')
        st.latex(r'N=(H/P)(W/P),\quad D_{patch}=P^2C,\quad S_{global}=N^2,\quad S_{local}=NM^2')
        st.write('H,W,C are image height, width, channels; P is patch side; M is window side on a token grid. '
                 'The 4×4 ramp 0…15 with P=2 begins with tokens [0,1,4,5] and [2,3,6,7]. '
                 'Row-major packing is reversible, without embedding or compression. Patch geometry and the separate '
                 '4×4 connectivity grid below are distinct experiments. Windows clip at boundaries, never wrap. '
                 'At shift1, local/shifted/two-layer paths count 64/36/144 versus 256 global pairs. '
                 'A possible path is not a learned attention weight. This is not Swin’s cyclic-shift mask implementation.')
        st.graphviz_chart('digraph {rankdir=LR; Image -> "Row-major patches" -> Tokens -> "Inverse reshape" -> Restored; "4x4 token grid" -> "Local windows" -> "Clipped shifted windows" -> "Two-layer reachability";}')
        st.caption('Failure cases: nondivisible patch geometry is rejected without implicit padding; invalid shift is rejected. Large grids use analytical counts, not dense allocations.')
        compute = lambda: windows(size, shift)
    else:
        height = st.slider('Synthetic four-band height',1,16,5,key='MP31_height')
        width = st.slider('Synthetic four-band width',1,16,7,key='MP31_width')
        settings.update(height=height,width=width,seed=31,dtype='float32',zero_residual=True)
        st.latex(r'\widehat X=B_{bicubic}(X)+\mathrm{PixelShuffle}_2(\mathrm{Conv}(F)),\quad PSNR=10\log_{10}(R^2/MSE)')
        st.write('This compact four-band RGBNIR model uses width16, two heads and unshifted 4×4 windows. '
                 'Replicated window padding is removed before the tail. The zeroed residual tail must equal the bicubic skip; '
                 'this synthetic shape check is not a trained checkpoint prediction. R=1 and MSE=0.01 gives PSNR=20 dB. '
                 'Historical validation below is separate from this control and from the seven hosted RGB models.')
        st.graphviz_chart('digraph {rankdir=LR; "4-band input" -> "Conv width16" -> "Unshifted windows" -> "Zero tail" -> "PixelShuffle x2" -> Add; "4-band input" -> Bicubic -> Add -> "4-band output";}')
        st.caption('Failure case: missing or inconsistent historical records produce a clear error, never invented metrics. Synthetic side lengths are capped at16.')
        compute = lambda: baseline(root, height, width)
    st.markdown('Primary sources: [Attention Is All You Need](https://arxiv.org/abs/1706.03762); '
                '[On Layer Normalization](https://arxiv.org/abs/2002.04745); '
                '[Swin Transformer](https://arxiv.org/abs/2103.14030); [SwinIR](https://arxiv.org/abs/2108.10257). '
                'These compact demonstrations are not full reproductions of those papers.')
    if not st.button('Run '+ident, key=ident+'_run'):
        return
    started = time.perf_counter()
    try:
        result = compute()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        st.error('Lesson could not run: '+str(exc))
        return
    elapsed = time.perf_counter()-started
    if ident == 'MP27':
        for name in ('input_Q','input_K','input_V','output'):
            matrix(name, result[name])
        heatmap('Allowed-key probabilities', result['probability'])
        matrix('Feature-scale experiment (fixed seed27, N=64, d=3)', result['scale_rows'])
        matrix('Analytical float32 score storage, one head only', result['memory'])
        st.write('Output=', result['output'][0][0], '; probability sum=', sum(result['probability'][0]))
    elif ident == 'MP28':
        for name in ('input','output','hand'):
            matrix(name, result[name])
        with st.expander('Fixed projection matrices'):
            for name, values in zip(('Q','K','V','O'), result['projections']):
                matrix(name, values)
        for i, values in enumerate(result['maps']):
            heatmap(f'Head {i+1} probability', values)
        st.write(f"Projection entries={result['parameters']}; head width={result['head_width']}; actual maps.nbytes={result['map_bytes']} (not peak RAM).")
    elif ident == 'MP29':
        for name in ('input','positions','actual_input','output'):
            matrix(name, result[name])
        matrix('Actual shape trace', [{'stage':k,'shape':str(v)} for k,v in result['trace'].items()])
        st.write('LN([1,3])=', result['layernorm_example'], '; residual-change RMS=', result['residual_change_rms'])
        st.write('Parameter entries including biases and LayerNorm=', result['parameters'])
    elif ident == 'MP30':
        for name in ('input','tokens','restored'):
            matrix(name, result[name])
        for name, values in result['maps'].items():
            heatmap(name+' connectivity (binary)', values)
        st.write('Exact round trip=', result['exact_roundtrip'], '; ordered pairs=', result['pairs'])
        matrix('Analytical one-head complete-window score storage, float32', result['memory'])
        st.caption('Memory excludes features, projections, gradients and work buffers; shifted edge counts differ.')
    else:
        record = result['historical']
        st.write('Precomputed historical validation · '+record['protocol'])
        matrix('Historical seed summary (dB)', record['summary'])
        matrix('Six recorded runs', record['rows'])
        matrix('Previously measured efficiency', record['efficiency'])
        st.write('Benchmark environment:', record['benchmark_environment'])
        st.write(record['scope']); st.warning(record['limitations'])
        st.write('Source SHA256:', record['source_hashes'])
        st.write('CNN exceeds this compact Transformer in the recorded mean gain; this does not isolate the cause. '
                 'Seed SD is not geographic uncertainty. Equal updates do not imply equal time or parameter count.')
        st.write('Executed synthetic control:', result['input_shape'], '→', result['output_shape'],
                 '; maximum difference from bicubic=', result['max_difference_from_bicubic'])
        matrix('Synthetic input band0 (not display RGB)', result['input'][0])
        matrix('Synthetic output band0 (not reflectance)', result['output'][0])
    sources = list((root/'stage_07_transformers').glob('MP*/*tools.py'))
    sources += [root/'app/transformer_labs.py', root/'src/rssr/models.py']
    settings['source_sha256'] = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    settings['seconds'] = elapsed
    settings['environment'] = {'numpy':np.__version__, 'torch':torch.__version__,
                               'device':'CPU', 'torch_threads':torch.get_num_threads()}
    settings['scope'] = 'Synthetic teaching run; historical MP31 data explicitly separated; no new scientific performance result'
    st.download_button('Download result and provenance', json.dumps({'settings':settings,'result':result},indent=2),
                       ident+'-lesson.json','application/json',key=ident+'_json')
    table = result.get('scale_rows', result.get('output', result.get('tokens')))
    if ident == 'MP31':
        table = result['historical']['summary']
    st.download_button('Download numerical table', pd.DataFrame(table).to_csv(index=False),
                       ident+'-table.csv','text/csv',key=ident+'_csv')
    st.success(f'{ident} bounded teaching computation completed in {elapsed:.4f} s. No trained-quality claim.')
