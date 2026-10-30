"""MP03: bounded interpolation lab using the canonical teaching implementation."""
import io
from time import perf_counter
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rssr.image_foundations import resize_hwc, compare_hwc


def render():
    st.subheader('MP03 · where do resized pixels come from?')
    st.caption('Synthetic grayscale, HWC float64 in [0,1]; no learned model, network or satellite reflectance.')
    fixture=st.selectbox('MP03 input',['Checkerboard','Horizontal ramp'],key='mp03_fixture')
    factor=st.selectbox('MP03 resize factor',[0.5,1,2,4],index=2,key='mp03_factor')
    st.latex(r'x=(j+\tfrac12)W_{in}/W_{out}-\tfrac12,\qquad y=(i+\tfrac12)H_{in}/H_{out}-\tfrac12')
    st.write('i and j are output row and column indices, starting at zero. Map pixel centers into the input, then clamp coordinates to its edges. W and H mean width and height. Nearest rounds each coordinate with floor(x+0.5); an exact tie selects the larger index.')
    st.latex(r'v=(1-a)(1-b)v_{00}+(1-a)b v_{01}+a(1-b)v_{10}+ab v_{11}')
    st.write('For bilinear interpolation, a=y−floor(y) and b=x−floor(x). The four v values are adjacent input pixels. First interpolate along each row, then between the rows. Nonnegative weights sum to one, so this method cannot exceed the input range. Edge neighbors repeat. No antialias filter is used: shrinking can alias.')
    st.markdown('**Worked example:** [0,10] enlarged from width2 to width4 has clamped coordinates [0,0.25,0.75,1]. Nearest gives [0,0,10,10]; bilinear gives [0,2.5,7.5,10]. At x=0.25, (1−0.25)×0+0.25×10=2.5. These inserted values are estimates, not recovered observations.')
    st.graphviz_chart('digraph {rankdir=LR; node[shape=box]; "8x12 synthetic input" -> "Half-pixel coordinates"; "Half-pixel coordinates" -> "Nearest: one neighbor"; "Half-pixel coordinates" -> "Bilinear: four weights"; "Nearest: one neighbor" -> "Exact-sized outputs"; "Bilinear: four weights" -> "Exact-sized outputs"; "Exact-sized outputs" -> "Row profile and disagreement";}')
    st.markdown('**Parameters and cost:** multiplying both dimensions by s multiplies pixel count by s². Nearest uses one sample per output channel; bilinear uses four weighted samples. Both cost O(Hout×Wout×C). Bicubic uses a larger cubic neighborhood and can overshoot; it is discussed in the lesson but is not executed by this two-method lab.')
    if not st.button('Run MP03 example',key='mp03_run'):
        st.info('Choose a pattern and scale, then run to measure both outputs.');return
    rows,cols=np.indices((8,12))
    source=(((rows+cols)%2) if fixture=='Checkerboard' else cols/11).astype(np.float64)[...,None]
    shape=(int(8*factor),int(12*factor))
    outputs={};timings={}
    for method in ['nearest','bilinear']:
        start=perf_counter();outputs[method]=resize_hwc(source,shape,method);timings[method]=perf_counter()-start
    columns=st.columns(3)
    columns[0].image(source[:,:,0],caption='Input: 12 × 8',clamp=True)
    for column,(method,out) in zip(columns[1:],outputs.items()):
        column.image(out[:,:,0],caption=f'{method}: {shape[1]} × {shape[0]}',clamp=True)
        data=io.BytesIO();Image.fromarray(np.rint(out[:,:,0]*255).astype(np.uint8)).save(data,format='PNG')
        column.download_button(f'Download MP03 {method}',data.getvalue(),f'mp03-{method}.png','image/png',key=f'mp03_download_{method}')
    diff=compare_hwc(outputs['bilinear'],outputs['nearest'])
    st.write(f'Executed output dimensions: {shape[1]} × {shape[0]}; {shape[0]*shape[1]} pixels. Each float64 output occupies {outputs["nearest"].nbytes} bytes (not total process memory).')
    st.dataframe(pd.DataFrame([{'method':m,'seconds':timings[m],'minimum':float(o.min()),'maximum':float(o.max())} for m,o in outputs.items()]),hide_index=True)
    st.line_chart(pd.DataFrame({m:o[shape[0]//2,:,0] for m,o in outputs.items()}),x_label='Output column',y_label='Middle-row intensity [0,1]')
    st.write(f'Method disagreement MAE={float(diff["mae"][0]):.6g}; MSE={float(diff["mse"][0]):.6g}. These compare two interpolations, not either output against ground-truth HR.')
    st.caption('PNG downloads contain the displayed full output, rounded to 8-bit grayscale; arithmetic and row profiles use unrounded float64.')
    if fixture=='Checkerboard' and factor==0.5:
        st.success('Measured aliasing example: nearest is constant 0; bilinear is constant 0.5. Neither preserves the original alternating pattern.')
    else:
        st.success('Both methods executed. Nearest repeats samples; bilinear mixes neighbors. Smoother appearance does not establish higher reconstruction accuracy.')
