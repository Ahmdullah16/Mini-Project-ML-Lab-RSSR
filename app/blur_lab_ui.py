"""MP05 bounded Gaussian experiment using the canonical NumPy filter."""
import io
from time import perf_counter
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from rssr.degradation_lab import gaussian, filter_hwc


def render():
    st.subheader('MP05 · Gaussian blur and boundary assumptions')
    pattern=st.selectbox('MP05 input',['Vertical step','Impulse','Constant'],key='mp05_pattern')
    sigma=st.slider('MP05 sigma (input pixels)',0.5,3.0,1.0,0.5,key='mp05_sigma')
    radius=st.slider('MP05 kernel radius',0,9,3,key='mp05_radius')
    boundary=st.selectbox('MP05 padding',['edge','reflect','constant'],key='mp05_boundary')
    st.latex(r'g(t)=\frac{\exp(-t^2/(2\sigma^2))}{\sum_{u=-r}^{r}\exp(-u^2/(2\sigma^2))},\quad K_{ab}=g(a)g(b)')
    st.latex(r'y_{ijc}=\sum_{a=-r}^{r}\sum_{b=-r}^{r}K_{ab}x_{i+a,j+b,c}')
    st.write('r is the integer radius, so kernel width is2r+1. σ controls spread in input-pixel units. Normalize the truncated weights to sum to one; their outer product gives a separable 2D kernel. Each output is a weighted local average, independently per channel. The implementation correlates the neighborhood; for this symmetric kernel correlation equals convolution. Output H×W×C is unchanged.')
    st.markdown('**Parameter effects:** increasing σ at fixed radius distributes more weight away from the center, until truncation limits the effect. Increasing radius retains more of the Gaussian tail. Radius0 gives the identity for every σ. Edge padding repeats edge values; reflect mirrors interior samples; constant pads with zero and can darken borders. Direct filtering costs O(HWC(2r+1)²); separable filtering can reduce arithmetic but this teaching implementation uses the full2D kernel.')
    st.markdown('**Worked example:** σ=1,r=1 gives normalized1D weights approximately[0.274069,0.451863,0.274069]. The center2D weight is0.451863²≈0.204180. A centered unit impulse produces these kernel weights in its neighborhood. A constant image stays constant with edge padding because the weights sum to one.')
    st.graphviz_chart('digraph {rankdir=LR; node[shape=box]; "sigma + radius" -> "Normalize 1D Gaussian" -> "Outer product K"; "Synthetic image" -> "Declared padding" -> "Weighted local average"; "Outer product K" -> "Weighted local average" -> "Same-size output + row profile";}')
    if not st.button('Run MP05 example',key='mp05_run'):
        st.info('Choose parameters, then run to compute the kernel and filtered output.');return
    source=np.zeros((25,33,1),dtype=np.float64)
    if pattern=='Vertical step': source[:,16:]=1
    elif pattern=='Impulse': source[12,16]=1
    else: source[:]=0.5
    start=perf_counter();kernel=gaussian(sigma,radius);out=filter_hwc(source,kernel,boundary);elapsed=perf_counter()-start
    for col,name,img in zip(st.columns(2),['Input','Blurred output'],[source,out]):
        col.image(img[:,:,0],caption=f'{name}: 33 × 25, fixed display range0–1',clamp=True)
    st.dataframe(pd.DataFrame(kernel));st.caption(f'Kernel sum={kernel.sum():.8f}; center weight={kernel[radius,radius]:.8f}; size={2*radius+1} × {2*radius+1}.')
    st.line_chart(pd.DataFrame({'input':source[12,:,0],'output':out[12,:,0]}),x_label='Column',y_label='Middle-row intensity')
    st.write(f'Output range=[{out.min():.6g}, {out.max():.6g}]; max absolute change={np.abs(out-source).max():.6g}; computation={elapsed:.6g} seconds.')
    st.caption('Raw float64 arithmetic is unclipped. Display uses0–1; PNG export rounds to8-bit. Larger-looking smooth regions are not recovered detail; blur suppresses local variation. This synthetic filter is not a calibrated satellite sensor model.')
    data=io.BytesIO();Image.fromarray(np.rint(np.clip(out[:,:,0],0,1)*255).astype(np.uint8)).save(data,format='PNG')
    st.download_button('Download MP05 output',data.getvalue(),'mp05-blur.png','image/png',key='mp05_png')
    st.download_button('Download MP05 kernel',pd.DataFrame(kernel).to_csv(index=False),'mp05-kernel.csv','text/csv',key='mp05_csv')
    st.success('Computed kernel and output; compare the row profile and boundaries before interpreting smoothness.')
