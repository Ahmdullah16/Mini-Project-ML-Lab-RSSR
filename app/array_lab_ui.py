"""Explicit-run MP02 demonstration using the canonical shared numerical API."""
import io
import numpy as np
import streamlit as st
from PIL import Image
from rssr.array_lab import brighten, masked_channel_stats


def render():
    st.subheader('MP02 · broadcasting and masked statistics')
    st.caption('Deterministic 2 × 2 RGB fixture; display values, not satellite reflectance. No network or training.')
    offset=st.slider('MP02 red-channel offset',-255,255,10,key='mp02_offset')
    exclude=st.checkbox('Exclude bottom-right pixel from statistics',value=True,key='mp02_exclude')
    st.latex(r'b_{ijc}=\min(255,\max(0,a_{ijc}+\Delta_c)),\quad \Delta=(d,0,0)')
    st.latex(r'N=\sum_{ij}m_{ij},\quad\mu_c=N^{-1}\sum_{ij}m_{ij}b_{ijc},\quad\sigma_c^2=N^{-1}\sum_{ij}m_{ij}(b_{ijc}-\mu_c)^2')
    st.write('Indices i,j select row and column; c selects RGB channel. The mask m is 1 for included pixels and 0 otherwise. Broadcasting repeats the three-entry offset across rows and columns. N is the included pixel count; variance uses N (population), not N−1 (sample estimator).')
    st.graphviz_chart('digraph {rankdir=LR; node[shape=box]; "2x2x3 RGB" -> "float64 + channel offset" -> "Clip 0..255" -> Output; Output -> "Boolean spatial mask" -> "Mean and population variance";}')
    if not st.button('Run MP02 example',key='mp02_run'):
        st.info('Choose parameters, then Run MP02 example to compute the output and worked arithmetic.');return
    source=np.array([[[10,20,30],[40,50,60]],[[70,80,90],[100,110,120]]],dtype=np.uint8)
    mask=np.ones((2,2),dtype=bool);mask[1,1]=not exclude
    output=brighten(source,[offset,0,0]);reference=brighten(source,[offset,0,0],loop=True)
    means,variances=masked_channel_stats(output,mask)
    loop_means,loop_variances=masked_channel_stats(output,mask,loop=True)
    assert np.array_equal(output,reference) and np.allclose(means,loop_means) and np.allclose(variances,loop_variances)
    columns=st.columns(2)
    columns[0].image(source,caption='Input 2 × 2 RGB',width=240)
    columns[1].image(output.astype(np.uint8),caption='Computed output, exact dimensions unchanged',width=240)
    st.write('Input pixels (row-major):',source.reshape(-1,3).tolist())
    st.write('Output pixels (row-major):',output.reshape(-1,3).tolist())
    st.write(f'Included pixels N={int(mask.sum())}; means={means.tolist()}; population variances={variances.tolist()}.')
    red=output[:,:,0][mask]
    st.write(f'Red worked example: μ=({" + ".join(str(v) for v in red)})/{len(red)}={means[0]:.6g}; squared deviations sum={float(((red-means[0])**2).sum()):.6g}; variance={variances[0]:.6g}.')
    st.success('Loop and vectorized output agree; masked means and variances agree.')
    st.markdown('**Interpretation and parameters:** only red receives the offset. Clipping can destroy differences, so a lower variance after clipping is not evidence of better image quality. Excluding a pixel changes the statistical population, not the output image. An empty mask is invalid. Float64 arithmetic avoids uint8 overflow; input occupies 12 bytes and the float64 output 96 bytes. Both implementations require work proportional to H×W×C; this tiny example does not establish a speedup.')
    st.markdown('**Check your understanding:** with offset10 and the last pixel excluded, red values are20,50,80. Their mean is50 and variance(900+0+900)/3=600. With all four pixels included, mean65 and variance1125. Explain why green and blue do not receive the red offset.')
    data=io.BytesIO();Image.fromarray(output.astype(np.uint8)).save(data,format='PNG')
    st.download_button('Download computed MP02 output',data.getvalue(),'mp02-broadcast.png','image/png',key='mp02_download')
