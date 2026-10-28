"""Interactive MP01 uses the canonical numerical helpers, not a duplicate model."""
import importlib.util
import io
import numpy as np
import streamlit as st
from PIL import Image


def render(root):
    spec=importlib.util.spec_from_file_location('mp01_site',root/'stage_01_python_numpy/MP01_digital_image_explorer/image_tools.py')
    tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
    st.subheader('MP01 · interactive input, operations and output')
    st.caption('Executed deterministic teaching fixture, not satellite imagery or an SR quality experiment.')
    source=tools.make_synthetic_image()
    amount=st.slider('Brightness offset (digital levels)',-255,255,20,key='mp01_brightness')
    x=st.slider('Pixel column',0,95,0,key='mp01_x')
    y=st.slider('Pixel row',0,63,0,key='mp01_y')
    output=tools.brighten(source,amount);gray=tools.to_grayscale(output)
    normalized=tools.normalize_rgb(output)
    cols=st.columns(3)
    cols[0].image(source,caption='Input: 64 × 96 × 3, uint8 RGB',width='stretch')
    cols[1].image(output,caption='Saturating brightness output',width='stretch')
    cols[2].image(gray/255,caption='Display-gray output, normalized only for preview',width='stretch')
    before=source[y,x].astype(int);after=output[y,x].astype(int)
    st.latex(r'b_c=\min(255,\max(0,a_c+\Delta)),\quad n_c=b_c/255,\quad g=0.299b_R+0.587b_G+0.114b_B')
    st.write(f'Pixel ({x}, {y}): input {before.tolist()} + offset {amount} → clipped {after.tolist()} → normalized {normalized[y,x].tolist()}.')
    st.write(f'Gray calculation: 0.299×{after[0]} + 0.587×{after[1]} + 0.114×{after[2]} = {gray[y,x]:.5f} digital levels.')
    st.markdown('**Why cast before addition?** An 8-bit integer stores only 0–255: unsigned arithmetic can wrap. The helper computes in signed 16-bit integers, clips, then returns uint8. Clipping is irreversible: a value pushed above255 loses its original excess. The gray weights sum to1, so neutral RGB remains neutral; these weights apply to encoded display values, not physical luminance or reflectance.')
    counts=tools.histogram_u8(np.rint(gray).clip(0,255).astype(np.uint8))
    st.bar_chart(counts,x_label='Rounded gray level (0–255)',y_label='Pixel count')
    st.caption(f'Histogram count sum={int(counts.sum())}; expected64×96=6144. Rounded bins are for this plot, not the floating-point gray calculation.')
    st.graphviz_chart('digraph {rankdir=LR; node [shape=box]; Input -> "int16 addition" -> "Clip 0..255" -> RGB; RGB -> "Divide by255"; RGB -> "Weighted gray" -> Histogram;}')
    st.markdown('**Parameters and trade-offs:** offset Δ is an integer in[−255,255], default20 for a visible teaching change; it is not tuned for reconstruction. Pixel coordinates select inspection only and do not crop the image. The fixed64×96 fixture uses18,432RGB bytes; normalization/grayscale use float32 intermediates. Operations and memory grow linearly with pixel count. Larger brightness offsets can saturate details rather than improve quality.')
    buffer=io.BytesIO();Image.fromarray(output).save(buffer,format='PNG')
    st.download_button('Download computed MP01 output',buffer.getvalue(),'mp01-brightness.png','image/png',key='mp01_output')
