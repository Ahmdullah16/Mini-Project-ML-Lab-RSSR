"""MP04 aligned synthetic reference comparison, using canonical float64 arithmetic."""
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rssr.image_foundations import compare_hwc


def render():
    st.subheader('MP04 · errors, masks and alignment')
    st.caption('A known 2 × 2 grayscale reference, in display levels 0–255. Synthetic controlled experiment; not satellite accuracy.')
    offset=st.slider('MP04 brightness error',-20,20,10,key='mp04_offset')
    shift=st.checkbox('Shift prediction one column (circular teaching boundary)',key='mp04_shift')
    exclude=st.checkbox('Exclude bottom-right pixel',key='mp04_exclude')
    st.latex(r'd_{ijc}=p_{ijc}-r_{ijc},\quad N=C\sum_{ij}m_{ij}')
    st.latex(r'\mathrm{bias}=\frac{\sum m d}{N},\quad \mathrm{MAE}=\frac{\sum m|d|}{N},\quad\mathrm{MSE}=\frac{\sum m d^2}{N},\quad\mathrm{RMSE}=\sqrt{\mathrm{MSE}}')
    st.latex(r'\mathrm{PSNR}=10\log_{10}(255^2/\mathrm{MSE})')
    st.write('p is the prediction and r the aligned reference; i,j,c identify row, column and channel. m is a Boolean spatial mask. N counts included channel samples. Bias can cancel opposite errors; MAE cannot. Squaring emphasizes larger errors. MAE, bias and RMSE use display levels; MSE uses squared levels; PSNR uses decibels. Identical arrays have zero MSE and infinite PSNR. The fixed range is 255, not the observed maximum minus minimum.')
    st.graphviz_chart('digraph {rankdir=LR; node[shape=box]; Reference -> "Known perturbation" -> Prediction; Reference -> "Check shape and alignment"; Prediction -> "Check shape and alignment" -> "Float64 subtraction" -> "Mask" -> "Signed / absolute errors" -> "Metrics and interpretation";}')
    st.markdown('**Worked example:** without shifting or masking, adding10 gives four errors of10: bias=10, MAE=10, MSE=(100+100+100+100)/4=100, RMSE=10, PSNR≈28.1308dB. A mask changes the evaluated population, not the image. Equal dimensions alone do not establish alignment. A displaced image can have the same histogram and still large pixel error.')
    if not st.button('Run MP04 example',key='mp04_run'):
        st.info('Choose the perturbation and mask, then run the controlled comparison.');return
    reference=np.array([[30,60],[90,120]],dtype=np.float64)[...,None]
    prediction=np.clip(reference+offset,0,255)
    if shift: prediction=np.roll(prediction,1,axis=1)
    mask=np.ones((2,2),dtype=bool);mask[1,1]=not exclude
    result=compare_hwc(prediction,reference,mask)
    mse=float(result['mse'][0]);mae=float(result['mae'][0]);bias=float(result['bias'][0])
    psnr=float('inf') if mse==0 else float(10*np.log10(255**2/mse))
    for col,label,img in zip(st.columns(2),['Reference','Prediction'],[reference,prediction]):
        col.image(img[:,:,0].astype(np.uint8),caption=label,width=220)
    st.write(f'Included pixels={result["valid_pixels"]}; bias={bias:.6g}; MAE={mae:.6g}; MSE={mse:.6g}; RMSE={np.sqrt(mse):.6g}; PSNR={psnr:.6g} dB.')
    rows=[]
    for i,j in np.ndindex(mask.shape):
        d=float(result['signed'][i,j,0]);rows.append({'row':i,'column':j,'included':bool(mask[i,j]),'reference':reference[i,j,0],'prediction':prediction[i,j,0],'signed error':d,'absolute error':abs(d),'squared error':d*d})
    frame=pd.DataFrame(rows);st.dataframe(frame,hide_index=True)
    st.write('Signed error table (prediction − reference; display levels):')
    def error_color(value):
        strength=min(abs(value)/50,1)
        endpoint=np.array([255,140,140] if value>=0 else [120,170,255])
        rgb=np.rint(255+strength*(endpoint-255)).astype(int)
        return f'background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); color: black'
    st.dataframe(pd.DataFrame(result['signed'][:,:,0]).style.map(error_color))
    st.caption('Fixed heatmap range −50 (blue) to +50 (red); zero is neutral. Masked pixels remain visible in this table but are excluded from metrics. No automatic contrast normalization.')
    st.download_button('Download MP04 error table',frame.to_csv(index=False),'mp04-errors.csv','text/csv',key='mp04_csv')
    data=io.BytesIO();Image.fromarray(prediction[:,:,0].astype(np.uint8)).save(data,format='PNG')
    st.download_button('Download MP04 prediction',data.getvalue(),'mp04-prediction.png','image/png',key='mp04_png')
    st.success('Computed against the known synthetic reference. A shift deliberately violates registration; the resulting errors illustrate that failure, not an SR quality ranking.')
    st.markdown('**Interpretation:** use both the signed map and scalar averages. A positive bias indicates systematic brightening; opposite signed errors can hide in the mean. SSIM requires an explicitly chosen spatial window and is not reported on this tiny fixture. SAM requires meaningful compatible spectral bands and is not appropriate for this single display channel. Real HR evaluation also requires matched geolocation, resolution, acquisition conditions and valid-pixel masks; none is established by this synthetic example.')
