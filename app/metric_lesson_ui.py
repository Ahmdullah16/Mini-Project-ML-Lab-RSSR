"""MP09–11 share a valid reference and canonical metric implementation."""
import numpy as np
import pandas as pd
import streamlit as st
from rssr.metric_lab import mse_psnr, ssim_map
from rssr.degradation_lab import gaussian, filter_hwc
from degradation_batch_ui import show_outputs


def render(ident):
    st.subheader(ident+' · reference-based metric laboratory')
    strength=st.slider('Perturbation strength',0.,1.,.5,.1,key=ident+'_metric_strength')
    window=st.selectbox('SSIM window width',[3,7,11],index=1,key=ident+'_metric_window')
    st.latex(r'\mathrm{MSE}=N^{-1}\sum(p-r)^2,\quad\mathrm{PSNR}=10\log_{10}(R^2/\mathrm{MSE})')
    st.write('p and r are prediction and known aligned reference; N counts all H×W×C samples. R=1 is fixed by the normalized encoding. MSE uses squared intensity units, RMSE its square root, MAE absolute differences, and PSNR decibels. Zero error means infinite PSNR. Example: errors[0.1,−0.1] give MSE0.01 and PSNR20dB; halving both errors gives MSE0.0025 and PSNR26.0206dB. Using R=255 without rescaling these0–1 arrays would incorrectly add48.1308dB.')
    st.latex(r'\mathrm{SSIM}=\frac{(2\mu_p\mu_r+C_1)(2\sigma_{pr}+C_2)}{(\mu_p^2+\mu_r^2+C_1)(\sigma_p^2+\sigma_r^2+C_2)},\quad C_1=(0.01R)^2,\ C_2=(0.03R)^2')
    st.write('μ is a local mean, σ² a population variance and σpr population covariance. Each uniform w×w window has w² samples per channel; no padding, border crop or color conversion is applied. Average valid-window scores over positions and channels. A32×48 input with w=7 gives a26×42 map. Larger windows change spatial context and the included centers. Constant identical patches have SSIM1; constant0.5 versus0.6 gives (0.6+0.0001)/(0.61+0.0001)≈0.983609. Stabilizers prevent division by zero; this is not the original Gaussian-window paper configuration.')
    if ident=='MP11':
        st.markdown('**Fair dashboard:** all rows use this same synthetic reference, range, dimensions and window. Identity is a sanity control, not an SR model. These distortions are not competing trained models. For multiple scenes, report scene-level summaries and uncertainty: mean PSNR generally differs from PSNR computed after averaging MSE. MSE values0.01 and0.04 give mean PSNR16.9897dB versus pooled-error PSNR16.0206dB. No scene-level uncertainty can be inferred from one generated image.')
    st.graphviz_chart('digraph {rankdir=LR; "Known RGB reference" -> "Controlled distortions" -> "Same-grid pairs"; "Known RGB reference" -> "Same-grid pairs" -> "Squared / absolute differences"; "Same-grid pairs" -> "Valid-window moments" -> SSIM; "Squared / absolute differences" -> "MSE and PSNR";}')
    if not st.button('Run '+ident+' metric lab',key=ident+'_metric_run'): return
    y,x=np.mgrid[:32,:48]
    ref=np.stack([.2+.6*x/47,.2+.6*y/31,.2+.6*((x//4+y//4)%2)],-1)
    blur=filter_hwc(ref,gaussian(1,3))
    noisy=np.clip(ref+strength*.1*np.random.default_rng(17).normal(size=ref.shape),0,1)
    predictions={'identity':ref,'blur':ref+strength*(blur-ref),'noise':noisy,'brightness':np.clip(ref+.2*strength,0,1)}
    if ident!='MP11':
        predictions={'brightness':predictions['brightness']}
    rows=[]
    for name,pred in predictions.items():
        mse,psnr=mse_psnr(pred,ref);score=ssim_map(pred,ref,window=int(window))
        rows.append({'case':name,'MSE':mse,'RMSE':np.sqrt(mse),'MAE':np.abs(pred-ref).mean(),'PSNR_dB':psnr,'SSIM':score.mean()})
    show_outputs(ident,{'reference':ref,**predictions},{'strength':strength,'window':window,'range':1,'seed':17,'reference':'synthetic aligned RGB','SSIM':'uniform population moments, valid windows'})
    st.dataframe(pd.DataFrame(rows),hide_index=True)
    st.write(f'Computed {len(rows)} reference comparisons; SSIM map={score.shape[1]} × {score.shape[0]} × {score.shape[2]}.')
    if ident=='MP09':
        st.write(f'Brightness MSE={rows[0]["MSE"]:.6g}; PSNR={rows[0]["PSNR_dB"]:.6g} dB.')
    if ident=='MP10':
        st.image(np.clip((score.mean(2)+1)/2,0,1),caption='Mean-channel SSIM: black=−1, gray=0, white=1; valid-window centers only',clamp=True)
    st.download_button('Download metric table',pd.DataFrame(rows).to_csv(index=False),ident+'-metrics.csv','text/csv',key=ident+'_metric_csv')
    st.success('Computed against the known synthetic reference. Values are not measurements of your upload or satellite scene.')
    st.caption('High similarity does not prove correct missing detail. Small geometric shifts, clipping, range and window changes alter scores. SAM is omitted because display RGB is not calibrated spectral reflectance; perceptual preference and scientific accuracy are separate.')
