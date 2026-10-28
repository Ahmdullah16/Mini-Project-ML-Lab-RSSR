"""Shared metric interpretation and explicitly separate synthetic laboratory."""
from pathlib import Path
import streamlit as st

def render(prefix):
    with st.expander('Understand these results · metrics and their limits'):
        st.markdown((Path(__file__).resolve().parents[1]/'docs/metric-instructor.md').read_text(encoding='utf-8-sig'))
    with st.expander('Synthetic metric laboratory · separate from your image'):
        from rssr.metric_teaching import KINDS,demonstrate
        kind=st.selectbox('Controlled distortion',KINDS,key=prefix+'_distortion')
        strength=st.slider('Distortion strength',0.,1.,.5,.05,key=prefix+'_strength')
        st.caption('64×64 synthetic RGB; fixed range[0,1], no border crop, noise seed17. Blur blends a replicate-padded5×5 box filter; noise σ=0.2a; brightness +0.3a; contrast gain1+a about0.5; right shift round(4a) with replicated edges; sharpening adds2a times the box-filter residual. Clip to[0,1] before scoring. a is the strength slider.')
        if not st.button('Run synthetic metric demonstration',key=prefix+'_run_metrics'):return
        r=demonstrate(kind,strength)
        cols=st.columns(3)
        cols[0].image(r['reference'].transpose(1,2,0),caption='Known synthetic reference',clamp=True)
        cols[1].image(r['distorted'].transpose(1,2,0),caption='Distorted synthetic output',clamp=True)
        cols[2].image(r['difference'],caption='Mean absolute RGB difference, fixed0–1 display',clamp=True)
        st.dataframe([{'metric':k,'value':v,'preferred':'higher' if k in ('psnr','ssim') else 'lower'} for k,v in r['metrics'].items()],hide_index=True)
        st.write(f"For this displayed pair: MSE={r['metrics']['mse']:.6g}, RMSE={r['metrics']['rmse']:.6g}, MAE={r['metrics']['mae']:.6g}. These quantify this controlled distortion only. Clipped scalar fraction={r['clipped_fraction']:.4f}.")
        st.caption('Observed scores are not guaranteed to vary monotonically with visual preference. Compare shifts, brightness and sharpening; zero strength should recover the reference. No upload/satellite accuracy or model ranking is implied.')
