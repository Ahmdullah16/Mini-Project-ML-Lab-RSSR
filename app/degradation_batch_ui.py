"""Related MP06–08 labs share bounded fixtures, previews and export conventions."""
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rssr.degradation_lab import degrade_hwc, spectrum


def show_outputs(ident, outputs, settings):
    for col,(name,array) in zip(st.columns(len(outputs)),outputs.items()):
        view=np.clip(array.squeeze(),0,1)
        col.image(view,caption=f'{name}: {array.shape[1]} × {array.shape[0]}',clamp=True)
        data=io.BytesIO();Image.fromarray(np.rint(view*255).astype('uint8')).save(data,format='PNG')
        col.download_button(f'Download {name}',data.getvalue(),f'{ident}-{name}.png','image/png',key=f'{ident}_{name}_png')
    st.caption('Fixed display range0–1. PNG previews clip/round to8-bit; raw floating-point values are exported separately. Synthetic fixtures, no external imagery or trained model.')
    raw=io.BytesIO();np.savez_compressed(raw,**outputs)
    st.download_button('Download raw arrays',raw.getvalue(),f'{ident}-arrays.npz','application/octet-stream',key=ident+'_raw')
    st.download_button('Download settings',json.dumps(settings,indent=2),f'{ident}-settings.json','application/json',key=ident+'_settings')


def render(ident):
    st.subheader(ident+' · interactive sampling, degradation and frequency laboratory')
    y,x=np.indices((32,48));checker=((x+y)%2).astype(float)[...,None]
    if ident=='MP06':
        scale=st.selectbox('Sampling factor',[2,4],key='mp06_scale')
        sigma=st.slider('Prefilter sigma',0.,2.,1.,.5,key='mp06_sigma')
        st.latex(r'z=k_\sigma*x,\quad y_{ijc}=s^{-2}\sum_{a,b=0}^{s-1}z_{si+a,sj+b,c}')
        st.write('x is the input H×W×C image, k a normalized Gaussian, s the integer reduction factor, and z the blurred image. Output dimensions are H/s × W/s; this lab requires divisibility. D here is area averaging over nonoverlapping s×s blocks. The separate point sampler takes x[si,sj] without filtering. σ=0 bypasses Gaussian blur but area averaging still filters; it is not pure decimation.')
        st.markdown('**Example:** a2×2 checkerboard [0,1;1,0] averages to0.5, while its upper-left point sample is0. Repeated sampling aliases fine texture into a false constant. At s=4, each output averages16 samples. More blur removes more high-frequency information; it cannot recover discarded detail. Edge replication is used for Gaussian padding; radius=ceil(3σ).')
        st.graphviz_chart('digraph {rankdir=LR; Input -> "Point sampling"; Input -> "Gaussian (optional)" -> "Area averaging"; Input -> "Area averaging without Gaussian";}')
        if not st.button('Run MP06',key='mp06_run'): return
        area=degrade_hwc(checker,scale,0,0);blurred=degrade_hwc(checker,scale,sigma,0)
        point=checker[::scale,::scale]
        show_outputs(ident,{'input':checker,'point':point,'area':area,'prefiltered':blurred},{'scale':scale,'sigma':sigma,'boundary':'edge','range':[0,1]})
        st.write(f'Point mean={point.mean():.6g}; area mean={area.mean():.6g}; output={48//scale} × {32//scale}.')
        st.success('Computed three sampling paths. These are synthetic LR generation methods, not super-resolution.')
    elif ident=='MP07':
        noise=st.slider('Noise standard deviation',0.,.2,.1,.05,key='mp07_noise')
        seed=st.number_input('Noise seed',0,999,17,key='mp07_seed')
        quality=st.slider('JPEG quality',10,95,75,5,key='mp07_quality')
        st.latex(r'y_{after}=D_2(x)+n_{LR},\quad y_{before}=D_2(x+n_{HR}),\quad n\sim\mathcal N(0,\sigma_n^2)')
        st.write('D₂ averages each2×2 block. Independent zero-mean Gaussian noise is added either before or after sampling, using the declared seed. σn is standard deviation in0–1 intensity units. Averaging four independent samples changes noise variance from σn² to σn²/4, so standard deviation becomes σn/2. With σn=0.1 the expected variances are0.01 versus0.0025; finite-sample measurements differ. Seeds make runs reproducible, not scientifically independent scenes.')
        st.markdown('**Compression is different:** JPEG quantizes transform coefficients after clipping/rounding to8-bit. Its quality control is an encoder parameter, not a calibrated noise variance or percentage of accuracy. The JPEG branch below starts from the same noisy LR image; report its error against that image as compression change, not HR accuracy. No Gaussian blur in this controlled order experiment; MP05/06 isolate blur.')
        st.graphviz_chart('digraph {rankdir=LR; Input -> "Area mean" -> "LR noise" -> "8-bit clipping" -> JPEG; Input -> "HR noise" -> "Area mean of noisy HR";}')
        if not st.button('Run MP07',key='mp07_run'): return
        source=np.full((32,48,1),.5);clean=degrade_hwc(source,2,0,0)
        after=degrade_hwc(source,2,0,noise,int(seed))
        before=degrade_hwc(source+np.random.default_rng(int(seed)).normal(0,noise,source.shape),2,0,0)
        encoded=np.rint(np.clip(after[:,:,0],0,1)*255).astype('uint8');buf=io.BytesIO()
        Image.fromarray(encoded).save(buf,format='JPEG',quality=quality);size=len(buf.getvalue());buf.seek(0)
        with Image.open(buf) as im: decoded=np.asarray(im,dtype=float)[...,None]/255
        show_outputs(ident,{'clean':clean,'noise_after':after,'noise_before':before,'jpeg':decoded},{'seed':int(seed),'noise_std':noise,'scale':2,'jpeg_quality':quality})
        st.write(f'Noise-after MSE={np.mean((after-clean)**2):.8f}; noise-before MSE={np.mean((before-clean)**2):.8f}; JPEG bytes={size}.')
        st.write(f'JPEG-only MSE against quantized encoder input={np.mean((decoded[:,:,0]-encoded/255)**2):.8f}. Raw noise is not clipped for the noise metrics.')
        st.success('Executed reproducible noise-order and JPEG branches. This is a controlled synthetic experiment, not a real sensor noise model.')
    else:
        cycles=st.slider('Horizontal cycles across48 pixels',1,20,6,key='mp08_cycles')
        cutoff=st.slider('Low-pass cutoff (cycles/pixel)',0.,.5,.1,.025,key='mp08_cutoff')
        st.latex(r'F_{uv}=\sum_{i=0}^{H-1}\sum_{j=0}^{W-1}x_{ij}e^{-2\pi\mathrm{i}(ui/H+vj/W)},\quad x=\mathrm{IFFT2}(F)')
        st.latex(r'\sum |x|^2=(HW)^{-1}\sum |F|^2,\quad \hat x=\mathrm{IFFT2}(F\,\mathbf1_{\sqrt{f_x^2+f_y^2}\leq f_c})')
        st.write('u,v index frequency bins; i,j index pixels; the upright imaginary unit satisfies i²=−1. NumPy uses an unnormalized forward DFT and a1/(HW) inverse. fx,fy and cutoff fc are cycles/pixel. A sinusoid with6cycles over48pixels has frequency0.125cycles/pixel and symmetric peaks at±0.125. DC is the mean component. A cutoff0.1 removes those peaks while retaining the mean0.5.')
        st.markdown('**Assumptions:** the DFT treats the image as periodic. Multiplying by a frequency mask corresponds to circular convolution, not edge-replicated spatial filtering. A sharp cutoff can ring near discontinuities; the integer-cycle sinusoid here avoids boundary discontinuity. Display log(1+|F|) reveals weaker components but is not physical power. No resolution or true detail is created.')
        st.graphviz_chart('digraph {rankdir=LR; Sinusoid -> FFT -> "Radial frequency mask" -> IFFT -> "Real output"; FFT -> "Shift + log magnitude";}')
        if not st.button('Run MP08',key='mp08_run'): return
        source=.5+.4*np.cos(2*np.pi*cycles*x/48);f,logmag=spectrum(source)
        fy=np.fft.fftfreq(32)[:,None];fx=np.fft.fftfreq(48)[None,:];mask=np.hypot(fx,fy)<=cutoff
        filtered=np.fft.ifft2(f*mask).real;roundtrip=np.fft.ifft2(f).real
        show_outputs(ident,{'input':source,'filtered':filtered},{'cycles':cycles,'cutoff_cycles_per_pixel':cutoff,'boundary':'periodic','fft_normalization':'backward'})
        st.image(logmag/max(float(logmag.max()),1),caption='Shifted log magnitude;0=black, largest log magnitude=white',clamp=True)
        st.line_chart(pd.DataFrame({'input':source[0],'filtered':filtered[0]}),x_label='Column',y_label='Intensity')
        residual=abs(np.square(source).sum()-(np.abs(f)**2).sum()/source.size)
        st.write(f'Inverse max error={np.abs(roundtrip-source).max():.3g}; Parseval residual={residual:.3g}; filtered mean={filtered.mean():.6g}; filtered standard deviation={filtered.std():.6g}.')
        st.success('FFT, inverse and low-pass output computed. Numerical residuals reflect floating-point rounding.')
