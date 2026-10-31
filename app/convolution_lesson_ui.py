"""MP17 native convolution lesson uses shared rssr.convolution_lab."""
import io, json
import numpy as np
import pandas as pd
import streamlit as st
from rssr.convolution_lab import correlate, output_size, receptive_field


def render():
    st.subheader("MP17 · Convolution playground: stride, padding and dilation")
    kernel=st.selectbox("MP17 kernel width",[1,3,5],index=1,key="mp17_kernel")
    stride=st.selectbox("MP17 stride",[1,2],key="mp17_stride")
    pad=st.selectbox("MP17 zero-padding",[0,1,2],index=1,key="mp17_padding")
    dilation=st.selectbox("MP17 dilation",[1,2],key="mp17_dilation")
    fixture=st.selectbox("MP17 input",["Center impulse","Horizontal ramp","Constant"],key="mp17_fixture")
    st.latex(r"H_o=\left\lfloor\frac{H+2p-d(k-1)-1}{s}+1\right\rfloor,\quad "
             r"W_o=\left\lfloor\frac{W+2p-d(k-1)-1}{s}+1\right\rfloor")
    st.latex(r"y[i,j]=\sum_{u,v}K[u,v]x[is+ud-p,js+vd-p]")
    st.write("H,W are input height/width (9×9), k is kernel width, p is symmetric zero-padding "
             "in input pixels, d dilation (spacing between sampled kernel taps), and s stride "
             "(step between output positions). Unlike strict mathematical convolution, "
             "the canonical code computes CNN cross-correlation without flipping the kernel. "
             "Output pixels are finite float64 teaching values.")
    st.write("Example: H=9, k=3, p=1, d=1, s=2 produces "
             "floor((9+2−2−1)/2+1)=5 output rows. "
             "A 3×3 average kernel has weights 1/9; a center unit impulse produces "
             "a 3×3 neighborhood of 1/9 at stride 1. At boundaries, zero-padding lowers "
             "the response of a constant field. Increasing stride reduces sampling density; "
             "dilation expands receptive span without increasing tap count.")
    st.graphviz_chart('digraph {rankdir=LR; Input -> "Zero pad" -> "Dilated cross-correlation" -> "Stride sample" -> Output; "k,s,d" -> "Receptive field";}')
    if not st.button("Run MP17",key="mp17_run"):return
    x=np.zeros((9,9),float)
    if fixture=="Center impulse":x[4,4]=1
    elif fixture=="Horizontal ramp":x[:]=np.arange(9)[None,:]/8
    else:x.fill(1)
    k=np.ones((kernel,kernel),float)/(kernel*kernel)
    try:
        expected=(output_size(9,kernel,stride,pad,dilation),)*2
        y=correlate(x,k,stride,pad,dilation)
    except ValueError as e:
        st.warning(f"Unsupported geometry: {e}");return
    assert y.shape==expected and np.isfinite(y).all()
    rf=receptive_field([(kernel,stride,dilation)])
    left,right=st.columns(2)
    left.dataframe(pd.DataFrame(x),hide_index=True)
    right.dataframe(pd.DataFrame(y),hide_index=True)
    st.write(f"Input=9 × 9; output={y.shape[0]} × {y.shape[1]}; "
             f"receptive-field span={rf['span']} input pixels; jump={rf['jump']} pixels; "
             f"output sum={y.sum():.8g}.")
    st.line_chart(pd.DataFrame({"input row":x[4,:],"input average":np.full(9,x.mean())}),x_label="Source column",y_label="Intensity")
    rows=[{"output_y":i,"output_x":j,"value":float(y[i,j])}
          for i in range(y.shape[0]) for j in range(y.shape[1])]
    st.download_button("Download MP17 output table",pd.DataFrame(rows).to_csv(index=False),
                       "mp17-output.csv","text/csv",key="mp17_csv")
    st.download_button("Download MP17 configuration",json.dumps({"kernel":kernel,"stride":stride,
        "padding":pad,"dilation":dilation,"fixture":fixture,"shape":list(y.shape),"boundary":"zero",
        "operation":"canonical cross-correlation","range":"float64 uncalibrated teaching"},indent=2),
        "mp17-settings.json","application/json",key="mp17_json")
    st.success("Actual canonical cross-correlation executed on deterministic synthetic input.")
    st.caption("No learned convolution or remote-sensing quality is claimed. Full photographic "
               "experiment, matched reference and learner mastery remain separate.")
