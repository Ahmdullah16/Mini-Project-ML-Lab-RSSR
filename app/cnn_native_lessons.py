"""Five native CNN teaching labs; use the repository's canonical models and operators.
Fixtures have known synthetic values; model weights are either explicitly initialized
for a shape/control check or trained only when a bounded optional Run is clicked.
"""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch.nn import functional as F
from rssr.cnn_labs import Reconstruction, UpsamplingCNN
from rssr.models import SRModel

def _fixture():
    y,x=torch.meshgrid(torch.arange(8),torch.arange(10),indexing="ij")
    return (((x+y)%3)/2).float()[None,None]

def _export(ident,rows,settings):
    df=pd.DataFrame(rows);st.dataframe(df,hide_index=True)
    st.download_button("Download measurements",df.to_csv(index=False),ident+"-measurements.csv","text/csv",key=ident+"_csv")
    st.download_button("Download provenance",json.dumps(settings,indent=2),ident+"-settings.json","application/json",key=ident+"_json")

def render(ident):
    st.subheader(ident+" · native CNN teaching experiment")
    torch.set_num_threads(2)
    if ident=="MP18":
        from stage_05_cnn_sr.MP18_cnn_edge_detection_laboratory.mp18_tools import sobel_components
        st.latex(r"G_x=K_x\star X,\quad G_y=K_y\star X,\quad \|G\|_2=\sqrt{G_x^2+G_y^2}")
        st.write("A 3×3 Sobel kernel is the weighted discrete derivative. Each coefficient is divided by 8. Replicate padding extends edge values; constant padding inserts zeros. A vertical 0→1 step has a central horizontal derivative of 0.5, while a constant field has zero replicate-padded gradient. This is a fixed numerical operator, not a trained detector.")
        padding=st.selectbox("MP18 boundary",["replicate","constant"],key="mp18_padding")
        fixture=st.selectbox("MP18 source",["Vertical step","Constant"],key="mp18_fixture")
        st.graphviz_chart('digraph {rankdir=LR; "2D fixture" -> "Sobel x/y" -> "Gradient magnitude";}')
        if not st.button("Run MP18",key="mp18_run"):return
        x=torch.zeros((8,10),dtype=torch.float32)
        if fixture=="Vertical step":x[:,5:]=1
        else:x.fill_(1)
        g=sobel_components(x,padding=padding);mag=torch.linalg.vector_norm(g,dim=0)
        rows=[{"column":i,"source":float(x[4,i]),"Gx":float(g[0,4,i]),"Gy":float(g[1,4,i]),"magnitude":float(mag[4,i])} for i in range(10)]
        st.write(f"Input={tuple(x.shape)}; output={tuple(g.shape)}; max magnitude={mag.max().item():.6f}.")
        _export(ident,rows,{"fixture":fixture,"padding":padding,"kernel":"normalized Sobel, fixed weights","purpose":"operator comparison; no measured edge-detector accuracy"})
    elif ident in ("MP19","MP21"):
        residual=ident=="MP21"
        st.latex(r"\hat X=F_\theta(X)" if not residual else r"\hat X=X+F_\theta(X),\quad \frac{\partial\hat X}{\partial X}=I+\frac{\partial F_\theta}{\partial X}")
        st.write("A canonical two-convolution model maps B×3×H×W to the same shape. Zero residual weights give exact identity only with a skip connection; the direct model returns zero. This test isolates architecture, not learned reconstruction quality. Parameter count and numerical MSE are from the actual initialized model.")
        fixture=st.selectbox(ident+" input",["Periodic pattern","Zeros"],key=ident+"_fixture")
        st.graphviz_chart('digraph {rankdir=LR; Input -> "Conv 3x3" -> ReLU -> "Conv 3x3" -> Output; Input -> "Optional residual add" -> Output;}')
        if not st.button("Run "+ident,key=ident+"_run"):return
        torch.manual_seed(19);model=Reconstruction(bands=3,residual=residual).eval()
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            a=_fixture().expand(-1,3,-1,-1).clone() if fixture=="Periodic pattern" else torch.zeros(1,3,8,10)
            output=model(a)
        rows=[{"case":"input","mean":float(a.mean()),"MSE_to_input":0.},
              {"case":"model output","mean":float(output.mean()),"MSE_to_input":float(((output-a)**2).mean())}]
        st.write(f"Input={tuple(a.shape)}, output={tuple(output.shape)}, parameters={sum(p.numel() for p in model.parameters())}, MSE={rows[1]['MSE_to_input']:.6f}.")
        _export(ident,rows,{"model":"rssr.cnn_labs.Reconstruction","residual":residual,"weights":"explicit zeros for architectural identity test","training":"none"})
    elif ident=="MP20":
        st.latex(r"\hat X=\operatorname{Conv}_5(\operatorname{ReLU}(\operatorname{Conv}_1(\operatorname{ReLU}(\operatorname{Conv}_9(\operatorname{Bicubic}_2(X))))))")
        st.write("SRCNN uses a 9×9 feature convolution, 1×1 mapping convolution, and 5×5 reconstruction convolution after bicubic ×2. For input 1×3×8×10, target is 1×3×16×20. Setting all convolution weights to zero yields zero output (not bicubic); random weights cannot be presented as trained model results.")
        st.graphviz_chart('digraph {rankdir=LR; LR -> "Bicubic x2" -> "9x9 features" -> "1x1 map" -> "5x5 reconstruction" -> HR;}')
        if not st.button("Run MP20",key="mp20_run"):return
        model=SRModel(bands=3,scale=2,kind="srcnn").eval()
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            a=_fixture().expand(-1,3,-1,-1).clone();pred=model(a)
        st.write(f"Source={tuple(a.shape)}, output={tuple(pred.shape)}, output max={pred.abs().max().item():.6f}; parameters={sum(p.numel() for p in model.parameters())}.")
        _export(ident,[{"case":"SRCNN zero-weight control","MSE_against_bicubic":float((F.interpolate(a,scale_factor=2,mode="bicubic",align_corners=False)**2).mean()),"output_max":float(pred.abs().max())}],{"checkpoint":"not loaded; zero-weight architectural control, NOT trained SR","implementation":"rssr.models.SRModel(kind=srcnn)"})
    elif ident=="MP22":
        mode=st.selectbox("MP22 upsampling",["shuffle","transpose","interpolate"],key="mp22_mode")
        st.latex(r"\operatorname{PixelShuffle}_r:\mathbb R^{B\times Cr^2\times H\times W}\rightarrow\mathbb R^{B\times C\times rH\times rW}")
        st.write("PixelShuffle rearranges channels spatially, transposed convolution learns overlap in a strided grid, and interpolation resamples before a convolution. For scale 2, three output bands need 12 pre-shuffle channels. The canonical residual skip always uses bicubic; zero learned weights give the same bicubic image for all three. This is a shape and identity check, not a training comparison.")
        st.graphviz_chart('digraph {rankdir=LR; LR -> "Conv head" -> "Select upsampling branch" -> Residual; LR -> "Bicubic x2" -> Addition; Residual -> Addition;}')
        if not st.button("Run MP22",key="mp22_run"):return
        model=UpsamplingCNN(mode=mode,residual=True).eval()
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            a=_fixture().expand(-1,3,-1,-1).clone();pred=model(a)
            ref=F.interpolate(a,scale_factor=2,mode="bicubic",align_corners=False)
        err=float((pred-ref).abs().max())
        st.write(f"Input={tuple(a.shape)}, output={tuple(pred.shape)}; max |output−bicubic|={err:.8g}; parameters={sum(p.numel() for p in model.parameters())}.")
        _export(ident,[{"mode":mode,"parameters":sum(p.numel() for p in model.parameters()),"max_absolute_residual":err}],{"weights":"explicit zero control","source":"rssr.cnn_labs.UpsamplingCNN","scale":2,"claim":"no trained SR comparison"})
    st.success("Executed canonical, bounded synthetic teaching fixture; no new trained-model or satellite claim.")
