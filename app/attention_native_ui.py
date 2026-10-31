"""On-site MP23–26 deterministic attention controls; no trained quality claims."""
import json
import torch
import pandas as pd
import streamlit as st
from rssr.attention_labs import InspectableAttention
from rssr.models import SRModel


def render(ident):
    st.subheader(ident+" · channel, spatial and residual attention")
    modes={"MP23":"channel","MP24":"spatial","MP25":"combined","MP26":"combined"}
    mode=modes[ident]
    strength=st.slider(ident+" feature amplitude",0.0,1.0,0.5,0.1,key=ident+"_amplitude")
    st.latex(r"z_c=(HW)^{-1}\sum_{h,w}F_{c,h,w},\quad A_c=\sigma(W_2\,\mathrm{ReLU}(W_1z))")
    st.latex(r"A_s=\sigma(\mathrm{Conv}_{7\times7}[\mathrm{mean}_c(F);\mathrm{max}_c(F)]),\quad F'=A_s\odot(A_c\odot F)")
    st.write("The input is B×C×H×W; channel weights broadcast over H,W and spatial weights over C. "
             "Sigmoid is in (0,1). The learned gates below are seeded, randomly initialized "
             "architecture demonstrations, not pre-trained image-restoration weights. "
             "Uniform mode multiplies each feature by 0.5; disabled mode is identity. "
             "The numerical output is a controlled change in a tensor, not a ground-truth SR metric.")
    st.graphviz_chart('digraph {rankdir=LR; Features -> "Global pooling" -> "Channel gate"; Features -> "Spatial pooling"; "Channel gate" -> Multiplication; "Spatial pooling" -> "Spatial gate" -> Multiplication; Multiplication -> Output;}')
    if not st.button("Run "+ident,key=ident+"_run"):return
    torch.manual_seed(23)
    layer=InspectableAttention(4).eval()
    x=torch.arange(4*6*8,dtype=torch.float32).reshape(1,4,6,8)/191*strength
    with torch.inference_mode():
        output,channel,spatial=layer(x,mode=mode)
        off,_,_=layer(x,mode="off")
        uniform,_,_=layer(x,mode="uniform")
    rows=[{"setting":name,"shape":str(tuple(t.shape)),"mean":float(t.mean()),"MSE_vs_disabled":float(((t-off)**2).mean())}
          for name,t in [("disabled",off),("uniform 0.5",uniform),(mode,output)]]
    st.write("Channel gate:",[round(float(t),5) for t in channel.reshape(-1)])
    st.write("Spatial range:",round(float(spatial.min()),6),round(float(spatial.max()),6))
    if ident=="MP26":
        torch.manual_seed(26)
        sr=SRModel(bands=3,scale=2,width=8,kind="attention",zero_residual=True).eval()
        with torch.inference_mode():
            inp=torch.ones(1,3,6,8)*strength
            pred=sr(inp)
        st.write(f"Zero-residual attention-SR control: input={tuple(inp.shape)}, output={tuple(pred.shape)}. "
                 "Model adds a bicubic skip; this is NOT trained inference.")
        rows.append({"setting":"zero-residual ×2 skip","shape":str(tuple(pred.shape)),
                     "mean":float(pred.mean()),"MSE_vs_disabled":None})
    table=pd.DataFrame(rows);st.dataframe(table,hide_index=True)
    st.download_button("Download measurements",table.to_csv(index=False),ident+"-attention.csv","text/csv",key=ident+"_csv")
    st.download_button("Download settings",json.dumps({"project":ident,"mode":mode,"amplitude":strength,
        "seed":23,"source":"rssr.attention_labs.InspectableAttention","claim":"random seeded gates, not trained SR"},indent=2),
        ident+"-settings.json","application/json",key=ident+"_json")
    st.success("Executed canonical seeded attention operator on a bounded synthetic tensor.")
