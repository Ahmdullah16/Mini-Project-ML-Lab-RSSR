"""Native MP12–16 teaching labs: bounded synthetic numerical experiments.

Reuse rssr.linear_lab, rssr.training_lab and rssr.pipeline_lab, never
a second model/SVD implementation or held-out research data.
"""
import io
import json
import math
import numpy as np
import pandas as pd
import streamlit as st

from rssr.linear_lab import approximate
from rssr.training_lab import linear_loss, initialize, mlp_loss, forward, finite_difference_error
from rssr.pipeline_lab import make_loader, weighted_mean, tiny_regression


def _downloads(ident, data, settings):
    """Reproducible small raw-number and settings exports, not frozen research results."""
    rows=pd.DataFrame(data)
    st.dataframe(rows, hide_index=True)
    st.download_button("Download numerical table", rows.to_csv(index=False),
                       f"{ident}-teaching.csv", "text/csv", key=f"{ident}_table_csv")
    st.download_button("Download settings/provenance", json.dumps(settings, indent=2),
                       f"{ident}-teaching.json", "application/json", key=f"{ident}_table_json")


def _mp12():
    st.subheader("MP12 · SVD, linear algebra and image matrices")
    fixture=st.selectbox("MP12 input matrix",["Diagonal rank three","Rectangular rank one"],key="mp12_fixture")
    rank=st.slider("MP12 retained singular directions (rank)",0,3,1,key="mp12_rank")
    st.latex(r"X=U\Sigma V^\top,\qquad X_r=\sum_{i=1}^r \sigma_i u_i v_i^\top")
    st.latex(r"\|X-X_r\|_F^2=\sum_{i=r+1}^{q}\sigma_i^2,\qquad \mathrm{MSE}=\|X-X_r\|_F^2/(HW)")
    st.write("X is an H×W numerical grayscale matrix; U and V contain orthonormal singular vectors, "
             "σ are nonnegative singular values ordered largest first, q=min(H,W), and r is the "
             "retained rank. Singular vectors and pixels are dimensionless here. The Frobenius norm "
             "is the square root of the sum of squared entries. For the diagonal matrix diag(3,2,1), "
             "rank 1 keeps only 3, so squared error is 2²+1²=5, norm √5, and MSE=5/9. "
             "Increasing rank cannot increase optimal squared reconstruction error.")
    st.write("The economy-SVD factor representation stores r(H+W+1) float64 numbers; the dense matrix "
             "stores HW. Small low-rank matrices can require MORE bytes. This counts numerical storage "
             "only; decompression, file headers and display clipping are excluded.")
    st.graphviz_chart("digraph {rankdir=LR; Matrix -> SVD -> \"Choose rank\" -> Reconstruction -> Residual -> \"Error / storage\";}")
    if not st.button("Run MP12",key="mp12_run"):return
    x=np.diag([3.,2.,1.]) if fixture.startswith("Diagonal") else np.outer([1.,2.,3.],[2.,-1.,4.,0.])
    r=min(rank,min(x.shape))
    estimate,detail=approximate(x,r)
    residual=x-estimate
    st.write(f"Input {x.shape[0]}×{x.shape[1]}; rank={r}; "
             f"error norm={detail['frobenius_error']:.6f}; "
             f"tail energy={detail['tail_energy']:.6f}; "
             f"MSE={np.mean(residual**2):.6f}.")
    st.write(f"Dense bytes={detail['dense_bytes']}; factor bytes={detail['factor_bytes']}. "
             "These are raw float64 values, not PNG sizes.")
    cols=st.columns(3)
    for col,(label,arr) in zip(cols,[("Input X",x),("Rank-r reconstruction",estimate),("Signed residual X−Xr",residual)]):
        col.caption(label)
        col.dataframe(pd.DataFrame(arr),hide_index=True)
    _downloads("MP12",[{"singular_index":i+1,"singular_value":float(v),
                         "retained":i<r} for i,v in enumerate(detail["singular_values"])],
               {"fixture":fixture,"rank":r,"shape":list(x.shape),
                "frobenius_error":detail["frobenius_error"],"tail_energy":detail["tail_energy"],
                "MSE":float(np.mean(residual**2)),"units":"arbitrary float64 teaching values",
                "source":"canonical rssr.linear_lab.approximate"})
    st.success("Executed actual canonical SVD. This is not super-resolution or evidence of learned detail.")
    st.caption("Failure case: rank zero returns zeros; full rank reconstructs to floating-point precision. "
               "No sensor reflectance calibration or test image was used.")


def _mp13():
    st.subheader("MP13 · Linear MSE and exact gradient descent")
    rate=st.selectbox("MP13 learning rate η",[0.01,0.1,0.5,1.1],index=1,key="mp13_rate")
    steps=st.slider("MP13 gradient steps",0,30,1,key="mp13_steps")
    st.latex(r"\hat y=Xw+b,\quad L=\frac1N\sum_i(\hat y_i-y_i)^2,\quad "
             r"\nabla_w L=\frac2N X^\top(\hat y-y),\quad \nabla_b L=\frac2N\sum_i(\hat y_i-y_i)")
    st.write("Use X=[1,2]ᵀ, y=[3,5]ᵀ, initial weight w=0 and bias b=0. "
             "The two squared errors are 9 and 25, so L₀=(9+25)/2=17. "
             "Its gradients are dw=−13 and db=−8. At η=0.1, the first step gives w=1.3, b=0.8; "
             "new predictions [2.1,3.4] and loss (0.9²+1.6²)/2=1.685. "
             "A high η can oscillate or diverge; iterations and rate are not accuracy guarantees.")
    st.graphviz_chart("digraph {rankdir=LR; \"X,w,b\" -> Prediction -> Residual -> \"MSE and gradients\" -> \"SGD update\" -> \"X,w,b\";}")
    if not st.button("Run MP13",key="mp13_run"):return
    x=np.array([[1.],[2.]],dtype=np.float64);y=np.array([[3.],[5.]],dtype=np.float64)
    w=np.zeros((1,1),dtype=np.float64);b=np.zeros(1,dtype=np.float64)
    records=[]
    for step in range(steps+1):
        loss,g=linear_loss(x,y,w,b)
        records.append({"step":step,"w":float(w.item()),"b":float(b.item()),
                        "MSE":loss,"dL_dw":float(g["w"].item()),"dL_db":float(g["b"].item())})
        if step<steps:
            w-=rate*g["w"];b-=rate*g["b"]
    st.line_chart(pd.DataFrame(records).set_index("step")[["MSE"]],x_label="Step",y_label="MSE")
    st.write(f"Initial MSE={records[0]['MSE']:.6f}; final MSE={records[-1]['MSE']:.6f}; "
             f"first gradients=({records[0]['dL_dw']:.0f},{records[0]['dL_db']:.0f}).")
    _downloads("MP13",records,{"learning_rate":rate,"steps":steps,"X":[1,2],"y":[3,5],
               "initial_weight":0,"initial_bias":0,"optimizer":"full-batch gradient descent",
               "loss":"mean squared error","source":"rssr.training_lab.linear_loss"})
    st.success("Computed every gradient using the canonical NumPy regression implementation.")
    st.caption("These two synthetic examples are not separate train/test regions; this is optimization teaching only.")


def _mp14():
    st.subheader("MP14 · NumPy neural network and backpropagation")
    hidden=st.selectbox("MP14 hidden neurons",[2,4,8],index=1,key="mp14_hidden")
    steps=st.slider("MP14 parameter updates",0,40,10,5,key="mp14_steps")
    st.latex(r"H=\tanh(XW_1+b_1),\quad \hat Y=HW_2+b_2,\quad "
             r"L=\mathrm{mean}((\hat Y-Y)^2)")
    st.latex(r"\delta_2=\frac2{NO}(\hat Y-Y),\quad "
             r"\delta_1=(\delta_2 W_2^\top)\odot(1-H^2),\quad "
             r"\frac{\partial L}{\partial W_1}=X^\top\delta_1")
    st.write("N=3 samples, D=1 feature, K hidden neurons and O=1 target. "
             "X has N×D shape, W₁ is D×K, H is N×K, W₂ is K×O and output is N×O. "
             "Tanh derivative is 1−tanh²(z); the second chain-rule term propagates output "
             "error through W₂. Loss uses the mean over all N×O scalar samples. "
             "For a scalar residual e=0.1, its squared contribution is 0.01 before averaging.")
    st.write("Gradients are checked by central differences "
             "(L(θ+ε)−L(θ−ε))/(2ε), ε=10⁻⁶. Finite differences detect chain-rule defects "
             "but may lose accuracy with very tiny ε. The simple three-point sine fixture is "
             "neither satellite training nor a scientific generalization study.")
    st.graphviz_chart("digraph {rankdir=LR; X -> \"W1,b1\" -> tanh -> \"W2,b2\" -> Prediction -> MSE -> Backprop;}")
    if not st.button("Run MP14",key="mp14_run"):return
    x=np.array([[-.7],[.2],[.8]],dtype=np.float64); y=np.sin(x)
    params=initialize(inputs=1,hidden=hidden,outputs=1,seed=17)
    check=finite_difference_error(x,y,params,epsilon=1e-6)
    records=[]
    for step in range(steps+1):
        loss,grads=mlp_loss(x,y,params)
        records.append({"step":step,"train_MSE":loss})
        if step<steps:
            for key in params:params[key]-=.05*grads[key]
    pred=forward(x,params)
    st.line_chart(pd.DataFrame(records).set_index("step"),x_label="Step",y_label="Training MSE")
    st.write(f"Max absolute gradient finite-difference error={check:.3g}; "
             f"initial MSE={records[0]['train_MSE']:.6f}; final MSE={records[-1]['train_MSE']:.6f}.")
    _downloads("MP14",[{"x":float(xx),"target_sin_x":float(yy),"predicted":float(pp)}
                        for xx,yy,pp in zip(x[:,0],y[:,0],pred[:,0])],
               {"hidden":hidden,"steps":steps,"seed":17,"lr":.05,"activation":"tanh",
                "gradient_check_max_abs_error":check,"train_curve":records,
                "source":"rssr.training_lab.initialize/mlp_loss/finite_difference_error/forward"})
    st.success("NumPy forward, backward and numerical gradients were computed on the same fixture.")
    st.caption("This synthetic fit has no independent held-out data. Gradient correctness does not establish SR quality.")


def _mp15():
    st.subheader("MP15 · PyTorch autograd versus the canonical NumPy chain rule")
    steps=st.slider("MP15 SGD updates",0,40,5,5,key="mp15_steps")
    lr=st.selectbox("MP15 learning rate",[0.01,0.05,0.1],index=1,key="mp15_rate")
    st.latex(r"L=\frac1N\sum_i(\tanh(XW_1+b_1)W_2+b_2-Y)_i^2,\qquad "
             r"\theta_{t+1}=\theta_t-\eta\frac{\partial L}{\partial\theta_t}")
    st.write("Autograd records the same operations as MP14 and differentiates scalar MSE. "
             "We initialize exactly the same NumPy float64 weights and cast them to tensors, "
             "then compare each autograd gradient with NumPy backpropagation. Gradient maximum "
             "absolute difference is an implementation parity check, not validation accuracy. "
             "torch.manual_seed(17) controls the dropout demonstration: train mode randomly "
             "drops activations with p=0.5 and rescales survivors by 1/(1−p)=2; eval mode is identity.")
    st.graphviz_chart("digraph {rankdir=LR; \"Canonical NumPy weights\" -> \"torch float64\" -> Forward -> MSE -> autograd -> SGD; \"Canonical NumPy gradients\" -> \"Compare gradient parity\"; autograd -> \"Compare gradient parity\";}")
    if not st.button("Run MP15",key="mp15_run"):return
    import torch
    torch.set_num_threads(2)
    x=np.array([[-.7],[.2],[.8]],dtype=np.float64);y=np.sin(x)
    p=initialize(inputs=1,hidden=4,outputs=1,seed=17)
    loss_np, g=mlp_loss(x,y,p)
    t={name:torch.tensor(v,dtype=torch.float64,requires_grad=True) for name,v in p.items()}
    tx=torch.from_numpy(x);ty=torch.from_numpy(y)
    predicted=torch.tanh(tx@t["w1"]+t["b1"])@t["w2"]+t["b2"]
    loss=((predicted-ty)**2).mean();loss.backward()
    differences={name:float(np.max(np.abs(t[name].grad.detach().numpy()-g[name]))) for name in p}
    optimizer=torch.optim.SGD(list(t.values()),lr=lr)
    rows=[]
    for step in range(steps+1):
        with torch.no_grad(): current=float(((torch.tanh(tx@t["w1"]+t["b1"])@t["w2"]+t["b2"]-ty)**2).mean())
        rows.append({"step":step,"training_MSE":current})
        if step<steps:
            optimizer.zero_grad(set_to_none=True)
            pred=torch.tanh(tx@t["w1"]+t["b1"])@t["w2"]+t["b2"]
            objective=((pred-ty)**2).mean();objective.backward();optimizer.step()
    torch.manual_seed(17)
    drop=torch.nn.Dropout(p=.5)
    ones=torch.ones(16)
    drop.train();training=drop(ones)
    drop.eval();evaluating=drop(ones)
    st.write(f"NumPy initial MSE={loss_np:.9f}; autograd initial MSE={loss.item():.9f}; "
             f"maximum derivative mismatch={max(differences.values()):.3g}; "
             f"eval dropout unchanged={bool(torch.equal(evaluating,ones))}.")
    st.line_chart(pd.DataFrame(rows).set_index("step"),x_label="Step",y_label="Training MSE")
    _downloads("MP15",[{"parameter":key,"max_gradient_difference":v} for key,v in differences.items()],
               {"dtype":"float64","seed":17,"steps":steps,"lr":lr,"torch_version":torch.__version__,
                "device":"CPU","dropout_training_output":training.tolist(),
                "dropout_eval_output":evaluating.tolist(),"train_curve":rows,
                "source":"rssr.training_lab.initialize/mlp_loss and torch.autograd"})
    st.success("Actual NumPy–PyTorch gradient equivalence and bounded optimizer steps executed.")
    st.caption("No GPU benchmark, trained remote-sensing checkpoint or independent held-out evaluation is claimed.")


def _mp16():
    st.subheader("MP16 · Deterministic DataLoader, validation and weighted minibatches")
    seed=st.number_input("MP16 random seed",min_value=0,max_value=999,value=16,step=1,key="mp16_seed")
    batch=st.selectbox("MP16 batch size",[1,2,3,4,5],index=1,key="mp16_batch")
    epochs=st.slider("MP16 CPU epochs",1,50,20,1,key="mp16_epochs")
    lr=st.selectbox("MP16 SGD learning rate",[0.01,0.1,0.3],index=1,key="mp16_lr")
    st.latex(r"\widehat L=\frac{\sum_b n_b L_b}{\sum_b n_b},\quad "
             r"L_b=\frac1{n_b}\sum_{i\in b}(\hat y_i-y_i)^2")
    st.write("An N=5 training set has minibatches [2,2,1] for batch size 2. "
             "If the first and second batch means are 2.5 and 1, weighted aggregation "
             "gives (2×2.5+1×1)/3=2, unlike the unweighted mean 1.75. "
             "The training curve averages losses measured before each optimizer update within an epoch; "
             "it is not a final-epoch fixed-weight full-dataset MSE. Validation uses two separate "
             "synthetic coordinates with the model in eval/inference mode after the epoch.")
    st.write("This demonstration trains two independent runs with the same seed and compares "
             "their parameter tensors exactly. No checkpoint file is persisted, and the "
             "existing production resumability test remains separately recorded in "
             "reports/mp16-pipeline.json. The values y=2x+1 are a synthetic functional relation; "
             "the split tests pipeline mechanics, NOT spatial generalization.")
    st.graphviz_chart("digraph {rankdir=LR; \"Seed and split\" -> DataLoader -> \"Mini-batch SGD\" -> \"Weighted training curve\"; \"Mini-batch SGD\" -> \"Eval on holdout coordinates\"; \"Seed and split\" -> \"Repeat exact run\" -> \"Parameter equality\";}")
    if not st.button("Run MP16",key="mp16_run"):return
    record=tiny_regression(seed=int(seed),epochs=epochs,batch_size=batch,lr=lr)
    frame=pd.DataFrame(record["history"])
    st.line_chart(frame.set_index("epoch")[["train_batch_weighted_MSE","validation_MSE"]],
                  x_label="Epoch",y_label="Mean squared error")
    st.write(f"Train samples={record['train_count']}; validation coordinates={record['validation_count']}; "
             f"last batch sizes={record['last_batch_sizes']}; "
             f"exact seed-matched parameters={record['repeat_identical']}; "
             f"final validation MSE={frame['validation_MSE'].iloc[-1]:.8g}.")
    _downloads("MP16",record["history"],{k:v for k,v in record.items() if k!="history"})
    st.success("Two genuine bounded CPU training runs executed, with deterministic comparison.")
    st.caption("No remote training, production checkpoint-resume rerun, real satellite data or independent regional test.")


RENDERERS={"MP12":_mp12,"MP13":_mp13,"MP14":_mp14,"MP15":_mp15,"MP16":_mp16}


def render(ident):
    if ident not in RENDERERS: raise ValueError("Unsupported mathematics lesson")
    RENDERERS[ident]()
