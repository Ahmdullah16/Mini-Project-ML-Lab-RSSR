### Objective and prerequisites

Reconstruct a larger image from a smaller observation while retaining geometry and compatible spectral values. Understand tensor shapes, convolution, residual addition and training/validation separation first. This family teaches different feature-processing paths; additional modules are hypotheses about useful context, not guaranteed improvements. All seven diagrams show implemented cat-v1 paths. Only a selected recorded checkpoint establishes trained configuration availability.

### Embedding and local residual features

Let input $y$ have shape $N\times C\times H\times W$. A padded3×3 convolution produces $f=E(y)$ of shape $N\times d\times H\times W$. For output channel j and position(u,v),

$$f_{n,j,u,v}=b_j+\sum_{c=1}^{C}\sum_{a=-1}^{1}\sum_{b=-1}^{1}W_{j,c,a,b}y_{n,c,u+a,v+b}.$$

This is the library cross-correlation convention; boundary padding is zero for these convolutions. It has $9Cd+d$ trainable parameters including biases. With C=3,d=16:9×3×16+16=448. The C branch computes $L(f)=f+W_2*\operatorname{ReLU}(W_1*f+b_1)+b_2$, with two padded3×3,d→d layers. ReLU keeps positive responses and zeros negative responses. Addition requires matching shapes. Each layer has $9d^2+d$ parameters; the pair has4,640 at d=16. Shared embedding/reconstruction remain present even when the additional C branch is absent.

### Feature attention: channel then spatial

For each feature channel j, average all HW locations: $z_j=(HW)^{-1}\sum_{u,v}f_{j,u,v}$. The gate is $g=\sigma(W_b\operatorname{ReLU}(W_a z+b_a)+b_b)$, with bottleneck $r=\max(1,\lfloor d/4\rfloor)$ and sigmoid $\sigma(t)=1/(1+e^{-t})$. Multiply $f'_{j,u,v}=g_j f_{j,u,v}$. A zero gate-logit gives g=0.5: feature0.8 becomes0.4, not automatically a better reconstruction.

Next compute channel mean and maximum of f′ at each spatial location; concatenate to shape N×2×H×W. A padded7×7 convolution2→1 followed by sigmoid forms the spatial gate m. Output is $m\odot f'$. Multiplication can suppress useful features as well as noise. The channel gate sees the entire processed tile; tiled and full-image contexts are therefore not generally identical. These gates differ from Transformer token self-attention.

### Window Transformer and gradient path

Replicate-pad spatial dimensions to multiples of window side w; each window contains t=w² tokens with d features. Add a learned positional tensor and apply LayerNorm before two-head self-attention. Per head, with key/query dimension $d_h=d/2$,

$$Q=ZW_Q,\quad K=ZW_K,\quad V=ZW_V,\qquad A=\operatorname{softmax}(QK^T/\sqrt{d_h})V.$$

Rows of the t×t attention matrix sum to1; softmax normalizes similarity scores, not calibrated probabilities of correctness. With w=4,d=16, each head has16tokens of width8; score matrices are16×16. Double w to8 and score entries grow from256 to4,096 per head/window. Total image cost also depends on how many windows exist.

The block adds attention output to its input tokens, then adds a feed-forward branch LayerNorm→Linear(d→2d)→GELU→Linear(2d→d). Residual additions preserve a direct gradient path. Windows are reassembled and padding removed. This implementation has fixed windows; it does not implement SwinIR's full shifted-window hierarchy.

### Fusion and reconstruction

When both C and T are active, they process the same embedding in parallel. Concatenate their d-channel outputs, then project2d→d with1×1 convolution. Its parameter count is $2d^2+d$ (528 at d=16). Apply A after fusion only in M7. M6 applies A after T; M4 after C; M2 directly after embedding.

The reconstruction convolution creates Cs² channels. PixelShuffle rearranges channels into spatial positions without adding parameters: N×Cs²×H×W becomes N×C×sH×sW. At C=3,s=2,d=16, the tail produces12channels and has9×16×12+12=1,740parameters. Add fixed bicubic interpolation $B(y)$ to the residual: $\hat x=B(y)+R(f)$. If one baseline value is0.40 and residual is0.03, output is0.43. If target is0.45, signed error is−0.02; this example is arithmetic, not a measured model result.

### Training and parameter decisions

Training uses the exact recorded configuration, not the UI architecture name. Adam learning rate controls update magnitude; excessively large values can destabilize optimization, while small values may require more steps. Batch size controls patches per update and activation memory; seeds change initialization/sampling. The current training command bounds steps1–10,000 and batch size1–32; these software bounds are not recommendations for scientific adequacy. One recorded CNN run uses lr0.001,batch4,1,000steps,seed17,validation every100steps and zero-initialized reconstruction weights. Do not copy those values onto every historical model.

The reconstruction loss is $K^{-1}\sum_i|\hat x_i-x_i|$ over K scalar values in the training patch. Its nonzero-error derivative is sign(error)/K; chain-rule differentiation propagates through the selected feature path. Optional consistency loss is present only for a configuration that enables it. Within-run best checkpoint uses minimum validation L1. Later protocol selection across runs uses development PSNR: these are two different stages. Fixed step limits are stopping budgets, not proof of convergence. No unrecorded regularization or perceptual/adversarial objective is implied.

Width d is an integer≥2; two-head Transformer paths require even width. Window w is a positive integer. Factory-native scales are2 or4, but trained compatibility comes only from the selected checkpoint. Requested larger scales may be cascades. Width often drives convolution cost quadratically; window attention memory increases with token count. No parameter setting is a proven universal quality optimum. Exact model parameter counts and selected steps appear with each recorded checkpoint.

### Interpretation and applications

This can support a supervised imagery-review prototype: compare reconstruction against bicubic, inspect roads/buildings/vegetation at the same crop and record provenance. It is not validated for cadastral measurement, safety decisions, crop diagnosis or automated change detection. Those applications require suitable independent labels, sensor calibration and task-specific validation. All seven families now have genuine RGB pilot checkpoints; M2–M6 use separately labeled100step compatibility pilots with pinned release assets. Independent geographic research remains incomplete; real cross-sensor superiority and architectural novelty remain unestablished. Related ideas: [RCAN](https://arxiv.org/abs/1807.02758), [SwinIR](https://arxiv.org/abs/2108.10257).
