# Detailed project supplement — working source

Existing teaching prose and recorded validation; not final scientific acceptance. Code cells are neither rendered nor executed. Complete parameter audits, new figures, individual mathematical acceptance and final typesetting remain pending. Author details are unavailable; supplied IEEE templates were reviewed (see reports/ieee-paper/evidence/supplied-template-review.json). No venue page limit is assumed.

# Shared mathematical reference

These are analytical teaching examples, not new model measurements. Image dimensions alone do not establish recovered physical detail.

## Shapes, scales and storage

An image tensor has shape $N\times C\times H\times W$: batch size $N$, spectral/display channels $C$, height $H$ and width $W$. A scale $s$ maps this to $N\times C\times(sH)\times(sW)$. Therefore the pixel ratio is $(sH)(sW)/(HW)=s^2$. A 256×128 input at ×8 produces 2048×1024, or 2,097,152 pixels. RGB float32 output alone occupies $3\cdot4\cdot2{,}097{,}152=25{,}165{,}824$ bytes (24 MiB), excluding model activations, masks, writer buffers and previews. This exceeds the configured hosted output-pixel limit. Larger scale increases storage quadratically; it does not guarantee higher accuracy.

The implementation accepts integer input dimensions 1–1024 per side and requested scales 2,4,8,16,32,64. Native models and cascades must be labeled separately. Six native ×2 stages yield $2^6=64$; intermediate errors and display quantization can accumulate. Scientific float32 export and display RGB are different paths.

## Residual reconstruction and gradient flow

The implemented residual form is $\hat x=B(y)+R_\theta(y)$, where $y$ is LR input, $B$ is fixed bicubic interpolation and $R_\theta$ is a learned residual. If $B(y)=0.40$, residual is 0.03 and target is 0.45, prediction is 0.43 and signed error is −0.02. This is a scalar illustration, not an observed prediction.

For $K$ valid scalar channel values, $L_1=K^{-1}\sum_i|\hat x_i-x_i|$. Away from zero error, $\partial L_1/\partial\hat x_i=\operatorname{sign}(\hat x_i-x_i)/K$. The chain rule gives $\partial L_1/\partial\theta=\sum_i(\partial L_1/\partial\hat x_i)(\partial R_{\theta,i}/\partial\theta)$. The fixed bicubic branch has no trainable parameters. A zero subgradient is a valid convention at exact equality. Loss units are the image-value units. A two-value error vector (−0.02,0.04) gives L1=0.03; each nonzero output derivative has magnitude0.5. Masked means require $K>0$; a fully invalid image is not a zero-error example.

## Metric derivation and aggregation

For that same error vector, MSE=$(0.02^2+0.04^2)/2=0.001$ and RMSE=$\sqrt{0.001}\approx0.03162$. With fixed range $R=1$, PSNR=$10\log_{10}(R^2/\mathrm{MSE})=30$ dB. The range is a declared protocol constant, not each image's measured maximum. Exact equality yields infinite PSNR mathematically and needs explicit handling in exports. Averaging tile PSNR differs from converting an aggregate MSE because logarithms are nonlinear. Valid geographic uncertainty resamples independent scenes/regions, not correlated patches or training seeds.

## Connection to implemented models and prior work

All seven cat-v1 paths have shared embedding and reconstruction convolutions. C means an additional convolutional backbone; A additional channel/spatial feature attention; T a Transformer that already contains self-attention. Width controls feature channels and often quadratic convolution cost; window side controls token count ($w^2$) and attention matrices ($w^4$ entries per head/window). Neither is a proven monotonic quality control. Exact historical settings and training provenance belong to each checkpoint, not to an architecture label.

[RCAN](https://arxiv.org/abs/1807.02758) motivates feature-channel reweighting; [SwinIR](https://arxiv.org/abs/2108.10257) demonstrates residual Transformer restoration. Our compact blocks are not faithful reproductions of either system. Combining these concepts is not by itself architectural novelty. Source contracts: `src/rssr/models.py`, the architecture registry and recorded checkpoint configuration. Complete per-model parameter and training derivations remain under development.


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

This can support a supervised imagery-review prototype: compare reconstruction against bicubic, inspect roads/buildings/vegetation at the same crop and record provenance. It is not validated for cadastral measurement, safety decisions, crop diagnosis or automated change detection. Those applications require suitable independent labels, sensor calibration and task-specific validation. M2/M6 lack trained bindings; real cross-sensor superiority and architectural novelty remain unestablished. Related ideas: [RCAN](https://arxiv.org/abs/1807.02758), [SwinIR](https://arxiv.org/abs/2108.10257).


# Understand these results

Reference-based accuracy is **unavailable** for uploads and acquired observations without a valid aligned HR reference. Neither the LR input nor bicubic interpolation is ground truth. A difference from bicubic is disagreement, not reconstruction error. The synthetic panel below has its own known reference and is unrelated to your uploaded image's accuracy.

These are evaluation metrics, not image-processing knobs. Editing a displayed score cannot improve an image. Training can minimize an L1 or squared-error loss through gradients with respect to model weights; reporting a score performs no such optimization.

## Error, units and aggregation

Let reference r and output p be C×H×W floating arrays with matching bands, radiometry and alignment. For N valid scalar samples across all bands define e_i=p_i−r_i. Here L=1 is the declared display range, not the observed maximum. No stretching enters evaluation. All pixels are valid, border crop is zero; these settings are specific to this synthetic demonstration.

$$\mathrm{MAE}=\frac{1}{N}\sum_i|e_i|,\quad\mathrm{MSE}=\frac{1}{N}\sum_i e_i^2,\quad\mathrm{RMSE}=\sqrt{\mathrm{MSE}}.$$

Take each signed difference, then absolute value or square, sum and divide by sample count. The square root restores the original units. MAE and RMSE have normalized intensity units; MSE has squared intensity units. Lower is better; for this bounded [0,1] data all three lie in [0,1]. Per-band RMSE uses only that band's H×W samples and may expose an error hidden by a pooled mean. Squaring penalizes large residuals more strongly. None of these errors proves correct geometry or physical calibration.

For errors [0.1,−0.1], MAE=0.1, MSE=(0.01+0.01)/2=0.01 and RMSE=0.1. [Metric definitions and explicit range conventions](https://scikit-image.org/docs/stable/api/skimage.metrics.html).

## PSNR: logarithmic squared error

$$\mathrm{PSNR}=10\log_{10}(L^2/\mathrm{MSE})=20\log_{10}(L/\mathrm{RMSE}).$$

The ratio compares squared full-scale amplitude with average squared error; the logarithm expresses it in decibels (dB). At L=1 and MSE=0.01, PSNR=20dB. Zero MSE gives positive infinity, not a numerical failure. Higher is better under identical settings. There is no finite upper bound; general unbounded predictions can produce negative values. Doubling RMSE reduces PSNR by approximately6.02dB. A changed data range changes the score without changing the image. Scores from different crops, bands, scenes, masks or preprocessing must not be ranked together. [PSNR implementation convention](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio).

## SSIM: local means, contrast and covariance

$$\mathrm{SSIM}=\frac{(2\mu_p\mu_r+C_1)(2\sigma_{pr}+C_2)}{(\mu_p^2+\mu_r^2+C_1)(\sigma_p^2+\sigma_r^2+C_2)}.$$

For each valid7×7 window, μ is its population mean, σ² its mean squared deviation, and σ_pr the mean product of centered p and r values. C1=(0.01L)² and C2=(0.03L)² stabilize flat regions. The first factor compares luminance; the second combines contrast and structure. Our implementation averages over channels and fully valid windows, with uniform weights and population moments. It is not the original paper's Gaussian-window implementation. No padding windows are scored. Images smaller than7pixels on either side have unavailable SSIM.

Identical windows have numerator=denominator, hence1. For constant reference0.5 and prediction0.6, the variance factor is1 and SSIM=(0.6+0.0001)/(0.25+0.36+0.0001)≈0.9836. Higher is preferred; the theoretical interval is[−1,1]. This near-one score can coexist with a visible brightness bias. Window choice and spatial shifts matter. SSIM and PSNR weight errors differently and need not track perceived sharpness. [Wang et al., 2004, original SSIM paper](https://ece.uwaterloo.ca/~z70wang/publications/ssim.html).

## Edge fidelity and spectral interpretation

$$D_x p_{cij}=p_{ci,j+1}-p_{cij},\quad E_D=\operatorname{mean}(|D_xp-D_xr|,|D_yp-D_yr|).$$

The displayed gradient MAE pools horizontal and vertical adjacent differences with equal weight per edge, without wraparound. Units are normalized intensity per pixel; lower is better, range[0,2] for bounded values. Reference[0,1] and output[0,0.8] have horizontal gradient error0.2. Constant offsets can produce zero gradient error despite wrong brightness. This is a declared diagnostic, not a standardized perceptual metric. Sharpening can increase edge contrast and also introduce halos/error.

For calibrated compatible spectral vectors at a pixel, SAM is arccos((p·r)/(||p||||r||)), averaged over valid nonzero-norm spectra in radians[0,π], lower preferred. Orthogonal vectors giveπ/2; multiplying a positive spectrum by2 gives0 despite wrong amplitude. Zero-norm vectors have undefined angles and are excluded. **SAM is not reported here for encoded display RGB.** Scientific use requires reviewed band semantics and calibration. LPIPS is also not computed: it requires a specified perceptual network/domain and belongs only to perceptual evaluation, not a substitute for reflectance accuracy.

## Pair eligibility before any scientific score

Record registration and residual alignment, CRS/grid, intersection mask, cloud/nodata handling, border crop, band order, units, radiometric calibration, range, metric version and aggregation. Report valid-pixel counts. Estimate generalization across independent scenes rather than treating correlated tiles as independent replicates. Small shifts can sharply reduce agreement even for similar-looking images. More output pixels or sharper texture do not establish recovered truth.


# Website validation update — 2026-10-29

The new metric laboratory was independently executed in both Live SR and Satellite Explorer for blur, noise, brightness, contrast, shift and sharpening. The reference is a deterministic64×64 synthetic RGB pattern, not an uploaded or acquired image. Tests cover exact zero-strength identity, direct NumPy recomputation of MAE/MSE/PSNR, invalid parameters and deterministic distorted pixels. The first test run found tiny zero-contrast roundoff; the algebra was corrected to preserve identity. The rerun passed17tests. Both-section AppTest executed all controls, verified no network and cleared stale output after parameter changes.

The shared seven-family availability view matches hosted artifacts through recorded training-checkpoint hashes. M1/M7 have pinned hosted RGB checkpoints; M2/M6 have local RGB pilots awaiting remote distribution; M3/M4/M5 verified historical checkpoints are four-band. This view does not complete seven-model execution. Scientific SAM and LPIPS are not reported for the synthetic encoded-RGB panel. Upload/satellite reference accuracy remains unavailable without genuine aligned HR.

The remaining mandatory requirements include full individual native lesson acceptance, compatible checkpoints for every supported mode, satellite multi-model comparison, real paired metrics, full scientific export validation and deployed desktop/mobile/restart verification. No new frozen research result, geographic generalization or public deployment is claimed.

## Satellite comparison execution follow-up

Two genuine cached hosted CNN checkpoints executed sequentially on one synthetic georeferenced5x3fixture, each exporting10x6 pixels with CRS32632 and unchanged extent. Source hashes match. An injected second-model error preserved the first output; targeted retry succeeded without changing it. Scale changes removed stale cached outputs. The UI exposes both Single model and Compare models workflows, per-model GeoTIFF/provenance downloads and measured runtime table/chart. This is display-RGB engineering evidence, not provider acquisition or scientific reflectance acceptance. Synchronized zoom, baseline/detail panels, disagreement maps, all-seven execution and scientific mask verification remain pending. See reports/satellite-comparison-validation.json.

PR40 was externally merged despite a failing Streamlit CI check: an old positional button selector clicked the new teaching control. The upload test now selects its stable key; the seven-section check passed locally. The follow-up still requires its own CI.


# Contribution audit

Initial contribution audit; closest verified references among reviewed sources, not an exhaustive novelty search

## Combining convolution, feature attention and Transformers improves SR

**Prior Work:** RCAN: channel reweighting; SwinIR: residual Transformer restoration

**Difference:** Compact cat-v1 feature paths and explicit seven-way implementation registry, rather than a demonstrated new operator

**Needed:** Component removal, matched complete budgets, identical data and multiple seeds with independent-region uncertainty

**Observed:** 39 historical bindings across five IDs; M2/M6 lack trained bindings. Partial-cost search finds no eligible M2 width.

**Supported Claim:** Implementation and artifact-compatibility contribution only; architectural novelty and seven-model superiority are not established

References: https://arxiv.org/abs/1807.02758, https://arxiv.org/abs/2108.10257

Versioned evidence: reports/historical-architecture-bindings.json, reports/architecture-budget-preflight.json

## Learned residual reliability produces a reconstruction gain

**Prior Work:** Residual reconstruction in SwinIR; project controls include fixed residual attenuation

**Difference:** Explicitly testing whether a learned gate adds benefit beyond simply reducing residual magnitude

**Needed:** Gain and offset controls, component-removal ablations, matched budgets and independent-scene replication before making a reliability claim

**Observed:** Existing pilot interpretation admits attenuation as an explanation; paper tables preserve all recorded pilot rows and seed dispersion.

**Supported Claim:** Negative/inconclusive pilot finding; learned reliability is not established

References: https://arxiv.org/abs/2108.10257

Versioned evidence: reports/ablation-pilot-runs.json, reports/paper-evidence.json

## Large multiscale output demonstrates recovered satellite detail

**Prior Work:** SwinIR restoration benchmarks require reference-based evaluation, not output-size evidence

**Difference:** Exact-dimension cascades, resource preflight and disk-backed export are engineering capabilities

**Needed:** Trained representative cascade/seam tests plus calibrated aligned independent cross-sensor references; spectral and downstream evidence where valid

**Observed:** Random-weight native-scale seam checks and synthetic reflectance conversion/export tests; these are not quality or transfer experiments.

**Supported Claim:** Bounded implementation evidence only; no measured physical resolving-power or cross-sensor improvement

References: https://arxiv.org/abs/2108.10257

Versioned evidence: reports/architecture-seam-smoke.json, reports/reflectance-job-validation.json

# MP01 · Original MP01 preserved

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP01 — Digital Image Explorer

**Learning status: Not started.** Implementation is prepared for review; automated checks passed as recorded in [VALIDATION.md](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP01_digital_image_explorer/VALIDATION.md). Learner exercises and explanations have not been assessed, so MP01 is not marked complete. The shared progress tracker is unchanged.

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

## Learning objectives

Understand pixels, image dimensions, RGB channels, grayscale, bit depth, NumPy arrays, and histograms from zero. Predict numerical results, inspect plots, and explain why normalization, overflow, and histogram conventions matter.

## Prerequisites

No machine learning knowledge or dataset is required. The notebook introduces indexing, variables, arrays, and functions as they appear. Use Python 3.11 or 3.12, a terminal, and NumPy, Matplotlib, and Pillow. JupyterLab is needed only for interactive notebook use. The headless runner needs only the three image libraries plus the Python standard library. CPU execution is sufficient.

## Exact execution instructions

From the repository root, create an environment and install only this project's dependencies (no PyTorch needed):



For automated checks, from that same MP01 directory:



`run_notebook.py` launches no Jupyter kernel. It executes this notebook's ordinary Python cells in sequence in a shared namespace, checks assertions, and captures Matplotlib figures using the Agg backend. It stops on an exception with a nonzero exit status. It does not support notebook magics, widgets, or general IPython display hooks. Use a fresh process as shown above for each run. This is the execution method validated in this environment; interactive Jupyter use was not available.

For a minimal headless environment, install `numpy matplotlib Pillow` and run the two automated commands above; Jupyter is unnecessary for that path.

## Mathematics

- RGB image: `I[r,c,k]`, shape `(H,W,3)`; `H×W` pixels and `H×W×3` samples. Pillow's size is `(W,H)`.
- Unsigned b-bit channel: `2**b` levels, from 0 to `2**b - 1`. uint8 RGB stores 24 bits per pixel.
- Normalization: `f = x/255`; inverse: `round(255*f)`.
- Display gray: `Y' = 0.299 R' + 0.587 G' + 0.114 B'`, applied to encoded RGB. This is not calibrated physical luminance.
- Histogram: count pixels at each integer gray value; counts sum to `H×W`, normalized counts sum to 1.

The notebook includes worked examples and explains every symbol. Helpers require explicit uint8 RGB conventions and reject ambiguous input rather than silently rescaling it.

## Experiments

The [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP01_digital_image_explorer/digital_image_explorer.ipynb) contains 13 small experiments:

1. Construct a six-pixel image and inspect dimensions.
2. Generate and save a deterministic 64×96 RGB image.
3. Inspect coordinates and a twelve-pixel crop.
4. Separate RGB channels.
5. Remove blue from a copy.
6. Compare eight-, two-, and one-bit quantization.
7. Normalize and recover an image exactly.
8. Compare uint8 overflow against safe clipped brightness.
9. Compare equal-average and weighted grayscale.
10. Count and plot a grayscale histogram.
11. Compare RGB channel histograms.
12. Mirror an image while preserving its histogram.
13. Verify a PNG round trip and NumPy view/copy behavior.

Five practice exercises follow, with hints and selected self-check values. The optional user-image cell is separate from these experiments.

## Optional satellite display image

Keep `USER_IMAGE_PATH = None` for a fully self-contained run. To inspect your own image, place a small display crop in this project's ignored `data/` directory and change the optional cell to, for example:



Supported input is an opaque 8-bit RGB, grayscale, or palette image readable by Pillow; the first frame is used and EXIF orientation is applied. Transparent and high-bit-depth images are rejected. File/format errors are surfaced rather than silently replacing your image with synthetic data.

This path inspects display pixels only. It does not preserve CRS, affine transforms, nodata masks, scientific precision, band meanings, or reflectance units. Do not convert scientific multispectral data through this helper. Export an intentional display visualization separately and retain your original raster; geospatial processing is outside MP01. Large files may exceed memory or Pillow safety limits. Document source and license; no user file is included or uploaded by this lesson.

## Deliverables and expected outputs

- [digital_image_explorer.ipynb](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP01_digital_image_explorer/digital_image_explorer.ipynb): teaching notebook with explanations, equations, 13 experiments, five exercises, and a conclusion.
- [image_tools.py](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP01_digital_image_explorer/image_tools.py): deterministic fixture, validation, normalization, grayscale, histogram, brightness, and optional loader helpers.
- [test_image_tools.py](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP01_digital_image_explorer/test_image_tools.py): 10 standard-library unittest tests for numerical results, shape/dtype checks, overflow, and file loading.
- [run_notebook.py](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP01_digital_image_explorer/run_notebook.py): dependency-light execution and figure capture.
- [VALIDATION.md](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP01_digital_image_explorer/VALIDATION.md): observed execution and limitations.

A successful run reports RGB shape `(64,96,3)`, 6144 pixels, 18432 raw bytes, grayscale shape `(64,96)`, histogram count total 6144, and probability total approximately 1. The hand-worked grayscale primary values are approximately 76.245, 149.685, and 29.070. Overflow demonstration prints 14 for wrapped `250+20`, and the safe version saturates to 255.

Expect 12 figures covering colors, crops, channels, quantization, normalization, brightness, grayscale, and histograms. The default optional branch prints that it was skipped. The runner creates `outputs/synthetic_rgb.png`, `outputs/synthetic_gray.png`, and `outputs/digital_image_explorer.executed.ipynb`. Generated files stay local under the existing ignore rules; the committed notebook has empty outputs for clean review. Test success should report `Ran 10 tests` and `OK`.

## References

- [NumPy beginners guide](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Matplotlib imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)
- [Pillow modes and concepts](https://pillow.readthedocs.io/en/stable/handbook/concepts.html)
- [ITU-R BT.601](https://www.itu.int/rec/R-REC-BT.601)

The synthetic fixture is created locally from explicit numerical gradients and color blocks. It has no external dataset source. No new repository license is assigned by this change.

## Completion criteria

- [ ] Explain every concept and work through the numerical examples yourself.
- [ ] Run the notebook in your learning environment and interpret all 13 experiments.
- [ ] Complete the five practice exercises with your own explanations.
- [ ] Review automated validation and any environment-specific failures.
- [ ] Document your observations and only then update progress with evidence.

Automated implementation validation alone does not establish that a learner has completed the project.

# MP01 — Digital Image Explorer

Welcome! A digital image is a grid of numbers. We will make our own tiny image, inspect its numbers, and connect them to colors. No dataset download or prior machine learning knowledge is needed.

**Learning goals:** explain pixels, dimensions, RGB, grayscale, bit depth, NumPy arrays, and histograms; make predictions and check them with small experiments.

All images below are original deterministic teaching data, not satellite observations. The optional final section can inspect your own display image. Exercises are invitations to learn, not evidence that you have completed the project.

## 1. Pixels and coordinates

A **pixel** is one sample at a grid location. A grayscale image can be modeled as $I(r,c)$: row $r$, column $c$. Height $H$ counts rows; width $W$ counts columns. Indices start at zero: $0\le r<H$ and $0\le c<W$.

For RGB, $I(r,c,k)$ has a third index $k$: 0 for red, 1 for green, 2 for blue. Its shape is $(H,W,3)$. One RGB pixel has three channel samples; it is still one pixel.

**Worked example:** a 2-row, 3-column RGB image has $2\times3=6$ pixels and $6\times3=18$ channel samples. Pillow reports `size` as `(width, height)`; NumPy reports `shape` as `(height, width, channels)`.

### Experiment 1 — Build a six-pixel image
Predict the shape and bottom-right color before running. A NumPy **array** is a rectangular collection of same-type numbers. Nested brackets here mean rows, pixels, then channel values.

### Experiment 2 — Generate and save a synthetic image

We create a 64-row, 96-column image with red and green gradients, a constant blue background, and six known color blocks. It uses no randomness. `image_tools.py` contains short reusable helpers; inspect them when curious. `outputs/` holds local generated files and is ignored by Git.

Predict: will Pillow's size be `(64, 96)` or `(96, 64)`?

### Experiment 3 — Zoom into pixels and crop a region

`rgb[r, c]` gives one pixel. A colon expresses a slice: `rgb[0:3, 0:4]` selects rows 0,1,2 and columns 0,1,2,3. The stop is excluded. `rgb[:, :, 0]` means every row, every column, red channel.

**Worked example:** rows 0 through 2 and columns 0 through 3 contain $3\times4=12$ pixels. They lie in our red block. Predict every triplet.

## 2. RGB channels

In additive RGB, $(0,0,0)$ is black, $(255,255,255)$ is white, and $(255,0,0)$ is full red for our 8-bit display encoding. An RGB triplet is a color encoding, not three geographic coordinates or automatically three physical satellite bands.

### Experiment 4 — Separate the channels
Predict: which channel has a horizontal gradient? Which has a vertical gradient? Displaying a channel with `cmap="gray"` shows its numerical strength, not its original hue. Fixed `vmin=0, vmax=255` avoids automatic contrast changes.

### Experiment 5 — Remove blue without changing the original

`.copy()` creates independent data. Setting one channel to zero changes colors, even though image dimensions stay the same. Predict whether the blue block becomes black and the white block becomes yellow.

## 3. Bit depth and numerical storage

A **bit** has two states. An unsigned $b$-bit channel can encode $2^b$ integer levels, numbered 0 through $2^b-1$. Eight bits give 256 levels (0–255). `uint8` means unsigned 8-bit integer. RGB with 8 bits per channel uses 24 bits per pixel, with $256^3$ possible triplets; an RGB image need not contain all of them.

The raw sample storage is $HWCb/8$ bytes, excluding metadata and compression. Our 64×96 RGB array uses $64\cdot96\cdot3=18432$ bytes. A PNG file's compressed size is a different quantity. Storage dtype also differs from sensor effective bit depth: a 12-bit sensor may store values in a 16-bit container.

### Experiment 6 — Reduce gray levels
Quantization maps continuous or finely spaced values to fewer levels. For $L=2^b$ levels spanning [0,255], we use $q(x)=255\,\mathrm{round}((L-1)x/255)/(L-1)$, then round for uint8 storage. Predict banding with two bits (four levels).

### Experiment 7 — Normalize numbers without changing the picture

Normalization here means $f=x/255$, changing uint8 [0,255] into floating-point [0,1]. **Worked example:** $128/255\approx0.50196$, not exactly 0.5. Recover the original with $x=\mathrm{round}(255f)$. Convert to float before arithmetic. Matplotlib expects RGB integers in [0,255] or RGB floats in [0,1].

### Experiment 8 — Brightness, clipping, and overflow

To brighten by $a$, use $y=\min(255,\max(0,x+a))$. **Worked example:** $250+20=270$, clipped to 255. Doing the addition in uint8 wraps modulo 256 instead: $270\bmod256=14$, which is dark! We demonstrate the wrong outcome deliberately and then use a wider signed type before clipping. Predict both results.

## 4. Grayscale

A grayscale image has one value per pixel, shape $(H,W)$. It cannot retain all three RGB values. A simple average is $(R+G+B)/3$. Our display-gray conversion uses $Y'=0.299R'+0.587G'+0.114B'$. The prime reminds us these are encoded display values: this is an approximate **luma** signal, not linear-light luminance or measured satellite reflectance.

**Worked example:** pure red gives $0.299\times255=76.245$; pure green gives 149.685; pure blue gives 29.070. Equal RGB averages would give all three 85. We keep float32 until deliberately rounding to uint8.

### Experiment 9 — Compare two grayscale recipes
Predict which primary color looks brightest with the weighted recipe. Green receives the largest weight.

## 5. Histograms

A histogram counts how often values occur; it does not record where they occur. For uint8 grayscale, define $h(v)=\sum_{r=0}^{H-1}\sum_{c=0}^{W-1}\mathbf{1}[I(r,c)=v]$, $v\in\{0,\ldots,255\}$. The indicator $\mathbf{1}$ is 1 when the condition is true and 0 otherwise. Therefore $\sum_v h(v)=HW$. The normalized histogram $p(v)=h(v)/(HW)$ sums to 1.

**Worked example:** for `[[0,0],[128,255]]`, counts are $h(0)=2$, $h(128)=1$, $h(255)=1$, all others zero. Probabilities are 0.5, 0.25, 0.25.

### Experiment 10 — Count first, then plot
Predict the sum of counts for our 64×96 image: 6144. We round weighted grayscale before counting exact integer bins.

### Experiment 11 — Compare RGB channel histograms

Each RGB channel has its own distribution. Predict a strong blue peak at 64: most background blue values are constant. Each channel histogram counts 6144 samples; combining all three would count 18432 and answer a different question.

### Experiment 12 — Same histogram, different arrangement

Reverse the columns of grayscale with `[:, ::-1]`. Predict: the picture changes position, but counts stay the same. A histogram cannot tell whether an object is on the left or right.

### Experiment 13 — Save/load round trip and NumPy memory

PNG is lossless: saving and reloading this RGB array should preserve every sample. Slices often share memory; `.copy()` makes independent storage. Predict whether changing `independent` will affect `loaded`.

## Optional — Your own satellite display image

The default path is `None`, so this section needs no file and performs no download. Set a local path to an **opaque 8-bit RGB PNG/JPEG or simple display TIFF**, or an 8-bit grayscale/palette image. The loader applies EXIF orientation and converts grayscale/palette display images to RGB. It uses the first frame only.

This is a display-image lesson. The loader rejects transparent and high-bit-depth modes rather than silently reducing them. Pillow RGB conversion does not preserve CRS, map transforms, nodata masks, physical units, or multispectral band meanings. Do not use it to preprocess scientific satellite rasters. Export an explicit visualization separately and retain your original data; geospatial processing comes in the later remote sensing stages. Large rasters may exceed memory or Pillow's safety limits: use a small display crop. Document your image source/license privately and do not commit it.

## Five practice exercises

Make a new cell for each task. Keep the original `rgb` unchanged. Predict, run, then explain the result in one or two sentences.

1. **Dimensions:** for an RGB array of shape `(20, 30, 3)`, calculate the number of pixels, channel samples, and uint8 bytes before creating it. Check with `.size` and `.nbytes`. Hint: dimensions multiply.
2. **Pixel detective:** inspect `rgb[0, 48]` and `rgb[0, 64]`. Name their colors. Explain why `rgb[64, 0]` would be outside this image. Hint: the last valid row is 63.
3. **Independent editing:** make a copy, draw a 5×7 yellow rectangle in it, and show the result. Verify that the original is unchanged. Hint: yellow is `[255,255,0]`; slices exclude the stop.
4. **Gray by hand:** calculate the equal average and weighted display-gray value for `[100,150,200]`. Compare with `to_grayscale(np.array([[[100,150,200]]], dtype=np.uint8))`. Explain why the two recipes differ.
5. **Histogram detective:** construct two different 2×3 uint8 grayscale arrays with identical histograms. Confirm equal counts and unequal arrays. Explain why histogram equality cannot prove two images are identical.

**Self-check values** (try first): exercise 1 gives 600 pixels, 1800 samples, 1800 bytes; exercise 4 gives average 150 and weighted value 140.75. Other exercises admit multiple answers. Record your own reasoning; running the notebook does not complete the exercises.

## Conclusion

A pixel is a spatial sample; RGB stores three values at each location. Array shape gives height, width, and channels, while dtype describes numeric storage. Eight-bit channels encode 256 levels. Normalizing changes representation; clipping avoids overflow; grayscale deliberately discards color information. Histograms summarize intensity counts but discard spatial layout.

The deterministic checks above connect these claims to small, inspectable examples. Our synthetic image makes no scientific claim about satellites, spectral fidelity, or model performance. The optional user-image branch has not been validated on your own imagery. Before moving on, answer the five exercises and explain the worked examples aloud.

## References

- [NumPy absolute beginners guide](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Matplotlib imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)
- [Pillow image modes](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)
- [ITU-R BT.601](https://www.itu.int/rec/R-REC-BT.601) for the luma weighting convention; this lesson applies it to display RGB without physical radiometric calibration.

## Recorded validation

# MP01 validation record

Implementation prepared for review. MP01 remains **Not started** in the learner progress record; no learner exercises have been assessed.

## Observed environment

Python 3.12.14; NumPy 2.3.5; Matplotlib 3.10.8; Pillow 12.3.0; CPU execution.

## Commands and outcomes

From the MP01 directory:



- All **10 tests passed**. They cover known synthetic pixels, shape/dtype, normalization endpoints and midpoint, independently computed primary-color gray values, histogram counts, brightness saturation and input immutability, invalid inputs, lossless PNG/grayscale loading, palette loading, EXIF orientation, rejected high-bit-depth/transparent data, and a missing path.
- The source notebook's **15 Python code cells ran sequentially** in a fresh Python process through the supplied runner. All assertions passed; **12 figures** were captured in a local executed notebook. The two generated PNG files were written successfully.
- The optional user-image cell printed its skip message. No user-supplied satellite image was available, so that branch was not executed on real satellite imagery. The loader was exercised with local generated fixtures in tests.

## Failures and limits

Jupyter's `nbformat`, `nbclient`, and `ipykernel` packages were unavailable. Attempting to install them into a scratch dependency directory failed with a package-proxy connection error, followed by pip's inability to find distributions. No interactive Jupyter kernel or nbclient execution is claimed. The fallback executes ordinary Python cells and captures Matplotlib figures; it is not a complete Jupyter implementation.

Matplotlib initially reported a read-only home cache; the lesson/runner explicitly use a temporary writable Matplotlib cache. Fontconfig emitted `No writable cache directories` during the first headless plot run. Plot rendering still completed and no cell or test failed. The six-pixel figure was visually inspected for correct colored blocks, axes, and labels.

Generated images and the executed notebook are ignored under `outputs/`, not committed. The source notebook retains empty output cells. No external datasets, training, downloads from notebook cells, scientific raster processing, or learner mastery validation occurred.

## Structural and scope checks

Validated all **309 relative Markdown links** in the local laboratory, including the new MP01 links. Checked notebook JSON structure, unique cell IDs, 13 experiment headings, 15 code cells, and empty source execution counts/outputs. This structural check is not nbformat schema validation. Checked changed source files for trailing whitespace. SHA-256 comparison against the pre-change snapshot confirmed every existing file outside MP01 remained byte-for-byte unchanged, including MP02–MP40 and the shared progress tracker. A repeat run with the exact README test and runner commands passed; the first-run Fontconfig warning did not recur.

# MP02 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP02 — NumPy Image Laboratory

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

## Learning objectives

- Indexing, slicing, reshaping, broadcasting, normalization, vectorized operations.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP01 — Digital Image Explorer](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP01_digital_image_explorer/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Compare elementwise arithmetic, clipping, and channelwise mean and variance.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Executed implementation experiment; learner mastery unassessed.** See [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP02_numpy_image_laboratory/numpy_image_laboratory.ipynb), [measured timings and provenance](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp02-array-lab.json), and `python scripts/mp02_experiment.py`. Loop/vectorized brightness matched exactly; masked statistics matched within 1e-12 on a checksummed 64x64 training crop. No test scene was used.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; the MP02 notebook, reusable array_lab module and experiment script are supplied.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Python and NumPy](https://numpy.org/doc/stable/user/absolute_beginners.html)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP02_numpy_image_laboratory/numpy_image_laboratory.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP02_numpy_image_laboratory/mp02_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP02_numpy_image_laboratory/test_mp02_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP02_numpy_image_laboratory/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP02_numpy_image_laboratory/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP02 — NumPy Image Laboratory
Implementation and a real training-crop experiment are executed; learner mastery remains unassessed.

Objective: indexing, slicing, reshaping, broadcasting, normalization, and vectorized operations. Read the [original brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP02_numpy_image_laboratory/README.md).

## Arrays before images
An array stores numbers in indexed positions. We use `image[row, column, band]`: H rows, W columns, C bands, shape `(H,W,C)`. Indexing starts at zero; slicing excludes its stop. A reshape changes the grouping of the same ordered values; a transpose changes axis order. These are different operations.

For a 1×2×3 array `[[[250,5,10],[1,100,255]]]`, adding channel offsets `[20,-10,2]` should give `[[[255,0,12],[21,90,255]]]` after clipping to [0,255]. A length-C vector broadcasts over H and W: the same offset applies at every pixel. Convert uint8 to floating point *before* addition, because uint8 cannot represent numbers above 255 or negative values.

For channel c, brightness is y[h,w,c] = min(M,max(0,x[h,w,c]+d[c])). M is the maximum allowed value; d is an offset in the same units as x. This operation illustrates arrays; it is not sensor calibration.

For a boolean H×W validity mask m and K valid pixels, mean[c] = sum(x[h,w,c] over valid pixels)/K. Population variance[c] = sum((x[h,w,c]-mean[c])² over valid pixels)/K. Variance has squared units. We use K, not K−1, because we describe the selected pixels rather than estimate an unbiased population variance. Empty masks are errors, not zero-valued observations.

## Index, slice, reshape, transpose
Predict each shape before running. Selecting one band removes an axis; selecting a one-band slice preserves it. A basic slice can share memory with its parent, so copy before changing a crop when the original must remain intact.

## Normalization and masks
For uint8 display data, x/255 maps to [0,1]. For scientific data, use declared units and calibration. Dividing each image by its own maximum changes the meaning of equal physical values across scenes. The pilot uses a fixed divisor of 10000 for numerical scaling; physical calibration is unverified.

In the example below, channel values [1,3,5] have mean 3 and variance (4+0+4)/3 = 8/3. The invalid pixel must not contribute.

## Real-data experiment and observed evidence
Run `python scripts/mp02_experiment.py` from the repository root after the pinned pilot download. It checks the source checksum, reads the upper-left 64×64 crop from the **training** NAIP scene, preserves all four bands and valid-pixel masks, and compares explicit loops with NumPy operations before timing. It does not read the test scene.

The experiment writes [mp02-array-lab.json](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp02-array-lab.json). The original executed run used all 4096 valid pixels. Brightness matched exactly; masked means/variances matched within 1e-12. The next cell reads recorded results, not a new timing run. One warm-up and five repetitions were used; medians include input validation/allocation and exclude raster IO. CPU contention and array size affect timings. Vectorization moves iteration into compiled array operations; it does not eliminate computation.

## Practice — attempt before reading solutions
1. Normalize the uint8 example to [0,1]. Convert the channel offsets into those units and show that rescaling the result by 255 matches the worked answer.
2. Change the validity mask to select only the top row. Predict each channel mean and variance, then verify.
3. Explain why `reshape(3,2,4)` is not a valid replacement for HWC-to-CHW transpose. Find a differing element.
4. Change one pixel in a view, then repeat with a copy. Explain which original array changes.

## Solution guide — generated examples, not learner submissions
1. Both values and offsets must be divided by 255; the clipping maximum becomes 1. Scaling only the values changes the operation.
2. Valid values are [1,3] and [2,4]; means are [2,3], population variances [1,1].
3. Reshape preserves flat ordering, whereas transpose preserves the semantic relationship between pixel and band coordinates.
4. Basic slicing shares the underlying data; `.copy()` allocates independent storage.

## Reflection and limitations
The executed crop showed the same mathematical outputs for both implementations, with shorter vectorized timings on this machine. Equality comes before speed: a fast calculation with overflow or an incorrect mask is still wrong. This experiment covers one crop, not all array sizes or hardware. Brightness transformations are educational and must not be confused with physical reflectance correction.

Local Jupyter and Colab setup: [setup guide](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/setup.md). Colab requires cloning this repository, installing it, and setting its root as the working directory before running. The optional real-data script must use the pinned manifest. Run offline checks with `python -m pytest tests/test_array_lab.py`.

References: [NumPy beginner guide](https://numpy.org/doc/stable/user/absolute_beginners.html), [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), and the [dataset card](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/dataset-cards.md). Implementation validation does not establish that a learner has completed the exercises.

## Recorded validation

# MP02 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 2 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (5 code cells). Source SHA-256: `f73fa80b8573d79439cb0e4f0446e6314786cd55c358b058057468629e0df0f2`.

Commands from this directory: `python -m pytest -q test_mp02_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP03 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP03 — Image Resizing Laboratory

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded notebook, reusable NumPy implementation, worked solutions, meaningful tests and real training-crop experiments are executed. See [measured evidence](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp03-mp04-experiments.json) and [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP03_image_resizing_laboratory/image_resizing_laboratory.ipynb). Learner mastery remains unassessed.

## Learning objectives

- Nearest-neighbor, bilinear and bicubic interpolation.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP02 — NumPy Image Laboratory](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP02_numpy_image_laboratory/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Use coordinate mappings and nearest-neighbor, bilinear, and bicubic interpolation weights.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Executed 2026-10-19; see measured evidence above.** Resize a checkerboard and photograph with three methods; deliver crops and an aliasing comparison.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; run `python scripts/image_foundations_experiments.py` for the implemented experiment.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Python and NumPy](https://numpy.org/doc/stable/user/absolute_beginners.html)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP03_image_resizing_laboratory/image_resizing_laboratory.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP03_image_resizing_laboratory/mp03_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP03_image_resizing_laboratory/test_mp03_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP03_image_resizing_laboratory/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP03_image_resizing_laboratory/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP03 — Image Resizing Laboratory
Implementation experiments executed; learner mastery remains unassessed. Objective: understand nearest-neighbor, bilinear and bicubic interpolation.

## Intuition and coordinates
An image samples a spatial signal on a grid. Resizing estimates values on another grid; extra output pixels are not extra measurements. Our arrays are H×W×C: H rows, W columns, C bands. Values are floating point in explicitly declared units.

For an output row j, map to input coordinate u=(j+1/2)H/Hout−1/2. Hout is the output height; j starts at zero. Columns use the same formula with W. This half-pixel convention aligns pixel-cell boundaries. Clamp coordinates outside the source to its nearest edge. Other conventions can yield different numbers, especially near boundaries.

Nearest neighbor takes floor(u+1/2), choosing the larger index at ties. Bilinear interpolation uses four neighboring pixels. With lower row i=floor(u), lower column k=floor(v), a=u−i and b=v−k, the weights are (1−a)(1−b), (1−a)b, a(1−b), ab. They sum to one and are nonnegative. C bands are interpolated independently. Bilinear output therefore stays within the contributing range.

For a one-row image [0,10] resized to width four, source coordinates clamp to [0,0.25,0.75,1]. Bilinear predicts [0,2.5,7.5,10]; nearest predicts [0,0,10,10]. Predict these values before running.

## Bicubic and aliasing
Bicubic combines a 4×4 neighborhood with cubic weights. Negative weights can produce overshoot near sharp edges; clipping before measuring error would change the experiment. This lesson uses the declared PyTorch bicubic implementation rather than claiming a new bicubic implementation. Its settings are align_corners=False and antialias=False.

Downsampling can make high-frequency patterns appear as false low-frequency structure: aliasing. Interpolation alone is not a general antialiasing filter. A one-pixel checkerboard reduced by two becomes a constant with our nearest tie rule; bilinear returns 0.5 for this particular pattern. That example does not prove bilinear prevents aliasing for all patterns. An explicit low-pass filter is needed before general downsampling.

## Real-data experiment and interpretation
Run `python scripts/image_foundations_experiments.py` from the repository root after the existing pinned pilot download. It verifies the training scene checksum, reads a fully valid 64×64 RGBNIR crop, averages nonoverlapping 2×2 cells to produce LR, then reconstructs with three methods. No test scene is used. Scaling is fixed stored values/10000, with physical calibration unverified.

On this crop bicubic had lower MSE in every band than bilinear and nearest. This is one example, not a universal ranking. The checkerboard illustrates aliasing; it is a synthetic diagnostic rather than a remote-sensing performance result. The next cell reads recorded evidence and shows the measured figure; it does not rerun the experiment.

## Practice
1. Calculate the bilinear value at the center of [[0,10],[20,30]]. Define all weights.
2. Resize a constant multiband image to an odd shape. Explain why the constant must be preserved.
3. Shift the checkerboard by one input pixel and repeat nearest downsampling. Predict the new constant.
4. Explain why a smooth resized image is not proof of recovered detail.

## Solution guide — read after attempting
1. a=b=0.5; each weight is 0.25, so the answer is 15.
2. The four weights sum to one independently in every band, including replicated boundaries.
3. The chosen checkerboard parity reverses, so constant zero becomes constant one. Sampling phase matters.
4. Interpolation estimates between samples; several different HR signals can share the same LR samples. Smoothness is not independent evidence of correctness.

## Reuse, execution and limitations
Implementation: `src/rssr/image_foundations.py`; checks: `python -m pytest tests/test_image_foundations.py`. See [setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/setup.md) for local Jupyter and Colab clone/install instructions. Run from notebooks/ or the repository root. Scientific georeferencing is not handled by this NumPy utility; use the separate GeoTIFF pipeline for CRS/transform-preserving export.

[Authoritative brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP03_image_resizing_laboratory/README.md), [dataset card](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/dataset-cards.md), [measured record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp03-mp04-experiments.json). The existing PyTorch comparison is an implementation check; it is not proof of physical reconstruction accuracy.

## Recorded validation

# MP03 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 6 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (4 code cells). Source SHA-256: `fd83d906df6f61572e4e60306c48ddedd7261ae272fae5df129f9b4d11f9cf98`.

Commands from this directory: `python -m pytest -q test_mp03_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP04 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP04 — Image Comparison Tool

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded notebook, reusable NumPy implementation, worked solutions, meaningful tests and real training-crop experiments are executed. See [measured evidence](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp03-mp04-experiments.json) and [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP04_image_comparison_tool/image_comparison_tool.ipynb). Learner mastery remains unassessed.

## Learning objectives

- Image differences, absolute error, MSE and visualization.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP03 — Image Resizing Laboratory](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP03_image_resizing_laboratory/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Compute signed difference d = x − y, absolute difference, and per-channel summary statistics.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Executed 2026-10-19; see measured evidence above.** Compare identical, shifted, and brightness-adjusted pairs; deliver difference maps and alignment notes.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; run `python scripts/image_foundations_experiments.py` for the implemented experiment.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Python and NumPy](https://numpy.org/doc/stable/user/absolute_beginners.html)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP04_image_comparison_tool/image_comparison_tool.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP04_image_comparison_tool/mp04_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP04_image_comparison_tool/test_mp04_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP04_image_comparison_tool/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP04_image_comparison_tool/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP04 — Image Comparison Tool
Implementation experiments executed; learner mastery remains unassessed. Objective: signed differences, absolute error, MSE and visual interpretation.

## What does error mean?
Let p[h,w,c] be a prediction and r[h,w,c] a reference on the same H×W×C grid, in the same units. Define d=p−r. Positive d means the prediction is larger. Cast integer inputs before subtraction: uint8 cannot represent −255.

For K valid pixels in a band, bias=sum(d)/K; mean absolute error (MAE)=sum(|d|)/K; mean squared error (MSE)=sum(d²)/K. Bias and MAE retain input units; MSE has squared units. Positive and negative errors can cancel in bias, but cannot cancel in MAE or MSE. For errors [−2,+2], bias=0, MAE=2, MSE=4. These statistics answer different questions.

A boolean H×W mask selects valid pixels for every band. An empty mask is an error. Equal array shapes alone do not establish alignment: images must refer to corresponding ground locations, compatible bands, times and radiometric units.

## Controlled comparisons before interpretation
We compare identical images, a brightness increase of +0.02, and a one-pixel horizontal shift. With no clipping, a uniform +0.02 change predicts bias=MAE=0.02 and MSE=0.0004. The shift preserves content but changes correspondence, demonstrating why registration matters.

The real training crop uses the same checksummed source and fixed scaling as MP03. A rolled shift wraps its first column; we exclude that column from **all three** comparisons, giving equal support of 4032 pixels. That exclusion prevents a synthetic wrap boundary from contaminating the comparison. It is not a registration method. No test scene is used.

## Maps and measured results
Run `python scripts/image_foundations_experiments.py` to generate both projects' evidence. Signed maps use a diverging color scale centered at zero, fixed to [−0.04,+0.04]. Absolute maps use [0,0.04]. All pairs share the same limits; invalid pixels are blank. Averaging signed differences over bands can conceal opposing band errors, so per-band numerical summaries accompany the maps.

Observed: identical images give zero error, brightness gives the predicted 0.0004 MSE per band, and shifted images have nonzero error with near-zero bias. Near-zero bias therefore does not imply a correct image. Changing display limits independently could conceal these differences.

## Practice
1. For errors [−1,+3], calculate bias, MAE and MSE.
2. Mask out the second pixel. How do all three statistics change?
3. Explain why subtracting images in different physical units is invalid even when their shapes match.
4. Describe how clipping could invalidate the brightness experiment's analytic prediction.

## Solution guide — generated answers, not learner work
1. Bias=1, MAE=2, MSE=5.
2. Only −1 remains: bias=−1, MAE=1, MSE=1.
3. Numerical subtraction requires compatible meanings and units; dimensions alone cannot validate this.
4. Saturated values change by less than the requested offset. The assumption of a constant +0.02 difference no longer holds.

## Execution, limitations and references
Use [local/Colab setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/setup.md), then run this notebook from notebooks/ or the repository root. Module: `src/rssr/image_foundations.py`; tests: `python -m pytest tests/test_image_foundations.py`. No data download happens in this notebook.

This experiment uses a known synthetic shift on one training crop. It does not estimate sensor registration, perceptual quality or geographic generalization. See the [authoritative brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP04_image_comparison_tool/README.md), [evaluation protocol](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/protocol.md), [dataset card](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/dataset-cards.md), and [measured record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp03-mp04-experiments.json).

## Recorded validation

# MP04 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 6 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (4 code cells). Source SHA-256: `69ccec42ce820c46e1be95ca53beec9633b4da2ce75d68f14fc20dc4c4e6e4bd`.

Commands from this directory: `python -m pytest -q test_mp04_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP05 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP05 — Gaussian Blur Simulator

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP05_gaussian_blur_simulator/gaussian_blur_simulator.ipynb) executed in a real kernel. [Measured experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp05-mp08-experiments.json) records the checksummed training crop and controlled settings. Reproduce with `python scripts/degradation_experiments.py`. Worked solutions are included; learner mastery remains unassessed.

## Learning objectives

- Convolution, Gaussian kernels, kernel size and blur strength.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP04 — Image Comparison Tool](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_01_python_numpy/MP04_image_comparison_tool/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Use G(x,y) proportional to exp(−(x²+y²)/(2σ²)); normalize discrete kernel weights to sum to one.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented and executed on synthetic fixtures and the recorded training crop.** Sweep sigma and kernel support on impulses and edges; deliver kernel plots and boundary-effect comparisons.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; provided in the linked notebook and experiment script.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Image processing](https://scikit-image.org/docs/stable/auto_examples/)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP05_gaussian_blur_simulator/gaussian_blur_simulator.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP05_gaussian_blur_simulator/mp05_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP05_gaussian_blur_simulator/test_mp05_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP05_gaussian_blur_simulator/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP05_gaussian_blur_simulator/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP05 — Gaussian Blur Simulator

Expanded engineering lesson; learner mastery unassessed.

## Intuition, definitions and worked mathematics
A blur replaces each sample by a weighted neighborhood. H and W count rows and columns; C counts bands. Our image is H×W×C in floating numerical units. For integer offsets i,j in [−r,r], the Gaussian weight is exp(−(i²+j²)/(2 sigma²))/Z. Sigma is a positive width in pixels, r is an integer radius, and Z is the sum of all retained weights. The support has width 2r+1. Normalization makes the weights sum to one, so a constant image stays constant under replicated or reflected boundaries.

The implementation computes correlation: y[h,w,c]=sum(k[i,j]x[h+i,w+j,c]). Convolution reverses the kernel offsets. For a symmetric Gaussian they agree; they need not agree for an asymmetric edge kernel. Edge replication copies the nearest pixel; reflection mirrors nearby values; constant padding inserts zeros. These assumptions affect edge pixels and must be recorded.

For sigma=1 and radius=1, the unnormalized 1D weights are approximately [0.60653,1,0.60653], normalized to [0.27407,0.45186,0.27407]. Their outer product gives the 3×3 kernel, with center about 0.20418. A centered unit impulse produces the kernel response. Greater sigma spreads influence only if the support is large enough; truncating to radius one cannot represent a very broad Gaussian.

## Real-data experiment and interpretation
The real experiment sweeps (sigma,radius)=(0.5,2),(1,3),(2,6), plots kernels and impulse/step-edge responses, and measures how much a fixed training crop changes. The corner of a constant image is preserved by replication but darkens with zero padding. This is a boundary artifact, not evidence of a dark object. Blur MSE against the original quantifies a change, not whether blur is scientifically appropriate.

Run `python scripts/degradation_experiments.py` explicitly from the repository root. The notebook below reads recorded measurements; it does not download data or execute a hidden research run. The source is a checksummed 64×64 NAIP training crop with fixed stored-value/10000 scaling, not verified reflectance.

## Practice — attempt before reading solutions
1. Why does a normalized blur preserve a constant image?
2. What happens when radius=0?
3. Why might sigma=5 with radius=1 be misleading?

## Separate solution guide
1. The weights sum to one, so weighted copies of the same constant sum to that constant, provided padding preserves it.
2. There is only one weight, equal to one, so the image is unchanged.
3. Most Gaussian mass is omitted; renormalization creates a short nearly flat kernel, not a broad sensor point-spread function.

These are generated examples, not evidence that a learner solved the exercises.

## Reuse, execution and limitations
Use `src/rssr/degradation_lab.py` and `python -m pytest tests/test_degradation_lab.py`. Follow [local and Colab setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/setup.md); run from notebooks/ or the repository root. The shared NumPy code preserves HWC channels and uses explicit boundaries. No physical sensor calibration or geographic generalization is established.

References: the authoritative stage brief, [dataset card](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/dataset-cards.md), [protocol](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/protocol.md), and [NumPy FFT documentation](https://numpy.org/doc/stable/reference/routines.fft.html). See [execution evidence](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/notebook-execution.json) and [experiment record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp05-mp08-experiments.json).

## Recorded validation

# MP05 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 4 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (4 code cells). Source SHA-256: `2a89ce25157b78aed66add153bb42def7c161fe0982b898e5363e11de4b610e6`.

Commands from this directory: `python -m pytest -q test_mp05_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP06 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP06 — Low-Resolution Image Generator

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP06_low_resolution_image_generator/low_resolution_image_generator.ipynb) executed in a real kernel. [Measured experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp05-mp08-experiments.json) records the checksummed training crop and controlled settings. Reproduce with `python scripts/degradation_experiments.py`. Worked solutions are included; learner mastery remains unassessed.

## Learning objectives

- Downsampling, scale factors, sampling and aliasing.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP05 — Gaussian Blur Simulator](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP05_gaussian_blur_simulator/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Model y = D_s(k * x); explain blur before decimation and integer scale factors.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented and executed on synthetic fixtures and the recorded training crop.** Generate ×2 and ×4 pairs with and without antialiasing; deliver a pairing manifest and exact shape checks.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; provided in the linked notebook and experiment script.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Image processing](https://scikit-image.org/docs/stable/auto_examples/)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP06_low_resolution_image_generator/low_resolution_image_generator.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP06_low_resolution_image_generator/mp06_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP06_low_resolution_image_generator/test_mp06_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP06_low_resolution_image_generator/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP06_low_resolution_image_generator/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP06 — Low-Resolution Image Generator

Expanded engineering lesson; learner mastery unassessed.

## Intuition, definitions and worked mathematics
A spatial scale s=2 or 4 means LR height H/s and width W/s, not H/s². Pixel count decreases by s². C bands remain unchanged. For synthetic pairing, HR and LR originate from the same scene and declared crop; a geographic split must happen before extracting patches.

The model y=D_s(k*x) means first blur x with kernel k, then sample with operator D_s. Unfiltered phase-zero decimation takes x[0::s,0::s]. Area averaging instead takes the mean of each s×s block. They are different forward models. Blur before decimation suppresses high spatial frequencies that would fold into lower frequencies. A filter is not a guarantee that all aliasing is removed.

For [[0,2],[4,6]], factor-two area averaging gives 3; phase-zero decimation gives 0. For a one-pixel checkerboard, decimation at an even factor produces a constant depending on sampling phase, although the original oscillates. This is aliasing. The mean of each 2×2 cell is 0.5, which describes local coverage but cannot recover the missing fine pattern.

A valid pair manifest needs source/checksum, crop, split, band order, scale, operator and file hashes. Changing any of these changes the experimental meaning. Our generator rejects dimensions not divisible by s instead of silently cropping.

## Real-data experiment and interpretation
The measured training-crop manifest contains explicit unfiltered decimation pairs and separate area-average pairs, with optional Gaussian blur, at ×2 and ×4. All pairs keep four bands. Shape and checksum records make the transformation reproducible. This is synthetic degradation of an aerial image; it does not create a real paired satellite dataset.

Run `python scripts/degradation_experiments.py` explicitly from the repository root. The notebook below reads recorded measurements; it does not download data or execute a hidden research run. The source is a checksummed 64×64 NAIP training crop with fixed stored-value/10000 scaling, not verified reflectance.

## Practice — attempt before reading solutions
1. What is the LR shape for HR 64×64×4 at ×4?
2. Why split scenes before patches?
3. Does sigma=0 in degrade_hwc mean no antialias filtering?

## Separate solution guide
1. 16×16×4; 256 spatial pixels rather than 4096.
2. Adjacent patches and repeated locations could otherwise leak between training and evaluation.
3. No. This function still area-averages. The separate decimation pairs are the unfiltered control.

These are generated examples, not evidence that a learner solved the exercises.

## Reuse, execution and limitations
Use `src/rssr/degradation_lab.py` and `python -m pytest tests/test_degradation_lab.py`. Follow [local and Colab setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/setup.md); run from notebooks/ or the repository root. The shared NumPy code preserves HWC channels and uses explicit boundaries. No physical sensor calibration or geographic generalization is established.

References: the authoritative stage brief, [dataset card](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/dataset-cards.md), [protocol](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/protocol.md), and [NumPy FFT documentation](https://numpy.org/doc/stable/reference/routines.fft.html). See [execution evidence](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/notebook-execution.json) and [experiment record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp05-mp08-experiments.json).

## Recorded validation

# MP06 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 4 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (3 code cells). Source SHA-256: `b5779aa05fcd78193b32aeceeb2fa85f1d12f95901ff2701b055f3b781a842e7`.

Commands from this directory: `python -m pytest -q test_mp06_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP07 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP07 — Synthetic Degradation Laboratory

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP07_synthetic_degradation_laboratory/synthetic_degradation_laboratory.ipynb) executed in a real kernel. [Measured experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp05-mp08-experiments.json) records the checksummed training crop and controlled settings. Reproduce with `python scripts/degradation_experiments.py`. Worked solutions are included; learner mastery remains unassessed.

## Learning objectives

- Blur, noise, downsampling and degradation combinations.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP06 — Low-Resolution Image Generator](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP06_low_resolution_image_generator/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Use y = D_s(k * x) + n with n drawn from a specified noise distribution; distinguish compression from additive noise.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented and executed on synthetic fixtures and the recorded training crop.** Compare blur–resize–noise orderings and JPEG quality levels; deliver a degradation grid and seed/configuration log.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; provided in the linked notebook and experiment script.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Image processing](https://scikit-image.org/docs/stable/auto_examples/)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP07_synthetic_degradation_laboratory/synthetic_degradation_laboratory.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP07_synthetic_degradation_laboratory/mp07_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP07_synthetic_degradation_laboratory/test_mp07_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP07_synthetic_degradation_laboratory/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP07_synthetic_degradation_laboratory/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP07 — Synthetic Degradation Laboratory

Expanded engineering lesson; learner mastery unassessed.

## Intuition, definitions and worked mathematics
Let x be H×W×C HR data. A synthetic LR observation is y=D_s(k*x)+n. The Gaussian kernel k has unit sum; s is an integer spatial reduction; n is additive noise in the same numerical units as y. In this laboratory, D_s area-averages nonoverlapping s×s cells. Noise is sampled independently per LR pixel and band from a normal distribution with mean zero and standard deviation tau.

Noise standard deviation has input units; variance tau² has squared units. A random seed selects a reproducible sequence, not an independent physical observation. Noise is not clipped here: Gaussian samples can be negative or exceed the display range. Clipping changes the noise distribution and must be a separate declared choice.

Order matters. Adding independent HR noise with variance tau² before averaging s² pixels reduces its LR variance to tau²/s²; adding noise of variance tau² afterward does not. Blur and noise are simplified modeling choices, not verified sensor physics. The same numerical sigma is measured in HR pixels and may correspond to different physical distances at different ground resolutions.

With a constant image 0.2 and no noise, any normalized blur plus area average should return 0.2. If tau=0.01 is added afterward, expected error mean is zero and expected error MSE is 0.0001 over repeated samples, but one finite sample will differ.

## Real-data experiment and interpretation
The real-crop sweep changes scale (2,4), Gaussian width (0,0.8) and LR noise standard deviation (0,0.01), keeping seed 17 and source fixed. Eight area-average conditions plus two unfiltered controls are saved with hashes. They demonstrate parameter effects, not fitted sensor degradation. Do not tune a research model on the held-out scene using these variants.

Run `python scripts/degradation_experiments.py` explicitly from the repository root. The notebook below reads recorded measurements; it does not download data or execute a hidden research run. The source is a checksummed 64×64 NAIP training crop with fixed stored-value/10000 scaling, not verified reflectance.

## Practice — attempt before reading solutions
1. What LR variance results from averaging independent HR noise of variance 0.0001 at ×2?
2. Why should a noisy result not always equal its expected MSE?
3. What changes if you clip negatives to zero?

## Separate solution guide
1. 0.0001/4=0.000025, assuming four independent samples.
2. An expectation averages hypothetical repeated draws; a finite realization fluctuates.
3. The distribution becomes biased near zero and is no longer the stated additive Gaussian model.

These are generated examples, not evidence that a learner solved the exercises.

## Reuse, execution and limitations
Use `src/rssr/degradation_lab.py` and `python -m pytest tests/test_degradation_lab.py`. Follow [local and Colab setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/setup.md); run from notebooks/ or the repository root. The shared NumPy code preserves HWC channels and uses explicit boundaries. No physical sensor calibration or geographic generalization is established.

References: the authoritative stage brief, [dataset card](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/dataset-cards.md), [protocol](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/protocol.md), and [NumPy FFT documentation](https://numpy.org/doc/stable/reference/routines.fft.html). See [execution evidence](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/notebook-execution.json) and [experiment record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp05-mp08-experiments.json).

## Recorded validation

# MP07 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 4 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (3 code cells). Source SHA-256: `3f119be5d4bb00cc92d02955ccdb789e66764d254c0faf987af03e2541b6b6d0`.

Commands from this directory: `python -m pytest -q test_mp07_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP08 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP08 — Frequency-Domain Image Explorer

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP08_frequency_domain_image_explorer/frequency_domain_image_explorer.ipynb) executed in a real kernel. [Measured experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp05-mp08-experiments.json) records the checksummed training crop and controlled settings. Reproduce with `python scripts/degradation_experiments.py`. Worked solutions are included; learner mastery remains unassessed.

## Learning objectives

- Fourier transforms, spatial frequencies, edge and texture information.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP07 — Synthetic Degradation Laboratory](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP07_synthetic_degradation_laboratory/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Apply the 2D discrete Fourier transform; relate spatial convolution to frequency multiplication.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented and executed on synthetic fixtures and the recorded training crop.** Analyze sinusoids, edges, and blurred images; deliver log-magnitude spectra and a filter-response explanation.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; provided in the linked notebook and experiment script.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Image processing](https://scikit-image.org/docs/stable/auto_examples/)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP08_frequency_domain_image_explorer/frequency_domain_image_explorer.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP08_frequency_domain_image_explorer/mp08_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP08_frequency_domain_image_explorer/test_mp08_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP08_frequency_domain_image_explorer/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP08_frequency_domain_image_explorer/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP08 — Frequency-Domain Image Explorer

Expanded engineering lesson; learner mastery unassessed.

## Intuition, definitions and worked mathematics
A discrete Fourier transform decomposes a 2D image into periodic spatial-frequency components. For H rows and W columns, F[u,v]=sum over h,w of x[h,w] exp(−2 pi i(uh/H+vw/W)). Here i²=−1, u and v are integer frequency-bin indices, and x is a single band. The forward NumPy FFT has no normalization; the inverse divides by HW.

Magnitude |F| describes frequency strength. Phase describes placement; magnitude alone cannot reconstruct the original image. FFT assumes periodic boundaries, so a mismatch between opposite edges can add spectral energy. fftshift moves the zero-frequency component to the display center. We display log(1+|F|) because the DC component may dominate; this visualization is not a physical logarithmic radiance measurement.

For a constant 4×4 image of value 2, F[0,0]=32 and all other bins are zero up to floating-point error. Parseval's relation is sum(x²)=sum(|F|²)/(HW) under this normalization. The inverse must reproduce x within numerical precision.

Gaussian smoothing suppresses high spatial frequencies. White noise adds broad frequency content. A spectrum alone does not identify which structures are physically real; textures, noise and boundary discontinuities can overlap in frequency.

## Real-data experiment and interpretation
The experiment compares the same real training red band, its Gaussian-blurred version, and a seeded noisy version. All frequency displays share fixed limits. It records inverse reconstruction error and both sides of Parseval's identity. Changes in high-frequency magnitude are visible, but this does not validate an SR model or distinguish useful detail from invented detail.

Run `python scripts/degradation_experiments.py` explicitly from the repository root. The notebook below reads recorded measurements; it does not download data or execute a hidden research run. The source is a checksummed 64×64 NAIP training crop with fixed stored-value/10000 scaling, not verified reflectance.

## Practice — attempt before reading solutions
1. What is the DC value of an 8×8 constant image valued 0.5?
2. Why does shifting an image affect phase?
3. Why not compare independently auto-scaled spectral plots?

## Separate solution guide
1. 64×0.5=32 under the unnormalized forward transform.
2. A translation changes the alignment of each periodic component while preserving magnitude for a circular shift.
3. Different color scales can make weak and strong components look equally intense.

These are generated examples, not evidence that a learner solved the exercises.

## Reuse, execution and limitations
Use `src/rssr/degradation_lab.py` and `python -m pytest tests/test_degradation_lab.py`. Follow [local and Colab setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/setup.md); run from notebooks/ or the repository root. The shared NumPy code preserves HWC channels and uses explicit boundaries. No physical sensor calibration or geographic generalization is established.

References: the authoritative stage brief, [dataset card](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/dataset-cards.md), [protocol](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/protocol.md), and [NumPy FFT documentation](https://numpy.org/doc/stable/reference/routines.fft.html). See [execution evidence](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/notebook-execution.json) and [experiment record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp05-mp08-experiments.json).

## Recorded validation

# MP08 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 4 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (4 code cells). Source SHA-256: `ca4d684abc599c3db41fcb668d48a03b2c68599268fc1225ac894fb3c1467189`.

Commands from this directory: `python -m pytest -q test_mp08_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP09 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP09 — MSE and PSNR from Scratch

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP09_mse_and_psnr_from_scratch/mse_and_psnr_from_scratch.ipynb) executed in a real kernel. [Measured experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp09-mp11-experiments.json) records the checksummed training crop and controlled settings. Reproduce with `python scripts/metric_experiments.py`. Worked solutions are included; learner mastery remains unassessed.

## Learning objectives

- Reconstruction error, dynamic range and PSNR.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP08 — Frequency-Domain Image Explorer](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_02_image_processing/MP08_frequency_domain_image_explorer/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Use MSE = mean((x − y)²) and PSNR = 10 log10(L²/MSE); handle zero MSE as positive infinity.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented and executed on synthetic fixtures and the recorded training crop.** Evaluate identical images and controlled noise levels; deliver hand-checked metric examples and a range-sensitivity table.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; provided in the linked notebook and experiment script.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Image metrics](https://scikit-image.org/docs/stable/api/skimage.metrics.html)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP09_mse_and_psnr_from_scratch/mse_and_psnr_from_scratch.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP09_mse_and_psnr_from_scratch/mp09_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP09_mse_and_psnr_from_scratch/test_mp09_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP09_mse_and_psnr_from_scratch/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP09_mse_and_psnr_from_scratch/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP09 — MSE and PSNR from Scratch

Expanded executable lesson. Prerequisites: earlier image-processing and array lessons. Learning goals: derive the calculation, predict edge cases, implement it, and interpret controlled real-image evidence. Learner mastery is not asserted.

## Setup and conventions
From repository root run `python -m pip install -e ".[dev]"`. Run `python scripts/metric_experiments.py` after downloading the pinned training scene with the portable quickstart. The notebook reads the committed numerical record; it does not silently download imagery. Arrays are HWC float64 for metrics.

## Mathematics and worked example
Let prediction p and reference r have H rows, W columns and C bands, all co-registered in identical units. N=HWC samples. MSE=sum((p-r)^2)/N has squared intensity units; RMSE=sqrt(MSE) has intensity units. PSNR=10 log10(L²/MSE) is in decibels, where L is the declared dynamic range, not the observed maximum of each image. Our normalized numerical samples use L=1. Identical images have MSE=0 and positive-infinite PSNR; this is a special case, not a division error.

For r=[0,0,0,0] and p=[0,0,0,2], squared errors sum to 4, so MSE=1. With L=2, PSNR=10 log10(4)=6.0206 dB. Cast unsigned integers before subtraction to prevent wraparound. Scaling both images and L by the same factor leaves PSNR unchanged. Changing only L from 1 to 255 artificially adds 48.1308 dB. Doubling Gaussian noise standard deviation approximately quadruples MSE and lowers PSNR by 6.0206 dB; finite samples fluctuate.

## Recorded real-data experiment
The source checksum, acquisition date, license, normalization, seed and settings are in the record below. The input is a 64×64 training crop of real four-band NAIP aerial imagery; these are synthetic distortions, not satellite sensor measurements. Reproduce with `python scripts/metric_experiments.py`.

## Practice
Predict before running: (1) MSE for errors [1,-1]? (2) PSNR change when only L doubles? (3) Can high PSNR prove that a generated road is real?

## Worked solutions
(1) Both squared errors are one, so MSE=1. (2) +6.0206 dB, a protocol change rather than model improvement. (3) No; an average pixel metric cannot certify individual structures or compensate for invalid reference alignment.

## Figure and interpretation
See [dashboard](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp09-mp11-dashboard.png). Blur and matched-MSE noise have equal PSNR but different SSIM. All error maps share a scale. Contrast and blur affect this low-texture crop less than the .01 noise case; this observation is not a universal ranking. Reflect: explain a failure mode that both metrics could miss.

## References and limitations
[Official scikit-image metric documentation](https://scikit-image.org/docs/stable/api/skimage.metrics.html), checked 2026-10-19, documents explicit ranges and SSIM settings. Our NumPy implementation uses declared uniform population moments and valid windows; it does not claim default-package equivalence. No alignment correction, learned perceptual metric or independent-scene claim is made.

## Recorded validation

# MP09 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 3 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (4 code cells). Source SHA-256: `466612ed5173a8261a4f6c987feb15a37e421c03578fb3440727909b8d56d0af`.

Commands from this directory: `python -m pytest -q test_mp09_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP10 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP10 — SSIM Evaluation Tool

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP10_ssim_evaluation_tool/ssim_evaluation_tool.ipynb) executed in a real kernel. [Measured experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp09-mp11-experiments.json) records the checksummed training crop and controlled settings. Reproduce with `python scripts/metric_experiments.py`. Worked solutions are included; learner mastery remains unassessed.

## Learning objectives

- Luminance, contrast and structural similarity.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP09 — MSE and PSNR from Scratch](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP09_mse_and_psnr_from_scratch/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Study SSIM = ((2μxμy+C1)(2σxy+C2))/((μx²+μy²+C1)(σx²+σy²+C2)).

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented and executed on synthetic fixtures and the recorded training crop.** Compare blur, contrast shifts, and noise under a fixed window and data range; deliver SSIM maps and PSNR disagreements.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; provided in the linked notebook and experiment script.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Image metrics](https://scikit-image.org/docs/stable/api/skimage.metrics.html)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP10_ssim_evaluation_tool/ssim_evaluation_tool.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP10_ssim_evaluation_tool/mp10_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP10_ssim_evaluation_tool/test_mp10_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP10_ssim_evaluation_tool/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP10_ssim_evaluation_tool/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP10 — SSIM Evaluation Tool

Expanded executable lesson. Prerequisites: earlier image-processing and array lessons. Learning goals: derive the calculation, predict edge cases, implement it, and interpret controlled real-image evidence. Learner mastery is not asserted.

## Setup and conventions
From repository root run `python -m pip install -e ".[dev]"`. Run `python scripts/metric_experiments.py` after downloading the pinned training scene with the portable quickstart. The notebook reads the committed numerical record; it does not silently download imagery. Arrays are HWC float64 for metrics.

## Mathematics and worked example
SSIM compares local luminance and local second-order statistics. For a window of n pixels, mu_x=sum(x)/n, variance_x=sum((x-mu_x)^2)/n, and covariance_xy=sum((x-mu_x)(y-mu_y))/n. We use population moments, uniform 7×7 windows, no padding, one score per band/window, and average these equally. The valid map has shape (H-6,W-6,C). This differs from implementations using Gaussian weights or sample moments.

SSIM=((2 mu_x mu_y+C1)(2 covariance+C2))/((mu_x²+mu_y²+C1)(variance_x+variance_y+C2)), with C1=(.01 L)² and C2=(.03 L)². Constants stabilize nearly flat regions. For constant windows x=.5,y=.6,L=1, variance and covariance vanish, the second ratio is one, and SSIM=(.6+.0001)/(.25+.36+.0001), approximately .98361. Identical windows have score one. Negative covariance may yield negative SSIM; it is not a probability.

Two errors can have equal MSE and PSNR but different spatial structure. The recorded experiment scales a noise realization to exactly the blur MSE; SSIM distinguishes their local statistics. This does not make SSIM a universal perceptual or scientific quality oracle. Multi-band averaging also hides which band failed.

## Recorded real-data experiment
The source checksum, acquisition date, license, normalization, seed and settings are in the record below. The input is a 64×64 training crop of real four-band NAIP aerial imagery; these are synthetic distortions, not satellite sensor measurements. Reproduce with `python scripts/metric_experiments.py`.

## Practice
Predict before running: (1) Map dimensions for 16×20×4 and window 7? (2) Why exclude a window containing nodata? (3) Is default-library SSIM comparable without checking its settings?

## Worked solutions
(1) 10×14×4. (2) Its moments would mix observations with missing-data fill values; mask whole windows, not only their centers. (3) No: range, window, weights, covariance normalization, border and band aggregation must agree.

## Figure and interpretation
See [dashboard](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp09-mp11-dashboard.png). Blur and matched-MSE noise have equal PSNR but different SSIM. All error maps share a scale. Contrast and blur affect this low-texture crop less than the .01 noise case; this observation is not a universal ranking. Reflect: explain a failure mode that both metrics could miss.

## References and limitations
[Official scikit-image metric documentation](https://scikit-image.org/docs/stable/api/skimage.metrics.html), checked 2026-10-19, documents explicit ranges and SSIM settings. Our NumPy implementation uses declared uniform population moments and valid windows; it does not claim default-package equivalence. No alignment correction, learned perceptual metric or independent-scene claim is made.

## Recorded validation

# MP10 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 3 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (4 code cells). Source SHA-256: `82209de472a6555c3dc5cbebc26b888426a50316938519034eb8df69be545d16`.

Commands from this directory: `python -m pytest -q test_mp10_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP11 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP11 — Super-Resolution Evaluation Dashboard

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP11_super_resolution_evaluation_dashboard/super_resolution_evaluation_dashboard.ipynb) executed in a real kernel. [Measured experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp09-mp11-experiments.json) records the checksummed training crop and controlled settings. Reproduce with `python scripts/metric_experiments.py`. Worked solutions are included; learner mastery remains unassessed.

## Learning objectives

- Metrics, error maps, comparison figures and result tables.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP10 — SSIM Evaluation Tool](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP10_ssim_evaluation_tool/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Define per-image PSNR and SSIM aggregation, error-map scaling, and fair comparison conventions.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented and executed on synthetic fixtures and the recorded training crop.** Design static metric tables, side-by-side crops, and error maps for identical, noisy, and blurred fixtures; deliver a dashboard specification and verified metric examples.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; provided in the linked notebook and experiment script.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Image metrics](https://scikit-image.org/docs/stable/api/skimage.metrics.html)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP11_super_resolution_evaluation_dashboard/super_resolution_evaluation_dashboard.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP11_super_resolution_evaluation_dashboard/mp11_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP11_super_resolution_evaluation_dashboard/test_mp11_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP11_super_resolution_evaluation_dashboard/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP11_super_resolution_evaluation_dashboard/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP11 — Super-Resolution Evaluation Dashboard

Expanded executable lesson. Prerequisites: earlier image-processing and array lessons. Learning goals: derive the calculation, predict edge cases, implement it, and interpret controlled real-image evidence. Learner mastery is not asserted.

## Setup and conventions
From repository root run `python -m pip install -e ".[dev]"`. Run `python scripts/metric_experiments.py` after downloading the pinned training scene with the portable quickstart. The notebook reads the committed numerical record; it does not silently download imagery. Arrays are HWC float64 for metrics.

## Mathematics and worked example
A comparison is valid only if predictions and reference share grids, bands, units, masks and crop policy. This static dashboard fixes each of these before measuring. Its first row displays the red band with a common range determined from the reference crop, the second row displays mean absolute error over four bands with a shared zero-to-maximum scale, and the third row displays mean band SSIM on valid windows with a common [-1,1] scale. Display stretching never changes metric inputs.

Aggregate at the level of the claim: patches within a scene are correlated; a hundred patches are not a hundred independent regions. For per-image MSE .01 and .0001 at L=1, image PSNRs are 20 and 40 dB. Their mean is 30 dB. PSNR of mean MSE is -10 log10(.00505)=22.9671 dB. These answer different questions. State which you report, and do not average missing-reference images into a fabricated score.

The dashboard is a controlled-distortion exercise on a training crop, not a leaderboard. Identical, blur, contrast and seeded noise cases expose numerical and display errors. Matched-MSE noise shows why one scalar is insufficient; band RMSE prevents an RGB display from concealing NIR errors.

## Recorded real-data experiment
The source checksum, acquisition date, license, normalization, seed and settings are in the record below. The input is a 64×64 training crop of real four-band NAIP aerial imagery; these are synthetic distortions, not satellite sensor measurements. Reproduce with `python scripts/metric_experiments.py`.

## Practice
Predict before running: (1) Why use a shared error color scale? (2) Should an unreferenced Sentinel estimate show PSNR? (3) What is the sampling unit for a geographic uncertainty claim?

## Worked solutions
(1) Independently stretched errors can make a poor result look equally good. (2) No, mark it unavailable and report provenance. (3) Independent scenes/regions, with repeated training seeds reported separately; this one crop cannot establish geographic uncertainty.

## Figure and interpretation
See [dashboard](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp09-mp11-dashboard.png). Blur and matched-MSE noise have equal PSNR but different SSIM. All error maps share a scale. Contrast and blur affect this low-texture crop less than the .01 noise case; this observation is not a universal ranking. Reflect: explain a failure mode that both metrics could miss.

## References and limitations
[Official scikit-image metric documentation](https://scikit-image.org/docs/stable/api/skimage.metrics.html), checked 2026-10-19, documents explicit ranges and SSIM settings. Our NumPy implementation uses declared uniform population moments and valid windows; it does not claim default-package equivalence. No alignment correction, learned perceptual metric or independent-scene claim is made.

## Recorded validation

# MP11 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 3 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (4 code cells). Source SHA-256: `1c5a3eb9da4e0333c35a4c4c8a04cda0214d48809c99e55405f555fcc5199033`.

Commands from this directory: `python -m pytest -q test_mp11_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP12 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP12 — Linear Algebra Visualizer

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP12_linear_algebra_visualizer/linear_algebra_visualizer.ipynb) passed a real isolated kernel. [Measured evidence](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp12-svd.json) includes checksums and actual outcomes. Reproduce with `python scripts/linear_experiment.py`. Worked solutions are provided; learner mastery remains unassessed.

## Learning objectives

- Vectors, matrices, tensor shapes, transformations and norms.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP11 — Super-Resolution Evaluation Dashboard](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_03_metrics/MP11_super_resolution_evaluation_dashboard/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Study matrix dimensions, inner products, norms, eigenvectors, and singular value decomposition.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented and executed; evidence linked above.** Approximate grayscale images at several SVD ranks; deliver reconstruction/error curves and a memory comparison.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; supplied in the linked notebook and experiment script.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Mathematics and PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP12_linear_algebra_visualizer/linear_algebra_visualizer.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP12_linear_algebra_visualizer/mp12_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP12_linear_algebra_visualizer/test_mp12_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP12_linear_algebra_visualizer/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP12_linear_algebra_visualizer/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP12 — Linear Algebra Visualizer

Learn to track matrix dimensions, interpret dot products and norms, connect eigenvectors to singular vectors, and measure rank/error/storage tradeoffs. Prerequisites: MP02 array shapes and MP09 MSE. This lesson runs a hand-checkable matrix first, then interprets a recorded real training-crop experiment. It does not assert learner mastery.

## Vectors, matrices and transformations

A column vector x in R^n contains n real numbers. A matrix A of shape m×n maps it to y=Ax of shape m. Each output y_i=sum_j A_ij x_j is a dot product between one row and the input. For A=[[2,0],[0,1]] and x=[1,2], y=[2,2]: the first coordinate stretches by two while the second is unchanged. Dot product x·y=sum_i x_i y_i measures alignment; Euclidean norm ||x||₂=sqrt(x·x) measures length. Orthogonal transformations R satisfy RᵀR=I and preserve this norm. A 90-degree rotation [[0,-1],[1,0]] maps [1,2] to [-2,1].

For an H×W image matrix X, its Frobenius norm is sqrt(sum_ij X_ij²). MSE(X,Y)=||X-Y||_F²/(HW). HWC and CHW images are rank-three arrays, not directly the two-dimensional matrices used below: select a band explicitly. Numerical units are fixed stored intensity divided by 10000; they are not calibrated reflectance.

## Eigenvectors and singular value decomposition

For a square matrix A, an eigenvector v≠0 satisfies Av=lambda v: its direction does not change, only its scale. A diagonal matrix has coordinate-axis eigenvectors. A general rectangular image needs singular value decomposition (SVD), X=U diag(s) Vᵀ. In economy form, with q=min(H,W), U is H×q, s has q nonnegative descending entries, and Vᵀ is q×W. Columns of V are eigenvectors of XᵀX with eigenvalues s²; U describes the corresponding output directions. SVD exists for rectangular real matrices, even when the matrix is singular. Repeated singular values can make individual singular vectors nonunique; the reconstruction is what matters.

Keeping the first k components gives X_k=sum_{j=1}^k s_j u_j v_jᵀ. It is an approximation of a known matrix, not an inference of missing high-resolution data. Its squared Frobenius error equals sum_{j>k}s_j². Among matrices of rank at most k, truncated SVD minimizes this error. No centering is performed: the bright mean intensity may dominate the first component.

Worked example: X=diag(3,2,1). Rank one retains diag(3,0,0), loses squared energy 2²+1²=5, and has MSE=5/9. Rank two loses energy 1; rank three reconstructs exactly. Rank zero is the all-zero approximation. The function handles each explicitly.

## Storage is not always reduced

A dense float64 image stores 8HW raw bytes. The rank-k factors store 8k(H+W+1) bytes: Hk for U, k singular values, and kW for Vᵀ. This excludes metadata, compression and executable code. Compression occurs only when k<HW/(H+W+1). For 64×64, dense storage is 32768 bytes; k=16 uses 16512 bytes, while k=32 uses 33024 bytes and is already larger. A low rank can also erase roads or texture, so byte reduction alone is not a scientific success criterion.

## Reproduce and inspect the real experiment

Run `python scripts/linear_experiment.py` after the pinned NAIP training imagery is available in data/raw. It selects the red band of the same 64×64 training crop used in earlier lessons. The source checksum is checked before use; no test data is evaluated. Eight ranks are measured, including zero and full rank. Each error is checked against discarded singular-value energy, and MSE must not increase with rank. All displayed reconstructions share the reference range; metric inputs are not clipped.

See [reconstruction, error and storage figure](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp12-svd.png). The numerical record below includes acquisition date, license, checksum, units and all singular values. A full-rank reconstruction reaches numerical precision, not an exact compressed representation. Interpret whether the error reduction between k=4 and k=16 justifies the added storage; inspect spatial detail rather than relying only on MSE.

## Practice before reading the solutions

1. A has shape 3×2 and B has shape 2×5. What is AB's shape, and does BA exist?
2. How much energy is lost at rank one for singular values [5,4,0]?
3. Is rank-64 factor storage smaller than dense storage for a 64×64 float64 image?
4. Why is an SVD reconstruction not evidence that a model recovers real satellite detail?

## Worked solutions

1. AB is 3×5. BA would require inner dimensions 5 and 3 to match, so it is undefined.
2. Squared Frobenius error is 4²+0²=16; Frobenius error is 4. MSE additionally divides by the number of entries.
3. No: 8×64×129=66048 bytes versus 32768 dense bytes, before metadata.
4. The complete HR matrix was supplied to SVD. No unseen LR-to-HR mapping was learned or tested. This is a linear-algebra approximation exercise.

Reflect: describe an image where a low-rank representation would perform poorly, and identify what spatial structure would be lost.

## References and limits

[NumPy SVD documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) defines economy/full shapes and return ordering. Results here use NumPy's decomposition, not a from-scratch eigensolver. One training crop cannot support a dataset-wide compression claim. Red-band grayscale is not luminance and does not replace multispectral evaluation.

## Recorded validation

# MP12 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 2 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (5 code cells). Source SHA-256: `448b224380c3b269b52ad676828983bf72200edf315ca60705da8235265d119e`.

Commands from this directory: `python -m pytest -q test_mp12_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP13 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP13 — Gradient Descent from Scratch

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP13_gradient_descent_from_scratch/gradient_descent_from_scratch.ipynb) passed a real isolated kernel. [Measured evidence](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp13-mp15-training.json) includes checksums and actual outcomes. Reproduce with `python scripts/training_experiments.py`. Worked solutions are provided; learner mastery remains unassessed.

## Learning objectives

- Derivatives, gradients, learning rates and optimization.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP12 — Linear Algebra Visualizer](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP12_linear_algebra_visualizer/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Differentiate mean squared linear prediction error; use θ_next = θ − η∇L.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented and executed; evidence linked above.** Fit a scalar regression problem with several learning rates; deliver loss curves and a finite-difference gradient check.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; supplied in the linked notebook and experiment script.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Mathematics and PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP13_gradient_descent_from_scratch/gradient_descent_from_scratch.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP13_gradient_descent_from_scratch/mp13_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP13_gradient_descent_from_scratch/test_mp13_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP13_gradient_descent_from_scratch/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP13_gradient_descent_from_scratch/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP13 — Gradient Descent from Scratch

Expanded executable lesson. Prerequisites: preceding mathematics lessons. Derive the calculation, check a hand example, reproduce the controlled experiment and interpret its limits. Learner mastery remains unassessed.

## Setup
Install the repository development dependencies. `python scripts/training_experiments.py` regenerates the experiment using existing checksum-pinned NAIP scenes; the notebook reads its committed record without downloading data. Arrays use sample-by-feature shape and float64.

## Mathematics and worked example
A derivative describes the local rate of change of a scalar function. For f(w)=w², f'(w)=2w. A gradient stacks partial derivatives for several parameters. To reduce a differentiable loss, gradient descent moves opposite its gradient: theta_next=theta-eta*gradient, where eta>0 is the learning rate. A step can overshoot; a decreasing direction is only a local statement.

For N scalar samples, prediction p_i=w*x_i+b and loss L=(1/N)sum_i(p_i-y_i)². The chain rule gives dL/dw=(2/N)sum_i(p_i-y_i)x_i and dL/db=(2/N)sum_i(p_i-y_i). Our general code uses X of shape N×D, W of shape D×O and b of shape O, returning predictions N×O. Mean squared loss averages all N*O entries, so its gradient normalization uses that count.

Worked example: x=[1,2], y=[3,5], w=b=0. Residuals are [-3,-5], L=17, dL/dw=-13, dL/db=-8. A simultaneous eta=.1 update gives w=1.3,b=.8, predictions [2.1,3.4] and L=1.685. Updating b before computing the w derivative would implement a different algorithm.

Central finite differences approximate a derivative by (L(theta+epsilon)-L(theta-epsilon))/(2epsilon). Too-large epsilon introduces truncation error; too-small epsilon magnifies roundoff. Compare several tolerances in float64 rather than demanding bitwise equality.

## Executed experiment
The measured experiment uses x in [-1,1], target 2x+.5, zero initialization and 60 full-batch steps. Only the learning rate changes: .001, .1, 1.1. The small rate progresses slowly; .1 converges; 1.1 destabilizes the intercept. For this centered problem the bias error multiplier is 1-2eta, whose magnitude exceeds one at eta>1. This bound is specific to this loss and scaling, not a universal neural-network learning-rate rule.

## Practice
1. What gradient does f(w)=(w-3)² have at w=0? 2. What happens with eta=0? 3. Why must input scaling be fixed before comparing rates?

## Worked solutions
1. -6, so a positive learning rate moves w toward three. 2. Parameters never change. 3. Feature scale changes curvature and therefore the stable step sizes; a learning-rate comparison with different scaling is confounded.

## Evidence and interpretation
See [training curves](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp13-mp15-training.png) and [complete record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp13-mp15-training.json). A decreasing training loss is an optimization observation, not proof of geographic generalization. The real red-to-NIR exercise has distinct scene splits and a training-mean baseline, but fixed numerical scaling is not physical calibration. Reflect on a case where the constant baseline could outperform the network.

## References
[PyTorch autograd tutorial](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) and [optimization tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html). Models here are compact teaching examples, not paper reproductions or final SR systems.

## Recorded validation

# MP13 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 3 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (4 code cells). Source SHA-256: `68ee2c6f95a19768b83cc37a05adb07feac0109249e247a60d9e666a369fb277`.

Commands from this directory: `python -m pytest -q test_mp13_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP14 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP14 — Neural Network from Scratch

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP14_neural_network_from_scratch/neural_network_from_scratch.ipynb) passed a real isolated kernel. [Measured evidence](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp13-mp15-training.json) includes checksums and actual outcomes. Reproduce with `python scripts/training_experiments.py`. Worked solutions are provided; learner mastery remains unassessed.

## Learning objectives

- Forward propagation, activation functions, loss and backpropagation.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP13 — Gradient Descent from Scratch](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP13_gradient_descent_from_scratch/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Use a = activation(Wx+b), squared-error loss, and the chain rule for backpropagation.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented and executed; evidence linked above.** Plan a small NumPy network on a tiny regression dataset; deliver a forward/backward derivation and finite-difference gradient checks.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; supplied in the linked notebook and experiment script.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Mathematics and PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP14_neural_network_from_scratch/neural_network_from_scratch.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP14_neural_network_from_scratch/mp14_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP14_neural_network_from_scratch/test_mp14_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP14_neural_network_from_scratch/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP14_neural_network_from_scratch/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP14 — Neural Network from Scratch

Expanded executable lesson. Prerequisites: preceding mathematics lessons. Derive the calculation, check a hand example, reproduce the controlled experiment and interpret its limits. Learner mastery remains unassessed.

## Setup
Install the repository development dependencies. `python scripts/training_experiments.py` regenerates the experiment using existing checksum-pinned NAIP scenes; the notebook reads its committed record without downloading data. Arrays use sample-by-feature shape and float64.

## Mathematics and worked example
A one-hidden-layer network maps X (N×D) to Z=XW1+b1 (N×H), then A=tanh(Z) (N×H), then P=AW2+b2 (N×O). W1 is D×H, b1 is H, W2 is H×O and b2 is O. Biases broadcast across samples, not features. Tanh is smooth, with derivative 1-A²; it saturates near ±1 and can make gradients small. Without a nonlinear activation, stacked affine layers collapse to one affine layer.

With L=mean((P-Y)²), set delta=2(P-Y)/(N*O). Then dW2=Aᵀ delta, db2=sum_samples(delta), hidden_delta=(delta W2ᵀ) elementwise (1-A²), dW1=Xᵀ hidden_delta, and db1=sum_samples(hidden_delta). This is the chain rule in matrix form. Every gradient has the same shape as its parameter.

Worked one-sample case: x=1,y=1,w1=0,b1=0,w2=2,b2=0. Hidden activation and prediction are zero, loss=1, output delta=-2. dw2=0, db2=-2, hidden delta=-4, dw1=-4 and db1=-4. Although dw2 is zero on this sample, gradients can still flow into the first layer because w2 is nonzero. Initializing all layers to zero can prevent useful learning and preserve symmetry.

The implementation computes each derivative explicitly with NumPy. Every parameter is checked against central finite differences, and parameters are restored after each perturbation. Checks use tiny float64 fixtures so a wrong transpose, missing mean divisor or wrong tanh derivative is observable.

## Executed experiment
The recorded network has one input, eight tanh hidden units and one output. It fits sin(3x) with 1000 full-batch SGD steps at .05. Even-indexed samples train; odd-indexed samples validate. This is an interpolation exercise, not independent geographic evidence. A separate real-image exercise maps red numerical values to NIR on one training scene and evaluates a different declared validation scene; compare it with a constant training-mean baseline. Neither task is super-resolution.

## Practice
1. What shape is dW1 for D=3,H=8? 2. Why sum bias gradients over samples? 3. Does a finite-difference pass prove the model will generalize?

## Worked solutions
1. 3×8, matching W1. 2. The same bias influences every sample, so the chain rule adds their contributions; averaging is already included in delta. 3. No. It checks local derivatives, not data quality, useful inductive bias or held-out performance.

## Evidence and interpretation
See [training curves](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp13-mp15-training.png) and [complete record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp13-mp15-training.json). A decreasing training loss is an optimization observation, not proof of geographic generalization. The real red-to-NIR exercise has distinct scene splits and a training-mean baseline, but fixed numerical scaling is not physical calibration. Reflect on a case where the constant baseline could outperform the network.

## References
[PyTorch autograd tutorial](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) and [optimization tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html). Models here are compact teaching examples, not paper reproductions or final SR systems.

## Recorded validation

# MP14 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 3 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (4 code cells). Source SHA-256: `0f0c346c8b137d80c7a2eac3b0a9d145880e357df6a7178a71693699bd5b6345`.

Commands from this directory: `python -m pytest -q test_mp14_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP15 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP15 — First PyTorch Neural Network

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP15_first_pytorch_neural_network/first_pytorch_neural_network.ipynb) passed a real isolated kernel. [Measured evidence](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp13-mp15-training.json) includes checksums and actual outcomes. Reproduce with `python scripts/training_experiments.py`. Worked solutions are provided; learner mastery remains unassessed.

## Learning objectives

- Tensors, autograd, optimizers and model training.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP14 — Neural Network from Scratch](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP14_neural_network_from_scratch/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Track tensor shapes, autograd gradients, minibatch loss, and optimizer updates.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented and executed; evidence linked above.** Plan a tiny PyTorch network and compare its gradients to a hand-worked example; deliver shape checks, loss curves, and a documented train/eval comparison.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; supplied in the linked notebook and experiment script.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Mathematics and PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP15_first_pytorch_neural_network/first_pytorch_neural_network.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP15_first_pytorch_neural_network/mp15_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP15_first_pytorch_neural_network/test_mp15_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP15_first_pytorch_neural_network/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP15_first_pytorch_neural_network/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP15 — First PyTorch Neural Network

Expanded executable lesson. Prerequisites: preceding mathematics lessons. Derive the calculation, check a hand example, reproduce the controlled experiment and interpret its limits. Learner mastery remains unassessed.

## Setup
Install the repository development dependencies. `python scripts/training_experiments.py` regenerates the experiment using existing checksum-pinned NAIP scenes; the notebook reads its committed record without downloading data. Arrays use sample-by-feature shape and float64.

## Mathematics and worked example
A tensor stores values and, when requested, participates in an autograd graph. Calling loss.backward() accumulates derivatives into leaf parameter .grad fields. An optimizer changes parameters using those gradients; zero_grad must clear previous gradients before the next independent update. Our comparison uses float64 CPU tensors so it can be checked against the preceding NumPy derivation, with identical initialization, sample order, full-batch loss and learning rate.

For one scalar prediction p=w*x+b at x=2,y=5,w=1,b=0, L=(2-5)²=9, dL/dw=2*(2-5)*2=-12 and dL/db=-6. SGD at .1 gives w=2.2,b=.6. backward computes derivatives; optimizer.step performs the update. They are separate operations.

model.train() and model.eval() change behavior of modules such as dropout and batch normalization. They do not themselves enable or disable gradient recording. torch.no_grad() or inference_mode() controls gradient tracking during inference. A network containing only linear layers and tanh behaves identically in train/eval mode; this lesson explicitly uses dropout to demonstrate a behavioral difference. Dropout with p=.5 zeros samples in training and scales retained values by two; evaluation is identity. This scaling preserves the expectation, not every individual sample.

## Executed experiment
The experiment trains the same 1→8→1 tanh network in NumPy and PyTorch for 1000 steps. Its ledger records the maximum final parameter difference, finite-difference error, train/validation loss curves and seed. Agreement validates this small implementation under its declared protocol; it does not imply reproducibility across GPUs, kernels or versions. Dropout is tested separately and is not inserted into the equivalence experiment.

## Practice
1. Why does backward twice without zero_grad change .grad? 2. Does eval disable gradients? 3. Why compare identical weights and sample order?

## Worked solutions
1. Gradients accumulate by design. 2. No; use no_grad or inference_mode for inference. 3. Different initialization or stochastic ordering would make numerical differences ambiguous and would not isolate autograd correctness.

## Evidence and interpretation
See [training curves](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp13-mp15-training.png) and [complete record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp13-mp15-training.json). A decreasing training loss is an optimization observation, not proof of geographic generalization. The real red-to-NIR exercise has distinct scene splits and a training-mean baseline, but fixed numerical scaling is not physical calibration. Reflect on a case where the constant baseline could outperform the network.

## References
[PyTorch autograd tutorial](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) and [optimization tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html). Models here are compact teaching examples, not paper reproductions or final SR systems.

## Recorded validation

# MP15 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 3 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (4 code cells). Source SHA-256: `ae56c39be6433d56146abc12425ab55b9ee1b4fb9a9a9a86979984592f454162`.

Commands from this directory: `python -m pytest -q test_mp15_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP16 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP16 — Reusable Training Pipeline

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded notebook covers weighted minibatch loss, Dataset/DataLoader, deterministic overfit, checkpoints and recovery. Measured evidence: [pipeline report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp16-pipeline.json). Learner mastery remains unassessed.

## Learning objectives

- Dataset, DataLoader, validation, checkpoints and reproducibility.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP15 — First PyTorch Neural Network](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP15_first_pytorch_neural_network/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Explain minibatch empirical risk and parameter updates; distinguish train and evaluation behavior.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Executed:** `python scripts/pipeline_experiment.py` trains a seeded tiny regression twice and compares uninterrupted production training with checkpoint recovery. The notebook specifies configuration, split limitations and real geographic-study evidence.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- [Expanded notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP16_reusable_training_pipeline/reusable_training_pipeline.ipynb) and the executable script `scripts/pipeline_experiment.py`.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Mathematics and PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP16_reusable_training_pipeline/reusable_training_pipeline.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP16_reusable_training_pipeline/mp16_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP16_reusable_training_pipeline/test_mp16_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP16_reusable_training_pipeline/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP16_reusable_training_pipeline/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP16 — Reusable Training Pipeline

Build a small reproducible training run, then understand the existing research pipeline's exact-resume contract. This is an expanded teaching implementation, not evidence of learner mastery.

## Inputs, outputs and assumptions
A dataset maps an index to an input/target pair. A DataLoader groups these pairs into minibatches. Here inputs and targets are float32 tensors shaped `[N, 1]`; the toy target is `y = 2x + 1`, with dimensionless units. The output is a trained linear model, a measured loss history and recovery checks. CPU, one loading process and a fixed seed make this small demonstration deterministic within its recorded environment.

Training updates parameters. Validation compares candidates without gradients. The held-out test stays unused until the protocol and model selection are frozen. Synthetic interpolation points are not independent geographic scenes.

## Mathematics and hand example
For batch B, mean squared error is L = sum((prediction_i - target_i)^2)/|B|. SGD updates each parameter theta to theta - learning_rate * gradient(L). For predictions [1, 3] and targets [2, 1], squared errors are [1, 4], so batch MSE is 2.5. If one batch contains two examples at MSE 2.5 and the final batch one example at MSE 1, epoch MSE is (2*2.5 + 1*1)/3 = 2, not the unweighted average 1.75. Always weight batch means by their example counts.

An epoch visits the dataset once; a step performs one optimizer update. With 16 examples and batch size 4, one epoch has four steps. Our research trainer instead samples with replacement for a specified number of steps; do not call those steps epochs.

## A tiny overfit with an explicit training specification
Run `python scripts/pipeline_experiment.py` from the repository root. It trains a one-input linear layer on 16 fixed points for 100 epochs using SGD, learning rate 0.1 and four examples per batch. Validation uses 15 different interpolation points and no optimizer steps. This noise-free toy should fit almost exactly; failure indicates a mechanics problem, not a research discovery.

The script executes the same seeded run twice and compares every parameter exactly. It separately uses the production SR trainer for an eight-step run versus a four-step run resumed to step eight. Checkpoint files are temporary; only measured reports are retained.

## What exact recovery must save
Weights alone reproduce predictions, but do not preserve the next training update. A resumable checkpoint stores model parameters, optimizer moments, the sampling generator, global random state, current step, best validation value and selected weights, configuration and dataset digest. `src/rssr/experiment.py` writes to a temporary file and replaces the destination after serialization. `last.pt` is resumable; `best.pt` is explicitly inference-only and is rejected for resume.

Changing the dataset digest, batch size or optimizer settings changes the experiment. The production trainer rejects those mismatches and allows only a larger target step count. This lesson tests epoch-boundary toy reproducibility and step-boundary production recovery; it does not promise arbitrary multi-worker DataLoader mid-epoch recovery or bitwise equivalence across platforms.

## Real-data connection and limitations
The existing `reports/geographic-study.json` documents six runs with independent scene splits: four training scenes and four validation scenes, three seeds, CNN and hybrid models, 1,000 steps each. Reuse those verified runs rather than silently training again. They use NAIP aerial imagery with synthetic degradation, not paired satellite ground truth. The mean validation PSNR gains were about 1.571 dB for CNN and 1.334 dB for hybrid; the hybrid did not win. The held-out scene remains unused. Source hashes, configuration and license are recorded in the study and dataset manifests.

The synthetic SR recovery fixture intentionally repeats a ramp across splits to isolate mechanics. It cannot measure generalization. Never report its validation loss as research accuracy.

## Practice — try before reading solutions
1. For 10 examples and batch size 4 with `drop_last=False`, list batch sizes and steps per epoch.
2. Explain why saving only weights cannot reproduce Adam's next step.
3. Why is `model.eval()` insufficient to disable gradients? What does `inference_mode()` add?
4. Propose a recovery acceptance test that catches changed data.
5. If the final batch has one example, why can averaging batch means bias the epoch loss?

## Worked solutions
1. Sizes 4, 4, 2; three steps. With `drop_last=True`, two steps and two omitted examples.
2. Adam carries moving averages of gradients and squared gradients, plus update counters. Those affect the next update even with identical weights.
3. Evaluation mode changes layers such as dropout; it does not turn off autograd. `torch.inference_mode()` prevents gradient tracking for evaluation computations.
4. Save a checkpoint, alter one prepared tile, then verify checksum validation rejects the run before an optimizer update. A changed split manifest must also fail its digest check.
5. Each example should have equal weight; an unweighted mean gives the one-example batch as much influence as a full batch.

## Reflection
The exact-resume checks establish a software property on a controlled CPU fixture. They do not establish scientific quality. Next, MP17 investigates convolution itself; retain this training/checkpoint machinery instead of rebuilding it.

References: [PyTorch beginner tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html), [project brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP16_reusable_training_pipeline/README.md).

## Recorded validation

# MP16 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 1 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `32e7368b4f7fe47e716f71a7a276f9aec391ed01dbfef426a60aa030caac70d8`.

Commands from this directory: `python -m pytest -q test_mp16_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP17 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP17 — Convolution Playground

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP17_convolution_playground/convolution_playground.ipynb), [report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp17-convolution.json) and [figure](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp17-convolution.png) cover impulse geometry, receptive-field support and licensed real-crop filtering. Learner mastery remains unassessed.

## Learning objectives

- Kernels, padding, stride, feature maps and receptive fields.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP16 — Reusable Training Pipeline](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_04_math_pytorch/MP16_reusable_training_pipeline/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Use output size floor((n+2p−d(k−1)−1)/s+1) and receptive-field recursion.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Executed:** `python scripts/convolution_experiment.py` compares four impulse settings with PyTorch, measures receptive-field support by autograd and filters a checksummed NAIP training crop. Reference errors are zero; measured span is 11×11 with 81 contributing pixels.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Executable script `scripts/convolution_experiment.py` and expanded notebook with hand examples, practice and worked solutions.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: CNN super-resolution](https://arxiv.org/abs/1501.00092)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP17_convolution_playground/convolution_playground.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP17_convolution_playground/mp17_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP17_convolution_playground/test_mp17_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP17_convolution_playground/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP17_convolution_playground/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP17 — Convolution Playground

Investigate kernels, padding, stride, dilation, feature maps and receptive fields. This expanded teaching experiment includes independent NumPy/PyTorch comparisons and a licensed aerial training crop. Learner mastery remains unassessed.

## Intuition and conventions
A kernel is a table of weights. Slide it over an image, multiply overlapping pixels by weights, and sum. Outputs form a feature map. Inputs here are single-channel HW float64 arrays, with zero padding and no bias. PyTorch uses NCHW; we add batch and channel axes for comparison.

Deep-learning libraries usually compute cross-correlation: weights are not reversed. Mathematical convolution reverses both kernel axes. This distinction disappears for symmetric mean kernels but matters for asymmetric derivative filters.

## Mathematics and hand example
For output indices i,j, y[i,j] = sum over a,b of K[a,b] X[i*s+a*d-p, j*s+b*d-p]. Here s is stride, d dilation and p padding. Values outside the input are zero. Dilation spaces taps; stride spaces output centers.

For input [[1,2,3],[4,5,6],[7,8,9]] and kernel [[1,0],[0,-1]], the first output is 1-5=-4. All four valid outputs are -4. Flipping the kernel gives +4.

For length n and kernel length k, output length is floor((n+2p-d(k-1)-1)/s)+1. Effective kernel span is d(k-1)+1. For n=7, k=3, p=1, s=2, d=2, the output is floor((7+2-4-1)/2)+1=3. Kernels larger than padded inputs are rejected.

## Controlled impulse experiment
Run `python scripts/convolution_experiment.py` from the repository root. An impulse is zero everywhere except one unit pixel. The asymmetric kernel contains 1 through 9. Four configurations vary stride, padding and dilation. Predict their shapes before reading results. Integer-valued computations must match PyTorch exactly; this is stronger than visually similar images. The impulse panels share a 0–9 scale.

## Receptive-field span and support
A receptive field identifies input locations influencing one output. Let r be the bounding span and j the distance between adjacent output centers in input coordinates. Start r=1,j=1. A layer updates r_new=r+(k-1)*d*j, then j_new=j*s. Use the old j in the first equation.

A 3×3 stride-2 layer gives r=3,j=2. A subsequent 3×3 dilation-2 layer gives r=3+2*2*2=11,j=2. The bounding box is 11×11, but autograd measures only 81 contributing pixels: dilation leaves gaps. Span is not support count. Padding affects alignment and boundary overlap, not this interior-span equation.

Positive linear kernels isolate geometric support. In a trained nonlinear network, zero weights or activation gates may suppress paths, so gradients need not expose every possible geometric dependency.

## Licensed real-data experiment
The script reads a checksummed 64×64 red-band crop from NAIP training ROI_00006, acquired 2021-09-17. NAIP is public domain; retain USDA/OpenSR attribution. Stored DN/10000 is numerical scaling, not calibrated reflectance. No new download or test scene is used.

Identity preserves pixels exactly. A normalized 3×3 mean smooths local contrast. A horizontal derivative responds to horizontal changes, often vertical edges. Zero padding introduces artificial boundary changes, visible in the derivative panel. Its signed color scale differs from the intensity panels. Derivative MSE relative to the input measures numerical change, not reconstruction quality. No learned model or SR improvement is claimed.

## Practice — solve before reading
1. Compute output length for n=8,k=3,p=1,s=2,d=1.
2. Describe a mean filter on a constant positive image, including boundaries.
3. Derive the span of three 3×3 stride-1, dilation-1 layers.
4. Why can stride-2 sampling miss a one-pixel feature?
5. Why is the measured support smaller than 121 pixels?

## Worked solutions
1. floor((8+2-2-1)/2)+1=4.
2. Interior stays constant because weights sum to one. Zero-padded boundaries darken because some inputs are zero.
3. Span increases 1→3→5→7, with jump one: a 7×7 box.
4. Output centers advance by two pixels; some alignments miss the feature. Prefiltering reduces aliasing but does not retain every detail.
5. Dilation leaves unsampled rows/columns even after composing these layers. The span uses extreme coordinates; support counts only contributing pixels.

## Reflection
Exact reference agreement establishes implementation mechanics, not usefulness for a task. Keep shape and geometry checks when introducing learned filters in later lessons.

References: [authoritative brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP17_convolution_playground/README.md) and [PyTorch Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).

## Recorded validation

# MP17 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 2 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (3 code cells). Source SHA-256: `0902685e7808f634c1f5c568d618c669b57ed40e6874fc6b0dbe5329bd18d3b7`.

Commands from this directory: `python -m pytest -q test_mp17_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP18 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP18 — CNN Edge Detection Laboratory

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP18_cnn_edge_detection_laboratory/cnn_edge_detection_laboratory.ipynb), [experiment record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp18-edges.json) and [figure](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp18-edges.png) compare fixed and learned Sobel operators. Learner mastery remains unassessed.

## Learning objectives

- Sobel filters and learned convolutional features.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP17 — Convolution Playground](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP17_convolution_playground/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Apply Sobel horizontal and vertical kernels; compute gradient magnitude sqrt(gx²+gy²).

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Executed:** `python scripts/edge_experiment.py` trains a small convolution to imitate known Sobel labels, validates on disjoint synthetic edge fixtures, compares kernels/maps on a licensed training crop and checks zero versus replicate padding. This is operator imitation, not independent semantic-edge accuracy.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Executable script `scripts/edge_experiment.py` and expanded notebook with definitions, hand examples, exercises and worked solutions.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: CNN super-resolution](https://arxiv.org/abs/1501.00092)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP18_cnn_edge_detection_laboratory/cnn_edge_detection_laboratory.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP18_cnn_edge_detection_laboratory/mp18_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP18_cnn_edge_detection_laboratory/test_mp18_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP18_cnn_edge_detection_laboratory/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP18_cnn_edge_detection_laboratory/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP18 — CNN Edge Detection Laboratory

Compare fixed Sobel derivatives with a learned convolution trained to imitate them. This is supervised operator imitation, not independent semantic-edge detection or evidence that learning beats Sobel. Learner mastery remains unassessed.

## Intuition, shapes and units
An edge is a local change in intensity. Gx responds to change across columns; Gy responds to change across rows. A vertical boundary therefore produces a horizontal derivative. We use single-channel NCHW float32 inputs, with two output channels for signed Gx and Gy. The fixed 3×3 kernels are divided by eight. Padding replicates boundary pixels; zero padding is compared separately.

Normalized Gx is [[-1,0,1],[-2,0,2],[-1,0,1]]/8 and Gy is its transpose. Magnitude M=sqrt(Gx²+Gy²) is nonnegative and loses sign. These are intensity-change units per pixel step, not calibrated physical gradients. We use cross-correlation, not reversed mathematical convolution.

## Hand example
In a 3×3 vertical step patch [[0,0,1],[0,0,1],[0,0,1]], Gx=(1+2+1)/8=0.5 and Gy=0, so magnitude=0.5. For a constant patch, both derivative sums vanish. At a zero-padded image boundary, a constant positive image suddenly meets artificial zeros and produces a false edge.

## Training specification and separation
Run `python scripts/edge_experiment.py`. The training fixtures are 32 seeded uniform-noise images, each 16×16. Their two-channel targets come from the declared fixed Sobel operator. Noise excites all coefficients, avoiding the ambiguity of training on only one straight edge. A randomly initialized Conv2d(1,2,3,bias=False), with 18 trainable coefficients, minimizes mean squared component error. Adam uses learning rate 0.03, full batch 32, 500 steps, seed 18, CPU and two threads. No scheduler or data augmentation is used. The final fixed-step model is inspected; validation does not choose a checkpoint.

Validation fixtures are a vertical step, horizontal step and diagonal line, distinct from the noise training inputs. The existing real NAIP training crop is an additional demonstration, not held-out research evaluation. No test scene or new download is used. Weights are regenerated deterministically by the script and are not committed.

Loss is mean((predicted_G - target_G)²) across batch, two components and all pixels. Comparing signed components prevents magnitude-only ambiguities: opposite signed derivatives have the same magnitude.

## What actually happened
The learned kernels match the known kernels to approximately 8.68e-9 maximum coefficient error. Synthetic validation component MSE is approximately 1.44e-17. This is expected because the target generator is exactly representable by the model; it is not a discovery of a better edge detector.

The real example is a checksummed 64×64 NAIP red-band training crop from ROI_00006, acquired 2021-09-17, public domain with USDA/OpenSR attribution. Stored values divided by 10000 are numerical scaling, not physical reflectance calibration. Fixed and learned magnitude panels share a display range. Their near equality shows transfer of the learned linear operator to this input, not accuracy against human boundary annotations.

Replicate padding produces zero derivatives on a constant image. Zero padding produces a maximum absolute component of 0.5 at its boundary. This artifact can dominate a visualization; always document padding. Sobel also responds to noise, texture and shadows, which may not be meaningful object boundaries.

## Practice — solve first
1. Why does a vertical boundary activate Gx rather than Gy?
2. Compute the magnitude for Gx=0.3,Gy=0.4.
3. How many trainable coefficients does this model have? What changes if bias is enabled?
4. Could this experiment show that learned edges beat Sobel? Explain.
5. Why is signed-component loss more informative than magnitude-only loss?

## Worked solutions
1. The intensity changes as column position changes, which is the x direction.
2. sqrt(0.09+0.16)=0.5.
3. Two output channels times one input channel times 3×3 equals 18. Two biases would increase it to 20.
4. No. Sobel generated every target. Independent annotated boundaries and a prespecified evaluation protocol would be needed for semantic accuracy claims.
5. Reversing both signs leaves magnitude unchanged. Component loss detects that error; magnitude-only loss cannot uniquely identify the derivative orientation.

## Reflection
Learned features depend on the objective and supervision. When labels are a known filter, training can recover that filter. Semantic edge detection is a different problem requiring independent labels. Keep this distinction when interpreting later CNN feature maps.

References: [project brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP18_cnn_edge_detection_laboratory/README.md) and [MP17 convolution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/notebooks/MP17.ipynb).

## Recorded validation

# MP18 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 1 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `f3da7bc6bdda2cb848da6aafacd5b53e7a6a05502d216ef6fb1be62e47bb3709`.

Commands from this directory: `python -m pytest -q test_mp18_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP19 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP19 — CNN Image Reconstruction

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded notebook and [executed experiment report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp19-mp22-cnn.json) provide mathematics, implementation, measured comparisons, exercises and solutions. Learner mastery remains unassessed.

## Learning objectives

- Supervised reconstruction using convolutional neural networks.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP18 — CNN Edge Detection Laboratory](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP18_cnn_edge_detection_laboratory/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Minimize mean squared reconstruction error through a compact convolutional encoder and reconstruction head.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Teaching experiment executed; see notebook for remaining scientific limits.** Plan identity reconstruction on tiny image patches and compare against a constant-output baseline; deliver learning curves, reconstruction crops, and shape checks.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Run `python scripts/cnn_lessons_experiment.py`; see the corresponding expanded notebook and reports/mp19-mp22-cnn.json.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: CNN super-resolution](https://arxiv.org/abs/1501.00092)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP19_cnn_image_reconstruction/cnn_image_reconstruction.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP19_cnn_image_reconstruction/mp19_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP19_cnn_image_reconstruction/test_mp19_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP19_cnn_image_reconstruction/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP19_cnn_image_reconstruction/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP19 — CNN Image Reconstruction

A reconstruction network maps an image back to an image of the same shape. This is a useful mechanics check before super-resolution. A constant-output baseline predicts the training-set channel mean everywhere. It is weak but exposes whether a network learned spatial structure.

## Mathematics, shapes and worked example

For N examples, C channels and H×W pixels, MSE = sum((prediction-target)²)/(NCHW). A convolutional encoder extracts features and a reconstruction head maps features back to C channels. Here two padded 3×3 convolutions surround ReLU, with width eight. For prediction [0.2,0.8] and target [0,1], MSE=(0.04+0.04)/2=0.04. Shapes stay N×3×H×W; zero padding can introduce border errors.

## Reproducible experiment

Run `python scripts/cnn_lessons_experiment.py` from the repository root. The script first checks a small synthetic input/output shape. It then uses four training and four validation NAIP scenes, split before tiling: 16 nonoverlapping 32×32 RGB patches per scene. HR stored DN are divided by 10000, not claimed as reflectance. SR inputs use Gaussian sigma 0.8 HR pixels and area x2 downsampling. Identity reconstruction uses HR as both input and target.

Training: CPU float32, two threads, seed 19, Adam learning rate 0.001, batch four, 300 updates, MSE, no augmentation or scheduler. Evaluate the final fixed step; no validation-based checkpoint search. Teaching weights are regenerated and not committed. The constant baseline uses training-channel means. SR baselines use bicubic with align_corners=False. Metrics use range one, border two, all RGB channels; equal patches per scene give equal scene weighting.

Data: existing checksummed NAIP/OpenSR crops, public domain with USDA/OpenSR attribution. No new imagery download. Final research weights/protocol/test results are untouched by these separate teaching exercises. No learner mastery claim.

## Observed result and interpretation

Identity reconstruction reached 36.144 dB versus 27.174 dB for the training-mean constant baseline on validation. It still loses detail and shows boundary errors. This demonstrates learnability, not useful super-resolution: input and target are the same image.

## Practice — solve first

1. Why compute the constant baseline from training images only?
2. What changes if every predicted pixel equals the target?
3. Why can a padded convolution make border errors on an identity task?

## Worked solutions

1. Using validation values would leak evaluation information into the baseline.
2. MSE is zero; PSNR is positive infinity for a positive declared range.
3. The boundary sees artificial zeros rather than the same neighborhood statistics as interior pixels.

## Limitations and next question

One seed and a short budget describe an engineering demonstration. Interpolation, padding, input distribution and unequal parameter counts affect results. Investigate these factors on validation before any new frozen research test. Generated answers do not assess learner mastery.

[Authoritative brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP19_cnn_image_reconstruction/README.md) · [Local/Colab setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md)

## Recorded validation

# MP19 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 2 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `f28928bd0af44d681c6f323156290ff1d1c04142456f749a5142d4714878b176`.

Commands from this directory: `python -m pytest -q test_mp19_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP20 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP20 — SRCNN Implementation

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded notebook and [executed experiment report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp19-mp22-cnn.json) provide mathematics, implementation, measured comparisons, exercises and solutions. Learner mastery remains unassessed.

## Learning objectives

- Feature extraction, nonlinear mapping and SR reconstruction.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP19 — CNN Image Reconstruction](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP19_cnn_image_reconstruction/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Map bicubic-upsampled input through feature extraction, nonlinear mapping, and reconstruction convolutions; minimize reconstruction MSE.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Teaching experiment executed; see notebook for remaining scientific limits.** Plan a compact SRCNN comparison with bicubic on fixed image pairs; deliver an architecture diagram, validation curves, and held-out metric table.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Run `python scripts/cnn_lessons_experiment.py`; see the corresponding expanded notebook and reports/mp19-mp22-cnn.json.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: CNN super-resolution](https://arxiv.org/abs/1501.00092)
- [SRCNN paper](https://arxiv.org/abs/1501.00092)
- [VDSR paper](https://arxiv.org/abs/1511.04587)
- [ESPCN paper](https://arxiv.org/abs/1609.05158)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

The independent held-out SRCNN table remains pending. This teaching model reports validation only and does not reuse the frozen research test for further selection.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP20_srcnn_implementation/srcnn_implementation.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP20_srcnn_implementation/mp20_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP20_srcnn_implementation/test_mp20_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP20_srcnn_implementation/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP20_srcnn_implementation/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP20 — SRCNN Implementation

SRCNN first upsamples the input with bicubic interpolation, then learns feature extraction, nonlinear mapping and reconstruction at the HR grid. This lesson uses a compact adaptation, not the original paper architecture or training schedule.

## Mathematics, shapes and worked example

Let y be LR and B(y) its bicubic x2 image. F1=ReLU(W1*B(y)+b1), F2=ReLU(W2*F1+b2), output=W3*F2+b3. The asterisk denotes library cross-correlation. Kernels are 9×9, 1×1 and 5×5 with widths 32 and 16. All convolutions are padded, unlike an exact original valid-convolution reproduction. Input 1×3×16×16 becomes 1×3×32×32. The 1×1 mapping changes channels but not spatial dimensions.

## Reproducible experiment

Run `python scripts/cnn_lessons_experiment.py` from the repository root. The script first checks a small synthetic input/output shape. It then uses four training and four validation NAIP scenes, split before tiling: 16 nonoverlapping 32×32 RGB patches per scene. HR stored DN are divided by 10000, not claimed as reflectance. SR inputs use Gaussian sigma 0.8 HR pixels and area x2 downsampling. Identity reconstruction uses HR as both input and target.

Training: CPU float32, two threads, seed 19, Adam learning rate 0.001, batch four, 300 updates, MSE, no augmentation or scheduler. Evaluate the final fixed step; no validation-based checkpoint search. Teaching weights are regenerated and not committed. The constant baseline uses training-channel means. SR baselines use bicubic with align_corners=False. Metrics use range one, border two, all RGB channels; equal patches per scene give equal scene weighting.

Data: existing checksummed NAIP/OpenSR crops, public domain with USDA/OpenSR attribution. No new imagery download. Final research weights/protocol/test results are untouched by these separate teaching exercises. No learner mastery claim.

## Observed result and interpretation

The 300-step compact SRCNN scored 37.737 dB on validation versus bicubic 42.964 dB. Preserve this negative result: a trainable architecture does not guarantee a better image after a tiny budget. This lesson reports validation only; its model was not added to the already frozen research test. The brief’s independent held-out SRCNN table remains unfulfilled, rather than relabeling validation as test.

## Practice — solve first

1. Why does this architecture work mainly on the HR grid?
2. Does a 1×1 convolution mix neighboring pixels?
3. Can its validation score be called a paper reproduction?

## Worked solutions

1. Bicubic upsampling happens before learned feature extraction, increasing spatial computation.
2. No; it mixes channels independently at each pixel.
3. No. Widths, padding, data, degradation and training budget differ.

## Limitations and next question

One seed and a short budget describe an engineering demonstration. Interpolation, padding, input distribution and unequal parameter counts affect results. Investigate these factors on validation before any new frozen research test. Generated answers do not assess learner mastery.

[Authoritative brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP20_srcnn_implementation/README.md) · [Local/Colab setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md)

## Recorded validation

# MP20 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 1 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `e6398589684f72233ff0ac78de987b5089cb5b43b3c0b6e248aa818733ffc203`.

Commands from this directory: `python -m pytest -q test_mp20_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP21 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP21 — Residual CNN Super-Resolution

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded notebook and [executed experiment report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp19-mp22-cnn.json) provide mathematics, implementation, measured comparisons, exercises and solutions. Learner mastery remains unassessed.

## Learning objectives

- Residual blocks, skip connections and deep feature extraction.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP20 — SRCNN Implementation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP20_srcnn_implementation/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Use residual blocks h_next = h + f(h) and image residual reconstruction x_hat = upsample(y) + g(y).

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Teaching experiment executed; see notebook for remaining scientific limits.** Compare matched direct and residual CNNs with fixed data and training budget; deliver convergence curves, quality metrics, and error crops.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Run `python scripts/cnn_lessons_experiment.py`; see the corresponding expanded notebook and reports/mp19-mp22-cnn.json.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: CNN super-resolution](https://arxiv.org/abs/1501.00092)
- [SRCNN paper](https://arxiv.org/abs/1501.00092)
- [VDSR paper](https://arxiv.org/abs/1511.04587)
- [ESPCN paper](https://arxiv.org/abs/1609.05158)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP21_residual_cnn_super_resolution/residual_cnn_super_resolution.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP21_residual_cnn_super_resolution/mp21_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP21_residual_cnn_super_resolution/test_mp21_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP21_residual_cnn_super_resolution/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP21_residual_cnn_super_resolution/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP21 — Residual CNN Super-Resolution

A residual model starts from a useful baseline and predicts a correction. A direct model predicts the entire HR image. The matched teaching comparison uses the same head, PixelShuffle tail, initialization seed, batches and update count; the skip addition is the difference.

## Mathematics, shapes and worked example

Direct: x_hat=g(y). Residual: x_hat=B(y)+g(y), with bicubic B. If target is 0.7 and B(y)=0.6, the desired residual is 0.1. A feature residual h_next=h+f(h) is distinct from this image-level skip. Its derivative includes an identity path I+df/dh. Here we isolate the image-level skip, not claim a deep residual-block experiment.

## Reproducible experiment

Run `python scripts/cnn_lessons_experiment.py` from the repository root. The script first checks a small synthetic input/output shape. It then uses four training and four validation NAIP scenes, split before tiling: 16 nonoverlapping 32×32 RGB patches per scene. HR stored DN are divided by 10000, not claimed as reflectance. SR inputs use Gaussian sigma 0.8 HR pixels and area x2 downsampling. Identity reconstruction uses HR as both input and target.

Training: CPU float32, two threads, seed 19, Adam learning rate 0.001, batch four, 300 updates, MSE, no augmentation or scheduler. Evaluate the final fixed step; no validation-based checkpoint search. Teaching weights are regenerated and not committed. The constant baseline uses training-channel means. SR baselines use bicubic with align_corners=False. Metrics use range one, border two, all RGB channels; equal patches per scene give equal scene weighting.

Data: existing checksummed NAIP/OpenSR crops, public domain with USDA/OpenSR attribution. No new imagery download. Final research weights/protocol/test results are untouched by these separate teaching exercises. No learner mastery claim.

## Observed result and interpretation

Matched residual learning scored 41.632 dB versus direct 34.222 dB after 300 steps. Both were below bicubic 42.964 dB. The skip helped this optimization setting but did not establish SR quality superiority. One seed does not measure training variability; the separate research study has three seeds.

## Practice — solve first

1. What happens when the residual head outputs zero?
2. Does adding a skip add trainable parameters?
3. Why keep the direct model’s negative result?

## Worked solutions

1. The output is exactly the chosen interpolation baseline.
2. The addition itself adds none; surrounding layers may differ in other designs.
3. It documents the measured comparison and avoids selecting only favorable outcomes.

## Limitations and next question

One seed and a short budget describe an engineering demonstration. Interpolation, padding, input distribution and unequal parameter counts affect results. Investigate these factors on validation before any new frozen research test. Generated answers do not assess learner mastery.

[Authoritative brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP21_residual_cnn_super_resolution/README.md) · [Local/Colab setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md)

## Recorded validation

# MP21 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 2 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `c230411243507315b457d9c8c5f68851fb080721b822d3e71c807c9cd9ad294e`.

Commands from this directory: `python -m pytest -q test_mp21_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP22 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP22 — Upsampling Architecture Comparison

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded notebook and [executed experiment report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp19-mp22-cnn.json) provide mathematics, implementation, measured comparisons, exercises and solutions. Learner mastery remains unassessed.

## Learning objectives

- Interpolation, transposed convolution and PixelShuffle.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP21 — Residual CNN Super-Resolution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP21_residual_cnn_super_resolution/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Compare interpolation weights, transposed-convolution output dimensions, and pixel shuffle H × W × (C r²) → rH × rW × C.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Teaching experiment executed; see notebook for remaining scientific limits.** Compare interpolation-plus-convolution, transposed convolution, and PixelShuffle at fixed scale; deliver shape checks, checkerboard artifact crops, timing, and parameter counts.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Run `python scripts/cnn_lessons_experiment.py`; see the corresponding expanded notebook and reports/mp19-mp22-cnn.json.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: CNN super-resolution](https://arxiv.org/abs/1501.00092)
- [SRCNN paper](https://arxiv.org/abs/1501.00092)
- [VDSR paper](https://arxiv.org/abs/1511.04587)
- [ESPCN paper](https://arxiv.org/abs/1609.05158)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP22_upsampling_architecture_comparison/upsampling_architecture_comparison.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP22_upsampling_architecture_comparison/mp22_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP22_upsampling_architecture_comparison/test_mp22_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP22_upsampling_architecture_comparison/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP22_upsampling_architecture_comparison/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP22 — Upsampling Architecture Comparison

Upsampling can interpolate features, spread features with a transposed convolution, or rearrange channels with PixelShuffle. All three create a larger spatial tensor, but they differ in parameters, computation and phase behavior.

## Mathematics, shapes and worked example

PixelShuffle rearranges N×(C r²)×H×W into N×C×rH×rW. For C=3,r=2, twelve channels become three. A transposed-convolution output length is (n−1)s−2p+d(k−1)+output_padding+1. Here n=8,s=2,p=1,d=1,k=3,output_padding=1 gives 16. This is not a matrix inverse of convolution. Kernel/stride overlap can differ across output phases, creating checkerboard patterns.

## Reproducible experiment

Run `python scripts/cnn_lessons_experiment.py` from the repository root. The script first checks a small synthetic input/output shape. It then uses four training and four validation NAIP scenes, split before tiling: 16 nonoverlapping 32×32 RGB patches per scene. HR stored DN are divided by 10000, not claimed as reflectance. SR inputs use Gaussian sigma 0.8 HR pixels and area x2 downsampling. Identity reconstruction uses HR as both input and target.

Training: CPU float32, two threads, seed 19, Adam learning rate 0.001, batch four, 300 updates, MSE, no augmentation or scheduler. Evaluate the final fixed step; no validation-based checkpoint search. Teaching weights are regenerated and not committed. The constant baseline uses training-channel means. SR baselines use bicubic with align_corners=False. Metrics use range one, border two, all RGB channels; equal patches per scene give equal scene weighting.

Data: existing checksummed NAIP/OpenSR crops, public domain with USDA/OpenSR attribution. No new imagery download. Final research weights/protocol/test results are untouched by these separate teaching exercises. No learner mastery claim.

## Observed result and interpretation

Validation PSNR: PixelShuffle 41.632, interpolation-plus-convolution 41.190, transposed convolution 39.300 dB; bicubic 42.964 dB. Constant-input responses expose phase patterns. These panels use separate color ranges and are qualitative. Equal steps are not equal parameter count or compute; measured counts and warmed CPU times are printed below.

## Practice — solve first

1. How many pre-shuffle channels are needed for RGB x4?
2. Why is transposed convolution not an inverse?
3. Can PixelShuffle also exhibit checkerboards?

## Worked solutions

1. 3×4²=48 channels.
2. A learned transpose-shaped linear operator does not undo information lost by arbitrary downsampling.
3. Yes. Independently learned subpixel phases can disagree; rearrangement alone does not guarantee smoothness.

## Limitations and next question

One seed and a short budget describe an engineering demonstration. Interpolation, padding, input distribution and unequal parameter counts affect results. Investigate these factors on validation before any new frozen research test. Generated answers do not assess learner mastery.

[Authoritative brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP22_upsampling_architecture_comparison/README.md) · [Local/Colab setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md)

## Recorded validation

# MP22 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 2 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `9f7025dc0ab38914fbdd1ee70bacbf47d579d4fa6de72ba42c4c22288cf17ce3`.

Commands from this directory: `python -m pytest -q test_mp22_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP23 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP23 — Channel Attention from Scratch

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded notebook and [experiment report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp23-mp25-attention.json) provide implementations, controlled comparisons, gate plots, exercises and solutions. Learner mastery remains unassessed.

## Learning objectives

- Global pooling, channel weights and feature recalibration.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP22 — Upsampling Architecture Comparison](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_05_cnn_sr/MP22_upsampling_architecture_comparison/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Use z_c = mean over spatial positions of F_c and channel gates sigmoid(W2 ReLU(W1 z)).

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented teaching experiment; measured evidence in the linked report.** Compare uniform, learned, and disabled channel gates; deliver gate plots and parameter-overhead measurements.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Run `python scripts/attention_lessons_experiment.py` and the corresponding notebook.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Attention](https://arxiv.org/abs/1807.02758)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP23_channel_attention_from_scratch/channel_attention_from_scratch.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP23_channel_attention_from_scratch/mp23_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP23_channel_attention_from_scratch/test_mp23_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP23_channel_attention_from_scratch/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP23_channel_attention_from_scratch/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP23 — Channel Attention from Scratch

Channel attention summarizes each feature channel and learns a multiplier. Feature channels are learned maps, not necessarily physical sensor bands.

## Mathematics and worked example

For F with shape N×C×H×W, z[n,c]=mean(F[n,c,:,:]). Two 1×1 convolutions implement sigmoid(W2 ReLU(W1 z)). The gate has shape N×C×1×1 and broadcasts spatially. If a channel is [2,4], its mean is 3; a gate of 0.5 returns [1,2]. Width 8 with bottleneck 2 has (8×2+2)+(2×8+8)=42 channel-gate parameters. Disabled means multiplier 1; uniform means fixed 0.5, not a learned distribution.

## Executed experiment and reproducibility

Run `python scripts/attention_lessons_experiment.py` from the repository root. This is same-grid RGB reconstruction, not super-resolution. A tiny shape fixture precedes training on four NAIP scenes, with four separate validation scenes, 16 fixed 32×32 patches each. USDA/OpenSR NAIP imagery is public domain; values are stored DN/10000, not claimed calibrated reflectance. No new imagery download or final-test access is required.

CPU float32, two threads, seed 23, Adam 0.001, batch four, 300 updates, MSE. Models share identical head/tail initialization and minibatch sequence. Evaluate the fixed last step, with no checkpoint selection. Validation PSNR uses range one, border two, equal-size patches and equal scene weighting. Inactive gate modules remain allocated to preserve initialization; the report separately counts active and allocated parameters. Inference timing uses five warmups and twenty observations on a 32×32 input; it is not whole-app latency.

The first figure compares separately trained reconstructions under one reference contrast scale. The second applies alternative gate modes to identical features from the trained combined model, plus synthetic step-edge/checkerboard fixtures. All mask heatmaps share [0,1]; these are not semantic labels. Training curves show sampled minibatch loss, not a full-dataset estimate. Read the actual measured table below; preserve unfavorable outcomes.

## Practice

1. Does pooling retain the location of the largest feature?
2. Why compare a fixed 0.5 gate?
3. Are gates probabilities over channels?

## Worked solutions

1. No. Spatial averaging discards location.
2. It separates simple attenuation from learned input-dependent weights.
3. No. Independent sigmoid gates need not sum to one.

## Limitations

One seed and a short reconstruction experiment cannot establish SR superiority, causality, explainability or geographic generalization. Equal update count is not equal computation. Learner mastery is unassessed. Frozen research weights and test results are untouched.

[Authoritative brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP23_channel_attention_from_scratch/README.md) · [Local/Colab setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md)

## Recorded validation

# MP23 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 2 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `400e5b247b4c6106e7515adfd92b6d68cbbaf5c12b97fdcfedbc25633d571e12`.

Commands from this directory: `python -m pytest -q test_mp23_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP24 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP24 — Spatial Attention from Scratch

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded notebook and [experiment report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp23-mp25-attention.json) provide implementations, controlled comparisons, gate plots, exercises and solutions. Learner mastery remains unassessed.

## Learning objectives

- Spatial attention maps and feature weighting.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP23 — Channel Attention from Scratch](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP23_channel_attention_from_scratch/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Apply a spatial sigmoid mask A and elementwise gated features F_out = A ⊙ F.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented teaching experiment; measured evidence in the linked report.** Compare channel-pooled spatial masks on edges and textures; deliver heatmaps and paired reconstruction comparisons.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Run `python scripts/attention_lessons_experiment.py` and the corresponding notebook.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Attention](https://arxiv.org/abs/1807.02758)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP24_spatial_attention_from_scratch/spatial_attention_from_scratch.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP24_spatial_attention_from_scratch/mp24_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP24_spatial_attention_from_scratch/test_mp24_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP24_spatial_attention_from_scratch/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP24_spatial_attention_from_scratch/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP24 — Spatial Attention from Scratch

Spatial attention pools across feature channels, then predicts a multiplier at each location. A bright mask does not establish a meaningful object or correct explanation.

## Mathematics and worked example

Mean and maximum across C produce two N×1×H×W maps. Concatenate them, apply a padded 7×7 convolution, and sigmoid to get N×1×H×W. This has 2×7×7+1=99 parameters. For two channels with values 2 and 4 at one pixel, pooled mean=3 and maximum=4. If the spatial gate is 0.25, that pixel becomes [0.5,1]. Zero padding affects border masks.

## Executed experiment and reproducibility

Run `python scripts/attention_lessons_experiment.py` from the repository root. This is same-grid RGB reconstruction, not super-resolution. A tiny shape fixture precedes training on four NAIP scenes, with four separate validation scenes, 16 fixed 32×32 patches each. USDA/OpenSR NAIP imagery is public domain; values are stored DN/10000, not claimed calibrated reflectance. No new imagery download or final-test access is required.

CPU float32, two threads, seed 23, Adam 0.001, batch four, 300 updates, MSE. Models share identical head/tail initialization and minibatch sequence. Evaluate the fixed last step, with no checkpoint selection. Validation PSNR uses range one, border two, equal-size patches and equal scene weighting. Inactive gate modules remain allocated to preserve initialization; the report separately counts active and allocated parameters. Inference timing uses five warmups and twenty observations on a 32×32 input; it is not whole-app latency.

The first figure compares separately trained reconstructions under one reference contrast scale. The second applies alternative gate modes to identical features from the trained combined model, plus synthetic step-edge/checkerboard fixtures. All mask heatmaps share [0,1]; these are not semantic labels. Training curves show sampled minibatch loss, not a full-dataset estimate. Read the actual measured table below; preserve unfavorable outcomes.

## Practice

1. Why use two pooled maps?
2. Is an edge heatmap segmentation accuracy?
3. What is broadcast over channels?

## Worked solutions

1. Mean captures average response; maximum captures a strong response. Their usefulness still needs measurement.
2. No. There are no semantic labels or segmentation evaluation here.
3. The same scalar at a spatial location multiplies every feature channel.

## Limitations

One seed and a short reconstruction experiment cannot establish SR superiority, causality, explainability or geographic generalization. Equal update count is not equal computation. Learner mastery is unassessed. Frozen research weights and test results are untouched.

[Authoritative brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP24_spatial_attention_from_scratch/README.md) · [Local/Colab setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md)

## Recorded validation

# MP24 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 2 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `257004b27a9dc03b40894c19bf580657502d3757654058d54f493b42af7d1c12`.

Commands from this directory: `python -m pytest -q test_mp24_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP25 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP25 — Combined Attention Module

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded notebook and [experiment report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp23-mp25-attention.json) provide implementations, controlled comparisons, gate plots, exercises and solutions. Learner mastery remains unassessed.

## Learning objectives

- Channel attention, spatial attention and CBAM-style integration.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP24 — Spatial Attention from Scratch](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP24_spatial_attention_from_scratch/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Compose channel and spatial gates as F_out = A_spatial ⊙ (A_channel ⊙ F), with dimensions broadcast explicitly.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented teaching experiment; measured evidence in the linked report.** Compare channel-only, spatial-only, and sequential CBAM-style gating on the same features; deliver gate visualizations and parameter/shape checks.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Run `python scripts/attention_lessons_experiment.py` and the corresponding notebook.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Attention](https://arxiv.org/abs/1807.02758)
- [CBAM paper](https://arxiv.org/abs/1807.06521)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP25_combined_attention_module/combined_attention_module.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP25_combined_attention_module/mp25_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP25_combined_attention_module/test_mp25_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP25_combined_attention_module/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP25_combined_attention_module/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP25 — Combined Attention Module

Sequential combined attention first weights channels, then computes a spatial gate from the already gated features. This implementation is CBAM-style, not a faithful CBAM reproduction: its channel branch uses average pooling only.

## Mathematics and worked example

F1=A_channel(F)⊙F; output=A_spatial(F1)⊙F1. The two gates broadcast over different axes. With scalar gates 0.5 and 0.25, a value 8 becomes 1. However swapping the modules generally changes the learned input-dependent gates; multiplication of fixed scalars commuting does not prove module order is irrelevant. The width-8 gate overhead is 42+99=141 parameters.

## Executed experiment and reproducibility

Run `python scripts/attention_lessons_experiment.py` from the repository root. This is same-grid RGB reconstruction, not super-resolution. A tiny shape fixture precedes training on four NAIP scenes, with four separate validation scenes, 16 fixed 32×32 patches each. USDA/OpenSR NAIP imagery is public domain; values are stored DN/10000, not claimed calibrated reflectance. No new imagery download or final-test access is required.

CPU float32, two threads, seed 23, Adam 0.001, batch four, 300 updates, MSE. Models share identical head/tail initialization and minibatch sequence. Evaluate the fixed last step, with no checkpoint selection. Validation PSNR uses range one, border two, equal-size patches and equal scene weighting. Inactive gate modules remain allocated to preserve initialization; the report separately counts active and allocated parameters. Inference timing uses five warmups and twenty observations on a 32×32 input; it is not whole-app latency.

The first figure compares separately trained reconstructions under one reference contrast scale. The second applies alternative gate modes to identical features from the trained combined model, plus synthetic step-edge/checkerboard fixtures. All mask heatmaps share [0,1]; these are not semantic labels. Training curves show sampled minibatch loss, not a full-dataset estimate. Read the actual measured table below; preserve unfavorable outcomes.

## Practice

1. Why inspect modes on the same learned features?
2. Do fixed gates imply the modules commute?
3. Does a lower feature norm mean better reconstruction?

## Worked solutions

1. It isolates gating effects from different upstream feature extractors; separately trained models answer a different question.
2. No. The second module recomputes a gate from the first module’s output.
3. No. Evaluate output error against a reference, and keep the metric protocol fixed.

## Limitations

One seed and a short reconstruction experiment cannot establish SR superiority, causality, explainability or geographic generalization. Equal update count is not equal computation. Learner mastery is unassessed. Frozen research weights and test results are untouched.

[Authoritative brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP25_combined_attention_module/README.md) · [Local/Colab setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md)

## Recorded validation

# MP25 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 2 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `adb2ef6c801fa72223ad76bf85678626a3cfa819435a0b4cbadea93b48d779ae`.

Commands from this directory: `python -m pytest -q test_mp25_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP26 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP26 — Attention-Based SR Network

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Expanded notebook and [measured comparison](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp26-attention-sr.json) connect the completed three-seed development study to learning curves, model shapes, compute overhead and limitations. Learner mastery remains unassessed.

## Learning objectives

- Residual attention blocks and SR performance comparisons.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP25 — Combined Attention Module](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP25_combined_attention_module/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Combine residual reconstruction with attention-gated features and a declared reconstruction loss.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Executed development comparison; see evidence and limitations.** Compare a residual CNN with its attention-equipped counterpart using matched splits and budgets; deliver learning curves, quality metrics, and compute-overhead measurements.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Run `python scripts/attention_sr_evidence.py` to export existing development records; read the expanded notebook without retraining.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Attention](https://arxiv.org/abs/1807.02758)
- [CBAM paper](https://arxiv.org/abs/1807.06521)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP26_attention_based_sr_network/attention_based_sr_network.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP26_attention_based_sr_network/mp26_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP26_attention_based_sr_network/test_mp26_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP26_attention_based_sr_network/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP26_attention_based_sr_network/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP26 — Attention-Based SR Network

An image-level residual network predicts a correction to bicubic interpolation. Attention changes the features used to predict that correction. This lesson connects the inspectable gates from MP23–25 with actual synthetic super-resolution experiments. The compact model is not RCAN or an exact CBAM reproduction.

## Mathematics, dimensions and worked example

Input y is N×4×H×W with RGB and NIR in fixed stored DN/10000 units. A 3×3 head maps four bands to sixteen feature channels. The local path has two 3×3 convolutions with a ReLU between them. Channel then spatial attention weights that local feature tensor. The tail predicts 4×2²=16 channels, then PixelShuffle rearranges them to N×4×2H×2W. Bicubic B(y) is added at the image output: x_hat=B(y)+tail(A(local(head(y)))).

The L1 training loss is mean absolute difference over batch, channels and spatial positions. If an HR value is 0.7 and bicubic predicts 0.6, the desired residual is +0.1. Zeroing the residual tail gives exactly bicubic, including its possible interpolation overshoot; do not silently clip evaluation values.

For width16, channel attention adds (16×4+4)+(4×16+16)=148 parameters and spatial attention adds 2×7×7+1=99, totaling247. Learned features need not correspond individually to sensor bands. The reconstruction skip is image-level; this small network is not a deep stack of residual attention blocks.

## Reproducible comparison

The completed development study used four training and four validation NAIP scenes, split geographically before extracting sixteen nonoverlapping 32×32 HR patches per scene. NAIP/USDA/OpenSR source and checksums are recorded in dataset manifests. Values are stored DN/10000, not verified reflectance. LR is Gaussian sigma0.8 followed by area x2 downsampling. CPU float32; seeds17,23,41; Adam0.001; batch4; 1000 updates; L1; no augmentation or scheduler. Validation L1 every100 updates selects the checkpoint. Metrics use range1, border2 and equal scene weighting; validation supports model development, not independent test claims.

Run `python scripts/attention_sr_evidence.py` to export the already recorded training curves and table, requiring the existing run records. The notebook itself reads the portable committed evidence and does not train or access held-out data. A fresh experiment uses `scripts/submission_study.py`; follow its resumable manifest/checkpoint checks and resource budget. Do not rerun training to obtain a preferred result.

Mean validation gains over bicubic were +1.5709 dB for CNN and +1.7947 dB for attention. Seed standard deviations were0.2818 and0.1873 dB; these are not geographic confidence intervals. The attention overhead is247 parameters. The sampled CPU timing happened to favor attention in this run; extra operations do not imply a reliable speedup, and a separate controlled repeated benchmark would be needed for that claim.

## Practice

1. How many tail channels would four-band x4 require?
2. Why does zero residual initialization make the model start at bicubic?
3. Does a three-seed gain on four validation scenes prove attention always helps?
4. Why can a measured model with more operations have a shorter observed latency?

## Worked solutions

1. 4×4²=64 channels before PixelShuffle.
2. The tail returns zero and the image-level skip contributes B(y).
3. No. Small-scene sampling, seed variation, the chosen degradation and model-selection protocol restrict the conclusion. Independent diverse test regions are needed.
4. Timing includes scheduling, kernel behavior and measurement noise. Compare repeated warmed measurements on identical hardware/input conditions; do not equate partial MAC counts with exact latency.

## Limits and interpretation

No new held-out test was used for this lesson. The separately frozen research result remains unchanged. The network and its average-only channel pooling differ from full published architectures. Spectral values are not calibrated reflectance, and NAIP synthetic downsampling is not an aligned satellite-pair benchmark. Full FLOPs exclude more than convolution/linear/attention MACs; process RSS is not exact allocator peak. Learner mastery is unassessed.

[Authoritative brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP26_attention_based_sr_network/README.md) · [Local/Colab setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md)

## Recorded validation

# MP26 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 1 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `126dd075f098089c90ab9c5c01a4c29c46180e14ea4bb6edbabe534f22ccd7f5`.

Commands from this directory: `python -m pytest -q test_mp26_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP27 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP27 — Self-Attention from Scratch

**Status: Complete (implementation and teaching deliverables)** · Learner mastery unassessed.

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

## Learning objectives and prerequisites

Explain queries, keys, values, row softmax and quadratic token-pair storage. Predict a two-key weighted output, implement it and diagnose numerical failures. Prerequisite: [MP26](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_06_attention/MP26_attention_based_sr_network/README.md), matrix multiplication and probability normalization.

## Implementation and mathematics

`Q` is N×d, `K` is M×d, `V` is M×dv. Compute A=softmax(QKᵀ/√d) across each row, then AV of shape N×dv. Self-attention normally uses N=M and projections from one sequence. The NumPy implementation is single-head float64 on CPU, without learned projections, positions or training. True mask entries permit keys; all-masked rows and nonfinite/empty/mismatched matrices are rejected. Inputs are not modified.

For Q=[1,0], K=[[1,0],[0,1]], V=[[10],[20]], logits are [1/√2,0], probabilities [0.669762,0.330238], output 13.302385. Subtracting each row maximum before exponentiation preserves the probabilities and avoids exponential overflow. Overflow before softmax is rejected.

Multiplying Q and K by c multiplies logits by c². The √d divisor moderates variance under independent unit-variance component assumptions; it does not make arbitrary feature scaling irrelevant. One dense float32 head uses 4N² score bytes, excluding Q/K/V, gradients and temporary buffers. Doubling tokens quadruples that quantity.

## Files and exact commands

- [Reusable implementation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/attention_tools.py), [six tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/test_attention_tools.py), [experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/experiment.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/run_notebook.py), [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/VALIDATION.md).
- [Canonical notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/self_attention_from_scratch.ipynb), with explanations, eight executable cells, plots, exercises and separate worked solutions. The notebook stays in the existing notebooks directory to preserve its canonical link; the local runner executes it in a real kernel.

From repository root, using the environment described in [setup](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/setup.md):



The notebook can inspect committed experiment evidence without local imagery. Re-running the experiment requires the existing checksum-pinned training raster under data/raw; see [rebuild instructions](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/frozen-reproduction.md). No script downloads automatically. Executed notebooks remain under ignored data/executed_notebooks.

## Executed experiment and interpretation

Seed27, 64 tokens, d=3; fixed values and Q/K multipliers 0.1,1,10,100, with/without √d. Inputs: a seeded Gaussian fixture, then the central valid 8×8 RGB crop of the first NAIP training scene. The source SHA, crop, license, Python/NumPy versions and CPU runtime are in the [measured report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp27-self-attention.json).

The scaled Gaussian entropy falls from 4.159 nats to effectively zero. At multiplier100 the real crop entropy is 1.879 with scaling versus0.709 without. Row sums differ from one by at most5.56e-16; all outputs are finite. These results show saturation, not SR accuracy. The 4096-token analytical score matrix is64 MiB (float32, one head); no matrix of that size was allocated.

![Feature-scale comparison](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp27-feature-scale.png)
![Analytical score storage](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp27-memory.png)

## Practice, reflection and limitations

Predict constant-value output, compute four-head memory, and explain why uniform attention is not an accuracy result before reading notebook solutions. Stable normalization and explicit shape contracts are useful; saturation still occurs. Next: positional encodings and multiple heads.

NAIP aerial imagery is public domain with USDA/OpenSR attribution retained; dataset card declares MIT. Values are fixed DN/10000, not verified reflectance. No validation/test samples, model optimization, satellite claim, performance superiority or learner assessment was used. The one-head primitive is not a full Transformer reproduction.

## Completion evidence

- [x] Explain dimensions and work through a checked mathematical example (notebook sections1–2).
- [x] Implement and execute the controlled feature-scale experiment (measured report).
- [x] Verify shape, mask, stability, permutation, memory and invalid-input cases (six tests).
- [x] Produce and interpret attention-scale and memory plots (notebook, figures).
- [x] Record source license, checksums, environment and limitations (report and validation).
- [x] Link actual evidence in the progress tracker; learner mastery remains unassessed.

## Reference

Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), scaled dot-product attention. This lesson implements the primitive, not the paper's full translation model.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/self_attention_from_scratch.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/attention_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/test_attention_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP27 — Self-Attention from Scratch

Learn how queries ask which keys match, how probabilities select values, and why dense attention needs quadratic token-pair memory. This is one untrained NumPy head, not a full Transformer or an SR result.

[Brief and commands](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/README.md) · [Validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/VALIDATION.md)

Prerequisites: matrix multiplication, softmax and MP26. All examples execute on CPU. No network calls or training. The recorded real example uses NAIP public-domain aerial RGB from the training split, normalized DN rather than calibrated reflectance.

## 1. Shapes and a hand calculation

$Q\in\mathbb R^{N\times d}$ holds queries, $K\in\mathbb R^{M\times d}$ keys and $V\in\mathbb R^{M\times d_v}$ values. $A=\mathrm{softmax}_{row}(QK^T/\sqrt d)$ has shape $N\times M$; output $AV$ has shape $N\times d_v$. In self-attention N=M and all projections originate from the same token sequence. Feature values here are dimensionless numerical representations.

Take Q=[1,0], keys [1,0] and [0,1], values 10 and 20. Logits are [0.707107,0], so probabilities are approximately [0.669762,0.330238]. The weighted value is 13.302385. It is between the input values because the probabilities are nonnegative and sum to one.

## 2. Stable row softmax

Subtract the largest logit before exponentiation. This does not change probabilities because the common exponential factor cancels. It avoids overflow for large finite logits; it cannot fix a matrix product that has already overflowed. The implementation rejects that input.

## 3. Masked attention and the zero-row error

True means an allowed key. False keys receive exactly zero probability. A row with no permitted key cannot define a probability distribution and raises ValueError. A diagonal mask makes each token select only its own value.

## 4. Feature scaling: predict, then inspect

Multiplying both Q and K by c multiplies the logits by c². Larger differences can make softmax nearly one-hot. Dividing by √d moderates feature-dimension variance under an independent unit-variance assumption; it does not cancel arbitrary feature rescaling.

We hold token count, values and seed fixed and compare scaled/unscaled attention at c=0.1,1,10,100. Entropy H=−Σp log(p), in nats, is log(64) for uniform rows and zero for one-hot rows. Entropy is not accuracy.

The recorded synthetic scaled entropy falls from about 4.159 to effectively zero as the multiplier grows. The real RGB crop is initially nearly uniform; at multiplier 100 the scaled entropy is 1.879 versus unscaled 0.709. Different feature distributions produce different saturation. No positional information, learned projections, SR quality or geographic generalization is demonstrated.

## 5. Memory growth without dangerous allocations

One float32 score matrix for h heads and N tokens needs 4hN² bytes. Doubling N multiplies storage by four. Doubling both image dimensions gives four times as many pixel tokens and sixteen times the score storage. These are analytical counts, not measured RAM. Training also stores gradients, activations and work buffers; the NumPy lesson uses float64.

## 6. Order and positional information

Without positional terms, permuting Q, K and V together permutes outputs in the same way. Attention alone does not know where a pixel was located. Test this property before adding positions in the next project.

## Practice before viewing solutions

1. Change both values in the hand example to 7. Predict the output.
2. How many MiB does a 1024-token, four-head float32 score tensor need?
3. Explain why a uniform attention map is not evidence of poor SR.
4. Propose a fair test of √d scaling: what must remain fixed?

## Solutions and reflection

1. Output is 7: convex weights sum to one. 2. 16 MiB, excluding other allocations. 3. There is no trained reconstruction task or HR comparison here. 4. Hold Q/K/V samples, seeds and dimensions fixed when toggling the divisor; separately study dimensions with a declared random distribution.

Learned: shape contracts and stable normalization make attention inspectable. Failure: large features can saturate weights even with √d scaling. Next question: how do positional encodings and multiple heads change representations? These worked solutions are teaching evidence, not a claim that a learner has demonstrated mastery.

Reference: Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), scaled dot-product attention. This project implements only that primitive.

## Recorded validation

# MP27 validation — 2026-10-23

Engineering and teaching deliverables are complete against the [brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/README.md). Learner mastery is unassessed. This is an untrained attention primitive, not a trained Transformer or an SR accuracy result.

## Commands actually run

From repository root in the existing isolated Python3.12.14 environment:



- Six tests passed: independent hand calculation, stable/translated softmax, mask/output shapes, permutation equivariance/input immutability, invalid matrices/masks, analytical storage and entropy.
- The bounded CPU experiment completed in0.122489 seconds; NumPy2.5.4. Sixteen feature-scale comparisons produced finite output and row-sum error at most5.56e-16. [Measured record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp27-self-attention.json).
- All eight notebook code cells passed in a real Jupyter kernel, exit0. [Execution record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/notebook-execution.json); executed notebook under ignored data/executed_notebooks. Two resulting figures were visually inspected: readable axes, legends, units and explicitly analytical storage.
- Source checksum and all crop masks checked. Used only a central8×8 RGB training crop; no test or validation evaluation, training, dataset download or paid compute.

## Limitations and failures

Large Q/K scaling saturates attention despite the √d divisor; this is an observed limitation, not a failed test. Real-image features do not follow the unit-variance Gaussian assumption. NAIP is aerial imagery with normalized digital numbers; physical calibration is unverified. The notebook reads recorded real-input evidence; re-running the experiment requires the pinned local raster. Memory figures count only one score tensor and are not peak-RAM measurements. The existing local Jupyter TCP transport warning appeared; kernel execution still completed successfully.

## Local-folder validation — 2026-10-25

Passed 6 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (8 code cells). Source SHA-256: `61a362548dde6913953d03281cf6f6a7e8057ebfe64de45a0193f98a1e812e18`.

Commands from this directory: `python -m pytest -q test_attention_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP28 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP28 — Multi-Head Self-Attention

**Status: Complete (implementation and teaching deliverables)** · Learner mastery unassessed.

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

## Objectives and prerequisites

Understand feature projections, head splitting, concatenation and output projection; hand-check shapes and parameter counts. Prerequisite: [MP27](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP27_self_attention_from_scratch/README.md) and matrix multiplication. This is a bias-free, untrained CPU NumPy implementation, not a full Transformer.

## Mathematics and worked example

For X[N,D], project Q=XWq, K=XWk, V=XWv using D×D matrices. Split into h heads with d=D/h; D must be divisible by h. Each head computes softmax(QjKjᵀ/√d)Vj of shape N×d. Concatenating gives N×D, then multiplication by Wo[D,D] gives N×D. Attention probabilities stack into h×N×N. No batch dimension, dropout, masks or positional encoding is included.

For N=3,D=4,h=2: two3×2 outputs concatenate to3×4; maps are2×3×3. Four bias-free projections contain4D²=64 entries, independent of head count at fixed D. With X=I₂ and identity projections, two scalar heads yield output[[p,0.5],[0.5,p]], where p=e/(e+1)=0.731059. The notebook independently checks this value.

## Files and commands

[Implementation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/multihead_tools.py) · [Tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/test_multihead_tools.py) · [Experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/experiment.py) · [Runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/run_notebook.py) · [Validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/VALIDATION.md) · [Canonical notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/multi_head_self_attention.ipynb).

From repository root in the [development environment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/setup.md):



The experiment reuses the pinned local training raster; [rebuild instructions](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/frozen-reproduction.md). It never downloads automatically. The notebook reads committed measured evidence without requiring imagery. Its seven cells run in a real Jupyter kernel; outputs remain in ignored data/executed_notebooks.

## Executed experiment

Seed28, width4, sixteen tokens; identical Q/K/V/O matrices across1,2,4 heads. A Gaussian fixture precedes a valid4×4 NAIP training RGB crop embedded to width4 by a fixed3×4 random matrix. The embedding adds12 entries separately from the64 attention-projection entries. All six conditions produced finite outputs and normalized maps. Map arrays occupy2048,4096,8192 bytes for1,2,4 heads in float64, excluding other allocations. See [source, settings and measurements](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp28-multihead.json).

![Hand-checked maps](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp28-hand-heads.png)
![Fixed-width head comparison](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp28-attention-maps.png)

Different subspaces create different attention maps even without learning. This is not evidence of semantic specialization or better SR. The real normalized crop remains close to uniform; input scale affects the comparison. Parameter count stays constant while map storage increases with heads. Holding per-head width instead would confound the comparison by increasing total width and parameters.

## Exercises, reflection and limitations

Before notebook solutions, predict D=4,h=3 behavior, zero output projection, and whether concatenation introduces positions. Answers: reject incompatible dimensions; zero output; no positional information. Learned: shape-preserving heads can compare different feature subspaces. Next: residuals, normalization and feed-forward networks in MP29.

All calculations use float64. The experiment is untrained and does not use validation/test data or report SR quality. NAIP aerial imagery is public domain with USDA/OpenSR attribution; the dataset card declares MIT. Values are DN/10000, physical calibration unverified. Source SHA and crop are in the measured report. No paid service or new dataset download.

## Completion criteria and evidence

- [x] Explain objectives and solve a small hand example (notebook sections1–2).
- [x] Implement and run one/multiple-head comparison (six conditions).
- [x] Verify numerical, shape, invalid-input and permutation contracts (five tests).
- [x] Deliver maps, shape trace and parameter counts with limited interpretation.
- [x] Document provenance, environment and deviations (report/validation).
- [x] Link evidence in progress; keep learner mastery unassessed.

## Reference

Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), multi-head attention. No full paper reproduction is claimed.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/multi_head_self_attention.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/multihead_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/test_multihead_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP28 — Multi-Head Self-Attention

Build feature projections, split heads, concatenate outputs and apply the output projection. [Brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/README.md) · [Validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/VALIDATION.md).

Prerequisite: MP27's checked single-head attention. No training, network or held-out evaluation. This is a bias-free NumPy float64 CPU demonstration, not a complete Transformer.

## 1. Follow the dimensions

For X[N,D], four projection matrices Wq,Wk,Wv,Wo each have D×D entries. With h heads, require D divisible by h and let d=D/h. Split XWq, XWk, XWv along their feature columns into h arrays[N,d]. Each head creates a probability matrix[N,N] using √d scaling and produces[N,d]. Concatenating gives[N,D], then multiplying Wo keeps[N,D].

For N=3,D=4,h=2: each head has width2, maps stack as[2,3,3], final output[3,4]. Bias-free attention uses4D²=64 scalar projection entries, independent of h when D stays fixed. More heads do not automatically mean more parameters; retaining width per head instead of total width would be a different experiment.

## 2. Two scalar heads by hand

Let X be the2×2 identity and all projections identity. Each head has width1, so its divisor is1. Head1 sees token features[1,0]: attention rows[p,1−p] and[0.5,0.5], where p=e/(e+1)=0.731059. Head2 sees[0,1] and reverses the pattern. The concatenated output is[[p,0.5],[0.5,p]].

## 3. Controlled one/two/four-head experiment

Keep total width4, tokens, projection matrices and seed28 fixed; change only the head partition. Use seeded Gaussian tokens first, then16 RGB pixel tokens from a4×4 valid NAIP training crop embedded by a fixed3×4 random matrix. The embedding is untrained and adds12 entries outside the64 attention projection entries. This experiment does not test semantic specialization.

The next cell reads previously measured evidence, so viewing the notebook does not repeat acquisition or experiments.

## 4. Inspect head maps on the same color scale

Compare the Gaussian fixture's maps below. All axes use0–1 probability. Different random projections/subspaces can yield different maps without learning anything useful. The real crop can remain almost uniform because its normalized values are small. Neither pattern establishes SR quality.

## 5. Parameters versus intermediate storage

For16 tokens each float64 attention map stores16×16×8=2048 bytes. At1,2,4 heads that is2048,4096,8192 bytes even though the projection count remains64. These measured array nbytes exclude other arrays, temporaries and gradients; they are not peak memory. A larger head count alone is not a quality improvement.

## Practice before solutions

1. What fails for D=4,h=3? 2. If Wo is zero, what is the output? 3. Does concatenation add token positions? 4. Why is comparing fixed head width across head counts confounded?

## Solutions, reflection and limits

1. Equal-width heads cannot partition four features into three groups. 2. Output is zero regardless of the maps. 3. No: concatenation joins features and has no coordinate information. 4. Total width and parameter count change, so this is not a head-count-only comparison.

Learned: head splitting preserves output shape but changes the feature comparisons. Failure case: incompatible dimensions must be rejected. Next: residual connections, normalization and a feed-forward block in MP29. No mask, positional encoding, dropout or learned task is included here. Learner mastery is unassessed.

Source: NAIP public-domain aerial imagery with USDA/OpenSR attribution; dataset-card MIT; stored DN/10000, physical calibration unverified. The measured report pins the source SHA and crop. No frozen test or validation sample is used.

Reference: Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), multi-head attention. Only the attention primitive is implemented.

## Recorded validation

# MP28 validation — 2026-10-23

Implementation and teaching criteria in the [brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/README.md) are complete; learner mastery unassessed.

Executed the three README commands using Python3.12.14 / NumPy2.5.4, CPU. All five unit tests passed: independent two-scalar-head hand solution; single-head reference equivalence; zero output projection/parameter counts; permutation equivariance and immutability; invalid dimensions/projections. The experiment produced six finite conditions using one checksummed valid training crop and a synthetic fixture; no validation/test data, optimization or acquisition.

All seven notebook code cells passed in a real Jupyter kernel, exit0; [execution record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/notebook-execution.json). Two plots were captured for visual review. The pre-existing local kernel TCP transport warning appeared, with no execution error.

The [measured report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp28-multihead.json) records seed, software, source hash/license/crop and all maps. Arrays have shapes[heads,16,16]; projection counts stay64 and float64 map sizes increase2048→4096→8192 bytes. These are actual array sizes, not measured peak RAM. Different random maps do not establish task benefit. No positions, biases, dropout, masks, trained model or SR metric is included; these are explicitly outside this primitive's scope.

## Local-folder validation — 2026-10-25

Passed 5 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (7 code cells). Source SHA-256: `6b8875a4d4e800cf6ccc8263a0b8aed2c7ea392dc8ba461a4f311bb749988868`.

Commands from this directory: `python -m pytest -q test_multihead_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP29 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP29 — Transformer Encoder Block

**Status: Complete (implementation and teaching deliverables)** · Learner mastery unassessed.

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

## Objectives and prerequisite

Combine multi-head attention, layer normalization, residuals and feed-forward projections; trace shapes and explain position sensitivity. Prerequisite: [MP28](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP28_multi_head_self_attention/README.md). This pre-normalized, untrained educational variant is not an exact reproduction of the original post-normalized Transformer.

## Mathematics and hand example

Start with X[B,N,D] and optional sinusoidal PE[N,D]. Set Z=X+PE; Y=Z+MHA(LN₁(Z)); O=Y+Linear₂(GELU(Linear₁(LN₂(Y)))). Attention keeps D; the FFN expands D→2D→D. Each residual adds tensors of identical shape. LayerNorm uses per-token feature mean and population variance, epsilon1e-5, learned gain and bias. LN([1,3]) with unit gain/zero bias is[-1,1]/sqrt(1+1e-5). Constant-token variance is zero but the denominator is finite.

Positions use sin(p/10000^(2i/D)) and cos(p/10000^(2i/D)) in paired feature columns; width must be even. At p=0,D=4: [0,1,0,1]. Without positions, permuting tokens permutes output; fixed-slot positions break that property when contents move but slots stay fixed. Positional amplitude relative to input matters.

## Files and commands

[Implementation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/encoder_tools.py) · [Tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/test_encoder_tools.py) · [Experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/experiment.py) · [Runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/run_notebook.py) · [Validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/VALIDATION.md) · [Canonical notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/transformer_encoder_block.ipynb).

From repository root in the [development environment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/setup.md):



No automatic download. Experiment uses the pinned local training raster from [rebuild instructions](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/frozen-reproduction.md); the notebook reads committed measurements and runs synthetic checks without imagery. Executed output stays in ignored data/executed_notebooks.

## Executed controlled comparison

Seed29, CPU float64, one thread, width4, FFN width8, dropout0, GELU, biases and LayerNorm eps1e-5. Identical parameter state across1/2/4 heads; positions off/on; seeded Gaussian tokens then16 embedded RGB tokens from a valid4×4 training crop. All twelve conditions produced finite output. The encoder has172 scalar parameters for each head count; the separate RGB embedding adds12. No optimization or SR metric.

![Encoder diagram](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp29-encoder-diagram.png)
![Feature responses](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp29-responses.png)

Actual trace: input, normalized features, attention, residual and output all[1,16,4]; FFN hidden[1,16,8]. [Measured report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp29-encoder.json) records full outputs, RMS changes, source checksum/license/crop and software. Response differences show representation sensitivity, not which configuration restores images better. Fixed order-one positions can dominate small normalized RGB features.

## Practice, reflection and limitations

Before notebook solutions, predict residual output when both learned branches are zero, calculate the normalization example, and identify why RMS is not accuracy. Answers: identity; the equation above; no reference target exists. Tests verify the identity path, finite gradients on constant tokens, position encoding, shape rejection and permutation behavior.

Learned: normalization and residual shape contracts are separate from attention. Failure cases: invalid heads, width, rank, empty/nonfinite tokens. Next: patches and window connectivity (MP30). NAIP is public-domain aerial imagery with USDA/OpenSR attribution (dataset card MIT); DN/10000 is not verified reflectance. No held-out/validation evaluation, training, masks, dropout, learned positional ablation or learner assessment.

## Completion criteria

- [x] Explain objectives and hand-check normalization/positions (notebook).
- [x] Implement and run positions/head-count comparison (twelve conditions).
- [x] Verify numerical, shape, edge and gradient checks (five tests).
- [x] Produce diagram, shape trace and interpreted response plots.
- [x] Record provenance, environment and pre-LN deviation (report/validation).
- [x] Link actual evidence in progress; learner mastery unassessed.

## Reference

Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762). The lesson's pre-LN ordering is explicitly different from the original post-LN model.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/transformer_encoder_block.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/encoder_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/test_encoder_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP29 — Transformer Encoder Block

Combine attention, normalization, residual paths and a feed-forward network. [Brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/README.md) · [Validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/VALIDATION.md). Prerequisite: MP28.

This is an untrained pre-normalized encoder, not a full paper reproduction. CPU float64, dropout0, no downloads or optimization. The measured crop is training-only NAIP aerial RGB, fixed DN/10000 with unverified physical calibration.

## 1. Equations, dimensions and normalization

For X[B,N,D], set Z=X+PE (or Z=X without positions). Pre-normalization means Y=Z+MHA(LN₁(Z)), then O=Y+W₂ GELU(W₁ LN₂(Y)+b₁)+b₂. LN operates over D features independently for each token using population variance, epsilon10⁻⁵ and learned gain/bias. FFN changes D→2D→D. Residual addition requires matching shapes.

For features[1,3], mean=2 and population variance=1, so normalized values are[−1,1]/√(1+10⁻⁵), approximately[−0.999995,0.999995]. This differs from sample variance and from normalization across tokens.

## 2. Encoder diagram and actual shape trace

At B=1,N=16,D=4 the attention branch retains[1,16,4], the feed-forward hidden tensor is[1,16,8], and the output is[1,16,4]. There is no spatial resizing in this block. Residual paths carry the original representation even if their learned branches output zero.

## 3. What positions change

For feature pair2i,2i+1 use sin(p/10000^(2i/D)), cos(p/10000^(2i/D)). At p=0 the width4 encoding is[0,1,0,1]. The implementation requires even width. Without positions, a token permutation yields the same output permutation. Adding encodings tied to fixed slots breaks that property when token contents are rearranged but slots are not.

## 4. Measured fixed-weight response comparison

The twelve recorded conditions combine two inputs (Gaussian and a pinned4×4 NAIP training crop),1/2/4 heads and positions off/on. Width4 and every learned parameter are held identical across head counts. The RGB→width4 random embedding is also fixed. No model is trained.

Plot RMS of the block's change relative to its actual input, not distance to a target image. Magnitude is not quality. Positional encodings are order-one while the numerical RGB values are small, so their effect is scale-dependent.

## 5. Residual and gradient sanity checks

If both branches are zero, output equals input exactly. At a constant token LN has zero variance, but epsilon keeps its denominator nonzero. Backpropagation must remain finite; this does not establish healthy gradients for arbitrary trained depth.

## Practice and separate solutions

Predict the shape after each residual, calculate LN([1,3]), and explain why output RMS cannot select an SR model. Then compare with the trace and hand solution above. Residuals stay[B,N,D]; RMS has no HR-reference comparison and can reward arbitrary scaling.

Reflection: pre-normalization exposes clear residual routes, while positions alter order sensitivity. A failed shape or nonfinite input is rejected explicitly. Next: patchification and window connectivity (MP30). Limitations: no training, positional ablation on a learned task, masking, dropout or dataset generalization. Learner mastery unassessed.

Reference: Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762). This pre-LN educational variant differs from the original post-LN encoder ordering.

## Recorded validation

# MP29 validation — 2026-10-23

Implementation/teaching criteria in the [brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/README.md) are complete; learner mastery unassessed.

Executed all three README commands in Python3.12.14, PyTorch2.14.1+cpu. Five tests passed: hand-computed population-variance normalization; zero-branch residual identity and hidden shape; permutation with/without fixed positions; positional values and finite gradients; invalid contracts. The bounded experiment used CPU float64/one thread and completed twelve finite conditions with identical weights across heads. No training, acquisition or held-out evaluation.

All six notebook code cells passed in a real Jupyter kernel, exit0; [execution record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/notebook-execution.json). The existing local TCP transport warning was emitted without execution failure. Diagram and response figures come from the executed notebook. [Measured data](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp29-encoder.json) pins source license/hash/crop, settings and output arrays.

All models have172 parameters; independent RGB projection adds12. Response RMS is not accuracy. The pre-LN ordering, sinusoidal positions added before the block and absence of dropout/masks are declared; no trained positional benefit or generalization claim.

## Local-folder validation — 2026-10-25

Passed 5 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (6 code cells). Source SHA-256: `dc7cf2cedd256734ad7d1b1539ae9c96283e1757fa739e34d426be9de0497a46`.

Commands from this directory: `python -m pytest -q test_encoder_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP30 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP30 — Image Patches and Window Attention

**Status: Complete (implementation and teaching deliverables)** · Learner mastery unassessed.

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

## Objectives and prerequisites

Implement reversible patch ordering, distinguish patchification from learned embedding, compare global/local/shifted connectivity and estimate attention storage. Prerequisite: [MP29](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP29_transformer_encoder_block/README.md).

## Mathematics, ordering and hand example

For H×W×C pixels and patch sizeP, N=(H/P)(W/P) tokens each holdP²C values. Require exact divisibility; no implicit padding. Token rows follow patch-grid row-major order. Within a token, rows, columns, channels flatten in that order. A4×4 single-channel ramp gives first2×2 token[0,1,4,5], second[2,3,6,7]. Inversion preserves original dtype and values. Shuffling tokens requires tracking their original spatial indices; a learned embedding is separate and need not be reversible.

Global score storage is N² entries, local M×M token windows use NM² for complete windows. Pairwise work is O(N²d) versus O(NM²d), excluding projections/FFN. For4×4 tokens and M=2:256 global versus64 local entries,1024 versus256 float32 bytes. Analytical counts are not peak RAM.

## Files and exact commands

[Implementation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/window_tools.py) · [Tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/test_window_tools.py) · [Experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/experiment.py) · [Runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/run_notebook.py) · [Validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/VALIDATION.md) · [Canonical notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/image_patches_and_window_attention.ipynb).

From repository root in the [development environment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/setup.md):



Experiment requires the existing pinned training raster; [rebuild instructions](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/frozen-reproduction.md). No automatic download. Notebook uses committed measured evidence plus synthetic checks; executed output is ignored under data/executed_notebooks.

## Executed experiments and interpretation

Synthetic ramp first, then a valid8×8 RGB training crop; patch sizes1,2,4. All six round trips were exact. Real pixels stay uint16, without resampling or calibration. Source checksum, license, crop and software are in the [measured report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp30-windows.json).

![Connectivity](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp30-connectivity.png)
![Analytical memory](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp30-memory.png)

On a4×4 token grid:256 global pairs,64 aligned-local pairs,36 shifted/clipped pairs,144 reachable pairs after local then shifted layers. Shifted boundaries connect some prior window neighbors but do not wrap opposite image edges. The two-layer paths are not fully global. This is a clipped-boundary educational adjacency model, not Swin's cyclic-shift/masking implementation, and it does not compute learned attention probabilities.

## Practice, reflection and limits

Predict tokens for32×32 pixels/P=4, diagnose shuffled patch positions, and compare two-layer versus global connectivity before notebook solutions. Answers:64; inverse order must be tracked;144<256 in this example. Learned: exact patch reconstruction is independent from model quality. Next: measured Transformer/CNN restoration comparison (MP31).

NAIP is public-domain aerial data, USDA/OpenSR attribution retained, dataset-card MIT. No reflectance inference, training, downloads, validation/test access or SR-quality metric. Adjacency allocation is capped at4096 tokens; larger storage estimates are analytical. Learner mastery unassessed.

## Completion criteria

- [x] Explain patch dimensions and hand-compute ordering/storage.
- [x] Execute synthetic and real round trips plus connectivity comparisons.
- [x] Verify exact values/dtype, order, boundary behavior and invalid inputs (five tests).
- [x] Produce interpreted connectivity/memory figures and positional notes.
- [x] Record license, source hash, software and clipped-window deviation.
- [x] Link evidence in progress without claiming learner assessment.

## References

[Swin Transformer](https://arxiv.org/abs/2103.14030) · [SwinIR](https://arxiv.org/abs/2108.10257). Only connectivity concepts are illustrated; neither network is reproduced here.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/image_patches_and_window_attention.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/window_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/test_window_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP30 — Image Patches and Window Attention

[Brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/README.md) · [Validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/VALIDATION.md). Prerequisite: MP29. Explore token ordering, exact reconstruction, connectivity and analytical storage without training or downloading data.

## 1. Patch shapes and positional order

For H×W×C pixels and square patch sizeP, N=(H/P)(W/P) tokens each containP²C numbers. We require exact divisibility and use row-major patch order; within each patch, rows, columns then channels are flattened. No padding or compression occurs. A learned embedding would add a matrix fromP²C to feature widthD and might lose information; that is separate from patchification.

Hand example: a4×4 single-channel image with values0…15 and P=2 yields first token[0,1,4,5], second[2,3,6,7]. Shuffling token rows changes reconstructed spatial positions unless the inverse permutation is recorded.

## 2. Global, local and shifted connectivity

Consider a4×4 token grid and2×2 windows. Global attention permits every pair (256). Local windows permit64 pairs. Shift the window boundary by one cell: clipped edge windows produce36 pairs and allow communication across some old boundaries. No cyclic wrap is allowed; first and last edge tokens are not neighbors.

This explicit clipped-boundary diagram is educational, not an implementation of Swin's cyclic-shift and mask machinery. Two layers (local then shifted) provide paths between144 ordered pairs, still fewer than global256. Connectivity only says a path is possible, not that learned weights use it.

## 3. Exact real-image round trips

The recorded experiment starts with a synthetic integer ramp, then a valid8×8 RGB crop from a checksum-pinned NAIP training scene. Patch sizes1,2,4 preserve exact dtype and values. No normalization or resampling is needed; stored uint16 values remain unchanged. The lesson reads recorded evidence; the separate experiment script checks the source hash before re-running.

## 4. Complexity and storage

Global attention scores need N² entries and O(N²d) pairwise work. For complete M×M token windows, each token sees M² keys: NM² entries and O(NM²d) pairwise work. Projections and feed-forward work are excluded. Fixed M makes this term linear in N.

At4×4 tokens, M=2: global256 entries/1024 float32 bytes; local64/256 bytes. These are analytical score counts, not peak RAM. Clipped shifted windows may have fewer entries. Do not allocate large dense adjacency matrices just to estimate memory.

## Practice and solutions

Predict token count for a32×32 image with P=4; explain why reversing patch order changes reconstruction; decide whether two shifted layers equal global attention. Answers:64 tokens; token order encodes location implicitly; no, the measured two-layer adjacency is144 rather than256 pairs.

Reflection: reversible tokenization and attention connectivity solve different problems. Explicit boundary conventions avoid accidental cross-edge communication. The next lesson compares an existing compact window-based SR baseline with a CNN.

Limitations: no learned embedding, softmax weights or SR accuracy; adjacency examples use clipped boundaries, not faithful Swin masks. No frozen evaluation or new acquisition. NAIP is public-domain aerial data, USDA/OpenSR attribution retained, dataset-card MIT; no reflectance claim. Learner mastery remains unassessed.

References: [Swin Transformer](https://arxiv.org/abs/2103.14030) for shifted windows; [SwinIR](https://arxiv.org/abs/2108.10257) for restoration. This lesson illustrates connectivity rather than reproducing either network.

## Recorded validation

# MP30 validation — 2026-10-23

Implementation/teaching criteria complete; learner mastery unassessed. Executed all three [README commands](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/README.md) with Python3.12.14/NumPy2.5.4 on CPU.

Five tests passed: exact multichannel roundtrip/dtype and hand ordering; non-wrapping shifted boundary connectivity; two-layer reachability; hand memory counts; invalid dimensions, missing divisibility and adjacency budget rejection. All six measured round trips on a synthetic ramp and checksummed training crop were exact. Connected pair counts256/64/36/144 are recorded in the [experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp30-windows.json).

All six notebook code cells passed in a real Jupyter kernel, exit0, with the existing local TCP transport warning. [Execution evidence](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/notebook-execution.json). Figures show connectivity and analytical storage; no measured peak-memory claim. No network, training, held-out evaluation or new dataset. This explicitly clipped shifted-window example is not a faithful Swin masking implementation.

## Local-folder validation — 2026-10-25

Passed 5 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (6 code cells). Source SHA-256: `30ebdc8e6418133f53d62b6c032f19adcd0dd0e398e4834e7a20fa018bb75fdd`.

Commands from this directory: `python -m pytest -q test_window_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP31 · Expanded existing lesson

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP31 — Transformer-Based SR Baseline

**Status: Complete (implementation and teaching deliverables)** · Learner mastery unassessed.

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

## Objectives and prerequisites

Evaluate a compact window-based SR model against a CNN with shared data/metrics and declared compute; distinguish parameter count from latency and analyze an unfavorable result. Prerequisite: [MP30](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP30_image_patches_and_window_attention/README.md).

## Reusable implementation and mathematics

The existing [SRModel/WindowBlock](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/src/rssr/models.py) is reused. RGBNIR input[B,4,H,W]→3×3 head/width16→unshifted4×4-token window block with two heads and learned positions→3×3 tail/16 channels→PixelShuffle×2/[B,4,2H,2W]. Add a bicubic skip. Replicated padding aligns boundary windows and is removed. This compact educational model lacks faithful shifted-window stacks/residual Swin groups; it is not official SwinIR.

PSNR=10log10(range²/MSE); range1/MSE0.01 gives20dB. Equal-weight scene means of tile metrics are retained; pooling MSE then converting to PSNR would be a different aggregation. Seed SD is descriptive training variation, not geographic confidence.

## Files and commands

[Evidence/benchmark code](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/benchmark_tools.py) · [Tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/test_benchmark_tools.py) · [Runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/run_notebook.py) · [Validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/VALIDATION.md) · [Canonical notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/transformer_based_sr_baseline.ipynb).

From repository root in the [development environment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/setup.md), with the package installed:



The exporter reads existing measured reports only; no imagery/checkpoint download, retraining, timing repetition or frozen-test access. It pins both source hashes and reconciles reported deltas against per-scene PSNR. Notebook executes a separate tiny untrained synthetic shape check and reads measured evidence.

## Declared benchmark and actual result

Four train/four validation NAIP scenes,16 fixed patches/scene, RGBNIR DN/10000 (not calibrated reflectance). Gaussian sigma0.8/area x2 LR; range1,border2,no clipping/luminance conversion. Adam0.001,batch4,L1,1000 updates; minimum validation L1 checkpoint every100; seeds17/23/41. Same updates are not equal parameters/time. Existing [development runs](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/submission-study.json) and [efficiency report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/development-benchmark.json) remain unchanged.

| Model | Parameters | Mean validation gain over bicubic | Sample seed SD |
|---|---:|---:|---:|
| CNN | 7552 | 1.570925 dB | 0.281768 dB |
| Educational window Transformer | 5392 | 0.618003 dB | 0.246293 dB |

![Measured comparison](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp31-baseline-comparison.png)

[Teaching export](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp31-transformer-sr.json) includes run/checkpoint/manifest IDs, configurations, training times and original CPU efficiency rows. Timing used input[1,4,64,64], five warmups/twenty repetitions and the recorded CPU/thread environment. Partial MACs exclude several operations; sampled process RSS is not allocator peak. Parameters alone cannot predict runtime.

## Failure analysis, practice and reflection

CNN exceeds this Transformer by0.952921dB mean validation gain under this protocol. Insufficient training, capacity, windows or small data are possible explanations, not isolated causes. No Transformer superiority is supported. A future test should predeclare one factor on development data; never retune/repeat the frozen test.

Before notebook solutions, compute the mean-gain gap and explain parameter/latency and seed/geography distinctions. Learned: architecture labels are not evidence of restoration benefit. Next: stage08 hybrid fusion with the existing baselines preserved.

Official pretrained SwinIR is a separate RGB/DIV2K-pretrained comparison with different degradation/budget; it is not pooled into this four-band table. NAIP is public-domain aerial data with USDA/OpenSR attribution (dataset card MIT). No real-satellite generalization, reflectance or learner mastery claim.

## Completion criteria

- [x] Explain common metric/shape conventions and a numerical PSNR example.
- [x] Implement and execute a reproducible benchmark export from already executed experiments.
- [x] Verify exact source arithmetic, parameter counts, finite odd-size output and bicubic zero-residual control (three tests).
- [x] Deliver measured table/plot, compute context and failure analysis.
- [x] Preserve provenance, source hashes, unequal-budget and architecture limitations.
- [x] Link validation/progress without conflating implementation and learner completion.

## References

[Swin Transformer](https://arxiv.org/abs/2103.14030) · [SwinIR](https://arxiv.org/abs/2108.10257). Compact educational baseline and official model remain explicitly separate.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/transformer_based_sr_baseline.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/benchmark_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/test_benchmark_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP31 — Transformer-Based SR Baseline

[Brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/README.md) · [Validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/VALIDATION.md). Prerequisite: MP30.

Use already executed CNN/window-Transformer development runs to assess a compact baseline honestly. This notebook does not retrain, download checkpoints, time benchmarks or repeat the frozen test. A tiny untrained synthetic shape check is separate from recorded accuracy.

## 1. Architecture and tensor contract

The existing educational model maps[batch,4,H,W] RGBNIR input through a3×3 convolution to width16; an unshifted4×4-token window attention block adds local context; a3×3 tail creates16 channels and PixelShuffle×2 restores four bands at[batch,4,2H,2W]. A bicubic skip supplies the base image. Replicated boundary padding is removed after window reconstruction. Windows have two attention heads and learned position parameters.

This is inspired by window-based restoration but lacks faithful shifted-window stacks and residual Swin Transformer groups. Official SwinIR is a separately recorded RGB/DIV2K-pretrained comparison, not this compact model.

## 2. Common metric and budget protocol

Four training/four validation NAIP aerial scenes,16 fixed patches per scene, RGBNIR, DN/10000 (not calibrated reflectance). Synthetic x2 LR: Gaussian σ0.8 then area downsampling. Metrics use range1,border2 and no clipping or luminance conversion. Aggregate tiles within each scene, then equal-weight scenes.

Both model families used seeds17,23,41, Adam lr0.001, batch4, L1 loss,1000 updates and minimum validation L1 selection every100 updates. Same updates do not equalize parameter count, training time or compute. This teaching export reads validation evidence only.

PSNR=10log10(range²/MSE). For range1,MSE0.01 the answer is20dB. Averaging per-scene PSNR is different from computing PSNR from pooled MSE; retain the source convention.

## 3. Baseline comparison including an unfavorable result

Across the three seeds the CNN's mean validation gain over bicubic is1.570925dB; the compact Transformer gains0.618003dB. Their seed sample SDs are0.281768 and0.246293dB. These are training variability over three seeds, not geographic confidence intervals. The experiment does not support Transformer superiority.

## 4. Parameters are not runtime

CNN has7552 parameters; compact Transformer5392. Read the separately measured CPU table below. It used float32, input[1,4,64,64], five warmups and twenty repetitions on its recorded machine/thread count. These times were not measured in this notebook. Sampled process RSS is not allocator peak. Partial MAC counts exclude normalization, activations, pooling, softmax, interpolation and elementwise work.

## 5. Failure analysis and next experiment

The compact Transformer underperforms this CNN under the declared budget despite fewer parameters. Possible explanations include insufficient updates, architecture capacity, window design or this small dataset; the present comparison does not isolate their causes. A fair next experiment would predeclare a single change and evaluate on development data, leaving the frozen test untouched.

The official pretrained SwinIR result cannot be pooled into this four-band equal-update table: it uses RGB, natural-image DIV2K pretraining and different degradation/budget. Small synthetic validation gains do not establish real satellite cross-sensor validity. Negative findings remain visible.

## Practice and separate solutions

Compute the gap between mean validation gains; explain why7552 versus5392 cannot predict latency; distinguish seed SD from geographic uncertainty. Solutions:0.952921dB favoring the CNN in this protocol; runtime also depends on operations, memory access and implementation; three training seeds do not sample new geographic regions.

Reflection: benchmarking requires shared metrics and explicit unequal budgets, not architecture labels. Next: explore satellite imagery in stage08, then combine branches in stage09 while preserving these baselines. Source data and artifact hashes remain in the reports. No new training/held-out evaluation or learner assessment occurred.

References: [Swin Transformer](https://arxiv.org/abs/2103.14030), [SwinIR](https://arxiv.org/abs/2108.10257). The shared educational implementation remains distinct from the official model.

## Recorded validation

# MP31 validation — 2026-10-23

Implementation/teaching criteria complete; learner mastery unassessed. Executed all three [README commands](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/README.md) in the existing CPU Python3.12.14/PyTorch2.14.1 environment.

Three tests passed: exact per-scene delta and summary reconciliation for six source runs; actual model parameter counts and finite odd-size7×9→14×18 output; zero-residual Transformer equals the declared bicubic skip. The benchmark export reuses previously executed four-band validation and CPU efficiency evidence and records source hashes. No training or frozen test was run.

All six notebook code cells passed in a real Jupyter kernel, exit0, with the existing local TCP warning. [Execution record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/notebook-execution.json). The figure shows the three observed seeds and descriptive sample SD. The CNN's mean gain1.570925dB exceeds the compact Transformer's0.618003dB; no superiority claim for Transformer. Measured parameters7552/5392 match current model construction.

Limitations: unshifted educational baseline, small four-scene validation set, three seeds, unequal parameter/time budgets, synthetic aerial degradation, normalized DN rather than reflectance. CPU latency and sampled RSS are historical measurements, not newly measured values. Official RGB SwinIR pretraining differs and is not combined with this table.

## Local-folder validation — 2026-10-25

Passed 3 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (6 code cells). Source SHA-256: `dbe74505107fcddffafd40566f6561ad5646d7010cf82cea5b457637a2f12eb3`.

Commands from this directory: `python -m pytest -q test_benchmark_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.

# MP32 · Complete implementation and teaching deliverables; learner mastery unassessed

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP32 — Satellite Image Explorer

**Status: Complete (implementation and teaching deliverables)** · Learner mastery unassessed.

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/README.md) · [Progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

## Objective and prerequisites

Interpret bands, spatial resolution, GeoTIFF metadata, affine coordinates and nodata before super-resolution. Prerequisite: [MP31](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_07_transformers/MP31_transformer_based_sr_baseline/README.md). The openly licensed example is NAIP aerial imagery, explicitly not satellite evidence.

## Mathematics and conventions

Arrays use CHW band/row/column order. Pixel centers map through x=a(c+0.5)+b(r+0.5)+c₀, y=d(c+0.5)+e(r+0.5)+f₀. CRS units are preserved. With 10-m north-up pixels and corner (100,200), row2/column3 maps to (135,175). Per-band means include only masked-valid finite samples; empty means are unavailable. Nodata is not a cloud mask. The notebook includes hand examples, exercises and separate solutions.

## Executed experiment and outputs

Synthetic masked two-band fixture first; then a SHA-256-verified cached NAIP training raster, bounded 128×128 crop at column238/row238. [Experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP32_satellite_image_explorer/experiment.py) records geometry, band statistics, license, source and environment in [measured JSON](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp32-raster-explorer.json). [Figure](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp32-band-views.png) compares RGB, NIR/red/green false colour and labeled band distributions. Reproduce with `python experiment.py` after restoring the pinned training raster; no automatic downloads. Run `python run_notebook.py` to execute the lesson from recorded evidence.

## Interpretation and limitations

Band means and histograms describe this crop only. NIR/red/green assigns NIR to screen red; color differences do not establish vegetation health or model accuracy. Fixed DN/10000 preview scaling is not verified reflectance. CRS/transform must change consistently with resampling; ×2 pixel density does not prove recovered physical detail. Acquisition and retrieval dates are different. This is historical aerial imagery, not fresh satellite acquisition, cloud-free evidence or cross-sensor validation. These conventions feed MP33 preprocessing and MP34 paired data.

## Completion criteria and validation

- [x] Explain affine coordinates, units, masks and band semantics with numerical examples.
- [x] Execute deterministic synthetic checks and pinned licensed-raster inspection.
- [x] Test coordinate centers/corners, window bounds, per-band nodata, nonfinite and empty-mask handling.
- [x] Produce metadata sheet, masked band views and measured distributions with limitations.
- [x] Record license, source checksum, environment, rerun commands, exercises and solutions.
- [x] Link [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP32_satellite_image_explorer/VALIDATION.md) and progress; keep learner mastery unassessed.

## References

[Rasterio georeferencing](https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html) · [Rasterio masks](https://rasterio.readthedocs.io/en/stable/topics/masks.html) · [Pinned dataset manifest](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/configs/naip-pilot-manifest.json).

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP32_satellite_image_explorer/satellite_image_explorer.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP32_satellite_image_explorer/mp32_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP32_satellite_image_explorer/test_mp32_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP32_satellite_image_explorer/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP32_satellite_image_explorer/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP32 — Satellite Image Explorer

[Brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP32_satellite_image_explorer/README.md) · [Validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP32_satellite_image_explorer/VALIDATION.md). Objective: read raster geometry, masks and spectral bands before interpreting an SR image. The measured example is licensed NAIP **aerial** training imagery; satellite-specific sensor calibration is not established by this lesson.

## 1. Coordinates are not array indices

An image is C×H×W: bands, rows, columns. Its affine transform maps column c and row r into map coordinates: x=a(c+½)+b(r+½)+c₀ and y=d(c+½)+e(r+½)+f₀ for pixel **centers**. Pixel corners omit ½. The CRS defines units and coordinate meaning. A north-up projected raster usually has a>0, e<0, b=d=0; rotated grids require the full transform.

Example: 10-m pixels, upper-left corner (100,200), north-up. Pixel (row=2,column=3) has center (135,175), while its upper-left corner is (130,180). Changing the array dimensions without updating the transform changes its geographic extent.

## 2. Nodata and band statistics

For band b, mean μ_b=(Σ valid x_b)/N_b. Validity is per band and excludes nonfinite values; an empty band returns unavailable, not zero. Nodata values are placeholders, not observations. A value of zero can be valid unless metadata/masks declare otherwise. A nodata mask is not a cloud mask.

Synthetic two-band 2×2 fixture: exclude the lower-left pixel (99). Remaining band-one values [1,3,5] average 3; band-two [2,4,6] average 4. Including 99 would bias the first mean to 27.

## 3. Measured raster inspection

The experiment verifies the training scene SHA-256, reads only the 128×128 window at column/row (238,238), and records its CRS, transform, pixel size, band count, nodata and valid statistics. No held-out scene is read. Acquisition time identifies the sensor observation; the manifest retrieval time belongs to the cached source acquisition, not a fresh request today. Re-run deliberately with `python experiment.py` once the pinned local raster is available; notebook execution reads the measured report.

## 4. Band choice changes the display, not the underlying sensor

RGB assigns red/green/blue bands to screen R/G/B. The false-colour panel assigns NIR/red/green to screen R/G/B, so red-looking areas can indicate higher NIR, not red visible light. Both previews divide stored DN by 10000 and clip to the **same fixed [0,1] display range**; this is a display convention, not reflectance calibration. Transparent pixels are invalid. Histogram x-axis: DN/10000; y-axis: valid pixel counts; line labels identify the bands. No universal brightness or vegetation threshold is inferred.

## 5. Exercises — try before opening the solutions

1. Compute the center of pixel (0,0) in the 10-m fixture. What happens to the affine pixel sizes after ×2 upsampling that preserves extent?
2. Why is an all-invalid band not a zero-valued band? Can a nodata mask remove all clouds?
3. Can stronger red in the false-colour view prove vegetation health? Can this cached aerial example establish current satellite conditions?

<details><summary>Solutions</summary>

1. (105,195). Pixel sizes become +5 and −5 in the same projected CRS, with the corner origin unchanged. More samples do not prove new physical detail.
2. No observations exist, so the mean is unavailable. Valid zeros are observations. Clouds need suitable quality masks/criteria beyond nodata.
3. No: screen red represents NIR in this mapping, and health requires calibrated measurements and context. Acquisition is historical and the sensor is aerial; neither current conditions nor satellite generalization is demonstrated.
</details>

## Interpretation and connection to SR

The actual report above shows the valid counts and band means for this single crop. Differences between bands and display mappings demonstrate why a three-channel preview cannot describe all four-band information. They do not identify land-cover causes. Preserve transforms, CRS, masks and band order when constructing HR/LR pairs; evaluation must compare compatible, aligned valid pixels. Next: MP33 preprocessing. Remaining scientific limits: one crop, unverified physical calibration, no per-pixel cloud mask, no cross-sensor accuracy or learner-mastery claim.

## Recorded validation

# MP32 validation — 2026-10-25

Implementation and teaching deliverables complete; learner mastery unassessed.

- `python -m pytest -q stage_08_remote_sensing/MP32_satellite_image_explorer`: 3 tests passed; six affine deprecation warnings. Checks center/corner geometry, cropped transforms, nodata, bounded windows, nonfinite and empty-mask statistics.
- `python stage_08_remote_sensing/MP32_satellite_image_explorer/experiment.py`: executed synthetic fixture and SHA-256-verified training crop. Four bands; EPSG:32611; 2.5-m pixels; 16,384 valid pixels per band. No download, optimization or test access.
- `python scripts/execute_project_notebooks.py --only MP32`: real kernel passed, 5 code cells. Source SHA-256 `f4c92dd218437ede150547a71c4e9f833e6efb09ba5b004b3a8c668e05b355f6`. Figure visually inspected; fixed-range previews intentionally retain dark values rather than silently applying different stretches.
- All 40 project layouts and 817 relative Markdown links passed before this validation update.

Evidence: [measured metadata/statistics](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp32-raster-explorer.json), [band views](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp32-band-views.png), [kernel record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Source license, versions, checksum and training-only selection are recorded. The historical project-folder test report remains an October25 earlier snapshot; this document records the expanded three-test scope.

The experiment is an aerial raster inspection, not satellite generalization. Masks do not establish cloud-free pixels; DN/10000 is not calibrated reflectance. Full cross-sensor work remains separate.

# MP33 · Complete implementation and teaching deliverables; learner mastery unassessed

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP33 — Satellite Preprocessing Pipeline

**Status: Complete (implementation and teaching deliverables)** · Learner mastery unassessed.

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/README.md) · [Progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

## Objective and prerequisites

Select bands, preserve invalid pixels, fit train-only normalization, and extract georeferenced patches. Prerequisite [MP32](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP32_satellite_image_explorer/README.md). CHW arrays and per-band masks are explicit. All examples use synthetic fixtures or pinned NAIP aerial training data, not satellite generalization evidence.

## Mathematics

Training population statistics: μ=Σx/N and σ=√(Σ(x−μ)²/N). Normalize z=(x−μ)/σ; constant bands use scale1; empty valid bands fail. For [1,3,5], μ=3, σ=√(8/3), z≈[−1.224745,0,1.224745]. Invalid fills remain masked. Freeze statistics before validation/test. Crop transform T_crop=T_parent×Translation(column,row); retain CRS and band semantics.

## Experiment and outputs

[Experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/experiment.py) verifies the training raster checksum, selects RGB+NIR, fits training-window statistics, produces float32 normalized data, and extracts 16 nonoverlapping valid32×32 patches from128×128 pixels. [Measured metadata sheet](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp33-preprocessing.json) records statistics, every patch transform, mask fraction, source and environment. `python experiment.py` re-executes with local pinned data; `python run_notebook.py` executes the lesson using recorded evidence. Original rasters and frozen SR preprocessing are unchanged.

## Interpretation and limitations

Near-zero means and unit population standard deviations validate the transformation, not reconstruction accuracy. The real crop has no invalid pixels; synthetic tests exercise nodata rejection and discarded edges. Per-band z-scores alter spectral angles and are not calibrated reflectance or compatible replacement inputs for existing DN/10000 models. No cloud-mask validation or cross-sensor claim. MP34 uses explicit geometry and validity to construct HR/LR pairs.

## Completion criteria

- [x] Explain band ordering, masks, normalization and crop transforms with worked numbers.
- [x] Execute synthetic fixtures and a checksum-pinned training-raster experiment.
- [x] Test train-only fitting, frozen application, constant/empty bands, band order, patch rejection and coordinates.
- [x] Provide preprocessing specification and metadata/invalid-pixel checks.
- [x] Include exercises, solutions, source license, environment and limitations.
- [x] Link [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/VALIDATION.md) and progress without claiming learner mastery.

## References

[Rasterio masks](https://rasterio.readthedocs.io/en/stable/topics/masks.html) · [Windowed reading](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html) · [Pinned manifest](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/configs/naip-pilot-manifest.json).

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/satellite_preprocessing_pipeline.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/mp33_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/test_mp33_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP33 — Satellite Preprocessing Pipeline

[Brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/README.md) · [Validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/VALIDATION.md). Objective: select bands, preserve validity, fit normalization only on training data, and extract correctly georeferenced patches. Prerequisite MP32. This lesson uses a cached aerial training crop and does not change the frozen SR pipeline.

## 1. Bands and validity before arithmetic

CHW means channels/bands, rows, columns. Band indices are zero-based; select [0,1,2,3] for the manifest's red/green/blue/NIR order, or [3,0,1] for an explicitly labeled false-colour display. Reordering is not color-space conversion. Use a CHW boolean mask so nodata can differ by band; intersect with finite values. Never infer nodata from darkness alone. The selection helper computes in float64 to avoid unsigned subtraction overflow.

Invalid normalized values are stored as zero **with a false mask**. Zero is a neutral fill, not an observation. Downstream losses and statistics must use the mask; partial patches are rejected by default.

## 2. Train-only standardization

For valid training samples in band b, μ_b=Σx/N_b and σ_b=√(Σ(x−μ_b)²/N_b), using population variance. Transform with z=(x−μ_b)/s_b, where s_b=σ_b unless σ_b=0, in which case s_b=1. Save these statistics and reuse them without fitting on validation or test. An all-invalid band cannot be fitted. The explicit split guard prevents accidental validation fitting, but cannot verify that a caller's label is truthful: preserve provenance too.

Hand example [1,3,5]: μ=3, σ=√(8/3)≈1.63299, z≈[−1.224745,0,1.224745]. A constant band [2,2,2] maps to zero. Values outside [0,1] are normal for z-scores. Unlike fixed DN/10000, z-scores are centered per band and alter spectral-vector angles; they are not physically meaningful reflectance or a replacement for the frozen model's expected inputs.

## 3. Patch coordinates and acceptance

For a crop starting at column c₀,row r₀, the new affine transform is T_crop=T_parent×Translation(c₀,r₀). Keep the CRS and band semantics unchanged. In a north-up 10-m fixture whose corner is (100,200), column2,row0 starts at (120,200). Pixel centers still add half a pixel. Rotated transforms use the full affine multiplication; do not guess north from the picture.

The extractor walks full nonoverlapping P×P patches in row-major order. Acceptance requires the fraction of pixels valid in **all selected bands** to exceed the threshold (default1). It reports rejected patches and incomplete bottom/right margins. Patching must follow geographic scene splitting, never precede it.

## 4. Measured cached-raster experiment

The experiment first checks the synthetic fixture, verifies the SHA-256 of the NAIP training raster, and reads a 128×128 four-band window at (238,238). Statistics are fitted on this training window only, then saved. Normalized output is float32; patch extraction uses float64 working arrays. Neither the original file nor model preprocessing is modified. Re-run with `python experiment.py` only when needed; this notebook reads the recorded result.

## 5. Exercises and solutions

1. Why must a false mask accompany a zero fill? 2. What are μ and σ for [1,3,5], and why is validation recentering leakage? 3. For the 10-m transform, compute the upper-left corner of a patch at row2,column2. 4. Is per-band z-scoring appropriate before physical spectral-angle comparison?

<details><summary>Solutions</summary>

1. Otherwise fabricated zeros affect losses/statistics. 2. μ=3 and σ=√(8/3); validation fitting changes the learned preprocessing using held-out information and masks distribution shift. 3. (120,180), with unchanged CRS and 10-m spacing. 4. No: different band offsets/scales change spectral directions. Use a declared scientifically compatible physical protocol instead.
</details>

## Interpretation and connection

Sixteen full 32×32 patches were accepted from the cached crop, with no invalid pixels there. Near-zero means/unit population standard deviations confirm the declared transform; they do **not** demonstrate improved SR quality. Synthetic masking tests cover rejection paths absent from this crop. The experiment preserves CRS and derives every patch transform. Next MP34 builds matched HR/LR pairs with explicit degradation. Limits: one aerial crop, no cloud-mask validation, no reflectance calibration, no test fitting, no learner-mastery claim.

## Recorded validation

# MP33 validation — 2026-10-25

Complete implementation and teaching deliverables; learner mastery unassessed.

`python -m pytest -q stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline`: 3 tests passed (3 affine deprecation warnings). Tests cover finite/nodata means, training-only fitting, fixed-stat application, constant/empty bands, zero-based band ordering, invalid patch rejection, full-patch edges and translated affine coordinates.

`python stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/experiment.py`: synthetic hand checks and pinned training-raster experiment passed. Sixteen valid32×32 four-band patches; normalized means within7.5e-9 of zero and standard deviations within6e-8 of one. These verify normalization, not SR quality. No download, training, validation/test fitting or frozen-result changes.

`python scripts/execute_project_notebooks.py --only MP33`: real kernel passed 5 code cells. Source SHA-256 `6f5c8b1412447204238645896f2fb8514cd45edcc2dd4847d22e6e86fa6328f0`. Notebook includes numerical examples, recorded metadata, exercises/solutions and interpretation.

[Measured record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp33-preprocessing.json) · [Kernel evidence](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). The data are cached NAIP aerial imagery, not a fresh satellite observation. Per-band standardization alters spectral angles and is not inserted into existing model inference. CRS, masks, license, checksum and environment are retained in the record.

# MP34 · Complete implementation and teaching deliverables; learner mastery unassessed

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP34 — Satellite HR–LR Pair Builder

**Status: Complete (implementation and teaching deliverables)** · Learner mastery unassessed.

[Home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/README.md) · [Progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

## Objectives and prerequisites

Construct reproducible aligned synthetic×2/×4 HR/LR pairs with explicit bands, masks, units and geographic transforms. Prerequisite [MP33](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/README.md). Scientific pair arrays remain distinct from clipped RGB display exports.

## Mathematics and controls

y=D_s(k*x)+n with CHW float input, Gaussianσ=.8 HR pixels (radiusceil(3σ), normalized kernel, replicate edges), area downsampling and optional seeded additive Gaussian noise. Recorded real pairs use noise0. Shapes must be divisible by s. T_LR=T_HR×Scale(s,s), retaining CRS and exterior corners. All HR values must be finite/valid; invalid patches are rejected before blur. No silent padding, cropping or clipping.

Hand example: [0,1;4,5] averages to2.5; a4×4 ramp averages to7.5 at×4. The lesson includes rotated-affine checks, center/corner distinctions and exercises with solutions.

## Executed experiment and deliverables

[Experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/experiment.py) runs synthetic checks, verifies the training-source checksum and builds two pairs from a64×64 four-band NAIP aerial crop. HR2.5-m spacing becomes LR5m/10m; both retain a160×160-m footprint in EPSG:32611. [Pairing manifest](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp34-pairing-manifest.json) records source/license, window, shapes, transforms, units, exact NPY hashes and environment. Rebuild ignored arrays with `python experiment.py`; run the lesson with `python run_notebook.py`. No data download or model training.

## Interpretation and limitations

Matching extents and shapes verify the synthetic geometry, not real sensor compatibility. Different bands, point-spread functions, timing, registration or calibration require a separate cross-sensor protocol. Fixed DN/10000 is not verified reflectance. Fully valid masks do not establish cloud-free observations. ×4 pairs cannot be fed silently to×2 models. This feeds MP35 geographic separation and the major project's declared synthetic pipeline.

## Completion criteria

- [x] Derive degradation, shape and transform equations with numerical examples.
- [x] Execute synthetic fixtures and pinned training-only×2/×4 pairing.
- [x] Check constants, area arithmetic, rotated corners, invalid pixels, shapes and seeded noise.
- [x] Deliver a pairing manifest with rebuildable artifact hashes and exact settings.
- [x] Record license, environment, interpretation, limitations and exercises/solutions.
- [x] Link [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/VALIDATION.md) and progress without learner-mastery claims.

## References

[Shared degradation implementation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/src/rssr/core.py) · [Rasterio georeferencing](https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html) · [Pinned manifest](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/configs/naip-pilot-manifest.json).

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/satellite_hr_lr_pair_builder.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/mp34_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/test_mp34_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.

# MP34 — Satellite HR–LR Pair Builder

[Brief](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/README.md) · [Validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/VALIDATION.md). Objective: construct reproducible synthetic ×2/×4 pairs with compatible dimensions, transforms and masks. Prerequisite MP33. The example source is aerial NAIP; synthetic pairing does not establish real satellite sensor correspondence.

## 1. The degradation equation

y=D_s(k*x)+n. HR x has C×H×W values; y has C×(H/s)×(W/s). The normalized Gaussian k uses σ=0.8 HR pixels, radius ceil(3σ), replicate boundaries; D_s takes a nonoverlapping s×s area mean. Noise n is optional seeded additive Gaussian in input units; the recorded real experiment uses noise0. No clipping is applied to scientific pair arrays. C, band ordering and numeric scaling stay fixed.

This is an explicit synthetic forward model, not a measured satellite point-spread function. Different real sensors can differ in spectral response, acquisition time, viewing angle, atmosphere, cloud cover and registration. Synthetic pairs isolate a controlled learning problem but cannot validate those differences.

## 2. Shape and footprint alignment

Hand calculation: the top-left2×2 block [0,1;4,5] averages to2.5. A4×4 image becomes2×2 at×2 or1×1 at×4. Require divisibility rather than silently crop or resize. If the HR pixel-to-map transform is T, use T_LR=T×Scale(s,s), keeping the CRS and upper-left corner. The four exterior map corners must agree. Pixel **centers** differ intentionally: each coarse sample summarizes an HR footprint.

For HR10-m north-up pixels and corner(100,200), the first HR center is(105,195); first×2 LR center is(110,190). Both cover the same4×4 HR outer extent. Rotated/skewed grids require the full affine multiplication.

## 3. Validity, noise and provenance

This bounded builder requires fully valid HR pixels. It rejects nodata instead of allowing an arbitrary fill to spread through the Gaussian kernel. For partially valid imagery, a separately specified support-aware degradation/mask protocol is needed; this implementation does not invent one. A valid mask is not a cloud mask. Split scenes geographically before making pairs, and preserve source license/checksum, crop offsets, bands, units, operator and seed.

Fixed seed gives identical noise for a repeated input/settings combination. It does not make the noise a physical sensor model. Unchanged source arrays and deterministic output hashes support reproducibility, not physical correctness.

## 4. Executed real-raster pairs

The experiment verifies the pinned training raster's SHA-256, reads a64×64 RGBNIR window at column238,row238, and divides stored uint16 DN by10000 without per-image stretching. σ=.8, noise0, and area×2/×4 produce32×32 and16×16 pairs. HR pixel spacing is2.5m; LR spacing is5m and10m. The common map footprint is160×160m in EPSG:32611. This describes the raster grid, not native optical resolution.

The manifest below records exact transforms, shapes, array-file hashes, operator and environment. Pair arrays live in ignored data/teaching/mp34 and can be rebuilt with `python experiment.py` from the pinned training source. Notebook execution reads the recorded manifest; it does not overwrite pairs or re-evaluate any model.

## 5. Exercises and separate solutions

1. Derive the top-left area×2 value of [0,1;4,5]. 2. A64×64 HR crop has2.5-m spacing: give LR shapes/spacings at×2 and×4. 3. Why not replace nodata with zero before blurring? 4. Does identical geographic extent prove accurate cross-sensor pairing?

<details><summary>Solutions</summary>

1. (0+1+4+5)/4=2.5. 2.32×32 at5m and16×16 at10m, both160m per side for this north-up grid. 3. The fill contaminates adjacent pixels through convolution; masks need support-aware treatment or the patch must be rejected. 4. No: spectral response, time, viewing geometry, registration and calibration can differ even with matching extents.
</details>

## Interpretation and connection

Both measured pairs have the intended shapes and exact matching exterior corners; smoothing/downsampling changes pixel ranges as recorded above. This is evidence of declared synthetic pairing, not improved SR fidelity. ×4 discards more samples and is not interchangeable with the hosted×2 models. Next MP35 checks geographic train/validation/test separation. Frozen test data, checkpoints and inference normalization remain unchanged; no learner mastery or real-satellite generalization claim.

## Recorded validation

# MP34 validation — 2026-10-25

Complete implementation and teaching deliverables; learner mastery unassessed.

`python -m pytest -q stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder`:3 tests passed;18 affine deprecation warnings. Verified constant preservation, input nonmutation, divisibility/scale rejection, hand-computed area averages, rotated exterior-corner equality, missing CRS/degenerate transform/invalid-pixel rejection, and noise-seed repeatability.

`python stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/experiment.py`: synthetic hand check and pinned training-raster experiment passed. Four-band64×64 HR produced32×32 and16×16 LR with5-m/10-m spacing; same160×160-m extent in EPSG:32611. Original source checksum verified. Ignored NPY outputs have recorded SHA-256 hashes and are rebuildable, not committed datasets.

`python scripts/execute_project_notebooks.py --only MP34`: real kernel passed 5 code cells. Notebook SHA-256 `d3ca4a6bfdb8cc6f4bf401c6bb3a3b26f1a1afe60127ca66623efc4adcfd49f0`. No frozen-result evaluation, new training or data download.

[Pair manifest](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp34-pairing-manifest.json) · [Kernel record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Lesson covers equations, examples, provenance, exercises/solutions and failure cases. No physical calibration, cloud-free or cross-sensor accuracy claim. Hosted models remain native×2 and cannot silently consume×4 pairs.

# MP35 · Complete implementation and teaching deliverables; learner mastery unassessed

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP35 — Geographic Dataset Splitting

**Status: Complete** · Implementation and teaching deliverables; learner mastery unassessed.

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

## Learning objectives and prerequisites

Assign regions before tiling, detect region and spatial leakage, and explain what geographic independence does and does not establish. Complete [MP34](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/README.md) first. A scene is a source image; a region groups related scenes. Keep every patch from one region in its declared split.

## Mathematics and worked example

Region sets T (training), V (validation), E (test) must have empty pairwise intersections. For WGS84 axis-aligned boxes A and B, define g=max(Bwest−Aeast, Awest−Beast, Bsouth−Anorth, Asouth−Bnorth). The existing pipeline accepts a cross-split pair only when g>b, with b=0.02 degrees. For A=[0,0,1,1] and B=[2,0,3,1], g=1: b=.02 passes, b=1 fails. Touching boxes fail. This is an angular envelope rule, not a geodesic-distance threshold; longitude degrees vary in physical length by latitude. Antimeridian-crossing boxes are explicitly unsupported.

## Executed experiment

Question: do the existing pinned scene declarations satisfy the pipeline's pre-tiling leakage rules? Reuse the original assignments, verify all nine cached file checksums, transform raster-header bounds to EPSG:4326, and audit every cross-split pair. No pixel arrays, new downloads, model inference, or test scores are used. File bytes are read only for checksum verification. No new random split is selected.

Observed: four training scenes, four validation scenes and one test scene; all 24 cross-split pairs pass. Original IDs and assignments remain unchanged. [Audit and environment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp35-geographic-splits.json), [train manifest](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp35-splits/train.json), [validation manifest](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp35-splits/validation.json), [test manifest](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp35-splits/test.json), [scene-center map](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp35-splits/map.svg).

## Run and deliverables

From the repository root in the development environment:



The [tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP35_geographic_dataset_splitting/mp35_tools.py) validate IDs, finite bounds and buffers before delegating to the unchanged shared pipeline. [Tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP35_geographic_dataset_splitting/test_mp35_tools.py), [experiment](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP35_geographic_dataset_splitting/experiment.py), [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP35_geographic_dataset_splitting/geographic_dataset_splitting.ipynb), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP35_geographic_dataset_splitting/run_notebook.py), and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP35_geographic_dataset_splitting/VALIDATION.md) provide MP01-style local evidence. Exports are metadata teaching artifacts; authoritative configs are not rewritten.

## Interpretation, exercises and limitations

Baseline failure example: randomly place patch 1 of each of two scenes in train and patch 2 in test. Both scenes leak across sets. Solution: assign scenes/regions first, then tile within each assignment. Exercise: can spatially separated scenes with identical region IDs pass? No; region identity is a separate constraint.

Sixteen patches from one scene are not sixteen independent geographic replicates. Within-split overlap is permitted. Box separation prevents one form of leakage, not sensor/date dependence or distribution shift. A single test region cannot establish broad geographic generalization. No change to the frozen benchmark or SR-quality improvement is claimed. Next question: cross-sensor transfer (MP36).

## Provenance and references

[Source manifest](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/configs/naip-development-manifest.json) pins OpenSR dataset revision and SHA256s. NAIP is aerial imagery, not satellite imagery. License: dataset-card MIT declaration; USDA NAIP public-domain source imagery attribution retained. Source CRS, transforms, shapes, acquisition dates and URLs are preserved in exports. Environment and source-manifest hash are in the audit record.

[Rasterio coordinate transformation documentation](https://rasterio.readthedocs.io/en/stable/api/rasterio.warp.html). No paper reproduction or new scientific benchmark is claimed.

## Completion criteria

- [x] Explain objectives and solve the small mathematical example.
- [x] Execute the declared cached-scene experiment.
- [x] Verify numerical, leakage and invalid-input cases.
- [x] Deliver split manifests, map and interpreted audit.
- [x] Record license, environment, limitations and frozen-assignment deviation from new split creation.
- [x] Link evidence and update progress. Learner understanding remains unassessed.

# MP35: geographic splitting before tiling
A scene is a source image; a region groups related scenes. Split regions before extracting patches, so nearby samples do not leak between training and evaluation. We audit the already frozen assignments rather than selecting a new favorable test set.

For region sets T,V,E require T∩V=T∩E=V∩E=∅. For boxes A,B define g=max(Bwest−Aeast, Awest−Beast, Bsouth−Anorth, Asouth−Bnorth). Accept only g>b, where b=0.02 degrees. This is an axis-envelope rule, not Euclidean/geodesic distance. Longitude degrees shrink physically toward the poles.

Worked example: A=[0,0,1,1], B=[2,0,3,1], g=1 degree. It passes b=.02 but fails b=1 because equality is rejected.

Exercise: change the second region to a. Why must the audit fail even if its box is distant? Solution: region identity is an independent leakage constraint; geometry alone is insufficient.

Interpretation: four train regions, four validation regions and one preserved test region pass 24 cross-split checks. No reassignment or test inference occurred. Sixteen patches from one scene remain one geographic replicate, not sixteen independent regions. Overlap rejection addresses leakage but does not establish sensor independence or broad generalization.

Exercise: if two patches from each of two scenes are split randomly, could both sets contain each scene? Yes: place the first patch in train and second in test. Scene-first assignment prevents this. Next question: how does performance transfer across sensors under a separately declared protocol? MP36 addresses that requirement.

Provenance: pinned OpenSR NAIP manifest and checksums in the report; NAIP is aerial, not satellite imagery. Dataset-card MIT declaration and USDA public-domain source attribution retained. These metadata exports reuse existing split declarations; they are not a newly selected benchmark.

## Recorded validation

# MP35 validation — 2026-10-25

Commands are listed in [README](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP35_geographic_dataset_splitting/README.md). Development Python 3.12.14.

- `python -m pytest -q stage_08_remote_sensing/MP35_geographic_dataset_splitting`: **12 passed**. Covers exact buffer equality, region reuse, overlap, duplicate IDs, invalid split/bounds, nonfinite/negative buffer, empty input and permitted within-split overlap.
- `experiment.py`: **passed**, nine SHA256-verified cached raster headers; 4/4/1 scenes and 24 cross-split pairs. [Audit](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp35-geographic-splits.json) records environment, declared buffer and all pair clearances.
- `run_notebook.py`: **passed-real-jupyter-kernel**, four code cells. [Execution record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json); executed outputs remain ignored under data/.
- Existing source manifest, assignments, raster pixels and frozen result artifacts remain unchanged. No image metrics or training ran.

The exported map depicts scene centers in angular coordinates, not a metric basemap. Passing this lesson does not validate broad geographic generalization or learner mastery.

# MP36 · Expanded numerical teaching component verified; real cross-sensor feasibility acceptance pending

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP36 — Cross-Sensor Dataset Investigation

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

The expanded numerical teaching component is now executed: seven tests, four real-kernel notebook cells and an analytical example report. Official dataset/version/license/size investigation is recorded, but registration/correction methodology and complete feasibility acceptance remain open. This is not a completed paired cross-sensor evaluation or learner completion.

## Learning objectives

- WorldStrat, Sen2Venµs, geometric and radiometric differences.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP35 — Geographic Dataset Splitting](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP35_geographic_dataset_splitting/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Relate ground sampling distance, registration error, radiometric scaling, and spectral response to apparent reconstruction error.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Partially executed; not complete.** Investigate WorldStrat and Sen2Venµs documentation, access, licenses, registration, and band compatibility before downloads; deliver a dataset comparison matrix and cross-sensor evaluation feasibility report.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.
- Remote sensing: preserve CRS, affine transform, band meaning, nodata, physical units, and source licenses. Split by region/scene before tiling.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; none is supplied in this scaffold.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Remote sensing](https://rasterio.readthedocs.io/en/stable/)
- [WorldStrat](https://github.com/worldstrat/worldstrat)
- [Sen2Venµs](https://zenodo.org/records/6514159)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP36_cross_sensor_dataset_investigation/cross_sensor_dataset_investigation.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP36_cross_sensor_dataset_investigation/mp36_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP36_cross_sensor_dataset_investigation/test_mp36_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP36_cross_sensor_dataset_investigation/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP36_cross_sensor_dataset_investigation/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.


## Executed numerical lesson and remaining gate

[Analytical report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp36-numerical-examples.json), [source feasibility matrix](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp36-feasibility.md) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP36_cross_sensor_dataset_investigation/VALIDATION.md) separate source facts, deductions and unresolved questions. Run from repository root:



Worked examples:10m/5m gives×2;10m/1.5m gives20/3 (not an integer native model scale). A5m registration offset divided by5m/pixel equals1pixel. For reference[.2,.4] and gain1.1/offset.01, apparent MSE is.0017 despite no SR reconstruction step. Equal-width toy spectral bins with values[.2,.8] produce.2,.8 or.5 under different response weights. The notebook defines symbols, units, exercises and solutions; weights are invented teaching values, not measured sensor response.

Next acceptance gate: verify official registration methodology and published correction, complete the source-specific band/timestamp/mask matrix, and prepare a concrete bounded resource proposal. The paper endpoint returned429; do not infer a registration tolerance. Real paired experiments additionally require the [v2 protocol gates](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/protocol-v2.md). No large download or training was authorized or executed by this lesson.

# MP36 — Cross-sensor feasibility and numerical pitfalls
This lesson distinguishes a dataset investigation from a completed cross-sensor study. No new imagery is downloaded or evaluated. Source review remains incomplete while the official paper/correction detail is unavailable.

Let dL and dH be metres per pixel. The geometric scale is s=dL/dH. Scale compatibility alone says nothing about radiometric or spectral compatibility. Work through10/5=2,20/5=4 and10/1.5≈6.667.

A displacement δ metres is δ/dH HR pixels. A5m offset on a5m grid is one pixel, even if both images have correct nominal resolution. Exercise: what is a2.5m offset? Answer: half a pixel. This predicts geometric displacement, not a universal pixel-error score.

If calibration gives y=a*x+b, the apparent residual is (a−1)*x+b even without reconstruction error. For x=[.2,.4], a=1.1 and b=.01, residuals are[.03,.05] and MSE=(.0009+.0025)/2=.0017. No clipping is hidden. Exercise: does lower MSE after test-fitted calibration establish SR quality? No: it leaks reference information.

Toy equal-width spectral bins produce band value B=sum(r_i*w_i)/sum(w_i). Identical band names can integrate different parts of a spectrum. Real response curves require wavelength-aware integration; these invented weights are not sensor measurements.

For spectrum[.2,.8], two single-bin sensors report.2 and.8; equal weighting reports.5. Thus spectral response mismatch can mimic model error. See reports/mp36-feasibility.md and the local README for verified source records, licenses, transfer sizes and unresolved methodology. Next: complete official registration/correction review and freeze a bounded inspection plan before asking for archive transfer. One inspected site cannot establish geographic generalization.

## Recorded validation

# MP36 partial validation — 2026-10-26

Status: In progress. The full investigation and paired-data evaluation are not complete.

- Local pytest: **7 passed**. Tests cover noninteger scale, invalid GSD/offset, cast-before-arithmetic, calibrated toy MSE, empty/nonfinite arrays, spectral response mismatch and invalid weights.
- `experiment.py`: executed synthetic analytical examples only; [record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp36-numerical-examples.json) includes environment and values.
- `run_notebook.py`: **passed-real-jupyter-kernel**, four code cells; [execution record](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Known local kernel TCP warning; process exited0.
- No dataset archive transfer, new test-image inspection, model training or frozen test recomputation.
- Primary official records support version/license/size findings. Paper/correction retrieval returned429, leaving registration-method review open. See [feasibility](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp36-feasibility.md).

Source citations and limitations do not become sensor performance evidence merely because these numerical tests pass.

# MP37 · Expanded implementation/teaching complete; learner mastery unassessed

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP37 - Feature Fusion Laboratory

**Status: Complete** (implementation/teaching scope). Expanded implementation and teaching checks; learner mastery is unassessed. Research-grade superiority is not claimed.

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/README.md) · [Progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

## Objectives and prerequisites

Explain sum, mean and concatenation/projection; derive shapes and costs; show how a projection can represent averaging; interpret a controlled fixture separately from historical training results. Reuse CNN/window features from MP26-MP31. [MP36](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_08_remote_sensing/MP36_cross_sensor_dataset_investigation/README.md) remains necessary before real cross-sensor claims, but is not needed for this synthetic tensor lesson.

## Mathematics and conventions

A,B are aligned floating-point tensors of shape N,C,H,W (batch, channel, height, width), on the same device/dtype. Feature values have learned arbitrary units, not physical reflectance. Sum S=A+B and mean M=(A+B)/2 retain C channels; concatenation Z=[A;B] has2C channels. A biased1x1 projection P maps Z back toC: Y=P_A A+P_B B+b at every pixel. Setting P_A=P_B=I/2 and b=0 reproduces averaging. This is an expressivity statement, not a claim that training discovers useful weights.

The projection has2C^2+C parameters and NHW*2C^2 MACs. A MAC is a multiply-accumulate pair; bias additions, feature copies, branch costs and activations are excluded. These are not full-model FLOPs. Averaging uses no trainable projection parameters. Derivatives of sum with respect to either branch are1; mean gives1/2. This changes gradient scale and can confound training comparisons.

Worked example: A=[1,2], B=[3,4]. Sum=[4,6], mean=[2,3], concatenation=[1,2,3,4]. A2x4 projection with half-identity blocks yields[2,3]. For C=8,H=5,W=7,N=1, the projection costs136 parameters and4480 MACs. Channel correspondence is assumed for sum/mean; spatial misalignment cannot be fixed merely by concatenating.



## Executed experiment and evidence

[experiment.py](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP37_feature_fusion_laboratory/experiment.py) uses the shared SRModel, CPU float32, seed17 and two threads. It copies every common parameter between the two fusion variants and initializes the concat projector as the exact mean. A synthetic1x3x5x7 input produces1x3x10x14 outputs. This isolates algebra/shape behavior; no training, checkpoint selection, imagery download or held-out inference occurs. Shared WindowBlock uses replicated boundary padding before cropping back to the original feature size.

[Report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp37-fusion.json) records actual output difference, shapes, counts, gradients and environment. It separately reads existing ablation-pilot-runs.json by variant/run ID with its canonical hash. Those historical real-NAIP validation results used three seeds,100 steps, one validation scene and unequal parameter counts. Their mean/seed SD are not new measurements, geographic confidence intervals or a matched-budget causal comparison. Larger trained-model differences cannot be attributed solely to fusion capacity. Frozen results are unchanged.

## Run and inspect

From repository root with the existing development environment:



Local files follow MP01: [notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP37_feature_fusion_laboratory/feature_fusion_laboratory.ipynb), [tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP37_feature_fusion_laboratory/mp37_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP37_feature_fusion_laboratory/test_mp37_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP37_feature_fusion_laboratory/run_notebook.py), [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP37_feature_fusion_laboratory/VALIDATION.md). Historical central notebooks stay unchanged. Synthetic fixtures have no third-party imagery license. Historical source licensing/provenance remains in the original pilot manifests; this report contains only derived numbers and run IDs.

## Practice and solutions

1. For N=2,C=4,H=3,W=5, derive concat shape and projection cost. **Solution:**2x8x3x5,36 parameters,960 MACs.
2. Can a1x1 projection implement A-B? **Solution:** P_A=I,P_B=-I,b=0; negative weights are allowed. It cannot perform a spatial shift by itself.
3. Why does averaging cancel opposite features? **Solution:** if B=-A, M=0, losing branch identity. Concatenation retains both before projection, but the projection can still discard them.
4. Does identical output under the mean initialization prove equal trained performance? **Solution:** no; parameterization, optimization, budgets and initialization differ. Use an independently registered matched-budget study for that claim.
5. What happens for integer inputs, empty spatial axes, NaNs or mismatched shapes? **Solution:** local tools reject them explicitly; no silent casting/broadcasting.

## Completion criteria

- [x] Written objectives, defined symbols, worked example and solved exercises.
- [x] Controlled synthetic experiment and historical validation interpretation recorded.
- [x] Relevant local tests and expanded real-kernel notebook pass.
- [x] Diagram, parameter/MAC convention, limitations and reproducible commands supplied.
- [x] Link final validation and update implementation status after kernel acceptance.

These criteria concern implementation/teaching deliverables, not learner mastery or completion of the larger research study.

# MP37: feature fusion

We compare aligned NCHW feature maps, using CPU float32. Sum and mean preserve C channels; concatenation preserves both branches as2C channels before a learned1x1 projection. These are teaching tensors, not sensor observations. See [lesson](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP37_feature_fusion_laboratory/README.md) for assumptions, diagram and exercises.

## Projection and gradients

P=[I/2,I/2] with zero bias exactly represents averaging. dM/dA=dM/dB=1/2. This equality does not imply that unconstrained training produces the same function.

## Controlled shared-model fixture and reused evidence

The next cell copies common weights between mean and concat models, sets the projector to the exact mean, and checks an odd rectangular input. Historical scores come from existing validation runs and are kept separate; no training or test evaluation runs here.

## Exercise and answer

For2x4x3x5 features, concat has2x8x3x5 elements; a2C->C projection has36 parameters and960 MACs. Why not call the larger historical score proof of better architecture? Because the one-scene pilot has differing parameter budgets and seed variation is not geographic uncertainty. A1x1 projection mixes channels at a pixel but cannot itself shift an incorrectly registered image.

## Recorded validation

# MP37 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 1 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `205d4450d3bd1d56f0312d9c96ae56688398414a28226efb73645b139050cb51`.

Commands from this directory: `python -m pytest -q test_mp37_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.


## Expanded acceptance — 2026-10-27 (supersedes introductory scope)

Seven tests passed in3.78s. The first collection attempt failed with Windows Application Control WinError4551 loading torch_python.dll; the subsequent unchanged experiment imported Torch and ran, and one unchanged pytest retry passed. No security policy or environment was altered. The expanded three-code-cell notebook passed a real kernel with exit0; the existing local TCP warning remains.

[Experiment report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp37-fusion.json): sum[4,6], mean/projected[2,3], branch gradients[.5,.5]. Shared-parameter hybrid fixture1x3x5x7 ->1x3x10x14 had maximum absolute output difference0 with the mean-initialized projector. Projection136 parameters/4480 MACs; whole fixture models3273 versus3137 parameters. This is algebraic equivalence, not trained quality improvement.

Historical validation statistics are read from unchanged ablation-pilot-runs.json with run IDs and source hash, separately labeled. No new training, acquisition or held-out evaluation ran. README provides the diagram, conventions, exercises/solutions, failure cases and interpretation. Implementation/teaching deliverables complete; learner mastery and broader scientific superiority remain unassessed. The authoritative notebook record updates MP37 only; original central notebook unchanged.

# MP38 · Expanded architecture component verified; matched-budget research criteria pending

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP38 — Hybrid CNN–Attention–Transformer SR Model

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Architecture teaching component expanded and executed; matched-budget quality study and error atlas remain pending. This is not learner completion. Historical central notebook evidence is preserved.

## Learning objectives

- Integrated architecture and reconstruction.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP37 — Feature Fusion Laboratory](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP37_feature_fusion_laboratory/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Trace CNN extraction, attention gating, Transformer context, feature fusion, and residual upsampling in one reconstruction equation.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Planned; not run.** Plan an integrated hybrid against matched CNN and Transformer baselines on fixed pairs; deliver an architecture specification, quality/compute table, and error crops when implemented.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.
- Remote sensing: preserve CRS, affine transform, band meaning, nodata, physical units, and source licenses. Split by region/scene before tiling.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Project-local experiment, expanded notebook and exact commands are provided below; remaining scientific acceptance is explicit.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Hybrid research](https://arxiv.org/abs/2205.04437)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP38_hybrid_cnn_attention_transformer_sr_model/hybrid_cnn_attention_transformer_sr_model.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP38_hybrid_cnn_attention_transformer_sr_model/mp38_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP38_hybrid_cnn_attention_transformer_sr_model/test_mp38_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP38_hybrid_cnn_attention_transformer_sr_model/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP38_hybrid_cnn_attention_transformer_sr_model/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.


## Executed architecture lesson — 2026-10-27

[experiment.py](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP38_hybrid_cnn_attention_transformer_sr_model/experiment.py) and [architecture report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp38-architecture.json) trace actual shared `rssr.models.SRModel` modules, without a second model implementation. CPU float32, seed38, two threads; RGB and four-band synthetic1xBx5x7 inputs. No image download, optimizer step, checkpoint selection or test evaluation. Feature values are arbitrary learned units; input fixture values are synthetic0..1, not calibrated imagery.

Let F=H(x) be the3x3 head output withC channels. Local branch L=F+CNN(F); context branch T=WindowBlock(F). Fuse Z=Conv1x1([L;T]) then apply channel/spatial attention A(Z). The residual is R=PixelShuffle2(Conv3x3(A(Z))); optional LR-only sigmoid gate g(x) is nearest-upsampled. Reconstruction is y=bicubic2(x)+upsample2(g(x))*R. Without the optional gate, its factor is1. The attention block reweights features; it is neither calibrated uncertainty nor a causal explanation. Window attention uses replicated padding and restores the original feature grid.



For B=4,C=8,H=5,W=7: head/local/context are1x8x5x7, concatenation1x16x5x7, projection1x8x5x7. The tail convolution produces1x16x5x7 and PixelShuffle gives1x4x10x14. Gate output before upsampling is1x4x5x7. PixelShuffle changes layout: each low-resolution location supplies four subpixel positions per output band; it does not prove new detail.

### Initialization and gradients

If tail weights/bias are zero, R=0 and output exactly equals bicubic, whatever the gate values. d(y)/d(g)=R, so the gate initially receives zero gradient, while the tail can receive nonzero gradients. Zero gate gradient at this initialization is therefore expected, not proof of a broken optimizer. Worked scalar example: baseline.4, residual.1, gate.5 gives.45; zero residual gives.4. No clipping is hidden in this equation.

### Practice and solutions

1. How many pre-shuffle channels for B=3 and scale2? **Answer:**3*2^2=12.
2. Does a high gate value certify accuracy? **Answer:**no; it is a learned multiplier without uncertainty calibration.
3. Can a zero-initialized residual establish SR improvement? **Answer:**no, it reproduces the interpolation baseline by construction.
4. Why is the historical hybrid/CNN table not the planned matched-budget comparison? **Answer:**architecture parameters/training controls differ; use the preregistered v2 accounting and independent regions before making that claim.

### Verification and remaining scientific work

Run from root: `python -m pytest -q stage_09_hybrid_research/MP38_hybrid_cnn_attention_transformer_sr_model`, then `python stage_09_hybrid_research/MP38_hybrid_cnn_attention_transformer_sr_model/experiment.py`, then the local runner. Tests cover dimensions, finite gradients, hook cleanup on failure, restored train/eval state and bicubic identity. Shape traces are actual forward-pass observations; parameter totals are implementation counts, not latency/FLOP measurements.

[Existing paper tables](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/paper/generated/results.tex) keep historical quality evidence separate. The planned matched-budget CNN/Transformer study, reference-error crops and complete quality/compute comparison remain unfinished under [protocol v2](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/protocol-v2.md). Do not mark MP38 complete from the architecture fixture alone. Synthetic tensors require no imagery license. No external paper reproduction or scientific novelty is claimed.

# MP38: integrated hybrid reconstruction

Trace the existing shared model rather than creating a second implementation. Features are NCHW float32. These synthetic checks are separate from the unfinished matched-budget research comparison. See [lesson](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP38_hybrid_cnn_attention_transformer_sr_model/README.md) for equations, diagram and exercises.

## Residual reconstruction

Output equals bicubic plus a gated residual. Zero tail implies zero residual, so the output equals bicubic exactly; this is an implementation check, not learned improvement. The sigmoid gate is not calibrated confidence.

## Repeat for RGB and four bands

The experiment has zero optimizer steps. It records shapes/counts and checks gradients; it does not produce accuracy or transfer evidence.

## Exercise and interpretation

With RGB and scale2, the pre-shuffle tail has12 channels. With baseline.4, residual.1 and gate.5, output is.45. Since dy/dg=R, the initial zero residual explains zero gate gradients. Neither a larger output nor a large gate certifies physical detail. Matched-budget comparisons and valid-reference error crops remain open; historical single-scene results cannot close those requirements.

## Recorded validation

# MP38 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 1 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `d5b0e1f2c242854575d54efff6c1b7515b4b410aa331b038b6b11430a8f30bb1`.

Commands from this directory: `python -m pytest -q test_mp38_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.


## Architecture component — 2026-10-27

Five tests passed7.85s; standalone experiment and expanded three-cell real Jupyter kernel passedexit0. Known local TCP warning remains. [Report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp38-architecture.json) records actual RGB/four-band module shapes,3524/3969 parameters, exact zero-residual bicubic identity, finite nonzero tail gradients and zero initial gate gradients. This is synthetic initialization behavior, not trained quality. No optimizer steps, data download or held-out evaluation. README adds the integrated equation, architecture diagram, worked example and solved exercises. Hooks are removed even on forward failure; model train/eval flag is restored.

MP38 remains In progress: matched-budget CNN/Transformer/hybrid quality comparison, reference error crops and complete quality/compute study remain under protocol-v2 gates. Original central notebook preserved; current project execution record updates MP38 only.

# MP39 · Expanded historical ablation analysis verified; geographic uncertainty pending

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP39 — Controlled Ablation Study

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Historical ablation analysis and numerical uncertainty lesson are expanded. Full geographic/matched-budget acceptance remains pending; learner mastery is unassessed.

## Learning objectives

- CNN, CNN + Attention, CNN + Transformer, full hybrid.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP38 — Hybrid CNN–Attention–Transformer SR Model](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP38_hybrid_cnn_attention_transformer_sr_model/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Estimate paired per-scene metric differences across variants; report seed variability and scene-level confidence intervals without pixel-level pseudo-replication.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Historical pilot available; broader research incomplete.** Reuse the24 recorded runs spanning eight variants and three seeds. Do not retrain or relabel the one-scene pilot as geographic validation. Matched-budget comparisons and regional uncertainty remain under protocol v2.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.
- Remote sensing: preserve CRS, affine transform, band meaning, nodata, physical units, and source licenses. Split by region/scene before tiling.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Project-local experiment, expanded notebook and exact commands are provided below; remaining scientific acceptance is explicit.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Hybrid research](https://arxiv.org/abs/2205.04437)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP39_controlled_ablation_study/controlled_ablation_study.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP39_controlled_ablation_study/mp39_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP39_controlled_ablation_study/test_mp39_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP39_controlled_ablation_study/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP39_controlled_ablation_study/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.


## Executed evidence analysis — 2026-10-27

[experiment.py](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP39_controlled_ablation_study/experiment.py) reads immutable ablation-pilot-runs.json, checks run identity, seeds, source manifest and parameter consistency, and writes a derived [table/report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp39-ablation-analysis.json). Source hash and run IDs make the lineage explicit. The pilot uses100 training steps per run, three seeds17/23/41 and one validation scene. Counts/budgets are displayed rather than described as matched. No new data, training or inference runs here.

### Estimands and units

For a model's seed scores q1..qK, mean q=(sum qk)/K and sample seed SD=sqrt(sum(qk-mean q)^2/(K-1)). Scores are dB; a difference from bicubic is subtraction in dB, not a percentage. For region j, a scientifically paired effect would be d_j=mean_seed(model_j)-mean_seed(control_j). Geographic uncertainty resamples independent regions j, keeping model/control paired. Three random initializations on one scene do not create three regions. Thus this pilot's geographic CI is explicitly null.

Worked seed example[1,2,3]dB gives mean2 and sample SD1; paired against baseline2 gives mean delta0. Separate synthetic scene example A=[2,3,4],B=[1,2,3] has constant effect1, so every bootstrap resample gives1. Its interval[1,1] is a mathematical check, not confidence about satellite reconstruction.

### Controls and negative findings

Compare CNN, CNN+attention, CNN+window and hybrid within the same historical protocol, then inspect addition/concatenation, constant attenuation, learned gate and consistency controls. The report retains every variant, including weak outcomes. The learned-versus-constant gate difference and consistency-versus-hybrid difference are descriptive contrasts. A constant residual multiplier can explain much of the apparent gate benefit; unequal parameters and short training prevent architectural superiority claims. Existing bicubic performance remains a relevant baseline even when learned components underperform it.

### Exercises with solutions

1. May sixteen tiles from one scene be bootstrapped as sixteen independent regions? **No.** Shared scene properties violate the intended geographic sampling unit.
2. Does a negative effect mean the component can never help? **No.** It describes the tested budget/data/protocol; retain the null/negative result without universal extrapolation.
3. Why reject repeated run IDs? **Because** duplicating observations silently changes weights and can make uncertainty appear smaller.
4. What would make the next comparison stronger? **Answer:** independent geographic groups, validation-only selection, matched parameter/MAC budgets, multiple seeds and the frozen v2 analysis rules before test access.

### Commands and acceptance

From repository root run `python -m pytest -q stage_09_hybrid_research/MP39_controlled_ablation_study`, the local `experiment.py`, then `run_notebook.py`. Source images and licenses are inherited from the original NAIP manifest; this derived table does not redistribute imagery. The toy bootstrap uses fabricated numbers explicitly labeled synthetic.

Historical table/negative-finding interpretation is now implemented; region-level uncertainty and matched-budget evidence remain unavailable. Status stays In progress until remaining scientific criteria are resolved. See [protocol v2](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/protocol-v2.md) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP39_controlled_ablation_study/VALIDATION.md).

# MP39: controlled ablation analysis

Reuse the historical24-run pilot; no retraining or frozen evaluation. Seed variation describes optimization sensitivity, not geographic uncertainty. See [lesson](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP39_controlled_ablation_study/README.md) for equations and controls.

## Negative findings and attenuation

Compare descriptive effects against a simple control rather than attributing all gain to architectural novelty. The parameter counts differ and the pilot has one scene.

## Synthetic uncertainty check

A=[2,3,4], B=[1,2,3] has constant paired effect1. This is a toy set of three independent units by construction, not the pilot's three seeds. A one-scene geographic CI must be refused.

## Exercise and solution

Why not use tiles or seeds to inflate n? They are not independent geographic regions. A short-budget negative finding must be retained, but cannot prove universal inferiority. The next study needs region grouping, matched budgets and validation-only selection under the already registered v2 protocol.

## Recorded validation

# MP39 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 1 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `9bd0a1920d39fa5cb545ba0eae9ae6161fb955ce50c26d6f35df36dd14db5194`.

Commands from this directory: `python -m pytest -q test_mp39_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.


## Expanded historical analysis — 2026-10-27

Seven tests passed2.35s; experiment and expanded three-cell real kernel passedexit0 (known TCP warning). [Derived report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp39-ablation-analysis.json) preserves all24 historical rows via source hash/run IDs, eight variants and three seeds. Geographic CIs remain null; synthetic constant-effect bootstrap is separately labeled. Duplicate seeds/run IDs, mixed manifests, changing parameter counts and nonfinite PSNR are rejected. No new training or frozen evaluation. README/notebook include formulas, controls, negative findings and solved exercises. MP39 remains In progress: independent-region uncertainty and matched-budget acceptance remain unfinished.

# MP40 · Expanded historical robustness/efficiency analysis verified; real cross-sensor/full compute acceptance pending

Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.

# MP40 — Robustness and Efficiency Benchmark

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/README.md) · [Stage overview](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/README.md) · [Progress tracker](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md)

Historical robustness/efficiency analysis expanded; real cross-sensor and complete compute/memory acceptance remain open. Learner mastery is unassessed.

## Learning objectives

- Blur, noise, cross-sensor shift, parameters, FLOPs, inference time and GPU memory.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP39 — Controlled Ablation Study](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP39_controlled_ablation_study/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Define severity-dependent metric changes, parameter counts, declared FLOP conventions, latency distributions, and peak memory measurements.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Historical synthetic sweeps/CPU benchmark available; cross-sensor study incomplete.** Reuse the existing development benchmark; derive curves/tables without re-running inference or claiming GPU results.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.
- Remote sensing: preserve CRS, affine transform, band meaning, nodata, physical units, and source licenses. Split by region/scene before tiling.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Project-local experiment, expanded notebook and exact commands are provided below; remaining scientific acceptance is explicit.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Hybrid research](https://arxiv.org/abs/2205.04437)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP40_robustness_and_efficiency_benchmark/robustness_and_efficiency_benchmark.ipynb), [local tools](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP40_robustness_and_efficiency_benchmark/mp40_tools.py), [tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP40_robustness_and_efficiency_benchmark/test_mp40_tools.py), [runner](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP40_robustness_and_efficiency_benchmark/run_notebook.py) and [validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP40_robustness_and_efficiency_benchmark/VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:



File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.


## Executed analysis and curve interpretation — 2026-10-27

[experiment.py](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP40_robustness_and_efficiency_benchmark/experiment.py) derives [30 condition rows and5 efficiency records](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp40-benchmark-analysis.json) from unchanged development-benchmark.json, pinned by canonical hash. [Robustness curves](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/figures/mp40-robustness.svg) show RGB/four-band protocols separately. Each point is the mean of four scene-paired model-minus-bicubic PSNR differences. Lines connect only recorded conditions; they are not continuous predictions. Selected best-validation seeds introduce optimism; these are development results, not fresh test evidence.

![Historical robustness curves](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/figures/mp40-robustness.svg)

Blur sweep uses sigma_HR=.4,.8,1.2,1.6 pixels with noise0. Noise sweep uses sigma_HR=.8 and noise SD0,.005,.01 in normalized-DN units. These two sweeps change one nuisance parameter at a time; do not connect unlike conditions on an unlabeled severity axis. Blur/noise simulation does not reproduce sensor spectral response, registration error or acquisition-time changes. Benefits near the training degradation can disappear under mismatch. Negative values mean bicubic has higher mean PSNR for that condition; no result is omitted for being negative.

## Mathematics and measurement conventions

For scene j and condition c, d_j(c)=PSNR_model,j(c)-PSNR_bicubic,j(c). Report mean_j d_j(c). A model can lose raw PSNR under noise while its relative delta changes differently because the baseline also degrades. Example: model30->25dB, baseline28->26dB yields delta+2->-1dB. A dB difference is not percent improvement.

Latency samples t1..tK are elapsed forward times after warmup. Median is a typical sampled latency; p95 describes a sample upper quantile, not a worst-case guarantee. The historical record uses CPU float32, two threads, batch1,64-square inputs,5 warmups and20 repetitions, excluding IO/tiling. Do not compare this with web end-to-end response time. An illustrative2ms versus4ms gives2x speed ratio only with equal workload; it does not imply2x reconstruction quality.

Parameters count stored trainable scalars. The recorded conv/linear/attention MAC subset uses2 counted FLOPs per MAC and explicitly excludes normalization, activations, pooling, softmax, interpolation and elementwise operations. It is NOT total FLOPs. Sampled process peak RSS includes Python/caches and can miss peaks; it is NOT allocator peak or model-only memory. GPU memory is unavailable because no GPU ran. Unknown measurements remain unavailable, never zero.

## Exercises and solutions

1. Why separate3-band and4-band rows? **Answer:**different data/metric and workload contracts make pooled rankings misleading.
2. Can low CPU median prove a free host completes65536-square output? **No.** Workload, IO, tiling, memory and host limits differ.
3. Can mild synthetic blur establish real cross-sensor robustness? **No.** It tests only that declared degradation family.
4. Why preserve negative delta points? **Answer:**they reveal degradation mismatch and prevent favorable-condition selection.
5. Should missing GPU memory be displayed0? **No.** Use unavailable/not applicable and state no GPU used.

## Run and remaining acceptance

Run the local tests, `experiment.py` and `run_notebook.py` from repository root as for earlier projects. The experiment reads existing reports and generates SVG; it does not acquire imagery, train or time models again. `benchmark_cpu` remains a separately tested timing helper, not an automatically rerun research benchmark. Historical imagery provenance/licenses stay with original manifests; this report redistributes derived numbers only.

Partial implementation acceptance: derived curves/table, source lineage, numerical tests, equations/exercises and real kernel. Full MP40 remains In progress for real cross-sensor shift, complete operation accounting and appropriately scoped peak memory evidence. No GPU is required merely to fabricate a missing GPU row. [Validation](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP40_robustness_and_efficiency_benchmark/VALIDATION.md) records actual checks; [protocol v2](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/protocol-v2.md) governs new studies.

# MP40: robustness and efficiency

Reuse existing development evidence. No new inference, training or GPU benchmark. Curves keep RGB/four-band protocols and blur/noise axes separate. See [lesson](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/stage_09_hybrid_research/MP40_robustness_and_efficiency_benchmark/README.md).

## Historical curves

Vertical value is mean scene-paired model-minus-bicubic PSNR in dB. Negative means the baseline is better in this recorded condition. Lines only connect sampled severities; this is not real cross-sensor validation.

## Efficiency table and limits

Convert seconds to milliseconds for display, retain band count/input shape. Sampled process RSS is not model-only or allocator peak. Counted MACs exclude operations listed in each source row.

## Exercise and solution

Model30->25dB, baseline28->26dB gives relative effect+2->-1dB. Do not convert dB to a percent score. Neither synthetic degradation curves nor local CPU latency establish physical detail, real sensor transfer or hosted large-job feasibility. Missing GPU/allocator measurements remain unavailable.

## Recorded validation

# MP40 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 1 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `75c5ebd18f2dcd51e4558d5c866cfaefa382a966281ea69c1b7d80a05e831723`.

Commands from this directory: `python -m pytest -q test_mp40_tools.py` and `python run_notebook.py`. Records: [folder tests](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-folder-tests.json), [kernel execution](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.


## Expanded historical benchmark analysis — 2026-10-27

Five tests passed4.84s; derived report/figure generation and three-cell real kernel passedexit0 (known TCP warning). [Report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/mp40-benchmark-analysis.json) preserves source hash,30 robustness conditions and5 efficiency records from unchanged development-benchmark.json. No model benchmark, training, acquisition or frozen inference was rerun. [Figure](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/reports/figures/mp40-robustness.svg) was visually checked via a rendered PNG: readable units, separate band/condition panels, consistent model colors and negative results retained. PNG preview is local temporary output; SVG is reproducible from the script.

Tests validate paired-first aggregation, band separation and rejection of duplicate/nonfinite scenes/conditions. Existing timing helper test verifies call counts and train/eval restoration. README/notebook explain dB contrasts, selected-seed optimism, median/p95, partial MAC/FLOP counting and sampled RSS limitations. MP40 remains In progress: real sensor shift, full operation counts and scoped allocator-peak evidence remain unavailable; no GPU measurement was invented.
