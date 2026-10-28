# Verified implementation and experiment contracts

## Architecture operations

Source: `src/rssr/architecture_registry.py`, `src/rssr/models.py`; canonical hashes in source audit. Let input have shape N×C×H×W, feature width d and native scale s. Shared embedding E is a padded3×3 convolution C→d. The bicubic baseline uses align_corners=False. Reconstruction is padded3×3 convolution d→Cs² followed by PixelShuffle(s), returning N×C×sH×sW. The residual is added to the bicubic baseline.

C is the additional local residual path L(f)=f+Conv3×3(ReLU(Conv3×3(f))). A first multiplies by a channel gate: spatial global average→1×1(d→max(1,floor(d/4)))→ReLU→1×1→sigmoid. It then forms channel mean/max of the channel-reweighted features, concatenates those2maps, applies7×7 convolution→sigmoid and multiplies again. Global pooling changes with tile context.

T replicate-pads height/width to multiples of window side w. It partitions windows into w² tokens of width d, adds a learned position tensor, applies LayerNorm and two-head self-attention, then a residual addition. A second residual adds LayerNorm→Linear(d→2d)→GELU→Linear(2d→d). Windows are reassembled and padding removed. This is fixed-window educational attention, not a shifted-window SwinIR reproduction.

| Registry | Feature path after E | Historical verified bindings |
|---|---|---:|
| M1 | L(f) |9|
| M2 | A(f) |0|
| M3 | T(f) |3|
| M4 | A(L(f)) |6|
| M5 |1×1 projection of concat[L(f),T(f)] |6|
| M6 | A(T(f)) |0|
| M7 | A(1×1 projection of concat[L(f),T(f)]) |15|

Counts describe exact recorded historical configurations. Parameter count must come from the chosen configuration; there is no universal parameter count per family. All paths retain shared convolutions. T already includes self-attention. Existing models.py RAM-assembling tiled helper must not be confused with the separate disk-backed large-output workflow.

## Training and checkpoint selection

`src/rssr/experiment.py` seeds deterministic Torch execution, uses Adam with recorded learning rate, samples training patches with replacement, and minimizes L1. Only configurations with positive consistency_weight add the implemented degradation-consistency L1 term. Do not attribute that term to all runs. Within-run best.pt is selected by minimum mean validation L1; final-evaluation.json records a separate later seed/configuration selection using validation PSNR. The fixed step budget is not convergence proof.

One inspected exact binding (run35931a32a3974ba2bdc5e212ee60b0fa): seed17,1000steps,batch4,Adam lr0.001,validation every100steps,consistency_weight0,4bands,native2,width16,window4,zero_residual=True. This configuration is illustrative of that executed run only; other rows retain their own full_training_config and selected_step. No new optimizer run was performed for this audit.

## Data, metrics and inference

`configs/final-evaluation.json` fixes Gaussian sigma0.8 plus area downsampling×2,16 deterministic nonoverlapping32×32HR patches from the original single test scene. Development geography is four train/four validation/one preserved test scene. Numerical DN/10000 scaling is not verified physical reflectance or atmospheric correction. Imagery licenses remain in the source manifests; do not transfer candidate dataset terms to NAIP without checking.

Metrics use range1,border2,no clipping,uniform7×7population SSIM and SAM radians with zero-norm spectra excluded. Per-tile scores are averaged within scene; PSNR is not recomputed from pooled MSE. No geographic CI from one test scene. MAE/global RMSE must only be added to manuscript tables when corresponding recorded values and aggregation are available; per-band RMSE does not automatically supply a differently aggregated global metric.

Input dimensions each≤1024; requested scales2,4,8,16,32,64 preserve rectangular/odd geometry. Cascades must be labeled; s² pixel growth is arithmetic, not resolving power. Host output cap1,048,576pixels;65536square output remains planned only. No new inference or frozen evaluation runs were required for these statements.
