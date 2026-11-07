# RSSR examiner and viva guide

Ahmdullah · Roll 264102307 · EE5302 L Introduction to Machine Learning Lab  
M.Tech Signal Processing and Machine Learning · Electronics and Electrical Engineering · IIT Guwahati  
8 November 2026

This guide explains the implemented project, its recorded results and how to demonstrate it accurately. Use the main 18 slides for the talk; slides 19–25 expand the seven model paths and slides 26–30 list the primary references. The central finding is bounded: the project implements and exercises compact CNN, attention and Transformer variants, but the recorded evidence does not establish hybrid superiority or calibrated cross-sensor generalization.

## Ninety second introduction

We study remote-sensing image super-resolution: predicting a denser image from a low-resolution input. Our model first builds a bicubic baseline. A learned path extracts features, processes them with CNN, attention or window-Transformer blocks, and predicts a residual correction. We add that correction to the baseline.

We implement seven combinations and a reusable inference workflow, with exact output dimensions, resource preflight, tiled disk-backed exports and provenance. The site also teaches 40 mini-projects through native demonstrations.

The experiments must be interpreted by protocol. On the frozen RGB study, the CNN and hybrid achieved 50.2614 and 50.2459 dB mean tile PSNR, compared with 48.5183 dB bicubic. These scores come from 16 tiles in one held-out scene. The early gate pilot did not beat bicubic. Our seven-model display comparison uses a small training example with unequal training budgets, so it is a software demonstration rather than an architectural ranking.

## Talk sequence

1. Slides 1–4: identify the task and define the seven feature paths. Explain that all families retain shared convolutional embedding and reconstruction.
2. Slides 5–9: explain local convolution, gates, window attention and PixelShuffle. Use the architecture appendix only if the examiner requests more detail.
3. Slides 10–11: explain scene splitting, synthetic degradation and reference metrics before presenting any score.
4. Slides 12–13: contrast the frozen RGB result with the earlier null result. Do not combine their tables.
5. Slides 14–16: present the actual demonstration, its limitations and output resource planning.
6. Slides 17–18: rehearse the app demonstration and state remaining scientific work.

Keep the 23 reference audit records available separately. Citation identity/concept verification is complete for that scope; full per-paper experimental extraction and replication are unfinished.

## Basic function and tensor flow

A function receives inputs, follows operations, and returns outputs. The model's `forward()` is the prediction recipe. Its input is an N×C×H×W tensor: batch size, channels, height and width. RGB display has C=3. Reflectance products can have different bands and units and must be processed through compatible scientific workflows.

The prediction is xhat=B_s(y)+R_s(G_i(E(y))). Here y is LR input; E embeds channels into d feature maps; G_i is the selected feature path; R_s reconstructs a residual at scale s; B_s interpolates the input to the same grid. Addition requires exactly matching shapes.

A trained checkpoint contains learned parameter values and configuration. Registering a model, finding a checkpoint, loading its weights, successfully executing inference and scientifically validating its accuracy are separate observations. A random-weight shape test proves only a limited software property.

## Convolution and the C path

A convolutional neuron forms z=sum(w_i*x_i)+b from a local receptive field and input channels. The same filter weights are reused at different positions. A3×3 kernel with d input and d output channels has 9 d²+d parameters, including biases.

The C block computes L(f)=f+Conv 2(ReLU(Conv 1(f))). ReLU returns max(0,z). There is no extra activation after the second convolution. The identity connection gives features a direct path around the learned correction. The local block has 18 d²+2 d parameters.

Increasing d provides more feature channels and increases many parameter/activation costs. It does not guarantee improved unseen-scene reconstruction. Kernel size, padding and stride determine receptive field and tensor shape; these convolutions use stride 1 and padding 1.

## Channel and spatial attention

The channel gate averages each feature channel over its spatial area. A small 1×1 convolution bottleneck and sigmoid produce weights, which multiply the features. The bottleneck has r=max(1,floor(d/4)) channels. Sigmoid(z)=1/(1+exp(-z)).

The spatial gate takes channel mean and maximum maps from the channel-gated features. It concatenates them, applies a 7×7 convolution and sigmoid, and multiplies again. Channel weights broadcast across space; spatial weights broadcast across channels.

These gates weight features. They are not semantic segmentation, uncertainty calibration or proof of which physical object the network understands. Global average pooling depends on the entire available tile, so attention outputs can change with tiling context.

## Window Transformer and head connections

Features are replicate-padded to a window multiple and partitioned into windows. A window of side w has L=w²tokens, each with d features. The code computes q=LayerNorm(z+P), where P is a learned positional tensor. Two-head self-attention uses q for Q/K/V inputs. Its output is added to original z. A second residual adds a LayerNorm–Linear–GELU–Linear feed-forward output. Reverse partitioning restores the image grid, and only the added padding is removed.

Each head projects tokens into queries, keys and values. Attention is softmax(QK^T/sqrt(d_h))*V, with d_h=d/2. Scores compare query and key features; normalization gives each query a row of weights summing to one. Values are mixed with these weights. Heads are concatenated and passed through an output projection.

This implementation has fixed windows and two heads. It does not reproduce the full shifted-window hierarchical SwinIR network. GELU appears in the feed-forward block; LayerNorm normalizes feature coordinates and has learned scale/bias. Larger windows increase attention-score memory approximately with L² per window.

## Fusion and the seven paths

M1 uses L(f). M2 uses A(f). M3 uses T(f). M4 uses A(L(f)). M5 projects concatenated L(f) and T(f) back from 2 d to d channels. M6 uses A(T(f)). M7 applies attention after that parallel fusion.

Transformer self-attention remains in T even when the additional A module is absent. “Transformer” also does not mean convolution-free: embedding and reconstruction convolutions remain in every family. The registry configuration at RGB3 bands, d=16,w=4,s=2 has parameter counts 6828,2435,4668,7075,9836,4915 and 10083 for M1–M7. These are untrained configuration counts, not every trained checkpoint's count.

## Reconstruction and scales

The reconstruction convolution maps d features into C*s² channels. PixelShuffle rearranges these channels into neighboring spatial locations and has no trainable parameters. Its output shape is N×C×sH×sW. The learned residual is then added to bicubic.

The input limit is 1024 per side; it is not an output limit. At requested scale s, Wout=s*Win and Hout=s*Hin. A256×128 input at×8 becomes 2048×1024. A1024-square input at×64 would become 65536-square, with 4,294,967,296 pixels and 12 GiB raw uint 8 RGB, before overhead. This extreme output was planned, not executed.

The UI offers×2/4/8/16/32/64. Larger scales use disclosed repeated native×2 steps. RGB stages clip and quantize between stages; this can accumulate artifacts. The free-host configured output limit is 1,048,576 pixels per model. Larger requests require an explicit local/Colab job and adequate disk/memory. A denser output grid does not establish increased physical resolving power.

## Metrics and recorded results

MSE is mean squared reference error; RMSE is its square root; MAE is mean absolute reference error. With declared range R, PSNR=10 log 10(R²/MSE). For errors[0.02,-0.02], MSE=0.0004, RMSE=0.02 and PSNR=33.98 dB when R=1. Zero MSE implies positive-infinite PSNR; it is not replaced by an arbitrary finite score.

SSIM compares local luminance, contrast and structure. The recorded implementation uses a uniform 7×7 window, population moments, K1=.01,K2=.03 and valid windows. It differs from the original Gaussian implementation. Edge differences describe spatial gradients; SAM measures spectral-vector angle only for compatible meaningful bands and valid nonzero vectors. SAM on normalized digital numbers is not proof of calibrated spectral fidelity.

The frozen RGB CNN and hybrid are close:50.2614 vs 50.2459 dB. One scene is not a geographic confidence interval. Official SwinIR has unmatched pretraining. Four-band and RGB protocols must not be combined into one ranking.

The early pilot's learned gate 45.2122 dB and fixed attenuation 45.0625 dB both remain below bicubic 45.8104 dB. Preserve this null result. Seed variation is not a substitute for independent geographic scenes.

The 14 new display requests reused seven trained checkpoints at×2 and×4 on one licensed 16×16 training crop. M1/M7 trained 1000 steps, M2–M6 trained 100 steps. The crop is dark and low in texture; it does not show convincing detail recovery. Heatmaps show RGB disagreement from Pillow bicubic, not HR error. There is no aligned HR reference for this demonstration.

## App demonstration and fallback

Open the existing Streamlit URL. Record whether the app loads and which revision it exposes. A wake/loading screen is not an inference result. Prepare the licensed input, inspect exact dimensions/resource preflight, choose an available trained RGB model and execute. Compare models on the same prepared pixels and show scale labels, runtime, provenance and downloads.

If network/hosting prevents the live run, state that limitation and present the recorded output bundle as precomputed evidence. Do not relabel it as a fresh public request. Preserve original model/scale labels when changing controls.

For Satellite Explorer, distinguish acquisition time from retrieval/refresh time. Check licensing, bands, CRS/transform, scale/offset, nodata/cloud provenance and atmospheric processing. A display RGB stretch is not scientific reflectance. No valid aligned HR means no reference-accuracy score. VENUS does not provide an eligible HR reference for Sentinel-2 B11/B12 merely because a dataset name matches.

Inputs, outputs and caches on stateless Community Cloud can disappear after restart. Local SQLite functionality and an offline-tested external adapter do not demonstrate live durable cloud storage. Keep credentials outside documents and screenshots.

## Short viva answers

**What is the contribution?** An integrated educational and engineering implementation with reproducible evidence and explicit limitations. Architectural novelty or superiority has not been demonstrated.

**Why use a residual?** It provides a simple interpolation baseline and lets the network predict a correction. It can simplify optimization but is not a guarantee of better quality.

**Does attention recover missing truth?** It changes feature weighting or token mixing. Truth recovery requires valid reference data and an appropriate independent evaluation.

**Why can a hybrid lose?** Extra capacity can overfit or be poorly optimized; unmatched parameters, training schedules and degradation can confound a comparison. The observed pilot null result is retained.

**Why not score against bicubic as HR?** Bicubic is an interpolation baseline. Difference from it is disagreement, not reference accuracy.

**What would establish generalization?** Independent regions, scene-before-tiling splits, train-only preprocessing, compatible paired sensors, matched budgets, multiple seeds, scene-level uncertainty and documented failure cases.

**Can this be used in industry now?** It can support teaching and exploratory visual inspection within its evidence scope. It has not been validated as an authoritative cadastral, physical measurement or scientific cross-sensor product.

## Source map and remaining work

Model operations: src/rssr/models.py and architecture_registry.py. Frozen scores: reports/final-heldout.json. Early pilot rows: reports/paper-evidence.json. Display runs: reports/seven-model-panels-20261108.json and its external output evidence. Citation identity/concept audit: reports/reference-audit-20261108.json. Acceptance: docs/acceptance_matrix.md.

Remaining work includes a broader source-bound failure atlas, full per-paper experimental extraction, renewed public all-model/satellite/mobile/export/restart checks, independent geographic/cross-sensor evaluation and configured durable external storage. Presentation completion does not close those gates.
