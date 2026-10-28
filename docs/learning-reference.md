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
