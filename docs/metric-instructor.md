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
