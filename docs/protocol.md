# Protocol synthetic-naip-v1

This is an engineering pilot on three real aerial images with synthetic LR inputs.
It is not a validated satellite cross-sensor experiment or an OpenSR leaderboard run.
Using benchmark HR imagery for this pilot disqualifies these scenes from a future
independent OpenSR test set for these trained weights.

Freeze scene assignments in configs/naip-pilot-manifest.json before patch extraction.
One training, one validation and one test scene are deliberately tiny. Read GeoTIFF
CRS/bounds and reject cross-split region reuse or WGS84 bounding boxes separated by
less than 0.02 degrees. This angular buffer is not a fixed distance and is unsuitable
near poles/antimeridian without a better projected-region policy.

Input: CHW RGBNIR stored uint16 values divided by a fixed 10,000, not reflectance. Do not apply
NDVI or claim reflectance consistency to these digital numbers. Validity requires
all four bands finite and unmasked. Only fully valid, nonoverlapping 32x32 patches
are accepted, up to 16 per scene in deterministic raster order. This is not a
representative random sample. Gaussian sigma=.8 HR pixels, radius=ceil(3 sigma),
replicated boundary and area downsampling produce x2 LR; no noise in pilot training.
There is no registration estimation: HR and synthetic LR share a known grid.

Train L1 with Adam, fixed learning rate .001, batch 4, seed 17, no augmentation,
no pretraining, float32 CPU, two threads. Smoke=2 steps, pilot=100, research template
=10,000 (not automatically executed). Validate every 20 steps or at final step.
Pick lowest validation L1; keep held-out test untouched until model selection is
recorded. Resuming saves optimizer, data-sampling RNG and Torch RNG.

Evaluation: no Y-channel conversion; all bands; no clipping before metrics; range=1;
border crop equals scale; per-tile MSE/RMSE/PSNR; SSIM uniform 7x7 population moments,
K1=.01/K2=.03 with only fully valid windows; SAM radians excludes zero-norm spectra.
Aggregate tile scores to scenes before comparing scenes. One validation scene cannot
support a credible geographic confidence interval. Undefined SSIM/SAM is null;
zero-error PSNR is infinite. JSON export of infinity requires downstream handling.

Classical baselines use PyTorch nearest/bilinear/bicubic, align_corners=False for
linear/cubic. They share the same pairs and borders. Educational neural models are
not SRCNN/RCAN/SwinIR reproductions. Parameter and training-step differences must
be reported. Exact MAC/FLOP accounting remains pending; do not invent totals.

Candidate hypothesis: an LR-only gate on the predicted residual, with a justified
synthetic consistency objective, may reduce spectral error under shift. This is
unproven. Existing attention, residual attenuation, blind degradation conditioning
and HAT/DiffFuSR overlap prohibit a novelty claim from architectural combination.
Before research: compare fixed scalar gate, learned gate, no gate, consistency off/on,
CNN, CNN+attention, CNN+window, full hybrid and addition/concatenation. Select weights
on validation only. Declare a spatial detail budget (e.g. no more than 0.1 dB PSNR
loss) and latency budget before testing; these are proposed thresholds, not results.

Expand only after a pilot demonstrates trainability. Use at least three independent
regions per evaluation group and three seeds where feasible. Bootstrap paired
scene differences, not pixels. A stronger pretrained baseline must disclose its
training data and pretraining. Freeze decisions in a dated file before test access.
