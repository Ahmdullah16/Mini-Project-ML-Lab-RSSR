# Preliminary research report: reliability gating in a compact SR pilot

Status: executed engineering pilot, not publication-ready research. No novelty or state-of-the-art claim is made. This draft describes only locally measured results.

## Question and context

Can an LR-conditioned gate improve reconstruction while limiting unsupported detail? A hybrid architecture alone does not establish a contribution. The current experiment tests whether a learned residual gate differs from simple constant attenuation, before attempting geographic or sensor-shift claims. The literature review and verified bibliography are in docs/literature.md and references.bib; detailed overlap analysis remains pending.

## Data and protocol

Three geographically separated NAIP aerial scenes were retrieved through a pinned OpenSR release. They were assigned to training, validation and test before extracting nonoverlapping patches. This is synthetic single-image x2 super-resolution of real aerial imagery, not a paired satellite cross-sensor experiment. Stored uint16 values were divided by the fixed numerical constant 10000; physical reflectance calibration is unverified.

The pilot uses 16 patches of 32x32 HR pixels per split, Gaussian blur followed by average-pool downsampling, and no noise. Optimization uses Adam, learning rate 0.001, batch size four, 100 steps, and seeds 17, 23 and 41. Checkpoints are selected by validation L1 at 20-step intervals. Eight variants cover CNN, attention CNN, CNN plus Transformer without attention, full hybrid, averaged additive fusion, learned gating, constant 0.5 gating, and a consistency loss weighted by 0.1. All share the data and step budget; parameter counts differ and are disclosed.

The gate sees LR input only. Consistency uses the known synthetic forward operator and has no claim to model cross-sensor physics. The held-out test was not evaluated. PSNR uses data range one and a two-HR-pixel border crop. We average tile metrics on one validation scene, then summarize variability across seeds. Seed standard deviation is not a geographic confidence interval.

## Results

See ablation-pilot.md and ablation-pilot-runs.json for the complete table, run IDs, configurations and hashes. Mean PSNR was 43.6869 dB for the hybrid, 45.2122 dB for its learned gate, 45.0625 dB for constant attenuation, and 43.6734 dB with the consistency objective. Bicubic interpolation reached 45.8104 dB and exceeded every learned configuration in all three seeds.

Most of the apparent gating benefit is also present with a constant multiplier. This weakens any claim that learned reliability explains the gain. The consistency variant did not improve mean PSNR over the basic hybrid in this small study. These are negative or inconclusive observations, not evidence that gating or consistency cannot help under other conditions.

## Limitations and next experiments

The training budget is extremely small, validation contains one scene, and the selected first patches are visually simple. Metric values depend on the fixed numerical scale and cannot be compared directly with published protocols. The RGB demonstration checkpoint is separate and is not a scientifically validated model for arbitrary uploads. Global channel attention makes tiled inference approximate.

Before final evaluation, expand independent training/validation locations, inspect texture and land-cover diversity, verify sensor units and alignment, and measure spatial detail, spectral distortion and inference costs jointly. Compare against a strong official baseline and controls for residual initialization. Predeclare selection rules, then use paired independent-scene differences and confidence intervals. Cross-sensor, downstream, robustness, FLOP and peak-memory experiments remain pending. No final test result is reported.
