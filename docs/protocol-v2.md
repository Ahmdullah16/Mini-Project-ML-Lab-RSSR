# Geographic and cross-sensor evaluation protocol v2

Status: **design registered; not executed; manifest and resource approval gates open**. Date:2026-10-26. This document must be committed before any new test imagery or results are inspected. It does not retroactively preregister old observations. [Research questions](research_questions.md) define the claims. Any amendments require a new version and explicit timing relative to data access.

## Data and leakage controls

Keep the original frozen NAIP test and all its outputs unchanged; report it only as the existing single-scene result. Previously viewed train/validation regions are development evidence, not fresh confirmatory test units. Select new regions using metadata-only coverage and eligibility rules, independent of pixels, model outputs or favorable scores. Freeze source versions/checksums, region IDs, footprints, site/date grouping, licenses and exclusions before opening new test pixels.

Target five outer geographic folds with at least15 eligible independent regions (at least3 per outer test fold). This target is not a claim that these data are available. Group overlapping/nearby scenes and repeated site/date observations into a single region before assignment. Reject cross-fold overlap and enforce a declared5km footprint buffer using a suitable metric projection or geodesic method, not constant degrees. Deterministically order groups by SHA256 of `rssr-v2-2026-10-26:` plus stable region ID, then allocate round-robin. Commit the resulting manifest and fold checks before decoding held-out imagery. No suitable manifest means execution blocked, not a reduced undisclosed study.

Within each outer training partition, reserve deterministic region groups for inner validation (at least2); fit normalization and attenuation only on training data and select settings/checkpoints only on inner validation. Outer test regions cannot influence architecture, preprocessing, stopping or hyperparameters. Acquisition/site overlap with pretraining must be investigated and unresolved overlap disclosed. Cross-sensor sites form a separately frozen target manifest and are not pooled with NAIP.

## Models, controls and compute

Primary comparison: hybrid versus matched CNN; attention-only and Transformer variants are secondary. Include bicubic and a gain/offset control y_hat=a*bicubic(y)+b, with parameters fitted using training data only and regularization/settings selected on inner validation. No test-fitted calibration. Add the official native×2 SwinIR checkpoint as an explicitly pretrained external control; disclose DIV2K/domain/degradation differences and unmatched pretraining compute.

Record parameter counts, estimated MACs for identical input shapes, wall time, hardware, optimizer, losses, batch size and checkpoint hash. Target parameter counts within5% and MACs within10% for the matched architecture contrast; widths are selected analytically before training. If no eligible configuration meets both, do not call that comparison compute-matched. Report it separately. Proposed bounded pilot schedule: seeds17,23,41, Adam, L1,1000 steps, batch4, learning rate0.001, validation every100 steps, minimum validation L1 checkpoint (earliest step breaks ties). Large/final training must wait for an explicit estimated-resource approval; these settings are a proposed pilot, not an executed final study or convergence claim.

## Metrics and interpretation

Synthetic×2 comparisons reuse the documented sigma0.8 Gaussian/area degradation with no added noise, fixed range1 and two-HR-pixel border crop, but record exact implementation/version hashes. Cross-sensor metrics require a justified common support, registered grids, valid masks, declared calibration and data range in its manifest addendum; do not transplant NAIP assumptions. No silent interpolation to match native scales.

Primary: per-scene mean tile PSNR, then paired scene difference (model minus CNN). Secondary: scene-aggregated MSE/RMSE/MAE/SSIM and model-minus-bicubic differences, with all settings inherited from the recorded [metric protocol](protocol.md) or explicitly amended before test access. Percentage RMSE improvement is100*(baseline-model)/baseline; zero baseline yields unavailable. Never percentage-normalize dB.

Planned additional diagnostics: Sobel-gradient error on the same valid aligned support (normalized-value/pixel units); radial power spectrum after per-channel mean removal and a separable Hann window, frequencies in cycles/HR pixel, with normalization and bin definitions frozen in code before testing. These are secondary diagnostics, not evidence of semantic detail recovery. SAM uses meaningful compatible calibrated spectra with documented zero-vector exclusions; per-band RMSE retains its units. Where only normalized DN is defensible, label it numerical rather than physical spectral evidence. Do not add LPIPS to scientific rankings. Downstream evaluation is gated on independent suitable labels.

## Uncertainty and multiplicity

For each model and scene, average its three seed-specific metrics first; report seed variability separately. Form paired differences on identical held-out scenes. Bootstrap scene/region units with replacement10,000 times using RNG seed20261026, retaining paired model values, and report the percentile95% interval of the mean difference. Never use patches or seeds as independent sampling units. Report raw paired mean effect and median, scene count, fold composition and sign pattern. Shared training sets across folds induce dependence; intervals are descriptive and can be optimistic. If eligibility leaves too few independent scenes, label estimates exploratory rather than silently changing the design.

Only hybrid-versus-matched-CNN mean PSNR is primary; other architectures/metrics are secondary exploratory comparisons. No post-hoc selection of best seed, fold or failure-free subset. Retain failed/unfinished runs with reasons; do not impute favorable scores. Performance per second and parameter tradeoffs must accompany any quality claim.

## Failure atlas and publication

Before test access, define atlas categories: edge halos, repetitive texture, vegetation, water/riverbanks, radiometric/clipping errors and registration mismatch. Select the worst and median scene by primary paired difference plus predetermined ROI coordinates for each category where metadata supports it; publish selection rules, not hand-picked successes. Use valid-reference error maps with units/colorbars; without HR, label maps disagreement. Provide aligned crops and identical contrast. Generated enhancements cannot be HR targets.

## Execution gate checklist

- [ ] Official dataset/version/license and feasible download size verified.
- [ ] User approval for large transfer/long training obtained with explicit estimate.
- [ ] New region/site manifest, hashes, folds and overlap/buffer checks committed before imagery inspection.
- [ ] Band/calibration/registration addendum and metric code tests frozen.
- [ ] Matched model parameter/MAC accounting and actual resource pilot recorded.
- [ ] Runs executed and reconciled with immutable report IDs.

Until these gates pass, no v2 accuracy result or completion claim is permitted. Reuse prior results only when their data, units, split and protocol match; otherwise show separate contextual tables.
