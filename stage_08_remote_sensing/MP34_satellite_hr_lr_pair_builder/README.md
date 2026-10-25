# MP34 — Satellite HR–LR Pair Builder

**Status: Complete (implementation and teaching deliverables)** · Learner mastery unassessed.

[Home](../../README.md) · [Stage](../README.md) · [Progress](../../docs/progress.md)

## Objectives and prerequisites

Construct reproducible aligned synthetic×2/×4 HR/LR pairs with explicit bands, masks, units and geographic transforms. Prerequisite [MP33](../MP33_satellite_preprocessing_pipeline/README.md). Scientific pair arrays remain distinct from clipped RGB display exports.

## Mathematics and controls

y=D_s(k*x)+n with CHW float input, Gaussianσ=.8 HR pixels (radiusceil(3σ), normalized kernel, replicate edges), area downsampling and optional seeded additive Gaussian noise. Recorded real pairs use noise0. Shapes must be divisible by s. T_LR=T_HR×Scale(s,s), retaining CRS and exterior corners. All HR values must be finite/valid; invalid patches are rejected before blur. No silent padding, cropping or clipping.

Hand example: [0,1;4,5] averages to2.5; a4×4 ramp averages to7.5 at×4. The lesson includes rotated-affine checks, center/corner distinctions and exercises with solutions.

## Executed experiment and deliverables

[Experiment](experiment.py) runs synthetic checks, verifies the training-source checksum and builds two pairs from a64×64 four-band NAIP aerial crop. HR2.5-m spacing becomes LR5m/10m; both retain a160×160-m footprint in EPSG:32611. [Pairing manifest](../../reports/mp34-pairing-manifest.json) records source/license, window, shapes, transforms, units, exact NPY hashes and environment. Rebuild ignored arrays with `python experiment.py`; run the lesson with `python run_notebook.py`. No data download or model training.

## Interpretation and limitations

Matching extents and shapes verify the synthetic geometry, not real sensor compatibility. Different bands, point-spread functions, timing, registration or calibration require a separate cross-sensor protocol. Fixed DN/10000 is not verified reflectance. Fully valid masks do not establish cloud-free observations. ×4 pairs cannot be fed silently to×2 models. This feeds MP35 geographic separation and the major project's declared synthetic pipeline.

## Completion criteria

- [x] Derive degradation, shape and transform equations with numerical examples.
- [x] Execute synthetic fixtures and pinned training-only×2/×4 pairing.
- [x] Check constants, area arithmetic, rotated corners, invalid pixels, shapes and seeded noise.
- [x] Deliver a pairing manifest with rebuildable artifact hashes and exact settings.
- [x] Record license, environment, interpretation, limitations and exercises/solutions.
- [x] Link [validation](VALIDATION.md) and progress without learner-mastery claims.

## References

[Shared degradation implementation](../../src/rssr/core.py) · [Rasterio georeferencing](https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html) · [Pinned manifest](../../configs/naip-pilot-manifest.json).

## Project-folder files

This folder now contains the [local notebook](satellite_hr_lr_pair_builder.ipynb), [local tools](mp34_tools.py), [tests](test_mp34_tools.py), [runner](run_notebook.py) and [validation](VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:

```bash
python -m pytest -q test_mp34_tools.py
python run_notebook.py
```

File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.
