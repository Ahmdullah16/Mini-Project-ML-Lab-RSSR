# MP33 — Satellite Preprocessing Pipeline

**Status: Complete (implementation and teaching deliverables)** · Learner mastery unassessed.

[Laboratory home](../../README.md) · [Stage overview](../README.md) · [Progress](../../docs/progress.md)

## Objective and prerequisites

Select bands, preserve invalid pixels, fit train-only normalization, and extract georeferenced patches. Prerequisite [MP32](../MP32_satellite_image_explorer/README.md). CHW arrays and per-band masks are explicit. All examples use synthetic fixtures or pinned NAIP aerial training data, not satellite generalization evidence.

## Mathematics

Training population statistics: μ=Σx/N and σ=√(Σ(x−μ)²/N). Normalize z=(x−μ)/σ; constant bands use scale1; empty valid bands fail. For [1,3,5], μ=3, σ=√(8/3), z≈[−1.224745,0,1.224745]. Invalid fills remain masked. Freeze statistics before validation/test. Crop transform T_crop=T_parent×Translation(column,row); retain CRS and band semantics.

## Experiment and outputs

[Experiment](experiment.py) verifies the training raster checksum, selects RGB+NIR, fits training-window statistics, produces float32 normalized data, and extracts 16 nonoverlapping valid32×32 patches from128×128 pixels. [Measured metadata sheet](../../reports/mp33-preprocessing.json) records statistics, every patch transform, mask fraction, source and environment. `python experiment.py` re-executes with local pinned data; `python run_notebook.py` executes the lesson using recorded evidence. Original rasters and frozen SR preprocessing are unchanged.

## Interpretation and limitations

Near-zero means and unit population standard deviations validate the transformation, not reconstruction accuracy. The real crop has no invalid pixels; synthetic tests exercise nodata rejection and discarded edges. Per-band z-scores alter spectral angles and are not calibrated reflectance or compatible replacement inputs for existing DN/10000 models. No cloud-mask validation or cross-sensor claim. MP34 uses explicit geometry and validity to construct HR/LR pairs.

## Completion criteria

- [x] Explain band ordering, masks, normalization and crop transforms with worked numbers.
- [x] Execute synthetic fixtures and a checksum-pinned training-raster experiment.
- [x] Test train-only fitting, frozen application, constant/empty bands, band order, patch rejection and coordinates.
- [x] Provide preprocessing specification and metadata/invalid-pixel checks.
- [x] Include exercises, solutions, source license, environment and limitations.
- [x] Link [validation](VALIDATION.md) and progress without claiming learner mastery.

## References

[Rasterio masks](https://rasterio.readthedocs.io/en/stable/topics/masks.html) · [Windowed reading](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html) · [Pinned manifest](../../configs/naip-pilot-manifest.json).

## Project-folder files

This folder now contains the [local notebook](satellite_preprocessing_pipeline.ipynb), [local tools](mp33_tools.py), [tests](test_mp33_tools.py), [runner](run_notebook.py) and [validation](VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:

```bash
python -m pytest -q test_mp33_tools.py
python run_notebook.py
```

File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.
