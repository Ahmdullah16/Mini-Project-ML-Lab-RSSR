# MP33 validation — 2026-10-25

Complete implementation and teaching deliverables; learner mastery unassessed.

`python -m pytest -q stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline`: 3 tests passed (3 affine deprecation warnings). Tests cover finite/nodata means, training-only fitting, fixed-stat application, constant/empty bands, zero-based band ordering, invalid patch rejection, full-patch edges and translated affine coordinates.

`python stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/experiment.py`: synthetic hand checks and pinned training-raster experiment passed. Sixteen valid32×32 four-band patches; normalized means within7.5e-9 of zero and standard deviations within6e-8 of one. These verify normalization, not SR quality. No download, training, validation/test fitting or frozen-result changes.

`python scripts/execute_project_notebooks.py --only MP33`: real kernel passed 5 code cells. Source SHA-256 `6f5c8b1412447204238645896f2fb8514cd45edcc2dd4847d22e6e86fa6328f0`. Notebook includes numerical examples, recorded metadata, exercises/solutions and interpretation.

[Measured record](../../reports/mp33-preprocessing.json) · [Kernel evidence](../../reports/project-notebook-execution.json). The data are cached NAIP aerial imagery, not a fresh satellite observation. Per-band standardization alters spectral angles and is not inserted into existing model inference. CRS, masks, license, checksum and environment are retained in the record.
