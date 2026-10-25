# MP32 — Satellite Image Explorer

**Status: Complete (implementation and teaching deliverables)** · Learner mastery unassessed.

[Laboratory home](../../README.md) · [Stage overview](../README.md) · [Progress](../../docs/progress.md)

## Objective and prerequisites

Interpret bands, spatial resolution, GeoTIFF metadata, affine coordinates and nodata before super-resolution. Prerequisite: [MP31](../../stage_07_transformers/MP31_transformer_based_sr_baseline/README.md). The openly licensed example is NAIP aerial imagery, explicitly not satellite evidence.

## Mathematics and conventions

Arrays use CHW band/row/column order. Pixel centers map through x=a(c+0.5)+b(r+0.5)+c₀, y=d(c+0.5)+e(r+0.5)+f₀. CRS units are preserved. With 10-m north-up pixels and corner (100,200), row2/column3 maps to (135,175). Per-band means include only masked-valid finite samples; empty means are unavailable. Nodata is not a cloud mask. The notebook includes hand examples, exercises and separate solutions.

## Executed experiment and outputs

Synthetic masked two-band fixture first; then a SHA-256-verified cached NAIP training raster, bounded 128×128 crop at column238/row238. [Experiment](experiment.py) records geometry, band statistics, license, source and environment in [measured JSON](../../reports/mp32-raster-explorer.json). [Figure](../../reports/mp32-band-views.png) compares RGB, NIR/red/green false colour and labeled band distributions. Reproduce with `python experiment.py` after restoring the pinned training raster; no automatic downloads. Run `python run_notebook.py` to execute the lesson from recorded evidence.

## Interpretation and limitations

Band means and histograms describe this crop only. NIR/red/green assigns NIR to screen red; color differences do not establish vegetation health or model accuracy. Fixed DN/10000 preview scaling is not verified reflectance. CRS/transform must change consistently with resampling; ×2 pixel density does not prove recovered physical detail. Acquisition and retrieval dates are different. This is historical aerial imagery, not fresh satellite acquisition, cloud-free evidence or cross-sensor validation. These conventions feed MP33 preprocessing and MP34 paired data.

## Completion criteria and validation

- [x] Explain affine coordinates, units, masks and band semantics with numerical examples.
- [x] Execute deterministic synthetic checks and pinned licensed-raster inspection.
- [x] Test coordinate centers/corners, window bounds, per-band nodata, nonfinite and empty-mask handling.
- [x] Produce metadata sheet, masked band views and measured distributions with limitations.
- [x] Record license, source checksum, environment, rerun commands, exercises and solutions.
- [x] Link [validation](VALIDATION.md) and progress; keep learner mastery unassessed.

## References

[Rasterio georeferencing](https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html) · [Rasterio masks](https://rasterio.readthedocs.io/en/stable/topics/masks.html) · [Pinned dataset manifest](../../configs/naip-pilot-manifest.json).

## Project-folder files

This folder now contains the [local notebook](satellite_image_explorer.ipynb), [local tools](mp32_tools.py), [tests](test_mp32_tools.py), [runner](run_notebook.py) and [validation](VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:

```bash
python -m pytest -q test_mp32_tools.py
python run_notebook.py
```

File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.
