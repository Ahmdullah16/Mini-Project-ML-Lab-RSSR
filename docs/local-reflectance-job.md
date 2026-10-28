# Local scientific reflectance job

Status: executed offline integration on synthetic GeoTIFF fixtures. This is an executable local step toward the scientific Sentinel-2 workflow, not a live acquisition adapter or a super-resolution model.

Use the existing environment with `PYTHONPATH=src`:

```text
python -m rssr.cli prepare-reflectance request.json reflectance.tif
python -m rssr.cli prepare-reflectance request.json reflectance.tif --execute
```

The first command prints the exact pixel window, output shape/grid, bytes to hash, raw float output bytes and a conservative array-memory allowance. It does not read pixel arrays, verify full-file hashes or create output. Header reads still perform local I/O. The second command prints the same plan before execution, verifies each declared SHA256, reads only the requested windows, converts once and exports an atomic no-overwrite masked float32 GeoTIFF. It does not perform atmospheric correction, interpolate grids, download imagery or run learned super-resolution.

## Request structure

`request.json` contains:

- `window`: `[column_offset, row_offset, width, height]`, explicit nonnegative integer offsets, dimensions1–1024. Out-of-bounds requests fail; no silent crop.
- `band_assets`: ordered list of1–13single-band local GeoTIFF assets. Each asset has `path` (relative to the request file or absolute) and lowercase `sha256`. Stored dtype must be uint16/int16/float32/float64; uint8 display sources are rejected.
- `scl_asset`: one-band uint8 SCL on the same full raster grid.
- `validity_asset`: uint8 raster with one band per spectral band;1means valid,0invalid.
- `saturation_asset`: uint8 raster with one band per spectral band;1means saturated,0not saturated. Neither mask accepts255as a substitute for boolean1. Set each raster's own nodata/coverage independently; do not mark valid zero-valued saturation pixels as nodata.
- `metadata`: the explicit product and band fields documented in [scientific Sentinel admission](scientific-sentinel.md). Omit grid fields to derive them from the actual cropped raster headers, or supply identical declarations; a conflicting declaration fails. Spectral nodata must agree with raster metadata. Original asset URL, wavelength, native resolution, provider scale/offset and review fields remain mandatory declarations, not inferred from filenames.

A compact structural example (replace asset paths and checksums with real reviewed inputs; these placeholders are not executable data):

```json
{
  "window": [1, 1, 3, 2],
  "band_assets": [{"path": "B04.tif", "sha256": "<actual SHA256>"}],
  "scl_asset": {"path": "SCL.tif", "sha256": "<actual SHA256>"},
  "validity_asset": {"path": "valid.tif", "sha256": "<actual SHA256>"},
  "saturation_asset": {"path": "saturated.tif", "sha256": "<actual SHA256>"},
  "metadata": {"product_id": "<reviewed product ID and remaining required fields>"}
}
```

All files must be self-contained GeoTIFFs, without unpinned external mask/auxiliary sidecars, on exactly the same full CRS/transform/width/height grid. Mixed native-resolution products are deliberately refused until a documented alignment step is implemented. File checksums pin inputs; they do not prove provider radiometry or mask accuracy. Do not change source files during a job.

## Values, masks and resource policy

The existing conversion computes dimensionless surface reflectance as `stored × scale + offset`, keeps finite negative and above-one values, and masks invalid pixels withNaN. The new reader intersects spectral validity, explicit quality validity, coverage masks from all quality rasters and saturation before that conversion. SCL4/5/6is the conservative accepted subset. It records the exact window, request checksum and each source checksum inside the output provenance.

Local source verification is capped at512MiB summed asset sizes; this bounds full-file hashing I/O, not downloads (there are none). Estimated working memory is64bytes per output band/pixel plus64MiB; this is a configured conservative allowance, not a measured RSS peak or complete GDAL guarantee. Execution checks free output space for raw float pixels plus32MiB reserve. Output TIFF/metadata overhead can vary. Very large jobs need a separate reviewed implementation; no scale or pixel dimensions are reduced.

## Actual evidence

`tests/test_reflectance_job.py` executed the existing CLI in fresh subprocesses: plan-only created no output; `--execute` produced the requested two-band3×2window with reflectance0.1from fixture DN2000, scale0.0001and offset−0.1. Source origin(500000,3000000),10m pixels, column1/row1 yielded output origin(500010,2999990). Quality coverage excluded one of six pixels in a separate case. These numbers are synthetic checks, not satellite observations.

Combined local-job and conversion suite:36tests passed2.34seconds. It includes checksum mismatch/no output, four invalid windows, shifted quality-grid rejection, nonbinary quality rejection, quality-coverage intersection, external-sidecar rejection, disk guard and duplicate-output refusal. [Validation record](../reports/reflectance-job-validation.json) pins source hashes and observed command/outcome.

Provider-specific STAC/XML reconciliation, explicit mixed-resolution mask alignment, actual product acquisition, calibrated model compatibility, real cross-sensor evaluation and website scientific controls remain unfinished. Keep these separate from the working RGB display demo and from the seven-model educational registry.
