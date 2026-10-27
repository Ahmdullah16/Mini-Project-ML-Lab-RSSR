# Scientific Sentinel-2 surface-reflectance admission

Status: local conversion/masking/export component implemented and fixture-tested. Live spectral acquisition, provider metadata reconciliation, quality-layer alignment and scientifically validated model inference remain unfinished. `fresh.py` continues the original uint8 visual RGB display workflow with item-level cloud filtering; it is not this scientific path.

## Radiometry and provenance

Use explicit per-band `reflectance = stored_value × scale + offset`. For original L2A encoding, the provider/product conversion may equivalently be `(DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE`. The actual metadata must determine both numbers. Do not derive a correction from acquisition date alone, add an offset after a provider already harmonized values, or assume every DN requires division by10000. Original product offsets and quantification values are documented by [Copernicus S2 Products](https://sentiwiki.copernicus.eu/web/s2-products). Consult actual product/asset metadata and reconcile provider transformations before admitting data.

The local API requires product/time/level/baseline/processor/provider/licence, radiometry source/review, harmonization status, mask source, band names/wavelengths/native resolutions/units, scale/offset/nodata/source asset/resampling, and common grid declarations. No scale/offset defaults are provided. These are caller declarations, not automatic proof of calibration. Unknown harmonization, L1C, display RGB, already-converted records and nonidentity conversion on decoded reflectance are rejected. Original records are copied, not mutated. Integer source samples are required for encoded DN. A converted output is tagged to reject a second conversion.

Example: DN2000, scale0.0001 and offset−0.1 yield reflectance0.1. DN500 yields−0.05 and DN15000 yields1.4. Finite values outside[0,1] are retained; display stretching/clipping must be a separate derivative. L2A is already atmospherically corrected; this layer never applies Sen2Cor. L1C needs a separately documented correction workflow before admission.

## Mask and grid policy

The conservative policy accepts SCL4/5/6 only and intersects every band's explicit valid mask, excludes any band's saturation mask, nodata and nonfinite samples. Clouds/cirrus/shadows/snow, no-data/defective and uncertain/dark classes are rejected by this policy. Classification is not guaranteed cloud-free truth. Quality mask origin must be recorded; quality flags and native processing are discussed in [Copernicus S2 Processing](https://sentiwiki.copernicus.eu/web/s2-processing).

All spectral and quality layers must declare the same CRS, affine transform and shape. Mismatches fail; there is no implicit interpolation. This deliberately leaves mixed10/20/60m alignment to a reviewed acquisition adapter. Upsampling a20m band must retain20m as its native resolution and record its resampling; it is not an independently observed10m band. SCL/quality masks need appropriate categorical/conservative alignment, not ordinary bilinear interpolation.

## API and observed tests

`rssr.surface_reflectance.prepare(values, scl, valid_masks, saturation_masks, metadata)` accepts bounded CHW inputs (1–13bands, each side≤1024), returns float32 values, a shared boolean valid mask, and provenance. Invalid pixels becomeNaN. `export(path, values, valid, metadata)` writes a float32 GeoTIFF with internal mask, CRS, transform, band descriptions and embedded provenance, then publishes by an atomic no-overwrite hard link. It does not export a super-resolved image or claim native resolving power.

24tests passed0.77s, including offset arithmetic, all12SCLcodes, mask/nodata/saturation intersection, L1C/unknown metadata rejection, decoded reflectance double-scaling rejection, mismatched grids, no-valid-pixel rejection,1×7and1024×1024inputs,oversize rejection, and actual GeoTIFF float/mask/grid/provenance round-trip with overwrite prevention. Tests use synthetic fixtures only; no satellite asset was downloaded or calibrated. No trained reflectance-model inference or aligned-HR metrics are claimed.

Reproduce with `PYTHONPATH=src`: `python -m pytest tests/test_surface_reflectance.py -q`. Provider-specific STAC/XML ingestion, metadata discrepancy resolution, measured alignment, live quality masking and full pipeline validation remain next tasks. The advanced architecture family, new M2/M6 training and final40-project acceptance are separate unfinished requirements.
