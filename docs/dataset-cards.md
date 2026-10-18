# Source review, checked 2026-10-18

## Pilot: OpenSR NAIP subset

Official distribution: https://huggingface.co/datasets/isp-uv-es/opensr-test
and https://github.com/ESAOpenSR/opensr-test. The dataset card declares MIT; source
NAIP imagery is public-domain aerial imagery, not satellite imagery. Retain USDA
NAIP and OpenSR attribution. USGS source information:
https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip

Only three HR files from release directory 100 are pinned by repository revision
and SHA-256 in configs/naip-pilot-manifest.json. Acquisition dates are parsed from
provider filenames and explicitly stored as date precision. Retrieval time is in
the manifest; provider availability time is unknown. All four RGBNIR bands remain
separate from RGB display. CRS, transform, nodata and extent are read from each
file. The fixed divisor 10,000 is an explicit numerical convention, not a claim of sensor calibration. Digital numbers are not assumed to be calibrated reflectance. No downloaded
imagery is committed. Total payload is 5,627,736 bytes. Splits precede patching.

## WorldStrat — assessed, not downloaded

Official code/data entrypoint: https://github.com/worldstrat/worldstrat
Paper: https://arxiv.org/abs/2207.06418 (Cornebise, Orsolic, Kalaitzis, 2022).
The official repository links Zenodo and Kaggle. Airbus HR imagery is CC BY-NC 4.0;
Sentinel-2/labels/weights CC BY 4.0; code BSD-3-Clause. Do not confuse code and data
licenses. Native HR 1.5 m and Sentinel-2 10 m are not exact x4 pairs. Multiple
observations make this naturally multiframe data. A future single-image adaptation
must pick the cloud-valid observation nearest HR acquisition, with provider ID as
a deterministic tie-break, and document time gap. Full archives exceed pilot needs.

## SEN2VENµS — assessed, not downloaded

Version 2.0.0, 2025-01-17: https://zenodo.org/records/14603764
Paper DOI: https://doi.org/10.3390/data7070096 (Michel, Vinasco-Salinas, Inglada,
Hagolle, Data 2022, 7, 96). V2 uses georeferenced GeoTIFFs; signed 16-bit samples
divide by 10,000 for surface reflectance. Sentinel-2 10 m to VENµS 5 m is x2;
20 m to 5 m is x4. SWIR B11/B12 lack an HR VENµS reference. Sentinel-2 Theia
products use Etalab Open Licence 2.0; VENµS products CC BY-NC 4.0; remaining files
CC BY 4.0. The smallest listed site FGMANAUS is about 121.2 MB but one site alone
does not permit geographic research validation. Do not randomly split its dates.

## OpenSR diagnostics — integration pending

https://esaopensr.github.io/opensr-test and official repository above. Distinguish
package versions and dataset releases. NAIP/SPOT/Spain examples are x4, VENµS x2
according to the official card. Harmonized HR products may embed calibration;
inspect fitting provenance before using them for an unbiased experiment. The card's
MIT label does not remove upstream Airbus/VENµS restrictions. The local metrics
are not advertised as verified OpenSR diagnostics.

## Fresh Sentinel-2 catalog

Earth Search v1 endpoint: https://earth-search.aws.element84.com/v1
Configurable AOI/date/collection/cloud filters are in configs/stac.json. Metadata
refresh downloads no image assets and cannot change a frozen research manifest.
Asset-specific licensing, scaling and quality masks must be validated before
scientific use. Acquisition time and provider-created time are recorded separately;
created is a provider field, not a guaranteed first-availability measurement.
