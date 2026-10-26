# MP36 preliminary feasibility investigation — 2026-10-25

Status: source investigation started; lesson and real cross-sensor validation remain incomplete. No data download, inference, or new benchmark evaluation ran.

## Dataset comparison

| Candidate | Provider-documented pairing | Immediate compatibility decision |
|---|---|---|
| WorldStrat | Sentinel-2 10 m with 1.5 m HR and multiple revisits | Ratio 10/1.5 ≈ 6.667, incompatible with existing native ×2 outputs without a separately declared task/grid. Temporal matching is not proof of simultaneous acquisition. |
| SEN2VENµS v1.0.0 | Same-day Sentinel-2 10/20 m and registered VENµS 5 m patches | 10/5=2 is a candidate for ×2 evaluation; 20/5=4 is not. Matching scale alone does not establish spectral or radiometric compatibility. |

[WorldStrat official repository](https://github.com/worldstrat/worldstrat), checked October 25: Airbus HR has CC BY-NC 4.0 terms; Sentinel-2/labels/weights use CC BY 4.0; code uses BSD-3-Clause. Raw and processed HR variants differ. Do not apply the code license to imagery. Public download links exist; actual transfer has not been tested.

[SEN2VENµS pinned record](https://zenodo.org/records/6514159), checked October 25: B2/B3/B4/B8 tensors are 128-square at 10 m and 256-square at 5 m; signed integers require division by 10000. Dataset documentation names Etalab v2 for Sentinel-2, CC BY-NC 4.0 for VENµS and CC BY 4.0 for remaining files, but repeats the 05m filenames in both sensor-license sections. Resolve that filename inconsistency against the LICENCE file and newer record before redistribution. FGMANAUS is listed at 65.7 MB; no archive was downloaded. A newer version is linked and remains to inspect.

## Mathematical interpretation and next protocol

These are deductions from documented sampling, not measured quality results. Let dL and dH denote metres per pixel: scale s=dL/dH. An alignment error δ metres corresponds to δ/dH HR pixels; for δ=5 m and dH=5 m, a one-pixel displacement can create edge residuals unrelated to SR quality. A calibration mismatch y=a*x+b creates error (a−1)*x+b even with perfect geometry. Sensor band labels alone do not establish equal spectral response.

Before any numerical study: inspect versioned provider metadata and license file; declare site-level train/validation splits independent of the frozen NAIP test; verify file hashes, acquisition dates, band ordering, masks, CRS and registration; define calibration, common support, border crop, metric range and per-site aggregation. RGB input requires an explicit B4/B3/B2 mapping. Do not reuse uncalibrated NAIP DN assumptions as physical reflectance evidence. Choose defaults on validation only.

Next deliverables: verified source matrix, numerical geometry/calibration examples and tests, expanded notebook and full feasibility decision. Real paired validation is a separate requirement from this documentation investigation. No generative image is a reference target.
