# Cross-sensor intake and bounded resource proposal

Status: proposed development inspection, not authorized acquisition or completed research. This adds an intake implementation without changing the registered v2 experiment or frozen NAIP results. MP36 remains In progress.

## Compatibility matrix

| Field | SEN2VENUS v2 candidate | WorldStrat v1.1 candidate | Admission requirement |
|---|---|---|---|
| Version | Record14603764 | Record15382551 | Pin archive and individual file hashes |
| Spatial sampling | 10m/5m gives native2; 20m/5m gives4 | 10m/1.5m gives20/3 | No silent resampling to fit existing native2 models |
| Bands | B2/B3/B4/B8 candidate; broader B8 response mismatch; SWIR has no HR reference | Exact selected HR/LR product bands still need file-level review | Explicit ordered mapping and spectral-response rationale; equal names are insufficient |
| Time | Provider describes same-day pairs; actual timestamps still need verification | Multiple temporally matched LR revisits; not a simultaneity claim | Timezone-aware timestamps, declared task-specific tolerance |
| Radiometry | Prior source review identifies MAJA L2A and int16/10000 convention | L1C/L2A and raw/processed HR variants must stay distinct | Document units, scaling and calibration; no NAIP DN assumptions |
| Grid | Georeferenced TIFFs; full registration/correction review unresolved | Actual pair grid/registration remains unverified | CRS, north-up grid, extent, dimensions, residual registration evidence |
| Masks | Pair-specific mask semantics and common support unverified | Pair-specific mask semantics and common support unverified | Verify cloud/nodata/invalid-mask meaning before scores |
| Licenses | Sentinel Etalab2; VENUS CC BY-NC4; other files CC BY4 | HR CC BY-NC4; LR/labels/weights CC BY4 | Retain sensor-specific attribution; code license does not license imagery |

Sources and prior retrieval limits: [feasibility review](../reports/mp36-feasibility.md), [SEN2VENUS record](https://zenodo.org/records/14603764), [WorldStrat record](https://zenodo.org/records/15382551). The latter was rechecked October26; SEN2VENUS returned429. Unverified cells are blockers, not inferred facts.

## Concrete proposal for later approval

Candidate: the previously listed SEN2VENUS v2 FGMANAUS archive, approximately121.2MB compressed, published MD5 `8b8e2bf764721c8b4a4ee88c76d3ca96`. This is one development site, permanently excluded from later confirmatory test selection once inspected. It cannot establish geographic transfer. Exact byte count and current URL must be confirmed from provider metadata before asking for transfer approval.

Proposed hard limits: at most130,000,000 response-body bytes across all attempts, no automatic retries; one archive; list ZIP entries before extraction; at most512MiB cumulative uncompressed extraction; reject traversal paths, links and oversized entries. Stop rather than raising limits if the site cannot fit. Reserve1GiB free disk. Inspect metadata first, then at most one pair only after registration/correction and split gates are resolved. No training, weights download, GPU, paid service, public redistribution or test metrics are included. CPU inspection budget10 minutes; decoded arrays≤256MiB. These are proposed caps, not measured usage or a runtime guarantee; enforcement in a downloader remains to implement before transfer.

WorldStrat archive acquisition is deferred: even the raw-HR and L2A selections are11.4GB and25.5GB respectively, before extraction. A documented small-pair distribution would need separate provenance and compatibility review. No full-dataset download is needed for the current metadata work.

## Executable metadata admission

`src/rssr/pair_audit.py` accepts a pair JSON and a separately frozen policy JSON:

```powershell
python -m rssr.pair_audit pair.json policy.json
python -m pytest tests/test_pair_audit.py -q
```

The pair schema is illustrated by the synthetic fixture in [tests](../tests/test_pair_audit.py). Supply genuine source/version/license/hash, region, timestamps, grid, mapped bands, radiometry/mask evidence and registration residual/evidence. Policy requires a protocol ID, integer native scale, extent tolerance in CRS units, maximum time delta in seconds, and maximum registration residual in HR pixels. There are deliberately no operational default tolerances. The zero-tolerance test policy is not a proposed sensor threshold.

Exit0 means `declarations_compatible`; exit2 means `blocked`. Neither verifies the declarations, checks footprint overlap across regions, certifies calibration, fits registration, or authorizes evaluation. Only development-inspection purpose is admitted. A one-pixel shift on a5m HR grid is5m: a correct dimension ratio alone cannot detect that shift, so bounds and measured registration remain separate checks.

## Executed and pending

October26:21 tests passed in0.08s on synthetic metadata. No imagery was opened, acquired, resampled or scored. Pending: full official correction review, justified tolerances, bounded downloader enforcement, acquisition approval, actual metadata/registration/mask validation, independent region manifest and all v2 scientific gates. This work does not close MP36 or the real cross-sensor requirement.
