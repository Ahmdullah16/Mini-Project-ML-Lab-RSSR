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


## Local archive implementation — 2026-10-27

`python -m rssr.archive_intake archive.zip --expected-md5 <published-md5>` lists and hashes a local ZIP without decoding imagery. Explicit extraction uses `--destination <new-folder> --entry <exact-file-name>` (repeat entry as needed). Archive cap130,000,000 bytes, declared and actual extraction cap512MiB, entry cap10,000 and free disk reserve1GiB are enforced. CRC checks are performed on selected entries; hashes are returned. Unsafe/Windows-reserved names, links, collisions, existing destinations, unknown selections and quota violations fail closed. Partial staging is cleaned after failure. Nested archives are not recursively expanded.

These limits bound archive/file bytes, not total interpreter RAM or wall time; ZIP central-directory parsing allocates metadata before the entry-count check. Network transfer enforcement and raster decoding remain separate unfinished steps. No downloader was executed or added in this milestone.21 targeted archive tests passed, including failure cleanup and Windows raw-name validation. The publisher-provided methodology/correction text has now been read; see the dated addendum in [feasibility](../reports/mp36-feasibility.md). Historical unknown cells above are superseded only for original-paper methodology; v2 file-level verification remains open.


## Single-attempt transfer implementation — 2026-10-27

`rssr.archive_transfer.transfer_archive` now requires an exact expected byte count and published MD5, accepts public HTTPS without embedded credentials, rejects redirects/encoded responses, writes a temporary file and records SHA256 after checksum verification. Existing destinations are preserved; failures clean temporary files. No automatic retry or resume. It enforces130,000,000 application-read bytes by default and1GiB disk reserve. The600-second deadline is checked between blocking reads, each with45-second socket timeout; this is not a hard process deadline. Reported bytes exclude TLS/HTTP overhead and OS buffering. The current cap policy therefore bounds application body reads, not all network wire traffic.

Seven offline tests passed0.11s. No live transfer was attempted. Exact provider byte count/final URL still need confirmation before requesting acquisition approval. Existing data.download has different retry semantics and is not used for this intake. CPU/raster array budgets remain separate unimplemented execution gates.
