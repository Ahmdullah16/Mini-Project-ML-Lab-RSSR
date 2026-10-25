# MP35 — Geographic Dataset Splitting

**Status: Complete** · Implementation and teaching deliverables; learner mastery unassessed.

[Laboratory home](../../README.md) · [Stage overview](../README.md) · [Progress tracker](../../docs/progress.md)

## Learning objectives and prerequisites

Assign regions before tiling, detect region and spatial leakage, and explain what geographic independence does and does not establish. Complete [MP34](../MP34_satellite_hr_lr_pair_builder/README.md) first. A scene is a source image; a region groups related scenes. Keep every patch from one region in its declared split.

## Mathematics and worked example

Region sets T (training), V (validation), E (test) must have empty pairwise intersections. For WGS84 axis-aligned boxes A and B, define g=max(Bwest−Aeast, Awest−Beast, Bsouth−Anorth, Asouth−Bnorth). The existing pipeline accepts a cross-split pair only when g>b, with b=0.02 degrees. For A=[0,0,1,1] and B=[2,0,3,1], g=1: b=.02 passes, b=1 fails. Touching boxes fail. This is an angular envelope rule, not a geodesic-distance threshold; longitude degrees vary in physical length by latitude. Antimeridian-crossing boxes are explicitly unsupported.

## Executed experiment

Question: do the existing pinned scene declarations satisfy the pipeline's pre-tiling leakage rules? Reuse the original assignments, verify all nine cached file checksums, transform raster-header bounds to EPSG:4326, and audit every cross-split pair. No pixel arrays, new downloads, model inference, or test scores are used. File bytes are read only for checksum verification. No new random split is selected.

Observed: four training scenes, four validation scenes and one test scene; all 24 cross-split pairs pass. Original IDs and assignments remain unchanged. [Audit and environment](../../reports/mp35-geographic-splits.json), [train manifest](../../reports/mp35-splits/train.json), [validation manifest](../../reports/mp35-splits/validation.json), [test manifest](../../reports/mp35-splits/test.json), [scene-center map](../../reports/mp35-splits/map.svg).

## Run and deliverables

From the repository root in the development environment:

```bash
python -m pytest -q stage_08_remote_sensing/MP35_geographic_dataset_splitting
python stage_08_remote_sensing/MP35_geographic_dataset_splitting/experiment.py
python stage_08_remote_sensing/MP35_geographic_dataset_splitting/run_notebook.py
```

The [tools](mp35_tools.py) validate IDs, finite bounds and buffers before delegating to the unchanged shared pipeline. [Tests](test_mp35_tools.py), [experiment](experiment.py), [notebook](geographic_dataset_splitting.ipynb), [runner](run_notebook.py), and [validation](VALIDATION.md) provide MP01-style local evidence. Exports are metadata teaching artifacts; authoritative configs are not rewritten.

## Interpretation, exercises and limitations

Baseline failure example: randomly place patch 1 of each of two scenes in train and patch 2 in test. Both scenes leak across sets. Solution: assign scenes/regions first, then tile within each assignment. Exercise: can spatially separated scenes with identical region IDs pass? No; region identity is a separate constraint.

Sixteen patches from one scene are not sixteen independent geographic replicates. Within-split overlap is permitted. Box separation prevents one form of leakage, not sensor/date dependence or distribution shift. A single test region cannot establish broad geographic generalization. No change to the frozen benchmark or SR-quality improvement is claimed. Next question: cross-sensor transfer (MP36).

## Provenance and references

[Source manifest](../../configs/naip-development-manifest.json) pins OpenSR dataset revision and SHA256s. NAIP is aerial imagery, not satellite imagery. License: dataset-card MIT declaration; USDA NAIP public-domain source imagery attribution retained. Source CRS, transforms, shapes, acquisition dates and URLs are preserved in exports. Environment and source-manifest hash are in the audit record.

[Rasterio coordinate transformation documentation](https://rasterio.readthedocs.io/en/stable/api/rasterio.warp.html). No paper reproduction or new scientific benchmark is claimed.

## Completion criteria

- [x] Explain objectives and solve the small mathematical example.
- [x] Execute the declared cached-scene experiment.
- [x] Verify numerical, leakage and invalid-input cases.
- [x] Deliver split manifests, map and interpreted audit.
- [x] Record license, environment, limitations and frozen-assignment deviation from new split creation.
- [x] Link evidence and update progress. Learner understanding remains unassessed.
