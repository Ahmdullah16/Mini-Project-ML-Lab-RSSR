# MP32 validation — 2026-10-25

Implementation and teaching deliverables complete; learner mastery unassessed.

- `python -m pytest -q stage_08_remote_sensing/MP32_satellite_image_explorer`: 3 tests passed; six affine deprecation warnings. Checks center/corner geometry, cropped transforms, nodata, bounded windows, nonfinite and empty-mask statistics.
- `python stage_08_remote_sensing/MP32_satellite_image_explorer/experiment.py`: executed synthetic fixture and SHA-256-verified training crop. Four bands; EPSG:32611; 2.5-m pixels; 16,384 valid pixels per band. No download, optimization or test access.
- `python scripts/execute_project_notebooks.py --only MP32`: real kernel passed, 5 code cells. Source SHA-256 `f4c92dd218437ede150547a71c4e9f833e6efb09ba5b004b3a8c668e05b355f6`. Figure visually inspected; fixed-range previews intentionally retain dark values rather than silently applying different stretches.
- All 40 project layouts and 817 relative Markdown links passed before this validation update.

Evidence: [measured metadata/statistics](../../reports/mp32-raster-explorer.json), [band views](../../reports/mp32-band-views.png), [kernel record](../../reports/project-notebook-execution.json). Source license, versions, checksum and training-only selection are recorded. The historical project-folder test report remains an October25 earlier snapshot; this document records the expanded three-test scope.

The experiment is an aerial raster inspection, not satellite generalization. Masks do not establish cloud-free pixels; DN/10000 is not calibrated reflectance. Full cross-sensor work remains separate.
