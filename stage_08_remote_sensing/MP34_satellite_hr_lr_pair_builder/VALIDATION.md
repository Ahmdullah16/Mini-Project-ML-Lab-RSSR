# MP34 validation — 2026-10-25

Complete implementation and teaching deliverables; learner mastery unassessed.

`python -m pytest -q stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder`:3 tests passed;18 affine deprecation warnings. Verified constant preservation, input nonmutation, divisibility/scale rejection, hand-computed area averages, rotated exterior-corner equality, missing CRS/degenerate transform/invalid-pixel rejection, and noise-seed repeatability.

`python stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/experiment.py`: synthetic hand check and pinned training-raster experiment passed. Four-band64×64 HR produced32×32 and16×16 LR with5-m/10-m spacing; same160×160-m extent in EPSG:32611. Original source checksum verified. Ignored NPY outputs have recorded SHA-256 hashes and are rebuildable, not committed datasets.

`python scripts/execute_project_notebooks.py --only MP34`: real kernel passed 5 code cells. Notebook SHA-256 `d3ca4a6bfdb8cc6f4bf401c6bb3a3b26f1a1afe60127ca66623efc4adcfd49f0`. No frozen-result evaluation, new training or data download.

[Pair manifest](../../reports/mp34-pairing-manifest.json) · [Kernel record](../../reports/project-notebook-execution.json). Lesson covers equations, examples, provenance, exercises/solutions and failure cases. No physical calibration, cloud-free or cross-sensor accuracy claim. Hosted models remain native×2 and cannot silently consume×4 pairs.
