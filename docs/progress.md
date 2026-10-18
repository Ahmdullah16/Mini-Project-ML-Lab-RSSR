# Progress

Last reconciled: 2026-10-18. No mini-project is marked learner-complete.

Implemented and executed: five 100-step CPU pilots (CNN, attention CNN, window
Transformer, hybrid, compact SRCNN), each using the same pinned three-scene NAIP
manifest. Three scenes are assigned train/validation/test before patch extraction.
Only validation was evaluated. A separate three-band CNN checkpoint exists for
the display demo. These are synthetic-degradation experiments on real aerial data,
not real cross-sensor satellite experiments.

Measured validation PSNR, one scene: bicubic 45.8104 dB; CNN 37.6707; attention
42.8686; window Transformer 30.5688; hybrid 43.2617; compact SRCNN 25.7763.
Every learned model underperformed bicubic. No final model or improvement claim.
See reports/pilot-runs.json and reports/pilot-summary.csv; full records/checkpoints
remain local in ignored data/runs/. Scaling is fixed stored-value / 10000, not
verified reflectance calibration; scores are not comparable with external protocols.

Implemented and checked: MP01's 10 unittest checks, exact four-step versus two-step
plus resumed training equality, 40 sequential syllabus IDs, 317 Markdown links
at the earlier structure check. Full pytest and real Jupyter execution are pending
the current validation pass, not inferred from these checks.

Fresh-data metadata: live Earth Search refresh found/added 10/10 items; repeat
found/added 10/0. Both are explicitly partial due to page cap. SQLite integrity
check returned ok. Initial SQL placeholder failure was fixed and remains logged.
Asset download, continuous scheduling and full incremental cursor recovery are pending.

Learning: original MP01 preserved with conditional Colab setup; MP02–MP40 introductory
offline notebooks implemented. Detailed project-specific solution guides, broader
mathematics, appropriate real experiments, plots and full completion criteria remain
pending. Learner mastery is unassessed for all 40 projects.

Blocked externally: public demo requires verified free hosting access/entitlement.
No hosting token found; current HF documentation requires a paid plan for new compute
Spaces. No paid service activated. Local demo validation is still pending.

Pending research: geographic multi-scene/multi-seed study, all controlled ablations,
sensor alignment/calibration, OpenSR diagnostics, strong official baseline,
complete literature extraction, manuscript/thesis/presentation deliverables.
