# Scientific claim register

Four categories are mandatory: **executed/measured**, **implemented/tested**, **planned**, **unavailable**. A file's existence is not evidence of scientific acceptance. Exact canonical source hashes and all frozen summary-row pointers are in [source audit](evidence/source-audit.json); table-specific derivations remain in [paper evidence](../paper-evidence.json).

| Major statement | Category | Exact evidence | Supported scope / forbidden inference |
|---|---|---|---|
| Frozen per-model PSNR/SSIM/SAM/per-band RMSE | Executed/measured | reports/final-heldout.json `/summary/0` onward, enumerated in source audit | One scene,16 tiles; separate3/4band protocols; no geographic CI |
| Early gate/attenuation outcome | Executed/measured | reports/ablation-pilot-runs.json; paper-evidence.json pilot rows and pointers | Seed dispersion, one validation scene; learned reliability not established |
| Development gains | Executed/measured | reports/geographic-study.json; paper-evidence.json geographic_validation rows | Four validation scenes; seed SD is not geographic uncertainty; multiple changes confound causality |
| Official SwinIR comparison | Executed/measured | configs/final-evaluation.json `/official_rgb`; frozen RGB rows | Different natural-image pretraining/degradation; not matched training budget |
| Seven cat-v1 paths | Implemented/tested | src/rssr/architecture_registry.py `REGISTRY` and `build`; models.py | Seven definitions, not seven trained scientific results |
|39 historical bindings across five IDs | Implemented/tested | reports/historical-architecture-bindings.json `/rows`; counts in source audit | Recorded prior checksum/load verification; this audit does not reopen weights |
| Partial-budget matching | Implemented/tested | reports/architecture-budget-preflight.json | Conv/Linear/MHA accounting only; M2 lacks eligible searched width; no full FLOPs match |
| Masked reflectance ingestion/export | Implemented/tested | reports/reflectance-job-validation.json; surface_reflectance.py | Synthetic fixtures; real-provider calibration/alignment unverified |
| Exact large-grid inference infrastructure | Implemented/tested | Existing scale reports and architecture-seam-smoke.json | Random-weight/native smoke is not trained cascade quality;65536square execution not observed |
| Public inference observation | Executed/measured | reports/public-access-20261028.json | One old-interface request; latest deployment/cold restart not established |
| Independent-region protocol v2 | Planned | docs/next_session.md preserved preregistered design | Five folds/≥15regions/three seeds are targets, not completed experiments |
| Physical detail recovery or cross-sensor advantage | Unavailable | No eligible aligned/calibrated independent paired evaluation | Do not claim superiority, physical resolution recovery or downstream benefit |
| Architectural novelty | Unavailable | docs/contribution-audit.json | No novelty from module combination or output pixel count |
| Supplied IEEE reference compliance | Unavailable | Missing conference_041818.pdf and IEEEtran_HOWTO.pdf | Existing IEEEtran compilation cannot substitute for reading supplied references |

This is a major-claim audit, not a completed sentence-by-sentence semantic or bibliographic review. The prospective reconstruction must keep this distinction explicit.
