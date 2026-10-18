# Submission readiness

Assessment date: 2026-10-19. **Not submission-ready and not publication-ready.** Failed means an acceptance criterion is incomplete or unverified; blocked means an external prerequisite is missing. Working engineering and notebook execution do not establish educational or research completeness.

| Original requirement | Status | Acceptance and evidence | Remaining limitation |
|---|---|---|---|
| 1a Resource authorization | passed | Bounded CPU runs; no paid service. `reports/environment-continuation.json` | No charges or GPU use; transfer overhead from the original session remains incompletely measured. |
| 1b Exact resource accounting | failed | Account for all wire bytes and compute. `reports/environment-continuation.json` | New COG transfers counted; historical retry overhead unknown. |
| 2 Audit and preserve | passed | Preserve source syllabus, old runs, inspect branches/PRs. `docs/import-manifest.json` | PRs 1–3 merged externally; current work is isolated on feat/submission-pipeline. |
| 3a 40 IDs and nine stages | passed | Unique authoritative IDs/titles/objectives and prerequisites. `docs/syllabus.json` | Structure checker verified IDs; mastery remains unassessed. |
| 3b Full teaching deliverables | failed | Every lesson has defined mathematics, worked examples, exercises, solutions and appropriate real experiments. `notebooks` | MP01–04 expanded; MP05–40 still introductory. |
| 4a Shared CLI/code | passed | Separate reusable data, training, metrics, inference, database modules. `src/rssr/cli.py` | Existing offline tests and local workflows; new fresh workflow has an explicit script. |
| 4b Portable reproducibility | failed | Fresh checkout/environment quickstart, pinned dependencies and relocatable assets. `scripts/portable_quickstart.py` | Relative prepared paths implemented; fresh-checkout execution in progress. |
| 5a Synthetic SR pipeline | passed | Licensed real subset through training/validation/inference/database. `reports/pilot-runs.json` | Executed engineering pilot only. |
| 5b Scientific cross-sensor pipeline | failed | Verified calibration, registration, masks and spectral compatibility. `docs/protocol.md` | No validated real paired cross-sensor study. |
| 6a Dataset provenance | passed | Pinned source versions, checksums, licensing and geographic split. `configs/naip-pilot-manifest.json` | Three NAIP scenes; fixed numerical scaling, not verified reflectance. |
| 6b Independent research scenes | failed | Multiple independent train/validation regions and suitable paired satellite data. `docs/dataset-cards.md` | Current research validation has one scene. |
| 7a Fresh acquisition-to-output | passed | Query, bounded download, mask/CRS validation, inference, provenance and database record. `src/rssr/fresh.py` | Sentinel acquisition 2026-10-16 executed; 1,900,544 bytes; no HR metrics; domain shift unvalidated. |
| 7b Incremental durable ingestion | failed | Watermarks, deduplication, retries, expired-cursor and scheduler recovery. `src/rssr/ingest.py` | Explicit cursor recovery works; automatic watermarks and robust scheduler recovery incomplete. |
| 7c Freshness/offline honesty | passed | Separate acquisition/availability/ingestion time; partial versus complete refresh. `src/rssr/catalog.py` | Read-only cached HTTP view tested. |
| 8a Local experiment database | passed | Foreign keys, migrations, integrity, normalized records, exports. `src/rssr/db.py` | Additive v2 inference_jobs migration tested; existing rows preserved. |
| 8b Hosted persistence | blocked | Restart preserves required server-side data. `docs/operations.md` | No authenticated free host/persistent storage provisioned. |
| 9a Literature verification | failed | Complete primary-source extraction matrix and bibliography. `docs/literature.md` | Initial verified references exist; detailed extraction/overlap remains incomplete. |
| 9b Defensible contribution | failed | Test hypothesis against attenuation and other controls under shift. `reports/ablation-pilot.md` | Constant attenuation explains much of pilot gain; no novelty established. |
| 10a Educational baselines/ablations | passed | Classical, SRCNN, CNN, attention, Transformer, hybrid and required controls. `reports/ablation-pilot-runs.json` | 24 measured runs across three seeds, with limits disclosed. |
| 10b Strong official baseline | failed | Compatible official/pretrained modern comparison. `docs/protocol.md` | Not reproduced. |
| 11a Metric protocol | passed | Range, border, bands, masks and aggregation explicit. `docs/protocol.md` | Synthetic protocol implemented; physical-unit metrics guarded. |
| 11b Research breadth | failed | Spectral, robustness, alignment, downstream when justified, geographic CI. `reports/ablation-pilot.md` | One scene cannot support geographic confidence intervals. |
| 11c Efficiency | failed | Parameters, latency, peak memory and declared FLOP/MAC conventions. `reports/efficiency-pilot.json` | CPU latency and parameters measured; peak memory/FLOPs incomplete. |
| 11d Final selection and held-out test | failed | Freeze selection then evaluate once without tuning. `src/rssr/experiment.py` | No final model selected; test remains unused. |
| 12a Resume and provenance | passed | Exact interrupted resume; configs, seeds, environment, hashes. `tests/test_experiment.py` | Existing equivalence test passed; relocatable prepared manifests tested. |
| 12b All notebooks clean exits | failed | All final notebook versions in isolated kernels with clean process exits. `scripts/execute_notebooks.py` | Earlier cell evidence exists; final clean-environment batch pending. |
| 13a Local inference demo | passed | Real upload/inference API and validated output dimensions. `reports/demo-http.json` | Local HTTP inference verified. |
| 13b Complete demo controls | failed | Samples, compatible models, AOI, metadata, downloads, scientific export. `app/demo.py` | Catalog/upload/basic comparison implemented; remaining controls in progress. |
| 13c Public inference deployment | blocked | Public URL completes inference and restart/persistence check. `docs/operations.md` | Authenticated host access/entitlement missing; no URL claimed. |
| 14a Submission artifacts | failed | Final report, slides, examiner guide, executed notebook bundle and manifest. `reports/research-draft.md` | Preliminary report and figures only; package incomplete. |
| 14b Consistent documentation | failed | README, briefs, progress and evidence agree. `README.md` | Known stale introductory-status text being reconciled. |
| 15a Local tests | passed | Offline regression suite and real-data smoke integration. `tests` | Last full suite 51 tests plus four subtests; new targeted tests pass; final full run pending. |
| 15b Final commit CI | failed | CI passes on actual submission commit. `.github/workflows/test.yml` | Final commit not selected; CI must be checked after publication. |
| 15c Review/checkpoints | passed | Validated commits, resumable exact next actions, no automatic merge. `docs/next_session.md` | Submission work checkpoint maintained; no institutional submission authorized. |

## Individual teaching acceptance

Every project requires defined symbols/shapes/units, worked mathematics, reusable code, synthetic and appropriate real experiments, interpretation, exercises, separate solutions, tests, setup, references and execution evidence. Learner mastery is separately unassessed for all 40.

| Project | Engineering lesson status | Acceptance |
|---|---|---|
| MP01 — Digital Image Explorer | expanded implementation; audit against all criteria still required | failed — full submission acceptance not established |
| MP02 — NumPy Image Laboratory | expanded implementation; audit against all criteria still required | failed — full submission acceptance not established |
| MP03 — Image Resizing Laboratory | expanded implementation; audit against all criteria still required | failed — full submission acceptance not established |
| MP04 — Image Comparison Tool | expanded implementation; audit against all criteria still required | failed — full submission acceptance not established |
| MP05 — Gaussian Blur Simulator | introductory implementation only | failed — full submission acceptance not established |
| MP06 — Low-Resolution Image Generator | introductory implementation only | failed — full submission acceptance not established |
| MP07 — Synthetic Degradation Laboratory | introductory implementation only | failed — full submission acceptance not established |
| MP08 — Frequency-Domain Image Explorer | introductory implementation only | failed — full submission acceptance not established |
| MP09 — MSE and PSNR from Scratch | introductory implementation only | failed — full submission acceptance not established |
| MP10 — SSIM Evaluation Tool | introductory implementation only | failed — full submission acceptance not established |
| MP11 — Super-Resolution Evaluation Dashboard | introductory implementation only | failed — full submission acceptance not established |
| MP12 — Linear Algebra Visualizer | introductory implementation only | failed — full submission acceptance not established |
| MP13 — Gradient Descent from Scratch | introductory implementation only | failed — full submission acceptance not established |
| MP14 — Neural Network from Scratch | introductory implementation only | failed — full submission acceptance not established |
| MP15 — First PyTorch Neural Network | introductory implementation only | failed — full submission acceptance not established |
| MP16 — Reusable Training Pipeline | introductory implementation only | failed — full submission acceptance not established |
| MP17 — Convolution Playground | introductory implementation only | failed — full submission acceptance not established |
| MP18 — CNN Edge Detection Laboratory | introductory implementation only | failed — full submission acceptance not established |
| MP19 — CNN Image Reconstruction | introductory implementation only | failed — full submission acceptance not established |
| MP20 — SRCNN Implementation | introductory implementation only | failed — full submission acceptance not established |
| MP21 — Residual CNN Super-Resolution | introductory implementation only | failed — full submission acceptance not established |
| MP22 — Upsampling Architecture Comparison | introductory implementation only | failed — full submission acceptance not established |
| MP23 — Channel Attention from Scratch | introductory implementation only | failed — full submission acceptance not established |
| MP24 — Spatial Attention from Scratch | introductory implementation only | failed — full submission acceptance not established |
| MP25 — Combined Attention Module | introductory implementation only | failed — full submission acceptance not established |
| MP26 — Attention-Based SR Network | introductory implementation only | failed — full submission acceptance not established |
| MP27 — Self-Attention from Scratch | introductory implementation only | failed — full submission acceptance not established |
| MP28 — Multi-Head Self-Attention | introductory implementation only | failed — full submission acceptance not established |
| MP29 — Transformer Encoder Block | introductory implementation only | failed — full submission acceptance not established |
| MP30 — Image Patches and Window Attention | introductory implementation only | failed — full submission acceptance not established |
| MP31 — Transformer-Based SR Baseline | introductory implementation only | failed — full submission acceptance not established |
| MP32 — Satellite Image Explorer | introductory implementation only | failed — full submission acceptance not established |
| MP33 — Satellite Preprocessing Pipeline | introductory implementation only | failed — full submission acceptance not established |
| MP34 — Satellite HR–LR Pair Builder | introductory implementation only | failed — full submission acceptance not established |
| MP35 — Geographic Dataset Splitting | introductory implementation only | failed — full submission acceptance not established |
| MP36 — Cross-Sensor Dataset Investigation | introductory implementation only | failed — full submission acceptance not established |
| MP37 — Feature Fusion Laboratory | introductory implementation only | failed — full submission acceptance not established |
| MP38 — Hybrid CNN–Attention–Transformer SR Model | introductory implementation only | failed — full submission acceptance not established |
| MP39 — Controlled Ablation Study | introductory implementation only | failed — full submission acceptance not established |
| MP40 — Robustness and Efficiency Benchmark | introductory implementation only | failed — full submission acceptance not established |