# Submission readiness

Assessment: 2026-10-19. **Not submission-ready and not publication-ready.** Passed means the stated criterion was observed; failed means incomplete/unverified; blocked means an external prerequisite is missing. Stateless inference and durable research storage are separate requirements.

| Original requirement | Status | Acceptance and evidence | Remaining limitation |
|---|---|---|---|
| 1a Resource authorization | passed | Bounded CPU runs; no paid service. `reports/environment-continuation.json` | No charges or GPU use; transfer overhead from the original session remains incompletely measured. |
| 1b Exact resource accounting | failed | Account for all wire bytes and compute. `reports/environment-continuation.json` | New COG transfers counted; historical retry overhead unknown. |
| 2 Audit and preserve | passed | Preserve source syllabus, old runs, inspect branches/PRs. `docs/import-manifest.json` | PRs 1–4 merged externally; preserved in feat/development-validation history. |
| 3a 40 IDs and nine stages | passed | Unique authoritative IDs/titles/objectives and prerequisites. `docs/syllabus.json` | Structure checker verified IDs; mastery remains unassessed. |
| 3b Full teaching deliverables | failed | Every lesson has defined mathematics, worked examples, exercises, solutions and appropriate real experiments. `notebooks` | MP01–MP15 expanded and executed; MP16–MP40 remain introductory. Learner mastery unassessed. |
| 4a Shared CLI/code | passed | Separate reusable data, training, metrics, inference, database modules. `src/rssr/cli.py` | Existing offline tests and local workflows; new fresh workflow has an explicit script. |
| 4b Portable reproducibility | failed | Fresh checkout/environment quickstart, pinned dependencies and relocatable assets. `scripts/portable_quickstart.py` | Fresh clone/isolated environment quickstart passed; Windows lock recorded. Linux dependency lock and legacy database path migration remain unverified. |
| 5a Synthetic SR pipeline | passed | Licensed real subset through training/validation/inference/database. `reports/pilot-runs.json` | Executed engineering pilot only. |
| 5b Scientific cross-sensor pipeline | failed | Verified calibration, registration, masks and spectral compatibility. `docs/protocol.md` | No validated real paired cross-sensor study. |
| 6a Dataset provenance | passed | Pinned source versions, checksums, licensing and geographic split. `configs/naip-pilot-manifest.json` | Nine checksum-pinned NAIP scenes, with four train/four validation/one preserved test; numerical scaling is not verified reflectance. |
| 6b Independent research scenes | failed | Multiple independent train/validation regions and suitable paired satellite data. `docs/dataset-cards.md` | Four independent validation scenes now evaluated; suitable real paired satellite data remains absent. |
| 7a Fresh acquisition-to-output | passed | Query, bounded download, mask/CRS validation, inference, provenance and database record. `src/rssr/fresh.py` | Sentinel acquisition 2026-10-16 executed; 1,900,544 bytes; no HR metrics; domain shift unvalidated. |
| 7b Incremental durable ingestion | failed | Watermarks, deduplication, retries, expired-cursor and scheduler recovery. `src/rssr/ingest.py` | Explicit cursor recovery works; automatic watermarks and robust scheduler recovery incomplete. |
| 7c Freshness/offline honesty | passed | Separate acquisition/availability/ingestion time; partial versus complete refresh. `src/rssr/catalog.py` | Read-only cached HTTP view tested. |
| 8a Local experiment database | passed | Foreign keys, migrations, integrity, normalized records, exports. `src/rssr/db.py` | Additive v2 inference_jobs migration tested; existing rows preserved. |
| 8b Hosted persistence | blocked | Restart preserves required server-side data. `docs/operations.md` | Optional S3/PostgreSQL artifact/provenance adapter prepared and offline-tested; external accounts, full schema migration and live durability checks pending. Stateless Streamlit demo does not wait for this. |
| 9a Literature verification | failed | Complete primary-source extraction matrix and bibliography. `docs/literature.md` | Initial verified references exist; detailed extraction/overlap remains incomplete. |
| 9b Defensible contribution | failed | Test hypothesis against attenuation and other controls under shift. `reports/ablation-pilot.md` | Constant attenuation explains much of pilot gain; no novelty established. |
| 10a Educational baselines/ablations | passed | Classical, SRCNN, CNN, attention, Transformer, hybrid and required controls. `reports/ablation-pilot-runs.json` | 24 measured runs across three seeds, with limits disclosed. |
| 10b Strong official baseline | failed | Compatible official/pretrained modern comparison. `docs/protocol.md` | Not reproduced. |
| 11a Metric protocol | passed | Range, border, bands, masks and aggregation explicit. `docs/protocol.md` | Synthetic protocol implemented; physical-unit metrics guarded. |
| 11b Research breadth | failed | Spectral, robustness, alignment, downstream when justified, geographic CI. `reports/ablation-pilot.md` | Four validation scenes and three seeds now measured, but robustness, downstream validity and geographic confidence intervals remain incomplete. |
| 11c Efficiency | failed | Parameters, latency, peak memory and declared FLOP/MAC conventions. `reports/efficiency-pilot.json` | CPU latency and parameters measured; peak memory/FLOPs incomplete. |
| 11d Final selection and held-out test | failed | Freeze selection then evaluate once without tuning. `src/rssr/experiment.py` | No final model selected; test remains unused. |
| 12a Resume and provenance | passed | Exact interrupted resume; configs, seeds, environment, hashes. `tests/test_experiment.py` | Existing equivalence test passed; relocatable prepared manifests tested. |
| 12b All notebooks clean exits | passed | All final notebook versions in isolated kernels with clean process exits. `scripts/execute_notebooks.py` | All 40 current notebooks passed real isolated Python 3.12 kernels, complete process exit 0, source hashes verified; local TCP warning documented. |
| 13a Local inference demo | passed | Real upload/inference API and validated output dimensions. `reports/demo-http.json` | Local HTTP inference verified. |
| 13b Complete demo controls | failed | Samples, compatible models, AOI, metadata, downloads, scientific export. `app/demo.py` | Streamlit and Gradio controls implemented; local browser inference and PNG download verified. On-demand Sentinel workflow uses temporary storage. |
| 13c Public inference deployment | blocked | Public URL completes inference and restart/persistence check. `docs/operations.md` | User will connect GitHub through Streamlit Community Cloud; app preparation and Linux CI underway. No public URL verified. |
| 14a Submission artifacts | failed | Final report, slides, examiner guide, executed notebook bundle and manifest. `reports/research-draft.md` | Preliminary report and figures only; package incomplete. |
| 14b Consistent documentation | failed | README, briefs, progress and evidence agree. `README.md` | README and MP05–MP11 briefs reconciled; older historical checkpoint notes explicitly superseded. |
| 15a Local tests | passed | Offline regression suite and real-data smoke integration. `tests` | 71 tests plus four subtests passed; two additional asset tests and affected storage/demo tests passed; final combined suite pending. |
| 15b Final commit CI | failed | CI passes on actual submission commit. `.github/workflows/test.yml` | CI succeeded for 79b8e5a. New Streamlit/lesson changes require CI on their published commit. |
| 15c Review/checkpoints | passed | Validated commits, resumable exact next actions, no automatic merge. `docs/next_session.md` | Submission work checkpoint maintained; no institutional submission authorized. |

## Teaching acceptance

MP01–MP15 have expanded lessons; MP16–MP40 remain introductory. Project completion checklists remain authoritative and learner mastery is unassessed. All current notebooks have real-kernel evidence; the expanded MP12–MP15 were rerun with exit 0 after the complete 40-notebook batch.
