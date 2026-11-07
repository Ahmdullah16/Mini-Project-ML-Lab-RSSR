2026-11-03 AURORA X: additive foundation/passport work In progress; no merge/deployment. VERIFIED bounded checks, NOT VERIFIED full M0–M6 acceptance. See [audit](aurora_x_audit.md) and [evidence](../reports/aurora-x-validation.json). Existing scientific/project statuses unchanged.

2026-11-02 integrated correction: PR59 merged as c0d59e1 after exact head a5b32150 passed tests and Streamlit in CI41487981111/41487971216. This supersedes pending correction notes below. All40 native interaction coverage remains separate from full W1/scientific/submission acceptance.

2026-11-02 follow-up: execution verification now also pins historical computational inputs (MP31/39/40) and feasibility-export dependency (MP32–36). Seven missing/changed/restored-input regression checks passed. Correction exact-head CI pending. [Input validation](../reports/native-lesson-input-validation.json).

2026-11-02 execution coverage: MP01–MP40 native checks passed in exact-head CI41435887407. Source-hashed per-lesson index now separates verified interaction from full content, learner and scientific acceptance. Public MP36 computation and corrected labels verified on47ad0bf; not an all40 browser claim.

2026-11-02 MP37–MP40 native controls implemented;14 numerical tests/four offline AppTests and source checks passed. Latest-head CI pending. All40 now have native dispatch, but this is not automatic full W1/scientific/submission acceptance. Historical seed spread, selected-seed robustness and CPU efficiency retain their limits; no new training or benchmarks. [Batch evidence](../reports/mp37-mp40-native-instructor.json).

2026-11-02 MP32–MP36 integrated through PR55 at26236d9 (external merge); exact headfd3fb38 passed both CI41434214186/41434206234. All five offline AppTests, all40 navigation and import recovery passed. Bounded native teaching verified; real paired scientific acceptance remains gated. Next MP37–MP40 on branch feat/mp37-mp40-native-hybrid.

2026-11-02 W1 MP32–MP36: native synthetic computations implemented with canonical tools; numerical tests passed. Offline AppTests and import recovery passed; latest-head CI pending. MP35 angular teaching buffer does not certify protocol-v2 metric separation. MP36 original methodology review is resolved, but versioned files, residual alignment, calibration and eligibility remain unverified; B11/B12 have no VENµS HR reference. [Batch evidence](../reports/mp32-mp36-native-instructor.json).

2026-11-01 MP27–31 native Transformer batch: implemented canonical worked examples and read-only four-band MP31 development comparison, with separately labeled synthetic shape check. Local numerical, all five offline AppTests, all40 navigation and import recovery passed. PR54 implementation head715a8c83 passed CI41282567158 tests and Streamlit; inspect the documentation follow-up head before merge. Full W1/40-project scientific acceptance is not implied. [Batch evidence](../reports/mp27-mp31-native-instructor.json).

2026-11-01 reviewed release: PR49-51 unique changes integrated through PR52, merged main3a10235 after exact-head173068 CI41279698496 SUCCESS. Local507 tests/17subtests and affected AppTests passed; post-merge50 notebook cells, import-recovery and seven-section AppTests passed. Public app reports3a10235 and MP26 control executed with measurements/settings downloads. No open PRs remain. Full scientific/40-project acceptance remains incomplete; these are bounded educational controls. Evidence: [PR stack review](../reports/pr-stack-review-20261101.json).

2026-11-01: PR50 exact head 847cc729 GitHub Actions 41232412252 SUCCESS; MP18–22 synthetic teaching tests passed. MP23–26 attention lab candidate added on stacked branch, using canonical gates with seeded controls (not trained). New CI pending. Full W1 and scientific acceptance incomplete.

2026-11-01: PR50 run 41229517239: tests success, Streamlit failure in scripts/check_cnn_native_lessons.py from incorrect MP19/MP21 button key case (mp19_run vs MP19_run). Corrected in ec88eba6; exact-head CI retest pending. No research results altered.

MP18–22 native bounded CNN lesson candidates added to stacked development branch, using canonical shared models, labeled zero-weight controls, explicit run and CSV/JSON. CI unverified; full scientific and learner completion pending. Evidence reports/mp18-mp22-native-instructor.json.

MP17 native convolution playground candidate added on stacked branch, with canonical correlate/output_size/receptive_field, explicit outputs and new CI checks. CI execution pending; never mark scientific/learner acceptance complete. Evidence: reports/mp17-native-instructor.json.

2026-10-31 MP12–16 native lab GitHub Actions run 41226551966 passed both tests and streamlit jobs at head 96930eb, including new five-lesson execution. Evidence: reports/mp12-mp16-native-instructor.json. Scientific project completion/public deployment still pending; next MP17–22. Documentation follow-up requires exact-head recheck.

2026-10-31 candidate MP12–16 native instructor labs added on review branch with canonical imports and static source checks; new AppTest/pytest/CI still pending. Evidence: reports/mp12-mp16-native-instructor.json. Do not mark W1 complete.

Current2026-10-30 update: public app is accessible and the user verified16×16→32×32 CNN inference. Do not reuse old access-blocker diagnoses. PR43 CI41088399016 passed and was externally merged; public7c231604 executed all8checkpoints covering7families at16x16→32x32. See reports/public-seven-family-20261030.json.

# Current acceptance matrix — 2026-10-28

This is the current acceptance index. Historical reports retain their original scope; newer evidence below supersedes older absence statements. A pilot is not research acceptance. Commands run from the repository with PYTHONPATH=src; use the development environment except app checks, which use the Streamlit environment.

| Requirement | Implementation | Verification command | Evidence | Status | Next action |
|---|---|---|---|---|---|
| Seven distinct trained paths | Registry; historical five families plus bounded M2/M6 training | `python scripts/train_missing_models.py` (resumes verified records) | `reports/historical-architecture-bindings.json`, `reports/missing-model-pilot.json` | Pilot evidence for all seven; not matched research | Pinned v0.3.0-rgb-pilots distribution verified; preserve pilots and resolve M2 matched-budget mismatch |
| Compatible multiscale comparison | Disk-backed native2 RGB cascades | `python scripts/check_trained_scales.py` | `reports/trained-rgb-capability.json` | 42/42 small RGB entries executed in trained-rgb-capability-v2.json; historical four-band weights preserved | Nine M3/M4/M5 RGB pilots executed; verify deployed version and each family |
| Exact geometry/resource disclosure | scaling.plan/cascade, CLI, host cap | `python -m pytest tests/test_missing_model_pilot.py -q` and existing scale tests | New 5x3 exports at six scales, four 65x133 native seam measurements | Small trained cases verified; max request preflight only | Expand trained square/max-size cases within measured resources; no 65536-square execution claim |
| On-site lessons | Native MP01–MP40 routes | Per-batch AppTests and pinned source index | reports/native-lesson-execution-index.json | All40 native interaction checks passed; no blanket scientific acceptance | Criterion-level content/visual review and public verification |
| Individual project scientific acceptance | Canonical README/VALIDATION and notebooks | `python scripts/check_project_folders.py` | Per-project validation; docs/progress.md | MP20/36/38/39/40 scientific gaps remain | Independent SRCNN evidence; v2 eligibility and matched study |
| Independent geographic study | Protocol-v2 and real ingestion adapters | Existing protocol/data tests; no new study executed | `docs/protocol-v2.md` | Blocked for research training by eligible manifest/acquisition gates | Verify provider pairing/license/bytes and approved resource envelope |
| IEEE paper | Existing paper/main.tex | Built-in compile_latex_document; `python scripts/audit_paper_sources.py --check` | source-audit.json; successful current editor compile | Working manuscript, not submission-ready | Latest all-page PDF review; real author/venue details; independent evidence |
| Detailed report/submission package | paper/supplement.md and report audit package | `python scripts/build_learning_supplement.py --check` | learning-content-manifest.json | Working report; slides/final packaging incomplete | Typeset, review and package without fabricated missing results |
| Site/deployment | Existing Streamlit app, new evidence cards | `python scripts/check_model_explorer.py` | Seven local navigation checks; earlier public-access-20261028.json | Current local changes verified; new deployed version/cold start unverified | Publish reviewed change, verify signed-out version/inference/restart when browser access available |
| Durable storage | Existing external adapters | Account-specific integration unavailable | docs/operations.md | External storage credentials/account absent | Configure authorized external store; stateless demo remains independent |
| CI/review | Existing workflow includes all native batches | Exact-head Actions checks | PR55 CI41434214186/41434206234; PR56 CI41435887407/41435879940 all successful | Integrated externally as26236d9 and47ad0bf | Verify each subsequent head; never bypass checks |


## Mandatory website requirements — 2026-10-29

| Requirement | Implementation | Verification command | Evidence | Status | Next action |
|---|---|---|---|---|---|
| W1 Complete all40 on-site lessons and actual outputs | Native MP01–MP40 routes, computed examples and historical records | Existing per-batch AppTests in CI41435887407 | reports/native-lesson-execution-index.json | All40 native interaction checks passed; full criterion-level content/visual acceptance remains open | Review remaining per-criterion pedagogy/visual coverage without rerunning frozen science |
| W2 Seven models in both sections | Shared model_availability registry, pinned execution catalog | pytest tests/test_metric_teaching.py; check_metric_ui.py | metric-teaching-validation.json | All seven have pinned hosted RGB artifacts; clean-download/local Live SR and satellite checks passed | Public Live SR all seven executed at7c231604; verify public Satellite Explorer execution separately |
| W3 Satellite model comparison | Shared-source Single/Compare, sequential execution, result preservation/retry | check_satellite_comparison.py | satellite-comparison-validation.json | All eight catalog checkpoints (seven families) executed offline; full visual/scientific acceptance pending | Common crop/zoom/bicubic/disagreement locally tested; next all7compatible models, scientific masks and public verification |
| W4 Deep metric explanations beside images | Shared metric_ui and metric-instructor.md | check_metric_ui.py; numerical tests | metric-teaching-validation.json | Six basic metrics + per-band/gradient example implemented; real paired-result integration incomplete | Add eligible paired-image path and domain-valid SAM; do not invent upload reference |
| W5 Controlled metric behaviour | Deterministic64square fixture,6controls, explicitRun | pytest tests/test_metric_teaching.py; check_metric_ui.py | metric-teaching-validation.json | Tested teaching panel in both sections | Visual desktop/mobile review; preserve reference separation |
| W6 Scientific honesty | Explicit unavailable reference accuracy, separate synthetic panel | Numerical/source/UI checks | metric-teaching-validation.json | Teaching separation verified; full scientific exports acceptance pending | Review registration/mask/band/unit gates for every future paired workflow |
| W7 Verification and finish | CI now runs metric UI checks | check_metric_ui.py, existing project checks | Per-feature records; checkpoint | Partial, not final acceptance | Full40output coverage, satellite comparisons, all7execution, live/mobile/download/retry validation |


## Interface presentation milestone — 2026-11-02

Shared theme/components, guest Overview, evidence-backed cards, roadmap and researcher profile implemented in `app/research_design.py`; original seven route IDs and lesson/inference logic preserved. Local offline design/navigation checks recorded in [interface evidence](../reports/interface-upgrade-20261102.json). This does not close authentication, mobile visual/accessibility review, public satellite/scientific or submission acceptance. No signup backend configured; see [interface scope](interface-upgrade.md). Exact-head CI must be inspected before release.
2026-11-07 report milestone:18-page detailed PDF and4-page IEEE PDF built and visually reviewed; seven feature-path diagrams, registry parameter CSV and nine publisher bibliography records added. Full560tests+17subtests and seven-section offline AppTest passed.20bibliography records is not20fully reviewed papers. Full submission and scientific gates remain open; see reports/submission-report-build-20261107.json.

2026-11-08 report integration milestone: seven model + five internal technical diagrams embedded in 23-page detailed PDF and editable Word report; source, full-size diagrams and checksum-indexed delivery ZIP produced. Six focused tests, 40-ID/999-link structure check, paper audit and seven-section offline AppTest passed. Full scientific/submission gates and new PR CI remain open. Evidence: [report integration](../reports/report-integration-20261108.json); build instructions: [report build](report-build.md).

2026-11-08 sequential tasks: PR66 latest-head CI passed both jobs and merge observed externally. Fourteen actual seven-model training-example display runs (2x/4x) now have source-bound outputs, provenance and report panels;25-page PDF/24-page Word preview built. This closes the bounded demonstration only; broader failure atlas, reference audit and scientific gates remain open. Evidence: [seven-model panels](../reports/seven-model-panels-20261108.json).

2026-11-08 reference milestone:23primary-source citation identities/concepts audited (22research works +1correction), all23cited in updated27-page report and5-page IEEE paper. Editable copies regenerated;12focused tests and offline source/citation checks passed. This closes citation identity/concept scope, not full per-paper experimental extraction or replication. Matrix: [reference review](../reports/reference-review-20261108.md); observations: [reference audit](../reports/reference-audit-20261108.json). Next: exact-head CI, then slides/examiner materials.
