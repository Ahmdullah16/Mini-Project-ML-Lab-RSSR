# Notebook consolidation follow-up — 2026-10-30

- Review branch: `feat/complete-portable-rssr-notebook-20261030`; initial notebook commit bf2b40c6b52c0a5dbbd556821cfdfa36b7893dee. Added root `RSSR_Complete_Project.ipynb` (40 roadmap chapters, 100 cells, source links, seven recorded distinct SHA-256 checkpoint families, six exact scale arithmetic plans, offline synthetic demonstration, reference gate, optional canonical native2 cascades, metadata-only bounded STAC, PNG/CSV/JSON teaching exports).
- Scope limitations: The chapters currently provide mathematical summaries, numerical fixtures, and canonical source pointers, not a fully reproduced deep derivation or all original experiments for every mini-project. Seven checkpoint identities are documented but have not been redownloaded/executed in this notebook. Real satellite acquisition and scientific GeoTIFF processing are optional and not completed here. No independent geographic study, valid real paired HR accuracy, novelty, public deployment or final paper acceptance is claimed.
- Static manifest check via GitHub file JSON parsing: 100 cells, 50 code cells, 40 ordered MP headings; no inline pip/subprocess calls. **Fresh-kernel nbclient execution and upload/satellite/model path tests NOT performed in this session**: current execution container cannot clone GitHub (no DNS/network), and connector has no compute kernel.
- NEXT: download notebook on a machine with pinned Python environment and run `jupyter nbconvert --to notebook --execute RSSR_Complete_Project.ipynb --output RSSR_Complete_Project.executed.ipynb --ExecutePreprocessor.allow_errors=False`; inspect output/stack traces and check all 40 code cells, then run upload path with PNG/JPEG and a real georeferenced GeoTIFF, and an explicitly bounded provider acquisition. Confirm canonical loader checkpoint compatibility and complete independent scientific acceptance tasks. Do not claim these tests passed until execution evidence exists. Preserve running Streamlit app.

# Recovered release milestone — 2026-10-30

- Latest user public check: accessible, LiveSR/Satelliteload, CNN16x16→32x32 ran; sevenfamiliesvisible but olddeployment onlyM1/M7executable. Do NOT repeat oldaccessblocker.
- Main152d730 contains externallymergedPR42; CI40820609698 SUCCESS; integrated987d522. No openPR at intake. Local plan928e6c6 committed BEFORE9M3/M4/M5RGB runs.
- Recovered uncommitted importsfix and7familyintegration. bootstrap refreshes projectimports on root/sourcefingerprint; regression reproduces missingvisitor_prose and stale satellitehelper then recovers.
-9newRGBpilotruns100steps/seeds17,23,41; reports/rgb-compatible-pilot.json. ExistingM2/M6reused.42/42smalltrainedscales executed reports/trained-rgb-capability-v2.json. Frozen test untouched.
- GitHub prerelease v0.3.0-rgb-pilots PUBLISHED with5portableweights + manifest; binaryassets outsideGit. reports/rgb-pilot-release-manifest.json pins hashes. Release screenshot ../../work/rgb-pilot-release-published.jpg. configs/hosted-models.json integratesM2–M6; hosted_assets permits onlyexactprojectreleaseprefix plus existingrawhost, requireschecksum/bytes.
- Actual local8catalogchecks covering7families: cleancheckpointdownloads,LiveSR and512x512→1024x1024 checks passed; fullcatalogLiveSRcomparison and satellitecomparison/retry passed. reports/seven-family-hosted-catalog-check.json, scale-seven-family-comparison.json, seven-family-satellite-validation.json.
- Fullpytest496passed+17subtests54.77s (62warnings), before latest documentation reconciliation. All40lesson AppTest and metricUI currentrun stillbeingcollected; do not mark them passed prematurely.
- NEXT: finish collection, report/sourceintegrity, commit/publishreviewPR, inspectactualheadCI; verifypublicdeployment/version+allfamilies andMini-projects/SatellitewithoutImportError. Then MP03onward fullinstruction, scientificmask/realpairprotocol/report/PDFreview. Noautomaticmerge.
- Environment: PYTHONPATH=src everycommand (devvenv has older editable checkout otherwise). Dev ../../work/clean-venv/Scripts/python.exe; app ../../work/streamlit-venv/Scripts/python.exe. Do not rerun completed training/frozenresults.

# Exact publication checkpoint — 2026-10-29

- PR41 CI40820234069 SUCCESS; externally merged e85f477, integrated910c733 after preserving detail-view follow-up. No agent remote merge.
- PR42 https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/pull/42 on feat/satellite-detail-views; exacthead3d39204041072a31b0b5a9b6f69e862707848dff; tree153b4319f51a63e8e85d0c84ba706085a5737197 exactly matches local910c733. CI40820609698 IN PROGRESS at last check. No passedCI/deployment claim for this head.
- FIRST NEXT: inspect PR42 actualheadCI; fix failures before new feature work; integrate external merge if any. Then progress compatible7model execution and MP03onward; scientificmask/generalization/publicverification remain open. This exact publication note is a local follow-up for the next milestone.

# Current detail-view milestone — 2026-10-29

- PR41 d257805c1b17d7d4b07698228ee469971df9db54 CI40820234069 SUCCESS; externally merged e85f477. PR40 positionalbutton failure is repaired and passing.
- Added bounded common-grid satellite preview: exact-grid bicubic baseline then area preview≤256, shared normalizedcrop centers/zoom1,2,4, divider and absolute disagreement (not accuracy). Fixed0–1 contrast, invalidblack, fullGeoTIFF unchanged. src/rssr/comparison_views.py rejects grid/CRS/extent mismatch and >hostcap BEFORE baseline allocation.
-6previewtests passed9.31s; two actualcachedmodels AppTest verified commoncontrols and preservedexporthashes, retries/scaleinvalidation stillpassed. reports/satellite-comparison-validation.json includes sources. Float32 casting fixed dividerendpoint roundoff found bytest.
- NEXT: publish this follow-up and inspect its CI. Then complete remaining compatibleRGB checkpoint integrations forM2–M6; all7visibility is not execution. Strengthen masks/nodata/common-source scientific validity; LiveSR shared view parity; MP03onward native lessons; finalreport/paper/publicbrowser acceptance remain open. No fresh satellite acquisition or frozenresult rerun.
- Commands: PYTHONPATH=src; devpython -m pytest tests/test_comparison_views.py -q; apppython scripts/check_satellite_comparison.py; devpython scripts/build_learning_supplement.py --check. Keep paper/main.tex open; noTeXchange thisturn.

# Resumed merge and verification — 2026-10-29

- Interrupted merge completed d3ed959; preserved newer satellite comparison, acceptance matrix and regenerated report. Original verification reports preserved; new completed local reruns stored separately as reports/scale-comparison-validation-20261029.json and reports/scale-streamlit-restart-20261029.json. Records contain successful final assertions (three published model comparison/cache/retry; two empty-cache process restart runs), not public deployment.
- PR40 failure diagnosis and key-based repair remain as below. Publish comparison/repair against origin/main e02c25a, then inspect actual-head CI. Do not mark CI passed from local tests.

# Latest follow-up — satellite comparison and CI repair, 2026-10-29

- PR40 head48a7bf08 CI40746483577 FAILED streamlit check_showcase.py positionalbutton after addedteachingcontrol; pytestjob passed. Externally merged e02c25a before repair. No agent merge. Fix uses scaled_liveprepare key; local7section check passed.
- New Single/Compare satellite workflow: samepreparedsourceSHA, sequentialpinnedRGBmodels, independentresults/errors, targetedretry, keyincludes source/spec/scale/divisor/pipeline, stale scale/source selections dropped, permodelGeoTIFF/provenance and runtimetable/chart. Boundhost requests only.
- scripts/check_satellite_comparison.py actually ran2cachedCNNmodels on same5x3 georeferencedfixture →10x6; CRS32632/extent retained. Injected2ndfailure preserves1sthash; targetedretry succeeds; scaleinvalidates. reports/satellite-comparison-validation.json. No providerrequest or frozenresult rerun.
- NEXT: publishrepair/newcomparison and inspectCI; add commonzoom/bicubic/details/disagreement, strengthen scientificmask/nodata tests and all7checkpoint availability. ContinueMP03onward. Prior metricpanel tests26plus41lesson tests passed; report/sourceintegrity passed. Publicdeployment, full40lessons and fullscientificcompletion NOTclaimed.

# Current checkpoint — metric instruction and shared availability, 2026-10-29

- New mandatory brief saved docs/mandatory-website-brief.md; all7groups in docs/acceptance_matrix.md. Full40onsiteoutput and7model execution remain mandatory, not fulfilled by visibility.
- PR39 exacthead730b2026 CI40744171262 SUCCESS; externally merged origin/main8f16439, integrated2a4e42f retaining newer checkpoint. No agent remote merge.
- Both Live SR/Satellite now show shared7family status from checkpoint identities: M1/M7 hostedRGB; M2/M6 localRGB pendingdistribution; M3/M4/M5fourband incompatibleRGB. Existing selectors remain pinned-hosted-only. No randomweights/bandconversion.
- Shared metric explanation and explicitRun64x64synthetic reference lab: blur/noise/brightness/contrast/shift/sharpening, fixedrange1, uniform7x7SSIM, MAE/MSE/RMSE/PSNR/perband/gradient; SAM/LPIPS unavailable. Source citations verified. Outputclears on parameterchange. Zero-contrast roundoff fixed before passing tests.
-26targetedtests1.95s; both-section AppTest6distortions pluszero/stale/availability passed;970links40IDs passed. reports/metric-teaching-validation.json records12actualsyntheticpairs. Detailed supplement regenerated fromsharedmetricprose and docs/website-validation-update.md; frozen results unchanged. paper/main.tex not edited thisturn.
- NEXT: publish validated milestone/inspect exactheadCI; implement Satellite Single/Compare workflows over ONE fresh_prepared GeoTIFF in app/satellite_scale_ui.py, sequential pinnedcompatiblemodels, preserve successes and retry failures, invalidate identities on pixels/model/scale/preprocessing. Existing fresh_result single-model UI must remain compatible with scripts/check_satellite_scale.py. Then complete7RGBcheckpoints and MP03onward. Full scientific mask/CRS/export, pairedmetrics, deployment/mobile and40lesson acceptance remain pending.
- Commands: PYTHONPATH=src; devpython -m pytest tests/test_metric_teaching.py tests/test_paper_source_audit.py tests/test_paper_evidence.py tests/test_instructor_site.py -q; apppython scripts/check_metric_ui.py; devpython scripts/build_learning_supplement.py --check; devpython scripts/audit_paper_sources.py --check. Devpython ../../work/clean-venv/Scripts/python.exe; apppython ../../work/streamlit-venv/Scripts/python.exe. Preserve existing trained artifacts; no frozen reruns.

# Exact publication state — 2026-10-28

- PR38 externally merged1b16c4b after CI40743598846 SUCCESS; integrated locally56997d1, preserving newer checkpoint/acceptance matrix. Agent did not merge remote PR.
- MP02 follow-up is PR39 https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/pull/39, branch feat/trained-seven-model-pilots, exacthead730b2026f751dc1c25c4201a8d53bff98f3b997f, tree61767529f9d6627e52810a79d1c6c94b6ff8eae0 matches local implementationd2c64a9. CI40744171262 IN PROGRESS at last check; do not claim passed.
- Next: inspect PR39 CI before modifying; preserve any external merge. Continue MP03 native lesson from canonical helpers, then remaining instructor coverage and compatible model scale gaps. All unfinished acceptance gates below remain. Last structural check40IDs/968links passed after local merge. This publication checkpoint is local and should accompany next reviewable milestone.

# Latest follow-up — MP02 native lab, 2026-10-28

- PR38 https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/pull/38 published; initial exacthead db81072ad3e69602479e7355d008207a35fd54c7 CI40743598846 SUCCESS. Local ba4bee8 and initial remote tree match0875dd4e6c308e569e4925e7e073d83e179edb71. Follow-up below requires new exact-head CI.
- MP02 now has native explicitRun fixture/broadcasting/mask/populationstatistics/math/example/diagram/PNG; MP01 also explicitRun.40lesson AppTest passed including both mask populations,5targetedtests passed0.18s. reports/mp02-native-instructor.json pins sources. No canonical notebook or frozen artifact regeneration.
- NEXT: verify follow-up PR38 headCI, continue MP03 native lab, then MP04 onward; separately complete compatible7model scale matrix and research/paper/report/deployment gates in docs/acceptance_matrix.md. No remote merge performed.
- Next commands: apppython scripts/check_learning_ui.py only after lesson edits; devpython -m pytest stage_01_python_numpy/MP02_numpy_image_laboratory/test_mp02_tools.py tests/test_instructor_site.py -q. Read MP03 canonical helpers before implementing.

# Latest milestone — trained missing models and RGB scale evidence, 2026-10-28

- Latest execution brief: attachment42356034; current acceptance index docs/acceptance_matrix.md. Preserve all prior scope.
- PR37 CI40694075637 SUCCESS and externally merged8a28fce; local integration1d8dcb7. Plan committed BEFORE training f6baa35.
- Six actual M2/M6 width16 RGB native2 pilots,100steps,seeds17/23/41, selected step100 for all; train0.723–1.570seconds. reports/missing-model-pilot.json pins config/checkpoints/validation/reload. Ignored checkpoints at data/missing-cat-pilot-v1/runs; preserve locally. No test pixels read; no dataset download. Strict resume checksum check passed without retraining.
-24/42 trained RGB scale entries executed: M1/M2/M6/M7 at2/4/8/16/32/64 on5x3 odd rectangle; TIFF dimensions and hashes recorded, native2 seams measured65x133. RSS sampled10ms is not exact peak. M3/M4/M5 verified weights have4bands; RGB executor requires3. No channel substitution. Full1024/65536 execution remains unverified. reports/trained-rgb-capability.json.
- Model Explorer shows separate local pilot provenance/validation tables; all7navigation checks passed with network denied. Paper updated in place and built-in compile SUCCESS; historical binding table preserved. No remote checkpoint/deployment claim.
- NEXT: finish targeted tests/source integrity, commit/publish milestone and inspect exact-head CI. Then add compatible M3/M4/M5 RGB pilot protocol (or four-band scientific executor), explicit Run controls and MP02 native lesson; continue40. Latest paper PDF visual review, formatted report/slides, durable accounts and deployed version/cold start remain open.
- Commands: set PYTHONPATH=src; devpython -m pytest tests/test_missing_model_pilot.py tests/test_instructor_site.py tests/test_paper_source_audit.py tests/test_paper_evidence.py -q; apppython scripts/check_model_explorer.py; devpython scripts/audit_paper_sources.py --check. Do not rerun completed training/scales merely to refresh dates. Devpython ../../work/clean-venv/Scripts/python.exe; apppython ../../work/streamlit-venv/Scripts/python.exe.

# Resume checkpoint — 2026-10-28

## Latest milestone — self-contained instructor, 2026-10-28

- PR36 externally mergedc5e3e16; CI40690309970 SUCCESS; integrated48594d6 preserving checkpoint. Latest user brief operationalized docs/instructor-site-brief.md; all previous scope retained.
- Supplied IEEE files NOW ACCESSIBLE at C:/Users/DELL/Downloads/ML LAB MINI PROJECT. reports/ieee-paper/evidence/supplied-template-review.json pins4files and actual review. Ignore .DS_Store. Earlier template-transfer blockers superseded; authors/venue still unknown.
- Site: ownGitHub/notebook links removed from visitor markdown/sidebar; internal notebooks preserved. MP01 executes canonical image_tools.py and displays deterministic input, brightness/grayscale output, per-pixel numeric derivation, histogram, flow graph and PNG download. All7cat-v1 paths have accurate native block diagrams; docs/model-instructor.md contains shared derivations, parameter effects and honest application limits, reused in detailed supplement.
- Paper: actual architecture family section/table and numerical conclusion added; compiled successfully. Test caught hybridRGBtranscription typo, corrected to50.2459from frozenrecord; no evaluation rerun. Latest PDF visual export/review blocked: computer-use tool denied ChatGPT access. Do not claim final PDF layout or deployedsite verified.
- Tests:47learning/paper tests0.37s;13instructor/MP01tests+4subtests0.38s;40lesson selections/linkabsence/MP01brightness/histogram/exactscale passed;7model checks passed. reports/instructor-site-validation.json records sources and limits.
- NEXT: verify currentPR/CI; complete MP02native inputs/outputs and explicit derivations/parameters using its verified helpers, continue individually through40, then deeper per-model training examples. Current40prose migration is not full instruction acceptance. Export/review currentpaper when UI available; finish detailed report typesetting and deployment verification. Preserve all untrainedM2/M6, geographic/crosssensor/resource gates.
- Reproduce changedchecks only: devpython -m pytest tests/test_instructor_site.py stage_01_python_numpy/MP01_digital_image_explorer/test_image_tools.py -q; apppython scripts/check_learning_ui.py; apppython scripts/check_model_explorer.py. Publication appended after commit.

## Preserved milestone — IEEE source audit, 2026-10-28

- Latest user master brief saved `docs/ieee-reconstruction-brief.md`; read `reports/ieee-paper/RESUME.md` next. User requires comparing3providedPDFs before manuscript modification. Exact conference_041818.pdf/IEEEtran_HOWTO.pdf absent; Downloads/document.pdf is the known earlier4page session export. Do not misidentify that as a separately transferred manuscript. New manuscript reconstruction held at this explicit gate; independent audit completed.
- Prior local954a390 native40lesson/report milestone was committed but unpublished at interruption. Preserve it. Origin/main89df7a7 confirmed after approved read-only fetch; PR35 CI40633080194 SUCCESS, no openPRs at intake. No remote merge performed.
- Added component matrix, major scientific claim register, architecture/training contracts, readiness/missing-evidence reports under reports/ieee-paper/. `scripts/audit_paper_sources.py` hashes16sources, validates6citation keys/TeXinputs/frozenprotocol, extracts exact12frozen summary rows with units and7historical model counts. This is source integrity, not full semantic/citation-quality acceptance.
- Tests:6passed0.29s. Initial defaultpytesttemp failed2setup cases because sandbox temp directory creation failed; workspace-local --basetemp resolved. Paper and lesson generators --check passed;7claims,958links passed. Existing manuscript unchanged from954a390, no repeat compile/training/inference/frozen evaluation. source-audit check addedCI.
- Critical finding: experiment.py chooses within-run minimum validationL1 checkpoint; final-evaluation.json later chooses runs/seeds using validationPSNR. Keep both stages explicit. Legacyattention bindsM4, notM2. M2/M6training absent.
- NEXT: finish publication/CI of preserved954a390 plus this audit; obtain exact2IEEE reference PDFs and intended manuscript if different, read/compare them, then reconstruct paper. Independently expand MP01 native parameter/math/diagram acceptance and continue7model teaching while transfer pending. Do not claim all40new teaching criteria satisfied by prose migration.
- Commands: PYTHONPATH=src; devpython scripts/audit_paper_sources.py --check; devpython -m pytest tests/test_paper_source_audit.py tests/test_paper_evidence.py -q --basetemp=../../work/pytest-paper-audit-NEW. Devpython ../../work/clean-venv/Scripts/python.exe. New publication details appended after commit.

## Preserved milestone — native teaching and contribution/report sources

- New requirements saved verbatim `docs/onsite-report-completion-brief.md` and linked from completion brief. PR35 head705e373 CI40633080194 SUCCESS; externally merged main89df7a765ca6def3838457760cf8fd7800bd7444, locally integratedc6ac750. No agent PR merge.
- `src/rssr/learning_content.py` extracts existing40lesson README/Markdown/validation prose; no code execution or code-cell viewer. Native UI includes shared mathematical reference and scale teaching plot. All40 render with explicit incomplete-coverage notice. This migration is NOT full new teaching acceptance.
- `docs/contribution-audit.json` shared by website and generated paper section/supplement. RCAN/SwinIR abstracts checked; no novelty or superiority inferred. HAT fetch403, no new claims. `paper/supplement.md` is working Markdown, not a final formatted report. `reports/learning-content-manifest.json` pins120lesson source hashes and shared sources. New CI generator/UI checks added.
- Validation `reports/onsite-learning-validation.json`:47pytest tests1.85s;40lesson AppTest selections plus exact rectangular and maximum arithmetic;7section/7model checks;943links,40foldercontracts,7claims; both generators --check passed. No frozen research or training rerun.
- Updated paper compiles successfully. Initial4page PDF exported to work/ieee-paper-20261028.pdf, allpages rendered/inspected; oversized tables and orphan reference found. Shrink-only table correction and audit punctuation compiled; final PDF reinspection pending after user-input UI guard. Do not label initial export final. pdftoppm available; pdftotext absent from PATH. Four requested Library IEEE files not found locally; transfer and authors/venue limit remain unverified.
- NEXT CONCRETE ACTION: export/reinspect corrected paper; then independently verify and expand MP01 native lesson against its actual numerical helpers, parameter table and diagrams, followed in order by MP02–40 and seven-model teaching pages. Existing prose remains a reusable starting point, not completed coverage. Preserve all scientific/resource gates below.
- Commands: `python scripts/build_learning_supplement.py --check`; `python -m pytest tests/test_learning_content.py tests/test_paper_evidence.py tests/test_evidence.py -q`; app environment `python scripts/check_learning_ui.py`. Publication commit/PR/CI to be appended after publication.

## Previous state (preserved provenance)

- Local branch `feat/mp27-attention`. Latest committed ingestion milestone `f46934e`; PR34 remote head `c97196b907ee43bb5e883edf92763ab548cb61a5`, exact-head Validation run40632438250 SUCCESS. Externally merged main `c1245a3`; agent has not merged any PR. Integrated external merge in42bec2b, preserving current milestone and newer checkpoint. No training or frozen evaluation rerun.
- Current milestone: corrected stale MP35 folder scope; MP36–40 display recorded teaching/scientific scope instead of an old numerical cutoff. MP35/37 teaching complete does not imply learner mastery or scientific generalization. Nested PNG/SVG/JPEG report figures now exposed. Validation: 3 evidence tests (0.23s), seven-section network-blocked AppTest,40 folder contracts,940 links,7 claim records. `reports/lesson-status-validation.json` pins source hashes. Implementation commitf08127a; integration42bec2b. Published draft PR35 https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/pull/35 on feat/verified-lesson-scopes; head705e3739a8323bf60a101835560c0e84badf413d, tree906d64faf2d2f8fa07b9cecfa149f296ac43352f exactly matches local42bec2b. Attached. Validation run40633080194 subsequently SUCCESS; PR35 externally merged89df7a7, integratedc6ac750. This checkpoint follow-up is local and should accompany the next milestone.

## Last verified evidence — reuse, do not regenerate

- PR33 externally merged880d96c, CI40572712377 SUCCESS; integrateda5c5ab5. Local reflectance job now accepts pinned self-contained GeoTIFFs, explicit same-grid window≤1024eachside, source-total≤512MiB, conservative quality intersection and atomic unclipped float32 export.36 tests passed2.34s including actual CLI subprocess output. `reports/reflectance-job-validation.json`; commands/schema `docs/local-reflectance-job.md`. Synthetic conversion/export only: no real-provider calibration or scientific SR inference claim.
- Seven-model cat-v1 registry implemented.39 distinct historical checkpoint bindings: M1=9,M3=3,M4=6,M5=6,M7=15; M2/M6=0. `reports/historical-architecture-bindings.json`. Full config/checksum/strict load/bit-exact synthetic verification;21 incompatible records excluded. No frozen test rerun.
-448 partial-cost candidates examined; M2 has no eligible width. Others: M1=20,M3=26,M4=20,M5=16,M6=26,M7=16. These match only the stated partial parameter/MAC tolerances, not complete FLOPs/runtime. `reports/architecture-budget-preflight.json`.7 untrained native2 seam smoke checks; attention uses global pooling and has no universal tiled equivalence claim.42 random-weight scale/export checks are implementation smoke, not trained capability evidence.
- All40 canonical folders have MP01-style files. MP20 independent SRCNN held-out table pending. MP36 numerical lesson, MP38 architecture, MP39 historical analysis and MP40 robustness teaching components executed; full scientific acceptance pending. MP37 teaching complete. Preserve original experiment reports/notebook records.
- Public browser actually executed older deployed CNN100step cached licensed example16x16→32x32, CPU displayed.002s. `reports/public-access-20261028.json` supersedes prior no-access observations. New local UI deployment version, cold-start recovery and durable storage unverified. Do not request same URL again. Local model cards verified desktop/phone; `reports/model-explorer-browser-20261028.json`.

## Next unfinished tasks, evidence first

1. Check latest PR/CI/main and preserve external merges. Publish checkpoint with next reviewable milestone; no automatic PR merge.
2. Produce trained scale/cascade capability entries from exact historical bindings, with resource preflight and representative seam/export checks. M2/M6 need bounded training design and resource review; do not relabel old weights. Advanced-family improvements/ablations in `docs/scientific-website-upgrade-brief.md` remain open.
3. Provider-specific reviewed metadata reconciliation, mixed-resolution quality alignment and real scientific ingestion remain unverified. Existing local ingestion command is ready. Protocol-v2 independent-region manifest/acquisition gates precede research training; no new held-out tuning. Dataset version/license/exact bytes/pair alignment and large-transfer authorization still pending.
4. Resolve MP20 and MP36–40 scientific criteria. Final40 independent fresh-kernel/output-hash batch belongs after substantive lesson changes. Existing `scripts/execute_project_notebooks.py`; do not repeat verified artifacts needlessly.
5. Paper all-page visual inspection and evidence-derived research/submission package, then remaining native website labs/UI and deployment verification. Paper compiled previously; IEEE uploaded template transfer and author/contact details unavailable. External S3/Postgres account configuration and live restart verification absent.

## Fixed requirements and resource limits

Read `AGENTS.md`, `docs/completion_brief.md`, `docs/scientific-website-upgrade-brief.md`, `docs/seven-model-completion-brief.md`, `docs/submission_readiness.md`. Latest brief supplements earlier requirements. Scientific reflectance vs display RGB vs optional generative enhancement must remain distinct. No fabricated model/quality/deployment claims.

Input width and height each≤1024, preserve rectangular/odd geometry. Scales2/4/8/16/32/64; exact output=s×input and pixel growth=s². Host execution cap1,048,576outputpixels; all choices visible with preflight/local job fallback. Disk-backed large output and bounded preview. Historical1024square→2048square ran;65536square only planned. Protocol-v2 is design only:5folds≥15regions,5km buffer,seeds17/23/41. No eligible frozen new manifest. Full training/large download require concrete resource approval; no paid services.

## Exact next commands / environment

Repo `outputs/Mini-Project-ML-Lab-RSSR`. Dev Python `../../work/clean-venv/Scripts/python.exe`; app Python `../../work/streamlit-venv/Scripts/python.exe`. PowerShell `$env:PYTHONPATH=(Resolve-Path src).Path; $env:PYTHONUTF8=1`.

```text
git status --short
git fetch origin
git log origin/main -3 --oneline
python -m pytest tests/test_evidence.py -q
python scripts/check_showcase.py
```

Run changed checks only. Ingestion checks when changed: `python -m pytest tests/test_reflectance_job.py tests/test_surface_reflectance.py -q`. GitHub connector publication helper `../../work/publish_text_entries.py` emits origin/main..HEAD text entries; compare local/remote tree SHA. Shell push previously hung. Attach each created PR.

## Publication checkpoint — 2026-10-28

- Local branchfeat/mp27-attention; preserved teaching milestone954a390, auditd753b0a. Published draft PR36 https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/pull/36, branchfeat/native-learning-ieee-audit, head60a8a184448934a9202fe8f33b036a896a8f0f6f, treebd201f9a2ec4de1185aa857dbfc4d13f751b93cb exactly matches locald753b0a. Attached. CI40690309970 IN_PROGRESS at last observation; no success inferred. No PR merge performed.
- Current checkpoint follow-up is local; include it in the next milestone. Tree clean before checkpoint edits. Independent publication recovered the previously interrupted work.
- Next commands: inspect PR36 exact-head CI and currentmain; read reports/ieee-paper/RESUME.md. User-reference gate still requires exact conference_041818.pdf and IEEEtran_HOWTO.pdf; confirm intended document.pdf if different from known export. Continue native lesson/model explanations independently; preserve all scientific/resource limits.

## Publication checkpoint — instructor milestone

Local implementation7400a2d onfeat/mp27-attention; published draft PR37 https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/pull/37 onfeat/self-contained-instructor, head0c1dacdde28c1d6c58fb86cae0128ed154b863dc. Remote treed00cefa7257987b8b48a81d1a3ad364522d75ba7 exactly matches local. Attached. CI40694075637 IN_PROGRESS at last check, no success inferred. No automatic merge or deployment verification. This checkpoint follow-up is local; carry into next milestone. Next: checkCI/main, implement MP02 native instructor using its verified numerical helpers, continue individual40lesson acceptance and report/PDF/deployment checks.
