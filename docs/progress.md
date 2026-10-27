# MP40 historical robustness and efficiency — 2026-10-27

[MP40](../stage_09_hybrid_research/MP40_robustness_and_efficiency_benchmark/README.md) now derives30 condition rows,5 efficiency records and a visually checked [curve figure](../reports/figures/mp40-robustness.svg) from existing development evidence. Five tests, report/figure generation and expanded three-cell real kernel passed. No new benchmark or frozen inference. MP38–MP40 expanded components are verified, but full scientific acceptance remains In progress. Unknown memory/compute/cross-sensor results remain unavailable.

# MP39 historical ablation analysis — 2026-10-27

[MP39](../stage_09_hybrid_research/MP39_controlled_ablation_study/README.md) expands the24-run historical analysis with source lineage, eight-variant table, explicit null geographic CIs, synthetic paired-bootstrap example and solved exercises. Seven tests and three-cell real kernel passed. [Report](../reports/mp39-ablation-analysis.json). No new training/inference; full status remains In progress because independent-region/matched-budget evidence is missing.

# MP38 architecture component — 2026-10-27

[MP38](../stage_09_hybrid_research/MP38_hybrid_cnn_attention_transformer_sr_model/README.md) now has an expanded architecture equation/diagram, worked examples, actual shared-model shape trace and zero-residual/gradient checks. Five tests, standalone synthetic experiment and three-cell real notebook passed. [Report](../reports/mp38-architecture.json). No training, downloads or frozen evaluation. Status remains In progress: matched-budget quality comparison and reference-error atlas are unfinished.

# MP37 feature fusion completed — 2026-10-27

[MP37](../stage_09_hybrid_research/MP37_feature_fusion_laboratory/README.md) implementation/teaching scope complete: expanded notebook, worked algebra/gradients, fusion diagram, parameter/MAC derivation, shared-model fixture and reused historical training comparison. Seven tests and a three-cell real kernel passed; transient initial Windows DLL error documented. [Report](../reports/mp37-fusion.json). No new training or test evaluation. MP36 remains In progress; MP38–MP40 and MP20 evidence gap remain. Learner mastery unassessed.

# Transfer implementation and paper reconciliation — 2026-10-27

Single-attempt archive transfer added with pinned length/checksum, temporary-file cleanup and no retries;7 offline tests passed. No live download. Source-access statements reconciled in paper/literature; correction reference added; generated bibliography/evidence refreshed. Three paper provenance tests and desktop LaTeX compile passed. [Validation](../reports/intake-paper-validation-20261027.json) separates software/build evidence from unexecuted scientific work. Exact archive URL/bytes, approval, v2 validation and PDF visual review remain open.

# Archive intake and source-methodology milestone — 2026-10-27

PR25 CI passed and external merge integrated. Local ZIP metadata inspection and selective extraction now enforce byte limits, path checks, new destinations and cleanup;21 tests passed. The [feasibility addendum](../reports/mp36-feasibility.md) resolves reading the published registration method/correction through publisher-provided full texts. Original method is not a measured v2 guarantee. No archives, training or frozen test inference ran. MP36 remains In progress; network transfer enforcement, version-specific verification, resource approval and actual paired evidence remain unfinished.

# Cross-sensor intake milestone — 2026-10-26

[Compatibility matrix and resource proposal](cross-sensor-intake.md) now distinguish verified source statements from pair-level unknowns. Metadata-only admission CLI passed21 targeted synthetic tests. No imagery download, registration fitting or new accuracy study ran. MP36 remains In progress; publisher access, calibration/registration policy and resource approval remain gates. Existing paper and frozen results preserved.

# Evidence-derived paper milestone — 2026-10-26

PR23 passed CI40280953883 and was externally merged at dffe38c; integrated locally at951aab0. [Working IEEE-style paper](../paper/README.md) now separates historical pilot, geographic validation, frozen single-scene test and unexecuted v2 design. Four tables and numerical interpretation are generated from unchanged source JSON, with hashes and row pointers in [paper evidence](../reports/paper-evidence.json). Three generator/provenance tests passed; seven claim-source hashes and891 relative links passed. No new training, imagery transfer or test inference ran. Desktop LaTeX compilation subsequently passed; PDF visual layout remains unverified. PR24 passed CI40282443346 and was externally merged07d9002. Build evidence: reports/paper-build-validation.json. Submission remains incomplete.

# MP36 numerical evidence milestone — 2026-10-26

PR22 passed CI40280123380 and was externally merged. MP36 remains **In progress**: seven numerical tests, analytical report and four-code-cell real notebook passed, illustrating registration/calibration/spectral mismatch without claiming sensor evaluation. [Evidence](../reports/mp36-numerical-examples.json), [validation](../stage_08_remote_sensing/MP36_cross_sensor_dataset_investigation/VALIDATION.md). WorldStrat v1.1 source-size audit added. Official SEN2VENµS paper/correction direct retrieval returned429; methodology gate remains open. No archives/training/frozen evaluation ran.

# Research design and CI repair — 2026-10-26

[Research questions](research_questions.md) and [protocol v2](protocol-v2.md) committed before new test inspection; execution remains gated. [Claim/evidence register](claim_evidence.json) pins7 source records with limitations. MP36 official-source review advanced to SEN2VENµS v2; no archives/training executed. PR21's tests job passed but Streamlit failed on a stale selector in the catalog validation script. Corrected keyed-control catalog check passed locally for all3 models; [new evidence](../reports/scale-model-catalog-check.json). Replacement CI pending; external merge is not a passing-CI claim.

# Exact-scale execution milestone — 2026-10-26

Implemented the corrected maximum INPUT contract (each side1..1024) and visible ×2/4/8/16/32/64 choices across the scale workflows. Explicit native ×2 cascades write incremental BigTIFF stages; preflight shows exact dimensions, pixels and configured estimates. Streamlit rejects above1,048,576 output pixels without resizing and exports exact-input/model local jobs. Scientific GeoTIFF remains float32; display cascades explicitly quantize between stages. [Execution guide](scale-execution.md).

Validation:58 targeted tests passed; eight scientific/fresh-cache tests rerun after final-output atomic rename passed. Three actual published models ran all six scales on small odd fixtures;1024×1024→2048×2048 executed; disk/full-tile seam comparisons recorded in [scale validation](../reports/scale-validation.json). Three-model UI comparison/cache/retry, exact-source satellite fixture, portable RGB job/Gradio callback and two-process empty-cache restart checks passed. Reports are local synthetic/implementation evidence, not new frozen scientific results. Full65536-square execution and public deployment remain unverified. MP36 and broader research remain unfinished.

# MP36 investigation started — 2026-10-25

Official source review and [preliminary feasibility matrix](../reports/mp36-feasibility.md) recorded. MP36 remains **In progress**: newer dataset version, license-file reconciliation, numerical examples/tests and full notebook are pending. No archive download or cross-sensor evaluation ran. MP35 is published in [draft PR20](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/pull/20).

# MP35 geographic split milestone — 2026-10-25

MP35 **Complete (implementation and teaching deliverables)**. Twelve tests, nine cached checksum/header inspections, 24 cross-split checks, split manifests, scene-center map and four-code-cell real-kernel notebook passed. [Audit](../reports/mp35-geographic-splits.json) and [validation](../stage_08_remote_sensing/MP35_geographic_dataset_splitting/VALIDATION.md). Existing 4/4/1 assignments and frozen results preserved. MP36–MP40, earlier exceptions and learner mastery remain unfinished.

# MP34 pairing milestone — 2026-10-25

MP34 **Complete (implementation and teaching deliverables)**. Three tests, synthetic/real training-only pairing experiment and five-code-cell notebook passed. Four-band64×64 HR generated aligned×2/×4 LR with exact common corners; hashes and metadata in [pairing manifest](../reports/mp34-pairing-manifest.json), [validation](../stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/VALIDATION.md). No new training/download/test evaluation. MP35–MP40 and earlier exceptions remain unfinished; learner mastery unassessed.

The updated seven-section showcase navigation/error/filter checks completed with zero navigation network access. Inference diagnostic and comparison behavior remain as recorded below. Frozen final-heldout.json and final-evaluation.json are unchanged from main. External public visibility/storage blockers remain.

# Inference diagnostic and MP33 — 2026-10-25

PR18 CI40191897702 passed and PR18 was merged externally0c7d91c. Preserved and integrated. Versioned training-only inference protocol3d41dd7 executed six actual runs; no preprocessing mismatch demonstrated. Input quantization and approximate hybrid tile context were measured, with no accuracy-gain claim. [Diagnostic interpretation](../reports/inference-diagnostic-v1.md). Added nonfinite-output guard and explicit clipping/range metadata; six affected diagnostic/comparison tests and actual three-model cache/retry check passed.

MP33 **Complete (implementation and teaching deliverables)**: train-only normalization, frozen application, per-band validity, georeferenced patch extraction, synthetic and pinned real training experiment, notebook, exercises/solutions and [validation](../stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/VALIDATION.md). Three tests and real kernel passed. Sixteen valid patches; existing frozen model preprocessing unchanged. MP34–MP40 and earlier unmet criteria remain unfinished; learner mastery unassessed.

# Explicit model comparison — 2026-10-25

Live SR now offers Compare models: three genuine published RGB×2 checkpoints, one decoded input, sequential bounded execution, input/checkpoint/version/settings identities, recoverable per-model errors, synchronized crop/zoom, shared preview-only contrast, individual PNGs and ZIP provenance. Unsupported standalone models/perceptual mode and absent HR are labeled honestly. [Actual inference validation](../reports/comparison-validation.json), [browser/export check](../reports/comparison-browser-check.json).

Two identity/alignment/ZIP tests passed; all three real models ran; selection changes preserved labels, matching cached results skipped recomputation, a controlled failure retained successful results and retry recovered. Seven-section/error/filter checks passed, six affected comparison/index/interpretation tests passed, 818 links valid. No improvement in fidelity claimed; no frozen results changed. Numerical equations/controlled example added beneath measured Results.

# MP32 and result interpretation — 2026-10-25

MP32 **Complete (implementation and teaching deliverables)**; three local tests, synthetic/real training-raster experiment and real-kernel notebook passed. [Validation](../stage_08_remote_sensing/MP32_satellite_image_explorer/VALIDATION.md). Learner mastery unassessed. MP33–MP40 and earlier exceptions remain unfinished.

Added reusable source-driven metric definitions, filtered observations and protocol/task isolation; overview navigation actions, derived run counts, reconstruction dimensions and satellite interpretation. Four affected index/interpretation tests passed. Seven-section navigation passed before the final additional filter assertions; see latest checkpoint for their outcome. No frozen metrics changed.

PR17 CI40175306388 succeeded and PR17 was merged externally as199bb10. Public URL checked October25 still redirects to not-found/no-access; available browser is signed out. Repository/branch/entry point/build logs cannot be diagnosed without the owner's Streamlit dashboard. No deployment mutation or new public inference claim.

# All project-folder deliverables populated — 2026-10-25

MP01 preserved; MP02–MP40 now contain a local notebook, tools, tests, runner and VALIDATION.md alongside README.md. Shared implementations are reused. All 40 IDs and folder layouts passed validation; 111 local tests +17 subtests passed; all 39 relocated notebooks passed real kernels. Two evidence-index tests and seven local showcase sections/error states passed. App links now target local notebooks. See [workflow](project_folder_workflow.md), [tests](../reports/project-folder-tests.json) and [execution](../reports/project-notebook-execution.json).

This completes the requested file-layout milestone, not every mini-project criterion. MP32–MP40 remain introductory; MP20 and other earlier checklist gaps remain open. MP27–MP31 implementation/teaching status remains complete. No frozen test results, datasets or training checkpoints were regenerated. Candidate remote CI and public verification remain pending.

# Stage07 implementation complete — 2026-10-23

MP27–MP31 **Complete (implementation and teaching deliverables)**; learner mastery unassessed. Added MP29 encoder/positions/shape trace, MP30 reversible patches and non-wrapping window connectivity, MP31 measured Transformer/CNN benchmark export and failure analysis. Each has local implementation/tests/runner/VALIDATION.md. Five, five and three new tests passed; all three six-cell notebooks passed real kernels; five new figures visually inspected. See [stage07](../stage_07_transformers/README.md). No new optimization, acquisition or frozen-test evaluation. MP32–MP40 and earlier unmet criteria remain unfinished.

PR16 head `c8a1757555b582d6c17203d5491bcc84066a7615` passed both Linux jobs, run39991113761, and was merged externally as75412c3. New stage07 follow-up candidate CI remains pending publication. Public access blocker below remains unresolved after the confirmed-URL retry.

# MP28 implementation complete — 2026-10-23

MP28 **Complete (implementation and teaching deliverables)**: local multi-head code, five tests, real-kernel seven-cell notebook, runner, measured one/two/four-head comparison, head maps and parameter counts. [Validation](../stage_07_transformers/MP28_multi_head_self_attention/VALIDATION.md). Learner mastery unassessed; MP29–MP40 remain incomplete. Both new lessons use existing training data only.

October23 public retry of the confirmed app URL redirected twice to Streamlit's not-found/no-access page. Public deployment/source identity cannot currently be verified; account visibility restoration is external. Local showcase navigation/error states passed after setting a consistent60-second AppTest timeout for cold imports; the earlier3-second timeout failure is not counted as a pass.

# MP27 implementation complete — 2026-10-23

MP27 **Complete (implementation and teaching deliverables)**: local implementation, tests, notebook runner, eight-cell lesson, hand example, real training-crop experiment, feature-scale and analytical-memory plots. Six tests and real Jupyter execution passed; [criterion-level validation](../stage_07_transformers/MP27_self_attention_from_scratch/VALIDATION.md). Learner mastery unassessed. MP01–MP26 remain expanded with their original exceptions; MP28–MP40 remain incomplete. No frozen result was changed.

Showcase PR15 head `04e7c2e536d8abe6e28b27d29f7a75325f86a0ac` passed both Linux jobs, run39989197874, including fresh data reconstruction, navigation, clean app installation and all-model inference. No agent merge or new public showcase verification.

# Research showcase checkpoint — 2026-10-23

Recovered and validated seven-section Streamlit showcase, shared measured-results index, source downloads, learning curves, project navigation and synchronized image zoom. Local 89 tests plus 4 subtests, all-model inference, fresh-cache restart and desktop/mobile checks passed on October 21; see [validation](../reports/showcase-validation.md). Publication and public verification of this showcase remain pending. No lessons were marked complete by this interface change.

# Verified portability fix — 2026-10-21

PR #14 code commit `104369288a54ee714a330f78eb028969484381a6` passed both Linux jobs (run39647819460): fresh reconstruction of all288 frozen tiles, database checks, clean deployment install and inference for all3 models. Local87 tests plus4 subtests and419 links/40 IDs passed. Only platform ZIP metadata and sub-micrometre derived-coordinate roundoff are normalized; exact frozen pixel-file hashes remain required. The preceding Linux failures are resolved on this branch.

Actual public hybrid inference, downloaded PNG/provenance and deployed source identity were verified after sleep/wake; see [public evidence](../reports/public-hybrid-validation.json). MP01–MP26 expanded and executed; MP27–MP40 and other [readiness](submission_readiness.md) gaps remain. See [latest checkpoint](next_session.md) before continuing. This is not a complete-submission claim.

---
Earlier checkpoint follows and is superseded where inconsistent.

# Latest continuation — 2026-10-21

PR #11 is a draft on `feat/submission-integration`. Published commit `1baf02121ca9814e3533ec9ed41957846cb5f7b8` passed both Linux CI jobs, run 39578629874, including all three hosted checkpoints and max-size inference. Its tree matches local `ce10a0f`. The pre-test local freeze `9109766` is preserved as tree-equivalent remote parent `f9e4880`; remote publication was later, not evidence of test timing. No PR was merged by the agent.

MP26 now expanded and executed successfully; its six recorded development curves and compute table are in `reports/mp26-attention-sr.json`. No new training or test evaluation. MP01–MP26 are expanded; MP27–MP40 still need full expansion, and earlier project-specific exceptions such as MP20's independent SRCNN test table remain explicit.

`scripts/rebuild_frozen_data.py` downloaded 16,883,208 raw bytes plus 142,290 manifest bytes into a new empty directory. All 144 RGB and 144 RGBNIR tiles matched frozen checksums and complete manifest structures. Original-byte manifests restored only after equality checks. SQLite integrity and foreign keys passed; repeat used zero raw bytes; controlled corruption was rejected, then restoration recovered successfully. Evidence: `reports/frozen-data-rebuild.json`; instructions: `docs/frozen-reproduction.md`. Linux rebuild added to CI, pending the next published head. Existing held-out result remains unchanged.

Next: publish this MP26/rebuild follow-up and verify its Linux CI; continue MP27–MP40, portable four-band research checkpoints, normalized external database, paired cross-sensor feasibility, final report/slides/package and final all-notebook gates. Existing public pilot remains available; new model selector has not been verified deployed. No new host/paid resource is needed. Durable external account credentials remain absent.

---
Earlier checkpoint (superseded where noted):

# Current checkpoint — 2026-10-21

Preserve branch `feat/submission-integration`. Main advanced to `5d6a2ea` through external merge PR #10 (frozen RGB artifacts); no open PRs at this check. Do not merge PRs or overwrite existing work. The source integration remains local until published and CI verified.

MP01–MP22 expanded and executed; MP23–MP25 have executed attention comparisons and expanded notebooks; all three passed real kernels. MP20's separate held-out SRCNN table remains unfulfilled; none of these statuses asserts learner mastery. MP26–MP40 still need full lesson expansion, even where underlying research experiments exist.

Research: 36 development records completed, with 30 new and six reused runs. Freeze protocol was committed at `9109766` BEFORE the single held-out evaluation. `reports/final-heldout.json` contains the executed results; do not retune or overwrite. RGB held-out mean PSNR: bicubic 48.5183, CNN 50.2614, hybrid 50.2459, official SwinIR 48.9897 dB. One scene and synthetic degradation cannot establish cross-sensor/general geographic validity. No hybrid superiority claim. Official SwinIR has different pretraining/degradation. Robustness/spectral/CPU-efficiency evidence is in `reports/development-benchmark.json`.

Public app https://mini-project-ml-lab-rssr.streamlit.app/ recovered at the same URL. Recorded tests passed sample, uploaded RGB, repeat-byte equality, maximum 512→1024, corrupt/transparent/oversized input rejection, and actual Sentinel acquisition with georeferenced downloadable output. This evidence is from the existing pilot deployment; no new deployment is claimed. Exact deployed source identity remains unverified.

New local Streamlit model selector preserves the pilot and adds genuine frozen RGB CNN/hybrid. Immutable artifact commit `b21b797a2c43f3a45879cd6778be4cd81600f2e7` contains downloadable SHA-verified weights and original-byte dataset manifests. All three models passed AppTest and actual maximum-size inference in the deployment-only environment; RSS after inference, not peak, was 259–283 MB. The new selector has not been verified on the public host. New code reports source identity and gives friendly bounded-upload errors. Preserve Gradio and stateless hosting; local caches/SQLite are not durable.

Ingestion scheduler/schema v3 is committed: leases, durable pending windows, page/cursor recovery, crash reconciliation and overlap watermark. Four scheduler tests and live metadata-only pagination passed; no external always-running scheduler was installed. No external-storage account credentials are configured. Offline S3/PostgreSQL tests do not establish live durability or complete the normalized external database requirement.

Validation before attention additions: 83 tests plus four subtests passed in 12.01 seconds. MP19–MP22 passed real kernels. Attention comparison ran CPU seed23, 300 updates, four train/four validation scenes: reconstruction PSNR off35.0253, fixed0.5 35.1894, channel35.1721, spatial35.6038, combined35.6899 dB. This is same-grid teaching reconstruction, not SR or a new research test. See `reports/mp23-mp25-attention.json`; focused attention tests passed.

Latest checks: 85 tests and four subtests passed in 34.13 seconds; 412 relative links/40 IDs passed; existing held-out result was preserved without repeat evaluation.

Next: commit validated checkpoint, integrate main's additive artifact merge, publish a reviewable draft PR and verify exact-head CI. Continue MP26–MP40, frozen dataset/checkpoint rebuild in a fresh checkout (Windows separators fixed in script consumers; original manifest byte/hash portability still needs validation), four-band portable checkpoints, normalized durable DB, real paired satellite feasibility, final report/slides/package and final all-notebook gates. Do not mark whole project submission-ready.

Environment: isolated `work/clean-venv` points editably at an older clone; set PYTHONPATH to current src. Use PYTHONUTF8=1 and writable work/tmp for Jupyter. CPU only; zero GPU. Keep within original 2 GB downloads/2 GPU-hours authorization; prior retry download overhead is incompletely measured. No new imagery/downloads for attention lessons.

---
Historical notes follow; the checkpoint above supersedes conflicting old status claims.

# Current progress — 2026-10-19

Latest resumable state: [next session](next_session.md). MP01–MP18 expanded; MP19–MP40 pending full implementation. Streamlit stateless deployment prepared with genuine public checkpoint; local inference/restart checks passed, public URL unverified. Durable external storage remains a separate incomplete requirement. See [deployment settings](streamlit-deployment.md).

Latest checks: 73 tests and four subtests passed without warnings; 40 IDs and 394 relative links passed. Deployment-only clean install and two empty-cache restart runs passed (sampled peak RSS at most 300.4 MB). Local browser Sentinel acquisition, inference and georeferenced download passed; public hosting and live external storage remain unverified.

MP16 expanded and executed: seeded regression and exact eight-step checkpoint recovery passed; see [MP16 evidence](../reports/mp16-pipeline.json). Streamlit checkpoint 6c118ea passed both Linux CI jobs (run 39247571971), then PR #7 was merged externally. Deploy from main.

MP17–MP18 expanded and executed in isolated kernels. Convolution agrees exactly with PyTorch for the tested cases; autograd confirms an 11×11 receptive span with 81 contributing pixels. Learned Sobel imitation has 8.68e-9 maximum kernel error; this is not independent edge accuracy. See [convolution](../reports/mp17-convolution.json) and [edges](../reports/mp18-edges.json). Full suite: 75 tests and four subtests passed without warnings in 14.44 s. No new imagery downloads or test evaluation.

The following is historical milestone evidence; newer counts and next actions above supersede older entries.

# Progress

## Latest validation — 2026-10-19

MP05–MP11 expanded and executed with exit 0; seven new degradation/metric tests and three inspected figures. Six geographic runs completed: mean CNN validation delta +1.5709 dB, hybrid +1.3337 dB; held-out test unused. See reports/geographic-study.md. New fresh-output paths are portable, with relocation/corruption/deduplication checks. Portable bundle launch and HTTP output downloads passed (reports/portable-demo-http.json).

Isolated Python 3.12 suite passed 64 tests and four subtests in 13.39 seconds without warnings. Earlier run without writable TEMP failed due sandbox paths; corrected environment passed. Cache fixture initially lacked a required ingestion timestamp; corrected. The GeoTIFF warning was caused by accidental alpha metadata in the test fixture; explicit non-alpha multispectral storage resolved it.

Next incomplete teaching task: MP12 Linear Algebra Visualizer. MP12–MP40 still need full implementations. Incremental watermark/scheduler recovery, real paired satellite evaluation, final selection/test, report/slides/package and public deployment remain incomplete. All 40 current notebooks passed the isolated batch with process exit 0; source hashes verified. Do not claim submission readiness.

## Historical milestone

Last reconciled: 2026-10-19. No learner mastery or completed research claim.

## Implemented and executed

- Original nine stages, all 40 IDs/titles/objectives, and MP01 retained. All 40 notebooks ran in real Jupyter kernels; source hashes and outputs were checked. See reports/notebook-execution.json. MP03-MP40 remain introductory lessons with substantial full-project criteria pending.
- MP02 expanded with defined mathematics, indexing/broadcasting examples, masked channel statistics, worked solutions, two tests and a real 64x64 four-band training-crop experiment. Loop/vectorized brightness matched exactly; masked statistics matched within 1e-12. See reports/mp02-array-lab.json. Learner exercises remain unassessed.
- Full suite: 43 tests and 4 subtests passed in 13.82 seconds. Two known GeoTIFF warnings remain. Exact interrupted-training resume is tested.
- Local Gradio HTTP inference passed: HTTP 200; 32x32 RGB input produced two 64x64 RGB outputs with matching pilot checkpoint hash. See reports/demo-http.json. Upload shape/dtype/size rejection tests passed.
- STAC metadata refresh is idempotent. Explicit --resume-job recovery supports saved GET/POST cursors and merged pagination bodies. Two recovery tests passed. Live resume added 9 items, remained partial at its page budget, and downloaded no assets. See reports/ingestion-resume.json.
- Five original 100-step CPU pilots preserved. A new eight-variant, three-seed pilot completed 24 runs; the ledger can be resumed without replacing successful runs. All checkpoint hashes were reverified with --resume. Test evaluation remains unused.

## Measured research outcome

One validation scene, 16 patches, synthetic x2 degradation of real NAIP aerial imagery. These are not cross-sensor satellite results. Three-seed mean PSNR: hybrid 43.6869 dB, learned gate 45.2122, constant 0.5 gate 45.0625, consistency objective 43.6734; bicubic 45.8104. Every learned run remains below bicubic. Constant attenuation explains much of the gating gain; novelty is not established. See reports/ablation-pilot.md, its run ledger, figures and preliminary research-draft.md.

Numerical scaling is fixed stored values / 10000, with physical calibration unverified. Seed variability is not geographic uncertainty; one scene cannot support a scene-level confidence interval. No final model has been selected.

## Failures and limits

The first notebook attempt failed because Matplotlib was absent; it was installed in the existing venv and rerun. All notebook cell outputs now pass, but Windows Jupyter/ZeroMQ emitted shutdown warnings (including a connection-reset assertion on one rerun). Explicit Python exit code on the final MP02 rerun was 0. Routine CI and Colab execution have not been verified by this session.

Known cached dependency wheels occupy 233,718,558 bytes; verified dataset files total 5,627,736 bytes. Prior retry/interrupted transfer overhead was not instrumented, so exact total network bytes cannot be claimed. All work used CPU, 0 GPU-hours. No new imagery was downloaded during this continuation.

## Pending or externally blocked

Public hosting needs an authenticated hosting account and verified free entitlement. HF documentation has a free ZeroGPU exception for eligible personal accounts; ordinary compute Spaces require a paid plan. No plan was activated or public URL claimed. The local application still lacks full catalog/AOI UI, model selection, licensed in-app sample selection and metadata download controls; scientific GeoTIFF export is available through the CLI.

Pending: full MP03-MP40 teaching content/solutions/real experiments, additional independent geographic scenes, real cross-sensor alignment/calibration, fresh asset acquisition/inference, automatic timestamp watermarks and scheduler recovery, strong official baseline, robustness/spectral/efficiency study, complete literature extraction, thesis/presentation deliverables, durable hosted storage, frozen final selection and held-out evaluation. The preliminary draft is not publication ready.

## Continuation verified 2026-10-19

PR #2 was already merged at c565276b1eda951b17d2dbabccc0f48e98e272af; the starting working tree was clean. Prior run records and checkpoints were preserved.

MP03 and MP04 now contain expanded mathematics, hand calculations, practice and separate solutions, reusable NumPy operations, real training-crop experiments and measured figures. Six new offline tests passed. Both notebooks passed real kernels with explicit exit code 0. The first attempt stalled under network restrictions; granting session network access resolved both kernel startup and Git fetch. Existing local-kernel TCP warnings remain.

MP03 used a checksummed 64x64 four-band training crop, area-average x2 degradation, and declared interpolation conventions. Bicubic had the lowest MSE on that crop. MP04 confirmed zero error for identical images and 0.0004 MSE for +0.02 brightness; a controlled shift demonstrated alignment sensitivity. No test scene or new download was used. Figures were visually checked. See reports/mp03-mp04-experiments.json.

The demo now exposes read-only catalog freshness and acquisition/availability/ingestion times. Two database tests passed, including byte-for-byte nonmutation and distinguishing a newer partial job from an older complete job. Local HTTP returned 19 cached observations, latest status partial, and no complete refresh. No live-sensing claim or asset download. See reports/catalog-http.json.

Full MP05-MP40 expansions, research generalization, fresh asset ingestion and public hosting remain unfinished. No new research-training result, learner completion or publication-readiness claim is made.
`python -m pytest`: 51 tests and 4 subtests passed in 11.47 seconds, with the same two known GeoTIFF warnings. Structure validation: 40 IDs and 365 relative links passed.

## Seven-model implementation milestone — 2026-10-27

Versioned C/A/T registry added without altering historical SRModel definitions. Twenty focused tests passed; seven structures are distinct and five historical structure mappings preserve exact outputs under strict state-dictionary loading. Training/evaluation/deployment status for the new registry configurations remains unverified; checkpoint and training provenance null. See docs/architecture-registry.md and reports/architecture-smoke-matrix.json. No frozen experiments rerun. Latest acceptance requirements preserved in docs/seven-model-completion-brief.md.

Training integration follow-up: versioned registry configs now route through existing train/checkpoint loader/evaluator/CLI and database model identity.9 tests passed8.83s, including seven one-step synthetic fixture runs and legacy resume regression. Initial fixture foreign-key failures were corrected by registering tiles; constraints preserved. Not a research training milestone; no deployed checkpoint added.

## Architecture budget/seam milestone — 2026-10-27

PR30 CI40505931536 passed and external merge981ec51 integrated locally12c3285. Added tested partial Conv/Linear/MHA accounting, executed448 candidate budget search and7 new-variant tiling checks. Four unit tests passed3.50s; search4.90s. M2 lacks an eligible partial-budget match in searched widths; no match/quality invented. See docs/architecture-preflight.md and reports/architecture-budget-preflight.json / architecture-seam-smoke.json. No training or frozen evidence reruns; all scientific acceptance gates remain.

Historical binding follow-up:39 distinct pilot checkpoints verified against original report SHA256/config, checkpoint/training identity, strict registry load and exact synthetic5x7 output equality.21 incompatible gate/fusion records excluded; M2/M6 still lack trained bindings.11 mapping tests passed0.11s. reports/historical-architecture-bindings.json records per-run provenance, no dataset/test reopening and no deployment mutation. New matched-budget training is not executed.
