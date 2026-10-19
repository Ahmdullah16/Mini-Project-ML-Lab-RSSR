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
