# Progress

Last reconciled: 2026-10-18. No learner mastery or completed research claim.

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
