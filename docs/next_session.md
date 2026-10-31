# MP12–MP16 native lesson branch — 2026-10-31

- Latest confirmed main at intake: d39bbd5 (PR47 externally merged). Branch `feat/mp12-mp16-native-lessons-20261031` was one commit ahead at intake with `src/rssr/pipeline_lab.py` only. No prior MP01–11 work repeated.
- Added bounded Streamlit MP12–16 lessons in `app/math_training_ui.py` and routed through existing `app/learning_ui.py`. Reused canonical `rssr.linear_lab.approximate`, `rssr.training_lab`, and `rssr.pipeline_lab`; local MP16 tool now reexports shared implementation rather than duplicating it.
- Lab demos: MP12 SVD rank/error/tail energy/storage; MP13 analytical MSE gradient and SGD trajectories; MP14 NumPy backprop plus finite difference; MP15 PyTorch float64 parity/dropout/train mode; MP16 deterministic seeded five-sample training, incomplete minibatch and weighted losses. Synthetic only; no dataset downloads, 1000-step reruns, held-out evaluation, training checkpoints or public deployment changes.
- New numeric tests `tests/test_math_training_lessons.py`, Streamlit offline/AppTest `scripts/check_math_training_lessons.py`, and CI invocation. Source inspection: all six checks passed in `reports/mp12-mp16-native-instructor.json`. Crucially **new Python pytest/AppTest/CI are NOT yet proven passed**. Current model's shell cannot clone GitHub because DNS resolution fails. CI of PR branch must supply execution evidence; do not claim all five accepted until passing exact-head workflow logs.
- NEXT: inspect PR and exact-head CI; repair failures; update evidence with actual job run IDs, complete smoke checks in installed environment; do not merge automatically. Then MP17–22 native lessons batch; preserve frozen scientific and checkpoint assets. Unmet satellite scientific masks, region-v2, reports/paper and public deployment remain.

# Resume checkpoint — batched lessons, 2026-10-31

## Verified state
- PR46 head1585a22 CI41221845231 SUCCESS; externally merged5fb37bff, integrated8cb7fa1 preserving checkpoint. No agent remote merge.
- New MP06–11 native labs implemented as two shared modules. Twenty-one canonical tests passed; both batched AppTests passed (exact sampling, seeded noise/JPEG, FFT inverse/Parseval/cutoff, MSE/PSNR hand examples, SSIM windows, shared-reference dashboard and stale clearing). reports/mp06-mp11-native-instructor.json. One combined review milestone; no per-project publication loop.
- User preference: minimize tokens by batching related projects, reading shared helpers once, running affected batch checks once, and performing full checks at release. Keep per-project evidence distinct; do not lower acceptance standards.
- Preserve earlier public7c231604 eight-checkpoint/seven-family16×16→32×32 execution in reports/public-seven-family-20261030.json. Satellite public inference/cold start remains separate.
- Preserve portable notebook source from external PR45 and fresh-kernel default50cells/opt-in51cells with seven32×32RGB TIFFs: reports/portable-notebook-execution-20261031.json. No notebook upload/satellite acceptance claim.
- This MP06–11 batch is validated locally; publish review PR and inspect its exact-head CI. Frozen research outputs unchanged.

## Next actions
1. Publish the validated MP06–11 batch and inspect actual-head CI; integrate any external merge without overwriting progress. No automatic merge.
2. Next related batch: stage_04_math_pytorch MP12–16. Inspect canonical shared numerical/training helpers together; add bounded native lessons and batch tests. MP01–11 UI implementation does not establish full40 scientific acceptance.
3. Public Satellite Explorer seven-model inference, downloads, mobile layout, cold-start and configured persistence remain unverified. Preserve stateless semantics and source calibration/masks.
4. Continue scientific paired-data/region-v2 eligibility and matched-budget gates, final paper/report/PDF review/slides/package. Never relax frozen splits or substitute pilots for geographic evidence. External durable storage account and real author/venue details remain missing.

## Commands and constraints
- Work from repo root. Set PYTHONPATH=src and PYTHONUTF8=1 each shell; dev venv otherwise resolves an older editable checkout.
- Dev Python: ../../work/clean-venv/Scripts/python.exe; app Python: ../../work/streamlit-venv/Scripts/python.exe.
- After lesson changes: apppython scripts/check_degradation_lessons.py and scripts/check_metric_lessons.py for this batch; full40 navigation remains in CI. Canonical stage02/03 tests passed; logs ../../work/mp06-08-apptest.log and ../../work/mp09-11-apptest.log.
- Integrity: devpython scripts/build_learning_supplement.py --check; devpython scripts/audit_paper_sources.py --check. Both passed after MP03; repeat only affected checks.
- Notebook artifacts stay outside Git: ../../work/notebook-validation-20261031 and ../../work/notebook-model-validation-20261031. Reproduce in fresh kernel/new output_directory; enable run_models and pinned_repo_path only for opt-in variant. No training or dataset download required.
- Preserve all frozen evidence and published v0.3.0-rgb-pilots release/checksums. 42 small RGB model-scale cases already executed; full65536-square output not executed.

Historical details: [checkpoint archive](next_session_history_2026-10-31.md). Current acceptance: [acceptance matrix](acceptance_matrix.md). Requirements: [completion brief](completion_brief.md), [mandatory website brief](mandatory-website-brief.md).
