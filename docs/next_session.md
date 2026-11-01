# PR stack integration checkpoint — 2026-11-01

- User explicitly authorizes merging reviewed PRs only after latest required checks pass; never bypass protection. Earlier no-auto-merge notes below are historical and superseded within this review scope.
- No open PRs at intake. PR48 reached main d05b695; PR49/50/51 were merged only into feature branches, leaving MP17–26 off main. Integration order48→49→50→51; consolidate unique changes, do not reimplement them.
- Isolated worktree: work/pr-review-20261101, branch review/pr-integration-20261101. Original outputs/Mini-Project-ML-Lab-RSSR workspace uncommitted MP12–16 work preserved untouched. Use the review worktree for this integration.
- Fixed periodic-pattern label and sequential attention equation/diagram; strengthened numerical UI assertions. Local full pytest507+17subtests, three affected AppTests,40IDs/979links, report/paper integrity passed. Source PR49/50/51 exact-head CI passed. reports/pr-stack-review-20261101.json records scope.
- NEXT: publish integration PR; require latest-head CI/review eligibility; normal merge only. Then run post-merge app/notebook checks, verify public version where accessible and update this checkpoint. Protection endpoint403 is API permission, not an app-access failure.
- Existing frozen research/checkpoints unchanged. MP17–26 are bounded educational controls, not fully accepted trained models. Remaining native work starts MP27; scientific geographic/real-paired, paper/report/slides, durable store and deployment acceptance remain open.
- Review venv commands from work/pr-review-20261101: set PYTHONPATH=src and PYTHONUTF8=1; ../clean-venv/Scripts/python.exe -m pytest; ../streamlit-venv/Scripts/python.exe scripts/check_attention_native_lessons.py (plus CNN/convolution checks). Reuse completed checks unless new changes justify reruns.

## Historical checkpoints (current state above takes precedence)

# MP23–26 native attention milestone — 2026-11-01

- PR50 exact head 847cc729 **CI success** run41232412252 (both tests and Streamlit including MP18–22), no merge. Prior PR48 and PR49 CI also passed; preserve stacked review chain.
- New stacked branch feat/mp23-mp26-native-attention-20261101 introduces app/attention_native_ui.py using canonical InspectableAttention and SRModel for MP23 channel gates, MP24 spatial gates, MP25 combined gates, MP26 zero-residual x2 shape/skip. Seeded random gate values explicitly NOT trained inference; no original frozen evidence changed. tests: scripts/check_attention_native_lessons.py wired to CI, not yet run.
- Next: inspect exact-head CI and correct failures, record job IDs; then MP27–31 native transformer lessons. Public satellite, scientific geographic test and final submission acceptance remain separate. Do not auto-merge.

2026-11-01: PR50 run 41229517239: tests success, Streamlit failure in scripts/check_cnn_native_lessons.py from incorrect MP19/MP21 button key case (mp19_run vs MP19_run). Corrected in ec88eba6; exact-head CI retest pending. No research results altered.

# MP18–22 native CNN teaching milestone — 2026-10-31

- Current stacked chain: PR48 MP12–16 (CI run 41227394028 SUCCESS, exact head 0a9db78), PR49 MP17 (CI run 41227558834 SUCCESS, exact head ecc9bac), new branch feat/mp18-mp22-native-cnn-20261031 based on PR49. Neither predecessor merged; no old verified work rerun.
- Added five bounded Streamlit MP18–22 experiments reusing canonical Sobel (local MP18), Reconstruction/UpsamplingCNN and SRModel source. Explicit Run, mathematical notation, shapes, diagrams, synthetic measurements, CSV/JSON. Zeroed parameters are labeled architectural controls, NOT trained SR performance. No frozen evidence or checkpoint touched.
- Tests/tests/test_cnn_native_lessons.py and scripts/check_cnn_native_lessons.py added to existing CI. **Actual exact-head CI pending**, not claimed passed. Evidence reports/mp18-mp22-native-instructor.json. Next: inspect PR exact-head test jobs, repair, update report; then progress MP23–26. No automatic merge. Full independent geographic/masked research remains incomplete.

# MP17 convolution native lesson — 2026-10-31

- Stacked development branch feat/mp17-native-convolution-20261031 on MP12–16 PR48. Preserved previous work and CI run171 success at prior head96930eb; newer documentation head not yet CI-confirmed.
- Added app/convolution_lesson_ui.py using rssr.convolution_lab for actual float64 9×9 impulse/ramp/constant, kernel 1/3/5, stride1/2, padding0–2, dilation1/2; exact output geometry, receptive field, comparison table, CSV and JSON exports. Explicit Run; no data download or learned checkpoint. Wired to app/learning_ui.py. Added numeric and offline AppTest checks in CI.
- Current MP17 source additions NOT yet executed under GitHub Actions. Read reports/mp17-native-instructor.json for static evidence. Next: inspect stacked branch CI and repair, then continue related MP18–22 native CNN teaching. No automatic merge. Scientific research and public deployment acceptance remain open.

# Validated MP12–MP16 instructor batch — 2026-10-31

- PR48 prior head 96930eb CI run171 / 41226551966 **SUCCESS**. Both jobs passed: tests job 124611801103 including full pytest and source/structure checks; Streamlit job 124611801412 including scripts/check_math_training_lessons.py and all existing app checks. This confirms actual on-site synthetic lesson execution, not independent scientific acceptance.
- Evidence report reports/mp12-mp16-native-instructor.json includes pinned prior head/run IDs. This documentation-only follow-up will get a *new head*, whose CI must be checked separately. Do not claim that exact new head passed until confirmed.
- Next unfinished native batch is MP17–22, preferably a separate stacked review branch. No code changes to MP01–16 after this checkpoint; all earlier training/checkpoint/frozen evaluation artifacts preserved. No automatic merge.

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
