# Resume checkpoint — 2026-10-31

## Verified state
- PR43 merged externally as 7c231604 after CI41088399016 passed. Public browser verified this version and all eight checkpoints (seven families) on one cached16×16→32×32 request. Evidence: reports/public-seven-family-20261030.json. No access blocker. No public satellite inference/cold-start claim.
- PR44 https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/pull/44 remains open: ff5f500a06fc0acf3da3b4147cbd342b18da2ddb CI41089876288 SUCCESS. MP03 native resizing implementation is validated; not yet merged. Do not auto-merge.
- External PR45 merged as dade80ed adds RSSR_Complete_Project.ipynb. Preserve it. Fresh-kernel default50cells passed4.20s; opt-in seven-model variant51cells passed22.25s and seven32×32×3TIFFs verified. reports/portable-notebook-execution-20261031.json. Upload/satellite paths not tested. Notebook fixtures are not full forty-project scientific acceptance.
- MP04 native comparison lab now implemented: signed error/mask mathematics, numeric examples, fixed-range heatmap, metrics, CSV/PNG exports, explicitRun/stale-output clearing. Six canonical tests passed20.13s. All40 navigation ran; initial MP04 computation failed missing optional matplotlib. Replaced gradient dependency with direct CSS; targeted MP04 and scale controls passed. reports/mp04-native-instructor.json. No frozen experiments regenerated.
- Local integration a5f622e preserved external PR45 and MP03/04. PR44 updated head7b33be095ca0ba380fa701a43dbcc9d87a4b8749; CI41221613538 tests job passed, Streamlit job still running at last inspection.
- MP05 Gaussian lab implemented and validated: four canonical tests0.43s, targeted UI radius0 identity, impulse center weight, zero-padding border darkening and stale clearing passed. reports/mp05-native-instructor.json. This follow-up still needs publication and its own CI.

## Next actions
1. Finish local integration, publish MP03/04 follow-up to existing PR44 (preserve notebook), inspect exact updated-head CI. Do not claim the new head passed from old CI.
2. Complete MP06 native downsampling lab using existing degradation helpers and canonical stage folder; continue remaining lessons individually. Read canonical helpers before changes. MP01–05 native labs do not imply full40 acceptance.
3. Public Satellite Explorer seven-model inference, downloads, mobile layout, cold-start and configured persistence remain unverified. Preserve stateless semantics and source calibration/masks.
4. Continue scientific paired-data/region-v2 eligibility and matched-budget gates, final paper/report/PDF review/slides/package. Never relax frozen splits or substitute pilots for geographic evidence. External durable storage account and real author/venue details remain missing.

## Commands and constraints
- Work from repo root. Set PYTHONPATH=src and PYTHONUTF8=1 each shell; dev venv otherwise resolves an older editable checkout.
- Dev Python: ../../work/clean-venv/Scripts/python.exe; app Python: ../../work/streamlit-venv/Scripts/python.exe.
- After lesson changes: apppython scripts/check_learning_ui.py; targeted canonical project pytest. Current local targeted log ../../work/mp04-targeted.log; full navigation initial failure ../../work/mp04-apptest.log.
- Integrity: devpython scripts/build_learning_supplement.py --check; devpython scripts/audit_paper_sources.py --check. Both passed after MP03; repeat only affected checks.
- Notebook artifacts stay outside Git: ../../work/notebook-validation-20261031 and ../../work/notebook-model-validation-20261031. Reproduce in fresh kernel/new output_directory; enable run_models and pinned_repo_path only for opt-in variant. No training or dataset download required.
- Preserve all frozen evidence and published v0.3.0-rgb-pilots release/checksums. 42 small RGB model-scale cases already executed; full65536-square output not executed.

Historical details: [checkpoint archive](next_session_history_2026-10-31.md). Current acceptance: [acceptance matrix](acceptance_matrix.md). Requirements: [completion brief](completion_brief.md), [mandatory website brief](mandatory-website-brief.md).
