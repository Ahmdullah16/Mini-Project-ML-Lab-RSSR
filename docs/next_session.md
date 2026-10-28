# Resume checkpoint — 2026-10-28

## Current state

- Local branch `feat/mp27-attention`. Latest committed ingestion milestone `f46934e`; PR34 remote head `c97196b907ee43bb5e883edf92763ab548cb61a5`, exact-head Validation run40632438250 SUCCESS. Externally merged main `c1245a3`; agent has not merged any PR. Integrate this external merge after committing current lesson-status correction. No training or frozen evaluation rerun.
- Current milestone: corrected stale MP35 folder scope; MP36–40 display recorded teaching/scientific scope instead of an old numerical cutoff. MP35/37 teaching complete does not imply learner mastery or scientific generalization. Nested PNG/SVG/JPEG report figures now exposed. Validation: 3 evidence tests (0.23s), seven-section network-blocked AppTest,40 folder contracts,940 links,7 claim records. `reports/lesson-status-validation.json` pins source hashes. Publication details follow below after commit.

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
