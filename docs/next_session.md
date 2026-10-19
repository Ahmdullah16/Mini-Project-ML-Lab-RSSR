# Updated checkpoint

Branch feat/development-validation; PR #4 merged externally and preserved.

## Latest validation — 2026-10-19

MP05–MP11 expanded and executed with exit 0; seven new degradation/metric tests and three inspected figures. Six geographic runs completed: mean CNN validation delta +1.5709 dB, hybrid +1.3337 dB; held-out test unused. See reports/geographic-study.md. New fresh-output paths are portable, with relocation/corruption/deduplication checks. Portable bundle launch and HTTP output downloads passed (reports/portable-demo-http.json).

Isolated Python 3.12 suite passed 64 tests and four subtests in 13.39 seconds without warnings. Earlier run without writable TEMP failed due sandbox paths; corrected environment passed. Cache fixture initially lacked a required ingestion timestamp; corrected. The GeoTIFF warning was caused by accidental alpha metadata in the test fixture; explicit non-alpha multispectral storage resolved it.

Next incomplete teaching task: MP12 Linear Algebra Visualizer. MP12–MP40 still need full implementations. Incremental watermark/scheduler recovery, real paired satellite evaluation, final selection/test, report/slides/package and public deployment remain incomplete. All 40 current notebooks passed the isolated batch with process exit 0; source hashes verified. Do not claim submission readiness.


Historical checkpoint follows; current next action is MP12, not MP09.

---
# Current continuation checkpoint — 2026-10-19

Branch: feat/development-validation. PR #4 was merged externally at 4745a49; merged into the local history without overwriting pending files. Preserve all data/runs and existing ledgers. Current working changes are not yet fully validated or committed.

Verified since the preceding checkpoint:
- Fresh HTTPS clone of cd3f61f with an isolated Python 3.12 environment (no system site packages); 55 tests plus 4 subtests passed. Portable quickstart downloaded 5,627,736 checksum-verified bytes, trained 100 CPU steps, produced a checkpoint, GeoTIFF and SQLite integrity result. See reports/clean-quickstart.json. All 40 clone notebook cell records passed; process exit status was lost across continuation, so a clean final batch exit is still pending.
- Actual fresh Sentinel-2 RGB acquisition-to-SR completed using 1,900,544 range-read bytes, with acquisition/ingestion/inference timestamps and database provenance. See reports/fresh-inference.json. A demo query repeated the download once because int/float AOI encodings differed; bbox normalization now prevents that duplicate. HTTP model selection and output download passed (reports/demo-controls-http.json). Public deployment remains unverified.
- 12 initialization diagnostics completed: CNN/hybrid, random/zero residual, 3 seeds, 1000 steps. All remain below bicubic on the single validation scene. See reports/initialization-study.json. No test evaluation or final model selection.
- Six geographically separated NAIP scenes downloaded (11,255,472 bytes), checksummed and assigned 3 train/3 validation before patches. Original test unchanged. See reports/geographic-expansion.json. scripts/geographic_study.py is now running 6 validation-only 1000-step zero-initialized CNN/hybrid runs across 4 train/4 validation scenes. Resume verifies completed run hashes; an interrupted individual run is not silently complete.
- MP05–MP08 now contain expanded mathematics, examples, experiments and worked solutions. Four degradation tests passed; scripts/degradation_experiments.py generated ten real training-crop pair configurations and two visually inspected figures. All four notebooks passed real kernels with explicit process exit 0 in this continuation. Local TCP kernel warning remains; first sandbox-blocked attempt was interrupted, not counted as a pass.

Next: finish geographic study; fix portable fresh-output paths and recovery; validate all pending demo/code changes; run final isolated tests/notebooks; update README/readiness and commit. Continue MP09 onward from authoritative briefs. MP09–MP40 remain introductory, not complete projects. Full educational acceptance, final research evaluation/report/presentation, hosted inference and final-commit CI are not complete. No claim of submission or publication readiness.

Existing workspace environments: ../../work/venv (editable current repo), ../../work/clean-venv (isolated, editable work/submission-clean clone). Set writable TEMP/TMP/Jupyter/IPython/Matplotlib caches under work/tmp. Network permission is needed for local notebook sockets in this sandbox. All runs CPU, zero GPU-hours. Added rasterio wheel about 30.7 MB; previous transfer retries were not instrumented, so exact total wire bytes are unknown. Resource cap remains 2 GB new downloads and 2 GPU-hours; no paid resources.

Historical notes below are superseded where they conflict with this checkpoint.

---

# Next session â€” 2026-10-19 checkpoint

Continue branch feat/image-foundations. PR #2 was already merged before this work; do not overwrite prior checkpoints. Local code, dataset and experiment records are preserved.

Next incomplete teaching task: expand MP05 Gaussian Blur Simulator from its authoritative brief. Define correlation versus convolution, kernel normalization, sigma/radius/boundaries; implement hand calculations, meaningful tests and synthetic plus existing training-crop comparisons. Then proceed to MP06 onward. Do not mark introductory notebooks complete.

Executed this continuation:
- tests/test_image_foundations.py: 6 passed; tests/test_catalog.py: 2 passed.
- scripts/image_foundations_experiments.py: both real training-crop experiments completed with two inspected figures and reports/mp03-mp04-experiments.json.
- scripts/execute_notebooks.py --only MP03 and --only MP04: real kernels passed, both explicit Python exit code 0.
- scripts/check_demo.py: HTTP inference checked again; scripts/check_catalog.py validates the new cached metadata endpoint when RSSR_CATALOG is configured.
- Network restrictions initially blocked Git fetch and Jupyter startup. Session network permission was granted; fetch then succeeded. Do not label that first stalled attempt as a notebook pass.

Local setup: workspace work/venv/Scripts/python.exe; TEMP/TMP and Jupyter/IPython/Matplotlib caches under work/tmp. From the repository root the environment is ../../work/venv. Reuse existing dependencies and ignored data/. No new imagery, dependencies or GPU time was consumed this continuation.

Launch local demo with RSSR_CHECKPOINT pointing at data/runs/b5f690d95fe4422b88f6f728ce6d6c2c/best.pt and RSSR_CATALOG at data/catalog.sqlite, then python app/demo.py. The catalog reader does not create, migrate or write the database. Cached partial jobs do not count as complete refreshes. The host needs persistent research storage; public deployment and remaining UI features remain pending.

The previous detailed checkpoint follows for commands, resources and research limitations. Its MP03 next-action text is superseded by the MP05 instruction above.

---

# Next session

Continue branch feat/notebook-demo-validation. Preserve completed work and data.
PR #1 was merged externally at 96b770559e3f866eed27cfc046dbbf61fc0f3382; no open PR existed at this continuation's start. Local implementation commit bf59f89 adds MP02, catalog recovery and ablation controls. A follow-up evidence commit and review PR contain the final continuation state. Do not merge automatically.

## Verified commands from repository root

Use the existing ../../work/venv/Scripts/python.exe from this checkout, or activate an equivalent environment. In the desktop workspace, work/venv is beside outputs/. Set TEMP/TMP to the writable work/tmp folder on restricted Windows; also set JUPYTER_RUNTIME_DIR, IPYTHONDIR and MPLCONFIGDIR below it.

- python -m pytest -q: 43 tests, 4 subtests passed, 2 GeoTIFF warnings, 13.82 seconds.
- python scripts/execute_notebooks.py: all 40 cell execution records passed. Windows wrapper returned nonzero after completion; see warning below. python scripts/execute_notebooks.py --only MP02 then ran the expanded lesson, with explicit Python exit code 0. Source hashes and absence of error outputs independently checked for all 40.
- python scripts/mp02_experiment.py: real training crop, brightness exact equality and masked-statistics tolerance passed, measured timing record saved.
- python scripts/ablation_pilot.py: 24 validation-only runs completed. python scripts/ablation_pilot.py --resume reverified all 24 hashes/configurations and did not retrain. Default refuses to overwrite an existing ledger.
- Local app/demo.py with RSSR_CHECKPOINT=data/runs/b5f690d95fe4422b88f6f728ce6d6c2c/best.pt; python scripts/check_demo.py completed HTTP inference. The local server may need restarting after the desktop session ends.
- Live refresh resumed prior job ed7874fc8c78498aa7de3e88d03f6e1a into job 5271e81960da4a6ba5ad096e9313222d: 9 found/added, partial. No assets downloaded. Continue explicitly with rssr refresh configs/stac.json --root data --resume-job 5271e81960da4a6ba5ad096e9313222d, or restart overlap if provider cursor expired.
- python scripts/research_figures.py generated and visually checked two measured figures. python scripts/check_structure.py verified 40 IDs and 361 links before final checkpoint docs.

## Artifacts and limitations

reports/ contains execution ledgers, MP02 timings, HTTP evidence, ablation summaries, environment versions, figures and an explicitly preliminary research draft. Ignored data/runs/ contains actual checkpoint/training/evaluation files. Keep data/catalog.sqlite and data/prepared/ together; tile paths are absolute, so reprepare after relocation. Original five-model runs are preserved in reports/pilot-runs.json.

The notebook dependency failure was fixed; Windows ZeroMQ shutdown messages persist and merit an isolated follow-up. Notebook cell success is verified; a clean exit across the complete 40-kernel batch is not claimed. No Colab or remote CI success is assumed.

New pilot uses 100 steps, 3 seeds, one validation scene; every learned run trails bicubic. No independent-scene CI or final test evaluation. Known wheel cache 233,718,558 bytes, source images 5,627,736 bytes, 0 GPU-hours; unmeasured prior network retries prevent exact transfer accounting. No paid resources activated.

## Precise next action

Expand MP03 from its authoritative image-resizing brief using the existing training/validation data: explain coordinate conventions, implement and test from-scratch nearest/bilinear interpolation, compare with declared bicubic, show measured figures, and run the expanded notebook. Continue MP04 onward with genuine project-specific experiments and solutions, not completion labels on scaffolds.

For research, inspect texture diversity and acquire additional independent geographic scenes within the remaining bounded download budget before increasing training. Estimate throughput from measured runs but do not extrapolate tiny 32px patches to full images without a pilot. Investigate zero-residual initialization as a predeclared control; retain old results. Do not tune on the test scene.

For public deployment, finish missing demo controls, then obtain authenticated access to a verified free host (e.g. eligible ZeroGPU account) through a secret store, not chat plaintext. Shell Git push previously failed in the credential helper; use connected GitHub tree/commit/branch tools for publishing. Never force-push or merge.
Latest full-suite result (2026-10-19): 51 tests, 4 subtests, 2 known warnings, 11.47 seconds. Structure check: 40 IDs, 365 links.
