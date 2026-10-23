Complete this existing project and turn its Streamlit app into a polished, evidence-backed research showcase.

Repo: https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR
App: https://mini-project-ml-lab-rssr.streamlit.app/

RESUME, DO NOT RESTART
Read AGENTS.md, the latest section of docs/next_session.md, submission_readiness.md, git status, recent commits and open PRs. Reconcile actual state before editing. Latest known state: MP01–MP26 expanded, MP27–MP40 incomplete; frozen held-out evaluation exists; live app exposes pilot CNN, CNN v2 and hybrid v2. Preserve all work, checkpoints, negative findings and frozen test results.

WORK EFFICIENTLY
Save this complete brief as docs/completion_brief.md. Maintain a compact docs/next_session.md containing branch/commit, dirty files, completed task IDs, validation evidence, blockers and exact next commands; archive historical narratives elsewhere. Read only relevant files/diffs, batch independent checks, reuse verified artifacts and avoid repeated audits, downloads, training or full-suite runs. Run affected checks per change and full acceptance gates at release. Keep chat updates brief. Efficiency must not reduce implementation or scientific quality.

PRIORITY 1 — PRESENT THE RESEARCH IN THE APP
Preserve the existing URL and inference functionality. Build clear navigation with:
1. Overview: project objective, motivation, approach, verified findings and actual completion status.
2. Live SR: all compatible checkpoints, licensed examples/uploads, LR/bicubic/SR comparison, synchronized zoom or before/after view, runtime, provenance and downloads.
3. Results: searchable/filterable measured tables, training curves, baseline comparisons, ablations, robustness, spectral metrics and efficiency charts.
4. Methodology & Data: architecture diagrams, equations, tensor shapes, dataset provenance, geographic splits and evaluation protocol.
5. Mini-projects: all 40 projects grouped into nine stages, with accurate status, explanations, outputs and notebook/code links.
6. Satellite Explorer: existing bounded acquisition/inference, timestamps, imagery preview and GeoTIFF/provenance downloads; explain freshness and domain limitations.
7. Downloads & Reproducibility: report, slides, result CSV/JSON, references, environment/setup instructions and artifact manifest.

Use a consistent professional theme, readable typography, restrained colors, good spacing, accessible contrast and responsive layouts. Show meaningful figures and short interpretations, not walls of JSON. Keep technical provenance available in expanders. Verify desktop/mobile layout and empty/error states.

Create one validated results/artifact index that reads existing measured reports; drive app charts, tables and downloadable reports from it. Include all recorded experiments through filters rather than overwhelming the landing page. Keep training/validation/test, RGB/multispectral, synthetic/cross-sensor and historical/final results separate. Show dataset, sample count, units, protocol and source run. Never invent missing values or hide negative results. Do not claim hybrid superiority when evidence does not support it. Never compute HR-reference metrics for uploads without valid references.

Lazy-load models, cache small immutable results and avoid training, bulk downloads or expensive computation on page navigation. Missing optional storage credentials must not disable the public showcase.

PRIORITY 2 — FINISH THE PROJECT
Complete MP27–MP40 and earlier unmet criteria against their briefs. Finish portable research artifacts, applicable cross-sensor work, normalized external-storage integration and remaining original requirements. Do not silently reduce scope; document concrete external blockers while completing independent work. Preserve the frozen held-out evaluation—no retuning or unnecessary repeat evaluation.

Produce the final report, presentation, examiner demo guide and reproducible submission package from actual evidence. Keep educational completion, engineering readiness and research limitations distinct.

PRIORITY 3 — VERIFY AND RELEASE
Test clean installation, artifact/checkpoint checksums, all changed notebooks, result-index consistency, navigation, each public model, uploads/rejections, downloads and bounded satellite inference. Verify final notebooks/tests/links and CI on the candidate release commit. Test restart recovery and external durability only where claimed.

Record the deployed source/checkpoint identity. Distinguish code committed, PR pending, deployed and publicly verified. Preserve the working app; use reviewable commits/PRs and existing merge restrictions. Do not claim a local preview is deployed.

Continue through implementation, not just planning. Ask only for exact missing access or resources; do not request secrets in chat or activate paid services. Checkpoint after each meaningful milestone so interruption loses minimal work.

Finish with verified changes, tests, deployed URL/version, submission artifacts and precise remaining blockers. Never mark the entire project complete merely because the website loads.