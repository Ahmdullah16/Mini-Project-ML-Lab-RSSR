# Continuation checkpoint

- Branch: `feat/research-showcase`; prior local commit `1176bc3`, remote validated code `1043692` (PR14 externally merged). Inspect git status for edits after this checkpoint.
- Authoritative scope: docs/completion_brief.md; latest user requests MP01-like implementation, notebook, tests and validation for every project. No automatic PR merge or paid resources.
- Recovered showcase: seven sections, measured-results index, artifact downloads, synchronized zoom, satellite preview. Gradio and frozen test unchanged.
- Actual local validation on October 21: 89 tests +4 subtests; seven-section/error-state AppTest; all3 checkpoints sample/max-size inference; two empty-cache restarts with equal output; desktop/mobile browser checks. See reports/showcase-validation.md. Public showcase verification pending.
- Teaching: MP01–MP26 expanded/executed; not all criteria met (MP20 independent SRCNN held-out table pending). MP27–MP40 introductory. Next implementation: MP27 self-attention with MP01-like local files and validation.
- Blockers/gaps: external storage account absent; normalized external DB, paired cross-sensor study, four-band portable weights, final report/slides/package unfinished. Do not retune/repeat frozen test.
- Next commands: `python scripts/check_structure.py`; publish checkpoint and verify exact-head CI; implement MP27 against its brief and execute its notebook/tests. Use work/clean-venv with current src on PYTHONPATH; deployment checks use work/streamlit-venv. Avoid rerunning unchanged artifacts/full audit.
- Historical narrative: docs/next_session_history_2026-10-21.md. Existing public hybrid identity: reports/public-hybrid-validation.json. New showcase is local, not yet deployed.
