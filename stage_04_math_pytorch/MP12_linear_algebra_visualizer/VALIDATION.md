# MP12 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](../../docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 2 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (5 code cells). Source SHA-256: `448b224380c3b269b52ad676828983bf72200edf315ca60705da8235265d119e`.

Commands from this directory: `python -m pytest -q test_mp12_tools.py` and `python run_notebook.py`. Records: [folder tests](../../reports/project-folder-tests.json), [kernel execution](../../reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.
