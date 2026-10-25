# MP13 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](../../docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 3 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (4 code cells). Source SHA-256: `68ee2c6f95a19768b83cc37a05adb07feac0109249e247a60d9e666a369fb277`.

Commands from this directory: `python -m pytest -q test_mp13_tools.py` and `python run_notebook.py`. Records: [folder tests](../../reports/project-folder-tests.json), [kernel execution](../../reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.
