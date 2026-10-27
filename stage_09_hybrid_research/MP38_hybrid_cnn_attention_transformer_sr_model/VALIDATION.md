# MP38 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](../../docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 1 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `d5b0e1f2c242854575d54efff6c1b7515b4b410aa331b038b6b11430a8f30bb1`.

Commands from this directory: `python -m pytest -q test_mp38_tools.py` and `python run_notebook.py`. Records: [folder tests](../../reports/project-folder-tests.json), [kernel execution](../../reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.


## Architecture component — 2026-10-27

Five tests passed7.85s; standalone experiment and expanded three-cell real Jupyter kernel passedexit0. Known local TCP warning remains. [Report](../../reports/mp38-architecture.json) records actual RGB/four-band module shapes,3524/3969 parameters, exact zero-residual bicubic identity, finite nonzero tail gradients and zero initial gate gradients. This is synthetic initialization behavior, not trained quality. No optimizer steps, data download or held-out evaluation. README adds the integrated equation, architecture diagram, worked example and solved exercises. Hooks are removed even on forward failure; model train/eval flag is restored.

MP38 remains In progress: matched-budget CNN/Transformer/hybrid quality comparison, reference error crops and complete quality/compute study remain under protocol-v2 gates. Original central notebook preserved; current project execution record updates MP38 only.
