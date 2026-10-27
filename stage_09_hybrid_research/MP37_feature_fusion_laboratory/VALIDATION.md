# MP37 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](../../docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 1 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `205d4450d3bd1d56f0312d9c96ae56688398414a28226efb73645b139050cb51`.

Commands from this directory: `python -m pytest -q test_mp37_tools.py` and `python run_notebook.py`. Records: [folder tests](../../reports/project-folder-tests.json), [kernel execution](../../reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.


## Expanded acceptance — 2026-10-27 (supersedes introductory scope)

Seven tests passed in3.78s. The first collection attempt failed with Windows Application Control WinError4551 loading torch_python.dll; the subsequent unchanged experiment imported Torch and ran, and one unchanged pytest retry passed. No security policy or environment was altered. The expanded three-code-cell notebook passed a real kernel with exit0; the existing local TCP warning remains.

[Experiment report](../../reports/mp37-fusion.json): sum[4,6], mean/projected[2,3], branch gradients[.5,.5]. Shared-parameter hybrid fixture1x3x5x7 ->1x3x10x14 had maximum absolute output difference0 with the mean-initialized projector. Projection136 parameters/4480 MACs; whole fixture models3273 versus3137 parameters. This is algebraic equivalence, not trained quality improvement.

Historical validation statistics are read from unchanged ablation-pilot-runs.json with run IDs and source hash, separately labeled. No new training, acquisition or held-out evaluation ran. README provides the diagram, conventions, exercises/solutions, failure cases and interpretation. Implementation/teaching deliverables complete; learner mastery and broader scientific superiority remain unassessed. The authoritative notebook record updates MP37 only; original central notebook unchanged.
