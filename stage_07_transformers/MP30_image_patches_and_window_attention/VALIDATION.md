# MP30 validation — 2026-10-23

Implementation/teaching criteria complete; learner mastery unassessed. Executed all three [README commands](README.md) with Python3.12.14/NumPy2.5.4 on CPU.

Five tests passed: exact multichannel roundtrip/dtype and hand ordering; non-wrapping shifted boundary connectivity; two-layer reachability; hand memory counts; invalid dimensions, missing divisibility and adjacency budget rejection. All six measured round trips on a synthetic ramp and checksummed training crop were exact. Connected pair counts256/64/36/144 are recorded in the [experiment](../../reports/mp30-windows.json).

All six notebook code cells passed in a real Jupyter kernel, exit0, with the existing local TCP transport warning. [Execution evidence](../../reports/notebook-execution.json). Figures show connectivity and analytical storage; no measured peak-memory claim. No network, training, held-out evaluation or new dataset. This explicitly clipped shifted-window example is not a faithful Swin masking implementation.

## Local-folder validation — 2026-10-25

Passed 5 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (6 code cells). Source SHA-256: `30ebdc8e6418133f53d62b6c032f19adcd0dd0e398e4834e7a20fa018bb75fdd`.

Commands from this directory: `python -m pytest -q test_window_tools.py` and `python run_notebook.py`. Records: [folder tests](../../reports/project-folder-tests.json), [kernel execution](../../reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.
