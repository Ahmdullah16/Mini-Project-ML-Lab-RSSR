# MP36 partial validation — 2026-10-26

Status: In progress. The full investigation and paired-data evaluation are not complete.

- Local pytest: **7 passed**. Tests cover noninteger scale, invalid GSD/offset, cast-before-arithmetic, calibrated toy MSE, empty/nonfinite arrays, spectral response mismatch and invalid weights.
- `experiment.py`: executed synthetic analytical examples only; [record](../../reports/mp36-numerical-examples.json) includes environment and values.
- `run_notebook.py`: **passed-real-jupyter-kernel**, four code cells; [execution record](../../reports/project-notebook-execution.json). Known local kernel TCP warning; process exited0.
- No dataset archive transfer, new test-image inspection, model training or frozen test recomputation.
- Primary official records support version/license/size findings. Paper/correction retrieval returned429, leaving registration-method review open. See [feasibility](../../reports/mp36-feasibility.md).

Source citations and limitations do not become sensor performance evidence merely because these numerical tests pass.
