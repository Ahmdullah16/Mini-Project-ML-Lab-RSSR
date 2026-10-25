# MP31 validation — 2026-10-23

Implementation/teaching criteria complete; learner mastery unassessed. Executed all three [README commands](README.md) in the existing CPU Python3.12.14/PyTorch2.14.1 environment.

Three tests passed: exact per-scene delta and summary reconciliation for six source runs; actual model parameter counts and finite odd-size7×9→14×18 output; zero-residual Transformer equals the declared bicubic skip. The benchmark export reuses previously executed four-band validation and CPU efficiency evidence and records source hashes. No training or frozen test was run.

All six notebook code cells passed in a real Jupyter kernel, exit0, with the existing local TCP warning. [Execution record](../../reports/notebook-execution.json). The figure shows the three observed seeds and descriptive sample SD. The CNN's mean gain1.570925dB exceeds the compact Transformer's0.618003dB; no superiority claim for Transformer. Measured parameters7552/5392 match current model construction.

Limitations: unshifted educational baseline, small four-scene validation set, three seeds, unequal parameter/time budgets, synthetic aerial degradation, normalized DN rather than reflectance. CPU latency and sampled RSS are historical measurements, not newly measured values. Official RGB SwinIR pretraining differs and is not combined with this table.

## Local-folder validation — 2026-10-25

Passed 3 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (6 code cells). Source SHA-256: `dbe74505107fcddffafd40566f6561ad5646d7010cf82cea5b457637a2f12eb3`.

Commands from this directory: `python -m pytest -q test_benchmark_tools.py` and `python run_notebook.py`. Records: [folder tests](../../reports/project-folder-tests.json), [kernel execution](../../reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.
