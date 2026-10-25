# MP29 validation — 2026-10-23

Implementation/teaching criteria in the [brief](README.md) are complete; learner mastery unassessed.

Executed all three README commands in Python3.12.14, PyTorch2.14.1+cpu. Five tests passed: hand-computed population-variance normalization; zero-branch residual identity and hidden shape; permutation with/without fixed positions; positional values and finite gradients; invalid contracts. The bounded experiment used CPU float64/one thread and completed twelve finite conditions with identical weights across heads. No training, acquisition or held-out evaluation.

All six notebook code cells passed in a real Jupyter kernel, exit0; [execution record](../../reports/notebook-execution.json). The existing local TCP transport warning was emitted without execution failure. Diagram and response figures come from the executed notebook. [Measured data](../../reports/mp29-encoder.json) pins source license/hash/crop, settings and output arrays.

All models have172 parameters; independent RGB projection adds12. Response RMS is not accuracy. The pre-LN ordering, sinusoidal positions added before the block and absence of dropout/masks are declared; no trained positional benefit or generalization claim.

## Local-folder validation — 2026-10-25

Passed 5 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (6 code cells). Source SHA-256: `dc7cf2cedd256734ad7d1b1539ae9c96283e1757fa739e34d426be9de0497a46`.

Commands from this directory: `python -m pytest -q test_encoder_tools.py` and `python run_notebook.py`. Records: [folder tests](../../reports/project-folder-tests.json), [kernel execution](../../reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.
