# MP27 validation — 2026-10-23

Engineering and teaching deliverables are complete against the [brief](README.md). Learner mastery is unassessed. This is an untrained attention primitive, not a trained Transformer or an SR accuracy result.

## Commands actually run

From repository root in the existing isolated Python3.12.14 environment:

```bash
python -m unittest discover -s stage_07_transformers/MP27_self_attention_from_scratch -p "test_*.py" -v
python stage_07_transformers/MP27_self_attention_from_scratch/experiment.py
python stage_07_transformers/MP27_self_attention_from_scratch/run_notebook.py
```

- Six tests passed: independent hand calculation, stable/translated softmax, mask/output shapes, permutation equivariance/input immutability, invalid matrices/masks, analytical storage and entropy.
- The bounded CPU experiment completed in0.122489 seconds; NumPy2.5.4. Sixteen feature-scale comparisons produced finite output and row-sum error at most5.56e-16. [Measured record](../../reports/mp27-self-attention.json).
- All eight notebook code cells passed in a real Jupyter kernel, exit0. [Execution record](../../reports/notebook-execution.json); executed notebook under ignored data/executed_notebooks. Two resulting figures were visually inspected: readable axes, legends, units and explicitly analytical storage.
- Source checksum and all crop masks checked. Used only a central8×8 RGB training crop; no test or validation evaluation, training, dataset download or paid compute.

## Limitations and failures

Large Q/K scaling saturates attention despite the √d divisor; this is an observed limitation, not a failed test. Real-image features do not follow the unit-variance Gaussian assumption. NAIP is aerial imagery with normalized digital numbers; physical calibration is unverified. The notebook reads recorded real-input evidence; re-running the experiment requires the pinned local raster. Memory figures count only one score tensor and are not peak-RAM measurements. The existing local Jupyter TCP transport warning appeared; kernel execution still completed successfully.

## Local-folder validation — 2026-10-25

Passed 6 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (8 code cells). Source SHA-256: `61a362548dde6913953d03281cf6f6a7e8057ebfe64de45a0193f98a1e812e18`.

Commands from this directory: `python -m pytest -q test_attention_tools.py` and `python run_notebook.py`. Records: [folder tests](../../reports/project-folder-tests.json), [kernel execution](../../reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.
