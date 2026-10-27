# MP40 validation

This project now has the MP01-style local file layout. Existing scientific results and original notebook execution evidence are preserved; see [progress](../../docs/progress.md). Expanded content through MP31 does not automatically satisfy every earlier project criterion. MP32–MP40 remain introductory until their full experiments are completed.

Local-folder test and execution outcomes are recorded below. No data acquisition, held-out evaluation or retraining is authorized by this file.

## Local-folder validation — 2026-10-25

Passed 1 local tests with pytest; the notebook passed a real Jupyter kernel from this project directory (2 code cells). Source SHA-256: `75c5ebd18f2dcd51e4558d5c866cfaefa382a966281ea69c1b7d80a05e831723`.

Commands from this directory: `python -m pytest -q test_mp40_tools.py` and `python run_notebook.py`. Records: [folder tests](../../reports/project-folder-tests.json), [kernel execution](../../reports/project-notebook-execution.json). Outputs are temporary under ignored data/executed_notebooks/project_folders. Original central notebook evidence remains historical. This validates the current lesson scope, not unfinished completion criteria.


## Expanded historical benchmark analysis — 2026-10-27

Five tests passed4.84s; derived report/figure generation and three-cell real kernel passedexit0 (known TCP warning). [Report](../../reports/mp40-benchmark-analysis.json) preserves source hash,30 robustness conditions and5 efficiency records from unchanged development-benchmark.json. No model benchmark, training, acquisition or frozen inference was rerun. [Figure](../../reports/figures/mp40-robustness.svg) was visually checked via a rendered PNG: readable units, separate band/condition panels, consistent model colors and negative results retained. PNG preview is local temporary output; SVG is reproducible from the script.

Tests validate paired-first aggregation, band separation and rejection of duplicate/nonfinite scenes/conditions. Existing timing helper test verifies call counts and train/eval restoration. README/notebook explain dB contrasts, selected-seed optimism, median/p95, partial MAC/FLOP counting and sampled RSS limitations. MP40 remains In progress: real sensor shift, full operation counts and scoped allocator-peak evidence remain unavailable; no GPU measurement was invented.
