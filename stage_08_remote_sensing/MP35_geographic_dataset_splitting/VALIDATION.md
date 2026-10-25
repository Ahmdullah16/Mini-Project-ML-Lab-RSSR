# MP35 validation — 2026-10-25

Commands are listed in [README](README.md). Development Python 3.12.14.

- `python -m pytest -q stage_08_remote_sensing/MP35_geographic_dataset_splitting`: **12 passed**. Covers exact buffer equality, region reuse, overlap, duplicate IDs, invalid split/bounds, nonfinite/negative buffer, empty input and permitted within-split overlap.
- `experiment.py`: **passed**, nine SHA256-verified cached raster headers; 4/4/1 scenes and 24 cross-split pairs. [Audit](../../reports/mp35-geographic-splits.json) records environment, declared buffer and all pair clearances.
- `run_notebook.py`: **passed-real-jupyter-kernel**, four code cells. [Execution record](../../reports/project-notebook-execution.json); executed outputs remain ignored under data/.
- Existing source manifest, assignments, raster pixels and frozen result artifacts remain unchanged. No image metrics or training ran.

The exported map depicts scene centers in angular coordinates, not a metric basemap. Passing this lesson does not validate broad geographic generalization or learner mastery.
