# Local showcase validation

Executed on 2026-10-21; recovered and checkpointed on 2026-10-23. No frozen test was recomputed.

- Full pytest: **89 tests and 4 subtests passed**, 12.41 seconds.
- `scripts/check_showcase.py`: seven sections, learning curves, artifact library, empty searches, missing upload and reversed acquisition dates passed with urllib downloads blocked.
- `scripts/check_model_catalog.py`: all three genuine checkpoints passed sample and 512-to-1024 tiled inference in the deployment environment. See [measurements](model-catalog-check.json).
- `scripts/check_streamlit.py`: separate empty-cache processes downloaded and verified the pinned checkpoint, rejected invalid inputs and produced identical output. See [restart measurements](streamlit-restart.json).
- Browser: desktop 1280×720 and mobile 390×844 inspected; mobile navigation, pilot inference and synchronized zoom worked. Corrected chart axis labeling, references path and top spacing. Restored viewport afterward.

The shared index separates study, split, bands and condition. Other historical reports remain downloadable source artifacts; not every report is normalized into a comparable metric table. Gradio and frozen results are preserved.

This is local validation, **not showcase deployment verification**. [Previous public hybrid evidence](public-hybrid-validation.json) remains valid for its recorded version. No new live satellite request or training ran for this UI milestone. MP27–MP40, earlier unmet criteria, portable four-band weights, normalized external storage, cross-sensor validation and final submission artifacts remain unfinished.
