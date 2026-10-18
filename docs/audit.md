# Initial audit — 2026-10-18

Target repository was empty, public, default branch main; connected GitHub permission
reported push access. Initialized minimal main at a9e83fb92379da6f1f21100476adf902c5bbc1c7,
then worked on feat/reproducible-rssr-lab. No merge or force-push is authorized.

Old repository read-only snapshot: a178568ee9e451f33d3062bd0f3e75dfac54e4e6.
Branches observed: main, docs/40-project-research-lab, mp01/digital-image-explorer.
Main contains all 40 briefs, nine stages and MP01 source/test/notebook files. Its root
README/progress says no implementation exists, contradicted by the MP01 files and
validation record. Preserve learner status while correcting implementation status.
Import hashes in import-manifest.json preserve provenance. No source license was
assigned by the old curriculum; no new blanket license is inferred here.

Environment: Windows, 8 logical CPU threads, roughly 382 GB free at audit. CIM hardware
queries were denied by the environment. Bundled Python 3.12.14 works; python on PATH
is an inaccessible Windows Store alias. CUDA unavailable in installed CPU PyTorch.
No HF_TOKEN/HUGGING_FACE_HUB_TOKEN/GITHUB_TOKEN/GH_TOKEN variable names were found;
values were never printed. Connected GitHub works independently of shell tokens.

Initial network attempt failed before network permission was granted. After permission,
Git clone and HTTPS downloads succeeded. Default temporary directory is unwritable;
workspace TEMP/TMP fixes venv bootstrap and fixture writes. Dependencies are installed
in work/venv outside the repository; no system installation was changed.

## Evidence status at audit

| Area | Initial state | Current evidence location |
|---|---|---|
| MP01 | Implemented upstream, old fallback execution only | 10 local unittest checks rerun successfully; kernel execution tracked separately |
| MP02–MP40 | Briefs only, missing implementations | Introductory notebooks added; full project experiments/solutions pending |
| Pipeline | Missing | New modules; see progress.md for actual validation |
| Real data | Missing | Three pinned OpenSR NAIP HR GeoTIFFs, 5,627,736 bytes, checksums passed |
| Fresh catalog | Missing | Live STAC ingest 10 items, repeat added 0, page cap reported partial |
| Research results | Missing | No research-evaluated model claimed |
| Public deployment | Missing | No hosting credential found; current HF compute Spaces require plan entitlement |

Failures retained: venv ensurepip temp permission error; first MP01 run had three
temporary-file errors (all pass after TEMP/TMP fix); first live ingest had a 17-column
versus 18-placeholder mismatch (fixed, rerun succeeded; failed job remains in DB).
