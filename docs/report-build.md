# Report and diagram build

Build artifacts outside Git (AGENTS.md). From a checkout with the project installed, install `requirements/report.txt`, then run:

```powershell
$env:PYTHONPATH='src'
python scripts/build_project_report.py --output ../submission-integrated-20261108
```

The builder generates twelve technical diagrams directly from the reviewed architecture specification, saves PNG and SVG assets plus a standalone landscape PDF, and inserts the images into the detailed TeX report. It instantiates untrained registry modules only to count parameters. It does not load weights, rerun experiments or read frozen image pixels. The source manifest hashes model definitions, diagram source and recorded table inputs.

Compile `RSSR_Final_Project_Report.tex` inside the output directory using Tectonic/XeLaTeX and BibTeX. The existing font setup uses Times New Roman and Consolas; these fonts must be available. The Linux font/build environment has not been verified. Two underfull paragraph warnings remain in the local build; no overfull boxes were reported.

For Word, supply Pandoc, an IEEE CSL file and the canonical bibliography:

```powershell
python scripts/export_editable_reports.py --source ../submission-integrated-20261108/RSSR_Final_Project_Report.tex --output ../submission-integrated-20261108/RSSR_Detailed_Report_Editable.docx --pandoc PATH_TO_PANDOC --bibliography references.bib --csl PATH_TO_IEEE_CSL
```

The exporter resolves image paths against the source directory, so the DOCX contains all twelve embedded images. Text, equations and numeric tables remain editable; diagrams can be edited separately as SVG or regenerated. The Word copy uses a single column for editing. `paper/main.tex` remains the IEEE publication source.

The standalone diagram PDF provides larger landscape versions for inspection/printing. The bibliography contains23primary-source identity/concept-audited records, including one publisher correction. This does not establish full per-paper experimental extraction or reproduction. Slides, source-bound model-output panels, complete reference review and scientific geographic/cross-sensor gates remain separate acceptance items.

## Actual seven-model display figures

Generate a new bounded demonstration (seven pinned trained models, native x2 and repeated x2→x2) with `python scripts/build_seven_model_panels.py --output ../new-seven-panels`. The existing asset cache is reused after hash verification. This uses the licensed training sample, not frozen test pixels. Reuse the evidence without rerunning inference using `python scripts/build_project_report.py --output ../new-report --comparison ../new-seven-panels`.

The cached16×16 input is dark and low in texture. The observed outputs cannot establish true-detail recovery. MAE against Pillow bicubic describes disagreement, not HR error. Native and cascade results, unequal training budgets, provenance and actual local pipeline runtime remain explicit.

To reuse already reviewed diagrams, pass `--reuse-figures ../previous-report-directory`. Model/generator source hashes must match its manifest, and image/PDF hashes must match its delivery-checksums.json; stale sources and unverified assets are rejected. The existing14comparison executions can be reused with `--comparison ../submission-seven-model-panels-20261108`.
