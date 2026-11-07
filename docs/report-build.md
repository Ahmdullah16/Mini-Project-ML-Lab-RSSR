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

The standalone diagram PDF provides larger landscape versions for inspection/printing. The bibliography contains twenty records; this build does not establish full-text verification of twenty papers. Slides, source-bound model-output panels, complete reference review and scientific geographic/cross-sensor gates remain separate acceptance items.
