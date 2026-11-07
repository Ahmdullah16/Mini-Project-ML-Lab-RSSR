# Working evaluation paper

## Detailed report build (7 November 2026)

`python scripts/build_project_report.py --output <external-output-directory>` creates a standalone detailed report TeX source, source-hash manifest, bibliography and seven-family parameter CSV from existing teaching sources and frozen result tables. Install the separate `requirements/report.txt` in a document environment. It uses Pandoc 3.9 through pypandoc-binary 1.17; check `pandoc --version` in that environment. Compile with Tectonic 0.17.0 or a compatible XeLaTeX environment. The present report selects Windows Times New Roman and Consolas; equivalent fonts must be explicitly configured on other systems. Initial Tectonic compilation downloads its TeX support bundle.

Local execution produced an 18-page detailed report and a four-page IEEE paper; all pages were rendered and visually inspected for layout. This does not close scientific or full submission acceptance. The diagrams summarize seven feature paths with shared internal derivations; further layer-level diagrams, full reference review, result-image panels and slides remain outstanding. Nine new references were retrieved from CVF publisher BibTeX records; the bibliography has 20 entries, but that is not a claim that every existing record or full paper has been independently verified. See `reports/reference-metadata-20261107.json` for retrieval scope.

The [detailed supplement source](supplement.md) now combines native lesson prose, worked examples and recorded validation for all40 projects with the shared mathematical reference and contribution audit. Generate it with `python scripts/build_learning_supplement.py`; `--check` verifies exact synchronization. The website reads the same lesson sources and audit. This is a working Markdown source, not a final typeset report or proof that every new teaching criterion is met. The same generator supplies the paper's contribution-audit section.

The four requested IEEE files in ChatGPT Library `/mini project` were not found in the accessible attachments or local Codex files on2026-10-28. Library access/transfer remains required to verify those exact templates. Existing IEEEtran draft compilation can proceed. No page limit or author identity is assumed.

[main.tex](main.tex) is an IEEE conference-style working draft, not a submission-ready paper. Author details, full literature extraction, reference-based failure atlas, new geographic study and cross-sensor evidence remain open. Compilation is separate from scientific acceptance and visual review.

From the repository root, regenerate only the derived tables and bibliography:

```powershell
python scripts/build_paper_evidence.py
python scripts/build_paper_evidence.py --check
python -m pytest tests/test_paper_evidence.py -q
```

These commands use the standard library for generation, pytest for tests, and no imagery, weights, network, training or inference. The root [bibliography](../references.bib) is authoritative; the paper copy is generated. Open this folder in the built-in LaTeX editor, or use an existing TeX environment with IEEEtran and `latexmk -pdf main.tex` from this folder. Terminal TeX installation is not required for the built-in compiler.

[Paper evidence manifest](../reports/paper-evidence.json) records canonical UTF-8/LF source hashes and JSON pointers for every generated table row. Source reports also feed the app evidence index; the generator does not maintain an independent hand-edited result database. Arithmetic is descriptive aggregation of recorded scores. Three-seed sample SD uses denominator n−1. Four-scene gains average paired scene differences first, then average across seeds. Neither is a geographic confidence interval. RGB and four-band frozen results remain separate.

## Claim-to-evidence map

| Paper statement | Evidence | Scope/limit |
|---|---|---|
| Pilot gate and attenuation comparison | [pilot runs](../reports/ablation-pilot-runs.json), generated pilot table | One validation scene, differing parameter budgets; no superiority claim |
| Later geographic validation | [geographic runs](../reports/geographic-study.json), generated development table | Four validation scenes, three seeds; both geography and initialization changed |
| Frozen held-out comparisons | [final held-out](../reports/final-heldout.json), generated separate band tables | One scene, sixteen tiles; pretrained control is unmatched |
| Metric definitions and sampling | [original protocol](../docs/protocol.md) | Synthetic degradation and normalized DN, not physical calibration |
| Prospective design | [protocol v2](../docs/protocol-v2.md) | Design-registered only; not executed or retrospectively preregistered |
| Exact-scale inference behavior | [scale validation](../reports/scale-validation.json), [execution guide](../docs/scale-execution.md) | Local tests; 65536-square output planned only |
| Cross-sensor source review | [feasibility](../reports/mp36-feasibility.md) | Registration/correction review remains blocked by publisher retrieval |

The project-wide [claim register](../docs/claim_evidence.json) retains measured/planned/unavailable statuses. New figures and the failure atlas must derive from compatible authorized evidence; synthetic fixtures must not be presented as real sensor outcomes. No frozen artifact is regenerated here.

Build status 2026-10-26: desktop compiler returned success for the source hashes recorded in `reports/paper-build-validation.json`. Three targeted tests and PR24 CI40282443346 passed. Visual PDF review remains pending.

## Supplied template update — 2026-10-28

The four named IEEE files are now accessible in the supplied local folder; earlier unavailable-template notes are superseded. [Review and hashes](../reports/ieee-paper/evidence/supplied-template-review.json) record the examined conference guidance/class. Added actual7path architecture scope and a numerical conclusion checked against frozen records; updated source compiled. Latest PDF visual inspection is pending because the computer-use tool denied access to ChatGPT. The detailed supplement includes the shared model-instructor derivations; it remains a working source, not a final formatted report.
