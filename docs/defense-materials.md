# Defense presentation and examiner guide

The defense package contains a30-slide editable PowerPoint, a slide PDF, and an editable four-page examiner/viva guide. Main slides1–18 cover the task, module operations, protocols, results, demonstration and limits. Slides19–25 contain editable diagrams for M1–M7. Slides26–30 list the23primary citation records; notes preserve source URLs and applicability limits.

The deck uses native PowerPoint text, diagram blocks/connections, a parameter table and two data charts with embedded worksheets. Comparison images are the previously executed training-example outputs. They are not new independent experiments. Chart values are displayed to four decimal places; complete source precision remains in the recorded JSON reports. Both charts use a zero baseline. Do not combine their different protocols into one ranking.

## Source inputs

- `reports/final-heldout.json`: frozen RGB mean tile PSNR for one scene/sixteen tiles.
- `reports/paper-evidence.json`: earlier pilot means and their source pointers.
- `reports/reference-audit-20261108.json`: citation identities and concepts.
- `src/rssr/models.py` and `architecture_registry.py`: implemented operations.
- Existing external `submission-seven-model-panels-20261108` artifacts: unchanged16×16input, seven-model comparisons and disagreement maps.
- [Examiner guide source](examiner-defense-guide.md) and [short app demonstration](examiner-demo.md).

The authoring source is `scripts/build_defense_deck.mjs`; `finalize_defense_deck.mjs` validates the exact native deck. The Windows orchestration helper is `scripts/build_defense_delivery.ps1`. It requires the configured bundled Node/Python/Artifact Tool runtime, the presentations skill directory, Arial, and Microsoft PowerPoint. It does not train models or download imagery.

## Local build and validation

Pass the runtime paths returned by `load_workspace_dependencies` to `build_defense_delivery.ps1` as `-NodeExecutable`, `-NodeModules`, `-PythonExecutable`; pass the presentations skill directory as `-SkillDirectory` and a fresh build directory as `-Workspace`. The helper stages an editable draft, creates explicit literal-data workbook snapshots, saves a PowerPoint copy, refreshes number caches and runs the packaged finalizer. It preserves other open presentations and refuses to overwrite a prior workspace.

Observed runtime limitation: the preview-authoring process returned exit1 after writing all30slides/previews without a reported authoring exception. This is not recorded as a successful clean one-step build. After inspecting the staged deck and all30previews, `-ResumeDraft` completed the remaining pipeline with exit0. The helper requires the expected draft, source map and30previews before accepting that explicit recovery step; unknown authoring failures are not marked passed.

PowerPoint can expand a decimal cache such as50.2614into a longer binary-float serialization. `refresh_slide_chart_caches.py` refreshes caches from actual embedded worksheet cells only when the numeric difference is≤1e-12; substantive changes are rejected. Embedded workbook bytes and all non-chart parts remain unchanged. Tests verify both properties. The finalizer then verifies exact cache/range consistency, source-independent native chart editability, fonts, geometry, package structure and first-party import.

Exact-final-deck PowerPoint opening and PDF export were tested. Native charts were resized/restored in a private copy and saved/reopened. Every slide and all guide pages were visually reviewed. This does not verify Google Slides rendering, public app execution, geographic research claims or cloud durability.

The Word guide is produced from the Markdown source with Pandoc and rendered through Microsoft Word for review. Keep generated PPTX/PDF/DOCX/PNGs and validation receipts outside Git, per AGENTS.md. Deliver the source and artifact checksums alongside them.
