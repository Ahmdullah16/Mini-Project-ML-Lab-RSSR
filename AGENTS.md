# Contributor and agent guidance

## Scope and teaching style

This file applies throughout the repository. Preserve nine numbered stages and unique MP01–MP40 project IDs. Detailed project briefs are proposed expansions of the existing syllabus for review. All projects begin Not started. Implement only the project or change requested; never infer permission to run large training jobs or download large datasets.

Write beginner-friendly explanations: define symbols, state assumptions, show a small numerical example, and describe expected inputs and outputs. Keep each project's objectives, prerequisites, mathematics, experiments, deliverables, references, and completion criteria current. Distinguish compact teaching models from faithful paper reproductions.

## Implementation conventions

Use readable Python, small functions, explicit paths or CLI arguments, and deterministic seeds where possible. Document tensor/image shapes, channel order, dtype, value range, color space, device, and boundary conventions. Cast integer arrays before arithmetic that may overflow. Validate shapes, input ranges, and zero/invalid cases.

Use a tiny synthetic fixture before real data. For training, document optimizer, loss, schedule, batch size, epochs, checkpoint selection, device, package versions, and seed variability. Add dependencies only when needed and record a resolved environment for reproducible experiments.

## Research integrity and data

Never fabricate results, citations, test passes, or completion evidence. Keep plans separate from observations. Use identical protocols for baseline comparisons and document alignment, border crop, metric range, aggregation, and compute budget. Report failures and limitations.

Separate training, validation, and test sets; split geospatial scenes/regions before tiling. Fit preprocessing on training data. Preserve CRS, transforms, band semantics, nodata, and reflectance units. Document source licenses and checksums. Never commit datasets, weights, generated outputs, secrets, or virtual environments.

## Validation and progress

Validate all relative Markdown links, including cross-stage prerequisite links, whenever structure changes. Ensure 40 unique sequential project IDs and statuses consistent between briefs and docs/progress.md. Run checks appropriate to implementation changes; document actual commands and outcomes. Documentation-only scaffolding must not add implementation notebooks or claim completed experiments.

Update docs/progress.md to In progress only when substantive work starts, and to Complete only after every project criterion is satisfied with linked evidence. Submit reviewable commits or pull requests; do not merge without an explicit request.
