# MP11 — Super-Resolution Evaluation Dashboard

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](../../README.md) · [Stage overview](../README.md) · [Progress tracker](../../docs/progress.md)

Expanded [notebook](super_resolution_evaluation_dashboard.ipynb) executed in a real kernel. [Measured experiment](../../reports/mp09-mp11-experiments.json) records the checksummed training crop and controlled settings. Reproduce with `python scripts/metric_experiments.py`. Worked solutions are included; learner mastery remains unassessed.

## Learning objectives

- Metrics, error maps, comparison figures and result tables.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP10 — SSIM Evaluation Tool](../../stage_03_metrics/MP10_ssim_evaluation_tool/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Define per-image PSNR and SSIM aggregation, error-map scaling, and fair comparison conventions.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Implemented and executed on synthetic fixtures and the recorded training crop.** Design static metric tables, side-by-side crops, and error maps for identical, noisy, and blurred fixtures; deliver a dashboard specification and verified metric examples.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; provided in the linked notebook and experiment script.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Image metrics](https://scikit-image.org/docs/stable/api/skimage.metrics.html)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](super_resolution_evaluation_dashboard.ipynb), [local tools](mp11_tools.py), [tests](test_mp11_tools.py), [runner](run_notebook.py) and [validation](VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:

```bash
python -m pytest -q test_mp11_tools.py
python run_notebook.py
```

File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.
