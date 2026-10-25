# MP20 — SRCNN Implementation

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](../../README.md) · [Stage overview](../README.md) · [Progress tracker](../../docs/progress.md)

Expanded notebook and [executed experiment report](../../reports/mp19-mp22-cnn.json) provide mathematics, implementation, measured comparisons, exercises and solutions. Learner mastery remains unassessed.

## Learning objectives

- Feature extraction, nonlinear mapping and SR reconstruction.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP19 — CNN Image Reconstruction](../../stage_05_cnn_sr/MP19_cnn_image_reconstruction/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Map bicubic-upsampled input through feature extraction, nonlinear mapping, and reconstruction convolutions; minimize reconstruction MSE.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Teaching experiment executed; see notebook for remaining scientific limits.** Plan a compact SRCNN comparison with bicubic on fixed image pairs; deliver an architecture diagram, validation curves, and held-out metric table.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Run `python scripts/cnn_lessons_experiment.py`; see the corresponding expanded notebook and reports/mp19-mp22-cnn.json.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: CNN super-resolution](https://arxiv.org/abs/1501.00092)
- [SRCNN paper](https://arxiv.org/abs/1501.00092)
- [VDSR paper](https://arxiv.org/abs/1511.04587)
- [ESPCN paper](https://arxiv.org/abs/1609.05158)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

The independent held-out SRCNN table remains pending. This teaching model reports validation only and does not reuse the frozen research test for further selection.

## Project-folder files

This folder now contains the [local notebook](srcnn_implementation.ipynb), [local tools](mp20_tools.py), [tests](test_mp20_tools.py), [runner](run_notebook.py) and [validation](VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:

```bash
python -m pytest -q test_mp20_tools.py
python run_notebook.py
```

File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.
