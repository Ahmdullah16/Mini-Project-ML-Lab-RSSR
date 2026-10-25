# MP18 — CNN Edge Detection Laboratory

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](../../README.md) · [Stage overview](../README.md) · [Progress tracker](../../docs/progress.md)

Expanded [notebook](cnn_edge_detection_laboratory.ipynb), [experiment record](../../reports/mp18-edges.json) and [figure](../../reports/mp18-edges.png) compare fixed and learned Sobel operators. Learner mastery remains unassessed.

## Learning objectives

- Sobel filters and learned convolutional features.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP17 — Convolution Playground](../../stage_05_cnn_sr/MP17_convolution_playground/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Apply Sobel horizontal and vertical kernels; compute gradient magnitude sqrt(gx²+gy²).

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Executed:** `python scripts/edge_experiment.py` trains a small convolution to imitate known Sobel labels, validates on disjoint synthetic edge fixtures, compares kernels/maps on a licensed training crop and checks zero versus replicate padding. This is operator imitation, not independent semantic-edge accuracy.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Executable script `scripts/edge_experiment.py` and expanded notebook with definitions, hand examples, exercises and worked solutions.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: CNN super-resolution](https://arxiv.org/abs/1501.00092)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](cnn_edge_detection_laboratory.ipynb), [local tools](mp18_tools.py), [tests](test_mp18_tools.py), [runner](run_notebook.py) and [validation](VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:

```bash
python -m pytest -q test_mp18_tools.py
python run_notebook.py
```

File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.
