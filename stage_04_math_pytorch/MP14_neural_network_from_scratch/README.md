# MP14 — Neural Network from Scratch

**Status: Not started** · **Curriculum: proposed for review**

[Laboratory home](../../README.md) · [Stage overview](../README.md) · [Progress tracker](../../docs/progress.md)

## Learning objectives

- Forward propagation, activation functions, loss and backpropagation.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP13 — Gradient Descent from Scratch](../../stage_04_math_pytorch/MP13_gradient_descent_from_scratch/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Use a = activation(Wx+b), squared-error loss, and the chain rule for backpropagation.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Planned; not run.** Plan a small NumPy network on a tiny regression dataset; deliver a forward/backward derivation and finite-difference gradient checks.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; none is supplied in this scaffold.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Mathematics and PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.
