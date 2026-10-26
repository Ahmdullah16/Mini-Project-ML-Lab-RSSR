# MP36 — Cross-Sensor Dataset Investigation

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](../../README.md) · [Stage overview](../README.md) · [Progress tracker](../../docs/progress.md)

The expanded numerical teaching component is now executed: seven tests, four real-kernel notebook cells and an analytical example report. Official dataset/version/license/size investigation is recorded, but registration/correction methodology and complete feasibility acceptance remain open. This is not a completed paired cross-sensor evaluation or learner completion.

## Learning objectives

- WorldStrat, Sen2Venµs, geometric and radiometric differences.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP35 — Geographic Dataset Splitting](../../stage_08_remote_sensing/MP35_geographic_dataset_splitting/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Relate ground sampling distance, registration error, radiometric scaling, and spectral response to apparent reconstruction error.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Partially executed; not complete.** Investigate WorldStrat and Sen2Venµs documentation, access, licenses, registration, and band compatibility before downloads; deliver a dataset comparison matrix and cross-sensor evaluation feasibility report.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.
- Remote sensing: preserve CRS, affine transform, band meaning, nodata, physical units, and source licenses. Split by region/scene before tiling.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- A documented script or notebook and exact run command added during implementation; none is supplied in this scaffold.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Remote sensing](https://rasterio.readthedocs.io/en/stable/)
- [WorldStrat](https://github.com/worldstrat/worldstrat)
- [Sen2Venµs](https://zenodo.org/records/6514159)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](cross_sensor_dataset_investigation.ipynb), [local tools](mp36_tools.py), [tests](test_mp36_tools.py), [runner](run_notebook.py) and [validation](VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:

```bash
python -m pytest -q test_mp36_tools.py
python run_notebook.py
```

File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.


## Executed numerical lesson and remaining gate

[Analytical report](../../reports/mp36-numerical-examples.json), [source feasibility matrix](../../reports/mp36-feasibility.md) and [validation](VALIDATION.md) separate source facts, deductions and unresolved questions. Run from repository root:

```bash
python -m pytest -q stage_08_remote_sensing/MP36_cross_sensor_dataset_investigation
python stage_08_remote_sensing/MP36_cross_sensor_dataset_investigation/experiment.py
python stage_08_remote_sensing/MP36_cross_sensor_dataset_investigation/run_notebook.py
```

Worked examples:10m/5m gives×2;10m/1.5m gives20/3 (not an integer native model scale). A5m registration offset divided by5m/pixel equals1pixel. For reference[.2,.4] and gain1.1/offset.01, apparent MSE is.0017 despite no SR reconstruction step. Equal-width toy spectral bins with values[.2,.8] produce.2,.8 or.5 under different response weights. The notebook defines symbols, units, exercises and solutions; weights are invented teaching values, not measured sensor response.

Next acceptance gate: verify official registration methodology and published correction, complete the source-specific band/timestamp/mask matrix, and prepare a concrete bounded resource proposal. The paper endpoint returned429; do not infer a registration tolerance. Real paired experiments additionally require the [v2 protocol gates](../../docs/protocol-v2.md). No large download or training was authorized or executed by this lesson.
