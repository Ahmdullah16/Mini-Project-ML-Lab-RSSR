# MP39 — Controlled Ablation Study

**Status: In progress** · **Curriculum: proposed for review**

[Laboratory home](../../README.md) · [Stage overview](../README.md) · [Progress tracker](../../docs/progress.md)

Historical ablation analysis and numerical uncertainty lesson are expanded. Full geographic/matched-budget acceptance remains pending; learner mastery is unassessed.

## Learning objectives

- CNN, CNN + Attention, CNN + Transformer, full hybrid.
- Explain the mathematical model below and its assumptions in your own words.
- Interpret the planned comparison using evidence and identify a limitation.

## Prerequisites

[MP38 — Hybrid CNN–Attention–Transformer SR Model](../../stage_09_hybrid_research/MP38_hybrid_cnn_attention_transformer_sr_model/README.md) and the earlier foundations in the roadmap.

Read the root setup guide. Use synthetic inputs first and document any external dataset source, license, and preprocessing. References are reading suggestions; a compact educational model need not reproduce a full published architecture.

## Mathematics

Estimate paired per-scene metric differences across variants; report seed variability and scene-level confidence intervals without pixel-level pseudo-replication.

Write down variable dimensions and units before coding. Work through one small numerical example and define edge cases; equations here specify topics to study, not measured results.

## Experiments

**Historical pilot available; broader research incomplete.** Reuse the24 recorded runs spanning eight variants and three seeds. Do not retrain or relabel the one-scene pilot as geographic validation. Matched-budget comparisons and regional uncertainty remain under protocol v2.

1. Define the question, baseline, controlled variables, input selection, and expected sanity checks before implementation.
2. Begin with a tiny deterministic fixture, then use a documented sample. Change one factor at a time.
3. Record settings, seeds, package versions, runtime/device, and observations. Training projects must keep validation and held-out test data separate.
4. Compare outcomes, explain failure cases, and record limitations. Do not tune on the test set.
- Remote sensing: preserve CRS, affine transform, band meaning, nodata, physical units, and source licenses. Split by region/scene before tiling.

## Deliverables

- The project-specific plots, tables, specifications, or reports named in the experiment above, when the project is implemented.
- Project-local experiment, expanded notebook and exact commands are provided below; remaining scientific acceptance is explicit.
- An experiment record with data provenance, configuration, checks, results, and interpretation. Keep large data, weights, and generated artifacts outside Git; link a reproducible artifact manifest instead.
- A short reflection: what was learned, what failed, and the next question.

## References

- [Stage reference: Hybrid research](https://arxiv.org/abs/2205.04437)

## Completion criteria

- [ ] Explain the learning objectives and solve the small mathematical example.
- [ ] Implement and run the planned experiment with a documented command and reproducible inputs.
- [ ] Verify relevant numerical, shape, and edge-case checks; record the actual outcomes.
- [ ] Produce the project-specific deliverables and interpret the baseline comparison without unsupported claims.
- [ ] Document data licenses, environment, limitations, and any deviations from the brief.
- [ ] Link evidence in the progress tracker and change status only after all criteria are met.

## Project-folder files

This folder now contains the [local notebook](controlled_ablation_study.ipynb), [local tools](mp39_tools.py), [tests](test_mp39_tools.py), [runner](run_notebook.py) and [validation](VALIDATION.md), following MP01's layout. Shared algorithms stay in src/rssr where applicable; local tools expose their API without divergent copies. Original notebooks/ snapshots and historical execution records are retained. This is the canonical notebook location for subsequent work.

From this directory, in the repository development environment:

```bash
python -m pytest -q test_mp39_tools.py
python run_notebook.py
```

File presence does not imply that all scientific or educational criteria are complete. The status and remaining criteria above remain authoritative.


## Executed evidence analysis — 2026-10-27

[experiment.py](experiment.py) reads immutable ablation-pilot-runs.json, checks run identity, seeds, source manifest and parameter consistency, and writes a derived [table/report](../../reports/mp39-ablation-analysis.json). Source hash and run IDs make the lineage explicit. The pilot uses100 training steps per run, three seeds17/23/41 and one validation scene. Counts/budgets are displayed rather than described as matched. No new data, training or inference runs here.

### Estimands and units

For a model's seed scores q1..qK, mean q=(sum qk)/K and sample seed SD=sqrt(sum(qk-mean q)^2/(K-1)). Scores are dB; a difference from bicubic is subtraction in dB, not a percentage. For region j, a scientifically paired effect would be d_j=mean_seed(model_j)-mean_seed(control_j). Geographic uncertainty resamples independent regions j, keeping model/control paired. Three random initializations on one scene do not create three regions. Thus this pilot's geographic CI is explicitly null.

Worked seed example[1,2,3]dB gives mean2 and sample SD1; paired against baseline2 gives mean delta0. Separate synthetic scene example A=[2,3,4],B=[1,2,3] has constant effect1, so every bootstrap resample gives1. Its interval[1,1] is a mathematical check, not confidence about satellite reconstruction.

### Controls and negative findings

Compare CNN, CNN+attention, CNN+window and hybrid within the same historical protocol, then inspect addition/concatenation, constant attenuation, learned gate and consistency controls. The report retains every variant, including weak outcomes. The learned-versus-constant gate difference and consistency-versus-hybrid difference are descriptive contrasts. A constant residual multiplier can explain much of the apparent gate benefit; unequal parameters and short training prevent architectural superiority claims. Existing bicubic performance remains a relevant baseline even when learned components underperform it.

### Exercises with solutions

1. May sixteen tiles from one scene be bootstrapped as sixteen independent regions? **No.** Shared scene properties violate the intended geographic sampling unit.
2. Does a negative effect mean the component can never help? **No.** It describes the tested budget/data/protocol; retain the null/negative result without universal extrapolation.
3. Why reject repeated run IDs? **Because** duplicating observations silently changes weights and can make uncertainty appear smaller.
4. What would make the next comparison stronger? **Answer:** independent geographic groups, validation-only selection, matched parameter/MAC budgets, multiple seeds and the frozen v2 analysis rules before test access.

### Commands and acceptance

From repository root run `python -m pytest -q stage_09_hybrid_research/MP39_controlled_ablation_study`, the local `experiment.py`, then `run_notebook.py`. Source images and licenses are inherited from the original NAIP manifest; this derived table does not redistribute imagery. The toy bootstrap uses fabricated numbers explicitly labeled synthetic.

Historical table/negative-finding interpretation is now implemented; region-level uncertainty and matched-budget evidence remain unavailable. Status stays In progress until remaining scientific criteria are resolved. See [protocol v2](../../docs/protocol-v2.md) and [validation](VALIDATION.md).
