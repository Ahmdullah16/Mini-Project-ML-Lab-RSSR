# Research questions and claim boundaries — v2

Declared 2026-10-26, before new v2 experiments or new test imagery inspection. This is a rigorous evaluation study, not an asserted novel architecture or state-of-the-art result. [Protocol v2](protocol-v2.md) is a design registration; execution remains gated on a frozen eligible manifest and resource approval.

## RQ1: Architecture under geographic holdout

Do attention/Transformer modules improve reconstruction over a CNN under geographic holdout with matched parameter and compute budgets? Primary contrast: hybrid minus parameter/compute-matched CNN. Null and negative findings are reportable outcomes. A higher pixel count or a visually sharper image cannot answer this question. Training initialization, loss, data, selection, seeds and budget must be controlled; unmatched legacy comparisons are contextual evidence only.

Primary endpoint: mean scene-level PSNR difference in dB under the same protocol. Report paired confidence intervals, absolute/per-scene differences, RMSE changes and efficiency. A prespecified 0.1dB practical margin is a study-design choice, not a universal threshold for perceptual significance. Superiority requires a lower paired interval above that margin; otherwise report uncertainty or lack of superiority, not equivalence.

## RQ2: Real cross-sensor transfer

Does synthetic-degradation training transfer to genuine real cross-sensor pairs? Evaluate unchanged trained weights on independently assigned paired sites, with documented spectral mapping, calibration, acquisition timing, masks and registration. Keep any target-domain adaptation a separate experiment with separate training sites. Synthetic NAIP evaluation and Sentinel RGB inference without genuine HR do not answer this question. MP36 feasibility must establish a suitable source before execution.

## RQ3: Metric agreement and failure modes

Do PSNR/SSIM agree with edge fidelity, spectral measures and justified downstream measures? Use paired per-scene rankings/differences; inspect disagreements rather than combining unlike units into one score. SAM and per-band physical errors require justified compatible calibrated bands. Downstream measures require independent labels and a declared task; absent labels mean unavailable. LPIPS is restricted to a separately labeled Perceptual experiment and is not scientific spectral evidence.

## What existing evidence supports

- Synthetic degradation of real NAIP aerial imagery is implemented. Existing fixed DN/10000 is not verified physical reflectance: [dataset card](dataset-cards.md).
- Four validation scenes and multiple seeds exist, but old architecture budgets differ: [geographic study](../reports/geographic-study.md).
- One frozen held-out scene has already been evaluated. It remains untouched and separate: [final result](../reports/final-heldout.json).
- Constant attenuation accounts for much of early pilot gain; attention causality is not established: [ablation pilot](../reports/ablation-pilot.md).
- No completed real paired cross-sensor evaluation: [feasibility](../reports/mp36-feasibility.md).
- Scale/export checks establish implementation properties, not recovered detail: [scale validation](../reports/scale-validation.json).

Every eventual paper claim must have a row in the [claim/evidence register](claim_evidence.json). Unexecuted protocol components remain planned, and no negative finding is removed to support a preferred architecture.
