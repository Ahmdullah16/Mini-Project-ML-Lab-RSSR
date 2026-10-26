# Working evaluation paper

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
