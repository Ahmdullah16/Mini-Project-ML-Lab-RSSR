# Manuscript comparison matrix — 2026-10-28

Basis: complete current `paper/main.tex` and included generated sections; previous four-page PDF inspection is recorded in [validation](../onsite-learning-validation.json). Current source includes a table scaling correction not present in that older export. Exact supplied IEEE reference requirements are unavailable. The table below therefore distinguishes requested criteria from verified reference comparison; it does not claim template compliance.

| Component | Existing draft | Requested criterion / reference comparison | Scientific gap | Required action |
|---|---|---|---|---|
| Title/authors | Evidence-bounded evaluation title; explicit author placeholders | Conference class present; supplied visual comparison unavailable | Contribution not yet definitive; author details absent | Retain cautious scope; obtain authors before finalization |
| Abstract | States synthetic NAIP, one test scene and no novelty; no numbers | User requests numerical principal findings; template unavailable | Can state existing protocol-specific values; cannot pool rankings | Rewrite after reference gate, deriving values from source-audit rows |
| Introduction | Research questions and scope only | User requests application motivation and supported contributions | Broad motivation and prior-work context incomplete | Expand cited context without claiming measured downstream benefit |
| Related work | SRCNN, SwinIR, WorldStrat, SEN2VENUS/correction; RCAN in audit | Numbered citation keys resolve; metadata comparison not complete | No comprehensive comparison table or exhaustive overlap assessment | Verify original sources and build comparison table |
| Methodology | Narrative residual/inference information | Explicit implemented methods requested | Seven paths, exact operations and training selection stages not fully in manuscript | Use experiment_protocol.md contracts; preserve distinctions |
| Mathematics | MSE, RMSE, MAE, PSNR, SAM equations; SSIM prose | IEEE equation numbering present; full template comparison unavailable | Degradation, attention, tokenization and loss derivations incomplete | Derive from code; independently check dimensions and boundary rules |
| Experimental setup | Broad Adam/L1 and scene descriptions | Reproducible exact run settings requested | Not a complete per-run table; different historical budgets | Derive settings from each exact bound run, not registry defaults |
| Results | Four protocol-separated tables; attenuation interpretation | Readable captions/units requested | Only one test scene; full RMSE/MAE/SAM presentation incomplete | Expand from available records only; no invented missing metrics |
| Ablations | Historical gate vs constant attenuation; no novelty claim | Component-removal and controls requested | No full matched-budget seven-model experiment; M2/M6 absent | Preserve negative result; prospectively design missing runs |
| Figures | No experimental reconstruction figure in main.tex | Traceable vector workflows/architectures and real outputs requested | Complete matching artifact provenance/crop/normalization audit not done | Validate artifacts before figure inclusion; no generative examples |
| Discussion | Negative/inconclusive interpretation scattered across sections | Dedicated scientific discussion requested | Causal explanations not established by changed data and initialization | Separate observations from hypotheses |
| Conclusion | No dedicated conclusion | Evidence-based conclusion requested | Cannot claim superiority or cross-sensor transfer | Summarize exact compatible results and limits after gate |
| References | Six cited keys resolve locally | Correct IEEE bibliography and metadata requested | Key resolution alone is not bibliographic verification | Verify each cited primary source; no fabricated author/publication data |
| Layout | Conference class; previous export four pages | Supplied conference/HOWTO comparison unavailable | Old export oversized tables and orphan reference; correction compiled | Export corrected source, inspect every page; do not alter class margins |

## Findings that change the next edit

1. **Two checkpoint selection stages must be distinguished.** `src/rssr/experiment.py` selects the within-run checkpoint by minimum validation L1. `configs/final-evaluation.json` records later seed/configuration selection using development scene-mean PSNR. Saying simply “best PSNR checkpoint” would be inaccurate.
2. **Historical attention is not attention-only.** The legacy attention model includes the convolutional local branch and binds to cat-v1/M4. M2/M6 have no verified historical trained binding.
3. **Public access wording is stale.** `reports/public-access-20261028.json` records one successful public inference on an older interface. The paper should distinguish that observation from unverified deployment of the latest code and cold-start durability. Do not replace it with a broad deployment-complete claim.
4. **Registry implementation is not measured quality.** Native scales2/4 exist in the factory, but a given trained checkpoint supports only its recorded scale/bands/configuration. The six requested output scales include cascades.
5. **The contribution audit is an initial comparison.** It is not an exhaustive novelty determination. Converting its dense prose to a table is pending the new manuscript-reference gate.
