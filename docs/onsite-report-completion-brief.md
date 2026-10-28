Resume our project from `docs/next_session.md` and all previously saved completion briefs. Inspect current git/PR/CI state, preserve verified work, and continue unfinished implementation without restarting.

Retain all previous requirements: seven distinct CNN/Attention/Transformer combinations; ×2, ×4, ×8, ×16, ×32 and ×64 scaling; scientific Sentinel-2 preprocessing; valid geographic evaluation; individually verified MP01–MP40; and a professional website.

**Add these requirements to the saved project brief and implement them:**

**1. Complete learning content directly on the website**

For every mini-project and each of the seven models, provide:

- Objective, prerequisites, application and connection to the main project.
- Fundamentals and complete step-by-step mathematical derivations, defining every symbol, assumption and tensor dimension.
- A worked numerical example with intermediate calculations.
- Every meaningful parameter: definition, units, configured value, valid range, reason for selection, and effect on output quality, stability, runtime and memory.
- Architecture and processing-flow diagrams, showing input → preprocessing → intermediate operations → output.
- Relevant training details: loss derivation, gradient flow, optimizer, learning rate, batch size, initialization, regularization, checkpoint selection and stopping criteria.
- Actual input/output examples, intermediate visualizations, measured results, failure cases and interpretation.
- References linked beside the concepts and methods they support.

Present this as readable website content with equations, diagrams, interactive plots and expandable explanations—not notebook viewers, code cells or links that visitors must open to understand the work. Distinguish executed results from teaching examples and planned experiments.

Keep shared concepts in a reusable on-site reference section, with direct links from each project. Ensure each project remains understandable independently. Verify that the documented mathematics matches the implemented code.

**2. Start the project report now**

Use my uploaded project format in ChatGPT Library `/mini project`. Previously identified files include `conference_041818.tex`, `conference_041818.pdf`, `IEEEtran.cls` and `IEEEtran_HOWTO.pdf`; verify their current contents and follow all applicable formatting instructions.

Do not assume Library paths exist locally. If the files are inaccessible in this Codex environment, record the transfer requirement and continue drafting against the existing IEEE paper rather than blocking all work.

Prepare an IEEE-format research paper plus a detailed project report/supplement for derivations, all 40 mini-projects and extended results. Respect any actual venue page limit; do not invent one.

Include:
- Abstract, problem statement, research questions and contributions.
- Verified related work with accurate citations.
- Dataset provenance, licences, geographic splits and preprocessing, including atmospheric-correction status.
- Mathematical formulation and seven model architectures.
- Native-scale versus cascaded inference methodology.
- Experimental protocol, baselines, ablations and reproducibility.
- Input/output figures, architecture diagrams, workflow diagrams, learning curves, quantitative tables, uncertainty and failure cases.
- Discussion, limitations, ethics, conclusions and future work.

Generate paper and website figures/tables from the same versioned evidence records. Include captions explaining the data, settings, units and interpretation. Compile and visually inspect the PDF; remove template examples, unsupported statements and unresolved citations. Never invent author details or results.

**3. Prioritise results and establish what is genuinely novel**

Do not claim novelty simply because CNN, attention and Transformers are combined, seven configurations exist, or output dimensions increase.

Create a contribution audit:
proposed contribution → closest prior work → precise difference → experiment needed → measured evidence → supported claim.

Test whether our changes improve reconstruction, spectral fidelity, robustness, efficiency or cross-sensor transfer under a fair protocol. Include component-removal ablations, gain/offset attenuation controls, matched-budget comparisons, multiple seeds and scene-level uncertainty.

Report negative findings and trade-offs. Never tune on the frozen test or select only favourable examples. If no architectural novelty is established, frame the work honestly as a reproducible comparative evaluation, with any supported methodological or empirical contribution.

Focus the abstract, results and conclusion on what we actually demonstrated, by how much, under which conditions and with what limitations. Aim for publication-quality evidence and presentation; do not promise publication acceptance or fabricate findings to obtain it. Do not submit to a venue without explicit instruction.

**4. Completion and continuity**

Test all website content, model controls, equations, figures, downloads and mini-project outputs. Mark missing evidence explicitly and finish independent work despite external blockers.

Before stopping, update `docs/next_session.md` with:
- Exact commit/branch/PR and CI status.
- Completed changes and evidence paths.
- Remaining experiments, report sections and website tasks.
- Commands, environments, failures and external blockers.
- The next concrete action.

Work efficiently: reuse verified artifacts, avoid repeating unchanged expensive experiments, and keep progress updates concise.