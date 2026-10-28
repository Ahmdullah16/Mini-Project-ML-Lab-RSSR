# MASTER CODEX PROMPT — IEEE Research Paper Reconstruction and Publication Preparation

## ROLE AND OBJECTIVE

Act as a coordinated team of:

1. An experienced IEEE research-paper editor.
2. A professor specializing in remote sensing, machine learning, and image super-resolution.
3. A researcher specializing in CNNs, attention mechanisms, Vision Transformers, and image reconstruction.
4. An expert in IEEEtran LaTeX formatting.
5. An experimental reproducibility and scientific-integrity reviewer.
6. A technical illustrator specializing in mathematical diagrams, model architectures, and scientific figures.

Your responsibility is to transform my existing research manuscript into a professionally organized, technically rigorous, IEEE-style conference paper, using my real project implementation and experimental evidence.

**Research project:**

Attention- and Transformer-Based Deep Learning for Super-Resolution of Remote Sensing Images

**Repository:**

Mini-Project-ML-Lab-RSSR

**Provided references:**

- `document.pdf` — my existing research paper.
- `IEEEtran_HOWTO.pdf` — IEEEtran formatting and typesetting guidance.
- `conference_041818.pdf` — IEEE conference template.

Read and compare all three documents before modifying anything.

The objective is not merely to make the document attractive. The final paper must demonstrate research methodology, reproducibility, mathematical correctness, experimental interpretation, and transparent scientific claims.

Never fabricate experiments, figures, citations, numerical measurements, or research novelty.

---

## PHASE 1 — COMPREHENSIVE PAPER AND REPOSITORY AUDIT

Read my entire existing manuscript.

Inspect my GitHub repository, including:

- Source code and model implementations.
- Dataset configuration and preprocessing scripts.
- Experiment configurations and execution records.
- Training and validation logs.
- Stored metrics and generated reports.
- Checkpoints and model weights.
- Dataset and geographic split manifests.
- Super-resolution outputs and baseline comparisons.
- Existing mathematical documentation.
- Previous research-paper source files.
- Relevant artifacts in `reports/paper-evidence.json`, if present.

Map every scientific statement in the existing manuscript to its supporting evidence.

Create a comprehensive comparison matrix:

| Component | Existing paper | IEEE reference requirements | Scientific gaps | Required action |
|---|---|---|---|---|
| Title | Audit | Compare | Identify | Fix |
| Abstract | Audit | Compare | Identify | Rewrite |
| Introduction | Audit | Compare | Identify | Expand |
| Related work | Audit | Compare | Identify | Improve |
| Methodology | Audit | Compare | Identify | Explain |
| Mathematical formulation | Audit | Compare | Identify | Derive |
| Experimental setup | Audit | Compare | Identify | Verify |
| Results | Audit | Compare | Identify | Present |
| Ablations | Audit | Compare | Identify | Validate |
| Figures | Audit | Compare | Identify | Generate if supported |
| Discussion | Audit | Compare | Identify | Strengthen |
| Conclusion | Audit | Compare | Identify | Add |
| References | Audit | Compare | Identify | Verify |

Use the actual repository evidence rather than assuming features described in project plans have been implemented or experimentally validated.

**Maintain four explicit evidence categories:**

1. Experimentally executed and measured.
2. Implemented and technically tested, but not scientifically evaluated.
3. Planned or proposed.
4. Unsupported or unavailable.

Do not merge these categories.

---

## PHASE 2 — RECONSTRUCT THE MANUSCRIPT IN THE IEEE FORMAT

Use `conference_041818.pdf` as the primary visual and conference-layout reference.

Use `IEEEtran_HOWTO.pdf` for technical LaTeX guidance.

The HOWTO document demonstrates journal mode, so do not blindly reproduce its journal headers or title-page settings in a conference manuscript.

Start from:

```latex
\documentclass[conference]{IEEEtran}
```

Preserve the applicable IEEE class defaults, including two-column layout, typography, margins, heading hierarchy, captions, bibliography formatting, and mathematical equation numbering.

Do not manually alter IEEE margins or font sizes simply to fit more content.

Use appropriate LaTeX packages for mathematical notation, graphics, tables, citations, and cross-references.

Use correct author and affiliation formatting. Preserve placeholders for information that has not been confirmed.

Generate a complete, compilable LaTeX project, not just manuscript text.

### Required manuscript structure

**Title and authors**

Use a concise, technically accurate title that represents the actual scientific contribution.

**Abstract**

Rewrite the abstract to explain:

- The remote-sensing super-resolution problem.
- The motivation and research gap.
- The dataset and evaluation protocol.
- The investigated architectures.
- The principal verified numerical findings.
- The main contribution and conclusion.
- Important limitations where they materially affect interpretation.

Avoid unnecessary administrative or repository-development details.

**Index Terms**

Use relevant technical keywords such as remote-sensing image super-resolution, convolutional neural networks, attention mechanisms, Vision Transformers, geographic generalization, and reproducibility.

### I. INTRODUCTION

Explain:

- What remote-sensing image super-resolution means.
- Why spatial resolution matters.
- Practical applications in land-cover analysis, agriculture, urban mapping, and environmental monitoring.
- Limitations of conventional interpolation.
- Motivation for convolutional neural networks.
- Motivation for attention and Transformer representations.
- Challenges of spectral preservation and geographic generalization.
- Research questions.
- The exact contribution supported by existing experiments.

Finish with a concise, defensible statement of contributions.

### II. RELATED WORK

Develop a properly cited technical discussion covering:

- Classical image interpolation.
- SRCNN.
- Residual and deep CNN super-resolution.
- EDSR, where relevant.
- RCAN and channel attention.
- SwinIR and Transformer-based reconstruction.
- Remote-sensing-specific super-resolution studies.
- Synthetic-degradation versus real cross-sensor evaluation.
- Existing remote-sensing datasets including NAIP, WorldStrat and SEN2VENUS, where appropriate.

Explain how the present study differs from prior research.

Include a compact comparison table with verified references, methodology, datasets, and limitations.

Do not claim that combining CNNs, attention and Transformers is inherently novel.

### III. PROBLEM FORMULATION

Develop the mathematics from first principles.

Clearly define high-resolution imagery, low-resolution observations, image channels, spatial scale, degradation, reconstruction, and learning parameters.

For an appropriate synthetic-degradation model, explain the relationship:

\[
Y = \mathcal{D}(H(X))+\varepsilon
\]

where the exact blur operator, downsampling procedure, and noise assumptions must be determined from the implemented experiment.

Define the reconstruction model:

\[
\hat X = F_\theta(Y)
\]

Explain the learning objective and its relationship to the experiment.

Include:

- Image tensor notation.
- Degradation operator.
- Scale factors.
- Reconstruction objective.
- Loss functions actually used.
- Gradient-based optimization where relevant.
- Model capacity and computational cost.

Do not present illustrative equations as though every corresponding method was implemented.

### IV. PROPOSED AND EVALUATED ARCHITECTURES

Inspect the code and document the actual architecture of each implemented model.

The project design includes seven architecture families:

1. CNN.
2. Attention.
3. Transformer.
4. CNN + Attention.
5. CNN + Transformer.
6. Attention + Transformer.
7. CNN + Attention + Transformer.

Distinguish implemented architecture definitions from models that have trained checkpoints and scientifically evaluated results.

For each supported architecture provide:

- Input and output dimensions.
- Model components.
- Layer-by-layer operation.
- Kernel sizes and feature dimensions.
- Residual connections.
- Attention mechanism.
- Transformer tokenization and attention equations, where applicable.
- Activation functions.
- Upsampling technique.
- Parameter count.
- Computational complexity, when measured or reliably calculated.
- Training objective.
- Expected inductive behavior.
- Limitations.

Explain channel attention, spatial attention, self-attention, and residual reconstruction only for components actually implemented or clearly identified as conceptual background.

Include a comparison table of model parameters and structural differences.

### V. DATASETS AND PREPROCESSING

Document:

- NAIP imagery.
- Source imagery characteristics.
- Number of scenes.
- Geographic regions.
- Train/validation/test partitions.
- Image patch extraction.
- Input channel configurations.
- Pixel-value normalization.
- Synthetic image degradation.
- Invalid-pixel handling.
- Dataset leakage prevention.
- Geographic holdout methodology.

Clearly distinguish normalized digital numbers from physically calibrated surface reflectance.

Describe the existing synthetic degradation accurately.

Do not claim real cross-sensor validation unless aligned, compatible, independently verified sensor pairs were actually evaluated.

If WorldStrat, SEN2VENUS, Sentinel-2, or other datasets are only proposed for future experiments, label them accordingly.

### VI. EXPERIMENTAL SETUP

Provide the experimental configuration required for reproducibility.

Include:

- Hardware and computing environment, if documented.
- Training epochs or optimization steps.
- Batch sizes.
- Optimizer and learning rates.
- Loss functions.
- Random seeds.
- Patch dimensions.
- Training and validation protocol.
- Evaluation metrics.
- Baseline configurations.
- Checkpoint selection criteria.
- Geographic holdout procedure.
- Controlled comparison conditions.

State when configuration details are unavailable.

Explain whether architectures were compared under matched parameter counts, matched compute, matched training data, and identical evaluation conditions.

### VII. QUANTITATIVE AND QUALITATIVE RESULTS

This must become one of the strongest sections of the paper.

Reuse all legitimate numerical evidence already available.

Preserve the distinction between:

- Early pilot experiments.
- Later development experiments.
- Frozen held-out test results.
- Three-band experiments.
- Four-band experiments.
- Planned future experiments.

Do not combine incompatible protocols into a single ranking.

#### A. Quantitative evaluation

Create professionally typeset tables for:

- Model parameters.
- PSNR.
- SSIM.
- RMSE.
- MAE.
- SAM, where valid.
- Improvement over appropriate baselines.
- Seed variability.
- Geographic uncertainty, only when supported.
- Computational requirements, when measured.

Use an appropriate precision consistent with the data.

Make the strongest and weakest results understandable without obscuring negative findings.

#### B. Visual comparisons

Where real, traceable image artifacts exist, create multi-panel comparison figures:

- High-resolution reference.
- Low-resolution observation.
- Bicubic interpolation.
- CNN reconstruction.
- Attention reconstruction.
- Transformer reconstruction.
- Hybrid reconstruction.
- Enlarged detail crops.
- Difference/error maps.

Only include outputs from real evaluation runs.

Use identical crop locations, consistent display ranges, valid masks and an appropriate reference.

Do not generate fictional result images or use a generative model to fabricate experimental evidence.

If the needed image files are missing, mark the figure as unavailable and record what is required.

#### C. Ablation study

Audit the existing residual gating experiments and architecture comparisons.

Distinguish:

- Learned gating.
- Fixed attenuation.
- Architecture component removal.
- Baseline behavior.
- Parameter-count effects.
- Random-seed variability.
- Geographic variability.

Do not interpret an improvement caused by fixed residual attenuation as proof of learned reliability.

Report negative or inconclusive outcomes.

#### D. Statistical interpretation

Explain the valid unit of independent sampling.

Separate scene-level variability from training-seed variability.

Do not treat multiple patches of the same scene as independent geographic test regions.

Use paired comparisons and confidence intervals only when supported by the available independent samples.

#### E. Scale-factor analysis

Document the implemented scale factors:

2×, 4×, 8×, 16×, 32× and 64×.

Distinguish native learned reconstruction from repeated cascaded inference.

Explain computational and memory scaling.

Do not claim that increasing output pixel dimensions proves recovery of physically missing geographic details.

---

## PHASE 3 — PRODUCE SCIENTIFIC DIAGRAMS AND FIGURES

Generate original, publication-quality vector diagrams directly from the real implementation.

Prioritize these figures:

**Figure 1 — Complete research workflow**

Remote-sensing imagery → preprocessing → dataset partitioning → degradation → training → inference → evaluation.

**Figure 2 — Model architecture**

Diagram showing the actual CNN, attention, Transformer, and fusion paths used in the implementation.

**Figure 3 — Dataset and degradation pipeline**

Explain HR–LR pair construction, patch dimensions, normalization, blurring, and downsampling.

**Figure 4 — Experimental design**

Show training, validation, held-out evaluation, model selection, and the separation between historical and prospective protocols.

**Figure 5 — Model comparison**

Create quantitative plots from actual experiment records.

**Figure 6 — Image reconstruction comparison**

Show traceable visual examples, only if the required artifacts exist.

**Figure 7 — Ablation analysis**

Visualize measured component-removal and gating results.

Do not add figures merely to fill pages.

Use vector PDF, TikZ, or similarly suitable publication assets where possible. Use high-resolution raster images when necessary.

Ensure all figures have:

- Readable labels at final IEEE column size.
- Correct mathematical notation.
- Informative captions.
- Appropriate units.
- Clear legends.
- Consistent terminology.
- In-text references.

---

## PHASE 4 — EXPAND MATHEMATICAL EXPLANATIONS

Every important model and evaluation equation should be technically justified.

Prioritize:

1. Degradation and inverse reconstruction.
2. Convolution and feature extraction.
3. Residual learning.
4. Attention mechanisms actually used.
5. Transformer operations actually used.
6. Upsampling and image reconstruction.
7. Implemented training loss.
8. MSE, RMSE, MAE, PSNR and SSIM.
9. SAM for compatible spectral data.
10. Computational and parameter complexity.
11. Statistical comparison and uncertainty, where valid.

Define every mathematical symbol.

Ensure equations are consistent with the source implementation.

Use numbered equations, proper notation, and cross-references.

Explain not only how an equation is computed, but what it measures and why it matters scientifically.

---

## PHASE 5 — SCIENTIFIC CONTRIBUTION AND NOVELTY AUDIT

Conduct a critical comparison with established literature.

Investigate whether the implemented methodology demonstrates:

- A new architectural component.
- A new training objective.
- A meaningful evaluation protocol.
- A reproducible baseline comparison.
- A measurable computational advantage.
- A novel finding about attention or Transformer behavior.
- A rigorously documented negative result.

Do not claim architectural novelty based only on combining existing components.

If current evidence supports an empirical evaluation or engineering contribution rather than a novel architecture, present it accurately.

Provide a structured novelty matrix comparing claims, prior art, available evidence and required further experiments.

The central scientific narrative must reflect the strongest defensible finding, even if some architectures do not outperform conventional baselines.

---

## PHASE 6 — WRITE THE DISCUSSION, LIMITATIONS AND CONCLUSION

### VIII. DISCUSSION

Explain:

- What the experiments demonstrate.
- What they do not demonstrate.
- Why certain models perform better or worse.
- Whether improvements are consistent across protocols.
- What parameter and training-budget differences imply.
- Whether attention and Transformers provide measured advantages.
- Implications for geographically unseen imagery.
- Implications for remote-sensing applications.

Use appropriate scientific caution.

### IX. LIMITATIONS AND FUTURE WORK

Discuss:

- Limited independent geographic test scenes.
- Synthetic degradation.
- Absence of validated cross-sensor transfer.
- Unmatched computational budgets where applicable.
- Dataset size and distribution shift.
- Spectral calibration limitations.
- Uncertainty estimation limitations.
- Limits of extremely large reconstruction scales.

Describe future experiments separately from completed results.

### X. CONCLUSION

Write a concise, evidence-driven conclusion stating:

- The problem addressed.
- What was implemented and evaluated.
- Principal verified numerical findings.
- What the evidence supports.
- Remaining open research questions.

Avoid generic claims of superiority or publication readiness.

---

## PHASE 7 — REFERENCES AND CITATION VERIFICATION

Use accurate IEEE citation formatting.

Verify bibliographic metadata for each cited work.

Include DOI or reliable publication URLs where available.

Do not invent citation details.

Ensure:

- Every in-text citation resolves.
- Every reference is relevant.
- Source titles and author names are correct.
- The bibliography is consistently formatted.
- References appear in the appropriate numbered order.
- Dataset sources and original methods receive attribution.

Use a BibTeX bibliography file.

Remove generic placeholder references from the template.

---

## PHASE 8 — PROFESSIONAL IEEE FORMATTING AND QUALITY CONTROL

Use the supplied IEEE conference template as the layout standard.

Check:

- Correct title and author blocks.
- Abstract formatting.
- Index Terms formatting.
- Roman-numbered main sections.
- Proper subsection hierarchy.
- Two-column typography.
- Figure-caption placement.
- Table-caption placement.
- Consecutive equation numbering.
- Appropriate citation formatting.
- Consistent units and abbreviations.
- No broken cross-references.
- No overlapping or clipped content.
- No excessive white space.
- No nearly empty final page caused by avoidable layout problems.
- No unreadable tables.
- No oversized or distorted figures.
- No unnecessary modifications to IEEEtran class dimensions.

Fix the existing paper's dense contribution-audit prose by converting it into meaningful, well-structured scientific writing and tables.

Improve page and column balance without compromising content or IEEE requirements.

Follow the target conference's actual page limit if one is supplied. Otherwise, do not invent a universal IEEE page-limit rule.

Compile the LaTeX source and inspect every page of the rendered PDF.

---

## PHASE 9 — REQUIRED DELIVERABLES

Organize all publication materials under:

`reports/ieee-paper/`

Suggested structure:

```text
reports/ieee-paper/
    main.tex
    references.bib
    figures/
    tables/
    evidence/
    paper_audit.md
    scientific_claims.md
    experiment_protocol.md
    publication_readiness.md
    missing_evidence.md
    README.md
    final-paper.pdf
```

Deliver:

**A. Complete IEEE paper**

A properly typeset, compiled PDF.

**B. LaTeX source**

Editable, maintainable, reproducible LaTeX files.

**C. Scientific figures**

Original diagrams and plots generated from supported technical content or experiment records.

**D. Verified bibliography**

Complete, consistent reference metadata.

**E. Evidence audit**

Map each major claim, table and experimental figure to its exact source.

**F. Publication-readiness report**

Clearly distinguish:

- Formatting complete.
- Paper content complete.
- Experimental evidence sufficient or insufficient.
- Reproducibility verified or pending.
- Novelty established or unestablished.
- Additional experiments required.
- Submission-specific requirements still unknown.

**G. Comparison report**

Explain how the revised manuscript differs from `document.pdf` and how it satisfies the relevant guidance from the two IEEE references.

---

## PHASE 10 — EXECUTION AND CONTINUATION

Do not stop after producing a plan.

Inspect the repository and begin implementing all changes that can be completed with available evidence and tools.

Reuse existing validated outputs before considering additional computation.

Do not silently retrain models or claim that unexecuted experiments have succeeded.

If a necessary experiment cannot be executed because of unavailable datasets, compute, checkpoints or permissions, document the blocker and complete the remaining manuscript work.

After modifications:

1. Compile the LaTeX manuscript.
2. Resolve compilation errors.
3. Check references and equation numbering.
4. Render and inspect every PDF page.
5. Review the numerical tables against the original records.
6. Verify scientific claims.
7. Verify that no results or images were fabricated.
8. Save the updated source and evidence reports.
9. Summarize completed work and remaining scientific blockers.

Do not declare the work accepted or publication-ready simply because the PDF compiles.

### RESUME AFTER CONTEXT OR TOKEN LIMIT

Maintain `reports/ieee-paper/RESUME.md` containing:

- Completed phases.
- Modified files.
- Verified evidence.
- Unresolved scientific issues.
- Compilation status.
- Remaining tasks.
- Exact next action.

When resumed, read this file and continue from the last unfinished task rather than restarting the entire workflow.

## FINAL ACCEPTANCE CRITERIA

The resulting manuscript should look like a professionally typeset IEEE research paper while remaining faithful to the actual project and evidence.

Its results, figures, mathematical descriptions and conclusions must be traceable, technically accurate, properly cited and reproducible where possible.

**Priority order: scientific integrity → research clarity → mathematical accuracy → experimental evidence → IEEE compliance → visual polish.**

Start the full manuscript audit and reconstruction now.
