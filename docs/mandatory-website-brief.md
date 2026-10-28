MANDATORY WEBSITE COMPLETION REQUIREMENTS

Implement these requirements in the existing project. They supplement the previous completion prompt and must be included in the acceptance matrix.

1. COMPLETE ALL 40 MINI-PROJECTS DIRECTLY ON STREAMLIT

Every MP01–MP40 must have a complete on-site lesson containing:
- Objective, prerequisites, applications and connection to the main project.
- Detailed concepts and step-by-step mathematical derivations, defining every symbol, assumption, unit and tensor dimension.
- Worked numerical examples with intermediate calculations.
- Parameter explanations: meaning, actual settings, valid ranges and effects on outputs.
- Architecture/workflow diagrams and intermediate processing stages.
- Actual input images/data, generated outputs, plots, tables and interpretation.
- Relevant verified research papers, with citations beside the methods they support and links to the original sources.
- Failure cases, limitations and reproducible experiment details.

Display all relevant outputs directly within Streamlit. Notebook links, source-code files and brief summaries do not satisfy this requirement. Use readable sections and expandable detail; keep optional downloads secondary.

Independently execute and validate each project. Distinguish measured research results, synthetic teaching examples and unavailable experiments. Never fabricate missing outputs.

2. ALL SEVEN MODELS IN BOTH LIVE SR AND SATELLITE EXPLORER

Provide the same shared model registry in both sections:
M1 CNN
M2 Attention
M3 Transformer
M4 CNN + Attention
M5 CNN + Transformer
M6 Attention + Transformer
M7 CNN + Attention + Transformer

Complete missing training, checkpoint integration and verification. Do not substitute random weights or rename an existing model to fill a missing variant.

All seven must appear with accurate checkpoint, training, band compatibility and availability status. Until a model is genuinely ready, show its blocker and disable execution rather than fake success. Full acceptance requires all seven working for the supported input mode.

Support ×2, ×4, ×8, ×16, ×32 and ×64 with explicit native/cascaded labels and resource checks. Do not silently change dimensions, bands or scale.

3. ADD FULL MODEL COMPARISON TO SATELLITE EXPLORER

Provide “Single model” and “Compare models” workflows in BOTH sections.

Satellite Explorer must acquire one common AOI/date/band selection, then apply all selected compatible models to exactly the same prepared pixels.

Show:
- Original input, bicubic baseline, each selected model and genuine HR reference when available.
- Synchronized crop/zoom, before/after sliders and identical display contrast.
- Full-image previews and enlarged detail regions.
- Valid error maps or clearly labelled disagreement maps.
- Model/scale comparison tables and charts.
- Runtime, output dimensions, checkpoint, preprocessing and provenance.
- Downloadable outputs and comparison results.

Preserve CRS, spatial extent and masks for scientific exports. Never silently convert scientific multispectral input into RGB to make an incompatible checkpoint run.

Run models sequentially where necessary for memory limits. Keep successful outputs if another model fails, and provide a targeted retry.

4. EXPLAIN METRICS DEEPLY BESIDE THE IMAGES

In both Live SR and Satellite Explorer, provide an “Understand these results” section covering SSIM, PSNR, MSE, RMSE, MAE and, where valid, SAM, per-band error and edge/gradient fidelity. LPIPS belongs only in perceptual evaluation.

For each displayed metric include:
- What it measures and what it does not measure.
- Formula, complete symbol definitions and derivation.
- Units, range, preferred direction and implementation settings.
- A small numerical example.
- The actual computed value and its interpretation for the displayed image.
- Relevant research reference.
- Limitations and situations in which the metric can mislead.

Explain that these are evaluation metrics, not image-processing knobs: changing a displayed PSNR or SSIM number does not improve an image. Explain separately when a related quantity is used as a training loss.

5. SHOW VISUAL DEMONSTRATIONS OF METRIC BEHAVIOUR

Add a bounded interactive teaching panel with a known reference image and controlled:
- Blur.
- Noise.
- Brightness/contrast changes.
- Small spatial shifts.
- Sharpening.

Let visitors adjust the distortion, then display reference/distorted images, a difference map and recomputed valid metrics. Label this as a synthetic teaching demonstration, separate from model benchmark results.

Explain observed changes rather than promising universal monotonic behaviour. Show why a sharp-looking image may be inaccurate, why misalignment reduces scores, and why PSNR/SSIM may disagree with perceived quality.

Use fixed declared data ranges and consistent evaluation rules. Keep display stretching separate from metric calculation.

6. SCIENTIFIC HONESTY

For uploads or satellite observations without valid aligned HR:
- Mark reference-based accuracy metrics unavailable.
- Do not use the LR input or bicubic output as supposed ground truth.
- Label differences from bicubic as disagreement, not reconstruction accuracy.
- Keep synthetic-reference demonstrations separate from the uploaded image’s results.

For valid paired data, document registration, masks, border treatment, bands, radiometric compatibility and aggregation. Do not rank models using incompatible studies or missing values.

7. VERIFICATION AND FINISH

Test:
- All 40 complete lesson pages and their real outputs.
- Seven-model visibility and checkpoint compatibility in both sections.
- Single-model and multi-model satellite workflows.
- Every supported scale and resource-limit response.
- Metric formulas, edge cases and teaching controls.
- Correct cache invalidation when input, model, scale or preprocessing changes.
- Mobile/desktop rendering, downloads and error recovery.

Maintain evidence for each requirement. Do not mark these features complete based only on screenshots, placeholder cards or navigation tests.

Continue implementation from the current checkpoint, update the final report with verified findings, and save exact next actions in docs/next_session.md before the token limit.