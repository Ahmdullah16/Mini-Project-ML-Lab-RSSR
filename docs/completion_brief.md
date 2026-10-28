Continue completing and improving this existing project:

Repository: https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR
App: https://mini-project-ml-lab-rssr.streamlit.app/

Read AGENTS.md, completion_brief.md, next_session.md, submission_readiness.md and current git state. Incorporate this request into the persistent brief. Preserve completed implementations, original images, checkpoints, recorded findings and frozen test results. Execute changes rather than only proposing them.

GOAL
Improve reconstruction clarity and build a polished research app that displays outputs from all available compatible models, with conceptual and mathematical explanations beneath each comparison. Complete remaining original requirements as well.

1. INSPECT THE PROVIDED EXAMPLES
Identify the original LR image, our SR output, ChatGPT-generated enhancement and explanatory illustration using provenance, dimensions and actual content—not filenames alone. If these attachments are unavailable in this Codex workspace, request their upload once while continuing independent work.

Compare identical spatial regions at matched display scale. Inspect buildings, roads, riverbanks, vegetation, blur, noise, halos, repeated textures and geometric changes. Do not treat the generated enhancement or AOI illustration as measured HR ground truth or training labels.

Explain the distinction between:
- More output pixels.
- Visually sharper texture.
- Accurate recovery of real spatial detail.

Do not promise ChatGPT-equivalent quality or physical detail recovery that the available data cannot support.

2. IMPROVE QUALITY THROUGH CONTROLLED EXPERIMENTS
First diagnose preprocessing and inference: band order, normalization, clipping, dtype, checkpoint selection, color conversion, degradation mismatch, padding, tiled seams and display resizing. Verify that the intended trained model actually runs.

Compare bicubic, existing CNN, attention, Transformer and hybrid models under matched input/scale conditions. Inspect training sufficiency and geographic/domain mismatch. Investigate suitable licensed pretrained restoration/SR models using official sources; record model identity, license, weights, training domain and limitations.

Maintain two clearly labeled modes:
A. Scientific reconstruction: prioritize fidelity, geometry, spectral consistency and reproducibility.
B. Optional perceptual enhancement: prioritize visual appeal while warning that plausible details may be synthesized. Keep this RGB-only unless independently justified for other bands.

Keep baseline SR, optional postprocessing and perceptual enhancement separately selectable and downloadable. Never hide sharpening or contrast adjustments inside the reported model output.

Investigate losses incrementally:
L = λrec Lrec + λedge Ledge + λspec Lspec + λcons Lcons.
Define each implemented term, its units, masks and weights. Use spectral terms only with meaningful compatible bands and a consistency operator only when justified. Consider perceptual/adversarial objectives only in the explicitly labeled perceptual experiment.

Tune on training/validation data only. Preserve the frozen benchmark; new experiments require a versioned protocol and appropriate independent evaluation. Do not tune against previously observed test scores. Retain unsuccessful experiments.

Use validation evidence to choose defaults. Do not select a model solely because its architecture is more complex.

3. ADD “COMPARE MODELS” TO THE EXISTING STREAMLIT APP
Extend the current showcase rather than rebuilding it. Allow users to select available models and run comparisons explicitly.

Display a responsive gallery:
Original LR → bicubic → CNN → attention/Transformer variants → hybrid → optional perceptual result → genuine HR reference, when available.

Only show outputs actually computed or clearly labeled precomputed examples. Label unavailable models honestly. Account for incompatible scale/band requirements; never silently resize outputs to manufacture a fair comparison.

Provide:
- Matched display scaling and synchronized crop/zoom.
- Before/after slider where practical.
- Full-image and detail-crop views.
- Individual downloads and a comparison bundle.
- Model/checkpoint, native scale, input/output dimensions, bands, runtime and provenance.
- Clear stage progress and recoverable errors.
- Lazy loading, sequential bounded execution and caching suited to the existing free host.

A model selection change must not mislabel an older result. Keep results keyed to input hash, model/checkpoint, preprocessing and settings.

4. PUT MATHEMATICS AND INTERPRETATION BELOW RESULTS
Create reusable metric-definition and interpretation components. For each displayed metric explain:
what it measures → formula → symbols/units → computed value → preferable direction → comparison → limitations.

For a valid aligned HR reference x and prediction x̂, define N as the number of valid scalar samples and show:

MSE = (1/N) Σ(x̂ᵢ − xᵢ)²
RMSE = √MSE
MAE = (1/N) Σ|x̂ᵢ − xᵢ|
PSNR = 10 log10(L²/MSE)

L is the declared data range, not an arbitrary per-image maximum. Handle exact equality, empty masks and invalid values explicitly.

Explain SSIM using:
SSIM(x,y) =
[(2μxμy + C1)(2σxy + C2)] /
[(μx² + μy² + C1)(σx² + σy² + C2)]
State the actual implementation’s window, constants, covariance convention, channels and aggregation.

Where bands are scientifically compatible:
SAM = arccos[(x·x̂)/(||x|| ||x̂||)]
State radians/degrees, zero-vector handling and averaging. Explain why RGB display color is not calibrated multispectral reflectance.

For efficiency, show measured runtime, parameters and memory with hardware, input size, warm-up and scope. Distinguish sampled RSS from peak memory and partial MAC counts from full FLOPs.

Show baseline differences:
ΔPSNR = PSNRmodel − PSNRbaseline
RMSE reduction (%) = 100(RMSEbaseline − RMSEmodel)/RMSEbaseline
Handle zero denominators. Do not present percentages of decibel values as meaningful quality improvements.

Show numerical substitution for a selected example or small crop, with formulas matching the actual implementation.

For arbitrary uploads without HR:
- Mark PSNR/SSIM/MAE/RMSE against HR as unavailable.
- A model-to-bicubic difference is disagreement, not ground-truth error.
- Edge energy, Laplacian variance or other sharpness descriptors must be labeled as descriptors that noise/oversharpening can increase.
- If displaying ||D(x̂)−y||, document D and explain that low LR-consistency error does not prove correct HR detail.

Do not create a single “accuracy percentage” from unrelated metrics.

5. EXPLAIN WHAT IS HAPPENING VISUALLY
Under every result, provide a short evidence-based interpretation:
- What changed relative to the input/baseline?
- Which measured properties improved or worsened?
- Are edges clearer, textures smoother, colors shifted or artifacts visible?
- Does this indicate measured fidelity or only visual sharpness?
- What cannot be concluded?

For image properties explain dimensions, scale factor, channels, dtype, range, pixel spacing and file size. Explain that ×2 produces four times as many pixels but does not prove twice the physical resolving power. Compression affects file size independently of quality.

Keep raw inference and display enhancement separate. Use shared contrast settings for fair previews. Never silently stretch every model independently.

Use accurately labeled heatmaps with colorbars:
- Prediction minus valid HR: reconstruction error.
- Prediction minus bicubic: baseline disagreement.
- Gradient/edge map: structural descriptor.

Do not describe attention maps as proof of causal explanation or confidence unless that interpretation is validated.

6. MAKE THE ENTIRE RESEARCH SHOWCASE PRESENTABLE
Improve existing Overview, Live SR, Results, Methodology & Data, Mini-projects, Satellite Explorer and Downloads sections.

Use readable typography, consistent spacing/colors, accessible controls, responsive cards and concise explanations. Show equations/details in expanders beneath clear plain-language findings.

Present all recorded studies through filters: baselines, ablations, training curves, robustness, spectral behavior, efficiency and failure cases. Retain dataset, split, scene count, units, protocol and source links. Do not pool incompatible studies or treat repeated patches/seeds as independent geographic evidence.

For every mini-project show objective, concept, experiment, output, interpretation and connection to the major project, with actual completion evidence.

For satellite outputs explain AOI, CRS, acquisition time, retrieval time, bands, cloud/nodata handling and georeferencing. Validate geographic labels against metadata. Do not infer north from image orientation. Compute distances geodesically or with an appropriate projection; a longitude degree is not a constant 111 km at every latitude.

7. COMPLETE, VALIDATE AND DELIVER
Continue all outstanding tasks from the current readiness checklist, including remaining lessons, portable checkpoints, applicable research studies, external persistence, final report, presentation and reproducible submission package. Keep externally blocked tasks explicit while completing independent work.

Generate app tables, explanations, report figures and downloads from one validated evidence source. Add meaningful checks for metric correctness, comparison alignment, model/result labeling, missing-reference handling, exports and UI failure states.

Verify clean installation, final notebooks, required tests, links and CI. Diagnose the existing app’s access/deployment issue using authorized access, then verify actual public inference and downloads. Do not declare a local preview publicly deployed.

Respect current review/merge restrictions and resource budgets. Ask only for concrete missing access or additional resources; never request secrets in chat or activate paid services without approval.

Work efficiently: targeted reads, reuse verified artifacts, affected tests per change, final full checks at release. Maintain compact resumable checkpoints after each milestone.

Finish with implemented changes, measured quality gains/tradeoffs, test evidence, verified app/version, submission artifacts and remaining blockers. Do not claim scientific accuracy, novelty or completion beyond the evidence.

## Scale correction — 2026-10-26 (supersedes earlier size limits)

1024×1024 is the maximum INPUT, independently for width and height. Accept rectangular and odd dimensions unchanged. Offer ×2, ×4, ×8, ×16, ×32, ×64; output width=s*input width and height=s*input height, pixels grow by s². 256×128 at ×8 is 2048×1024; 1024×1024 at ×64 is 65536×65536. Never silently resize, crop, lower scale or label interpolation learned SR.

Use verified native models or explicitly labeled cascades, disclose interpolation and intermediate quantization. Show dimensions, pixels, estimated memory/storage and configured/measured availability BEFORE execution. Large local jobs require overlapped tiles, seam checks, incremental disk-backed outputs and separately bounded previews. Keep all scales visible on the free host; rejected requests need a runnable local/Colab job with identical input/model/settings. Do not claim untested large hosted execution. Test odd, rectangular, square and maximum input contracts, exact exported sizes, resource rejection and seams. Preserve frozen scientific results; scaling is not evidence of recovered detail.


## Research-first priority update — 2026-10-26

The [latest complete user brief](research_upgrade_brief.md) overrides conflicting earlier requirements. Prioritize defensible evidence, then submission package, then interface/deployment. Commit research questions and protocol v2 before new experiments/test inspection; retain null findings and frozen results. Large downloads, long training and paid resources require explicit approval with size/compute/cost/purpose. Existing bounded implementation checks remain authorized.

## Latest expanded acceptance requirements — 2026-10-27

The [seven-model completion brief](seven-model-completion-brief.md) preserves the latest user request verbatim and supplements this brief. It requires distinct C/A/T configurations, an honest seven-by-six scale matrix, final independently checked MP01–40 outputs, interactive lessons, evidence-backed IEEE submission materials and deployment evidence. See [architecture definitions and status](architecture-registry.md). Previous scientific gates and input≤1024-per-side/exact-scale rules remain. Implementation smoke tests are not trained models or final research acceptance.

## Latest website/scientific upgrade — 2026-10-28

The [latest full user brief](scientific-website-upgrade-brief.md) adds a versioned advanced family with ablations, a separate scientific Sentinel-2 reflectance path, and a results-first native website. Preserve educational models, historical checkpoint compatibility and frozen evidence. [Scientific admission status](scientific-sentinel.md) distinguishes implemented local conversion from unfinished live provider validation. No advanced-family, seven-trained-model, real cross-sensor or public-readiness completion is implied.


## Additional requirements — 2026-10-28

The [on-site teaching, report and contribution-audit brief](onsite-report-completion-brief.md) supplements all prior scope. Complete derivations, parameter explanations, measured examples and references must be readable on-site for all40 lessons and7 models. Start a shared-evidence report supplement now; do not claim novelty from combinations or enlargement. Library `/mini project` files are not mounted here: targeted attachment search found none of the four named IEEE files. Transfer remains required; no venue page limit or author details inferred.


## IEEE reconstruction request — 2026-10-28

The [master reconstruction brief](ieee-reconstruction-brief.md) supplements the prior scope. Its explicit requirement to read and compare the existing PDF, conference template and HOWTO gates new manuscript reconstruction until the exact reference PDFs are accessible. Continue independent evidence audit under reports/ieee-paper/. Existing paper/ remains canonical until a reviewed migration; no duplicate diverging manuscript.
