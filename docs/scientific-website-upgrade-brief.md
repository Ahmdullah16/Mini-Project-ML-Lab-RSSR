Upgrade and complete our existing remote-sensing super-resolution project.

Repository:
https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR
Current app:
https://mini-project-ml-lab-rssr.streamlit.app/

Continue from the latest git state and docs/next_session.md. Read AGENTS.md and existing completion/research briefs. Preserve working code, checkpoint compatibility and frozen research evidence. Implement and verify changes rather than only proposing them.

GOAL
Build a professional, results-first research website with seven distinct model variants, multiscale inference, all 40 mini-projects explained on-site, and defensible satellite validation. Visitors must not need notebooks, terminals or code execution knowledge.

1. SEVEN ADVANCED MODEL VARIANTS
Provide:
M1 CNN
M2 Attention
M3 Transformer
M4 CNN + Attention
M5 CNN + Transformer
M6 Attention + Transformer
M7 CNN + Attention + Transformer

Audit and reuse the existing architecture registry and historical checkpoint bindings.

Define:
C = convolutional feature backbone;
A = additional channel/spatial attention;
T = Transformer blocks, including their inherent self-attention.

Disclose shared embedding/reconstruction convolutions. Do not call Transformer “attention-free” or label CNN+attention as attention-only.

Develop a versioned advanced configuration family using justified improvements such as deeper residual feature extraction, residual attention and shifted-window Transformer context. Preserve the educational variants. Every added component requires an ablation; do not add complexity merely to claim improvement.

Give each variant separate configuration, training provenance, checkpoint identity and status: implemented, trained, evaluated, deployable. No random-weight output presented as trained SR and no seven labels pointing to the same checkpoint.

2. MULTISCALE SUPPORT
Offer ×2, ×4, ×8, ×16, ×32 and ×64 for supported inputs up to 1024 pixels per side. Consider larger powers of two only after resource preflight and successful bounded tests.

Exact output dimensions must be sW × sH. Preserve aspect ratio, including odd/rectangular inputs.

Distinguish:
- Native trained scale.
- Cascaded learned SR.
- Interpolation.
- Experimental perceptual enhancement.

Reuse existing tiled/disk-backed scaling. Add resource estimates, overlap/seam validation, bounded previews and atomic exports. Never silently crop, downsize or reduce the requested scale.

Make oversized jobs available through a documented external/local execution path while keeping their results viewable on-site. State that ×64 creates 4096 times as many pixels; it does not prove 64 times better spatial resolution.

Display nominal output pixel spacing separately from experimentally validated resolving power.

3. SCIENTIFIC SENTINEL-2 PIPELINE
Audit current fresh.py: it uses sentinel-2-l2a visual uint8 RGB and item-level cloud filtering. Retain that as a clearly labelled display mode.

Implement a separate scientific mode using actual spectral reflectance bands and provider metadata:
- Product ID, acquisition time, processing level/baseline and processor.
- Band wavelengths, native resolution, CRS, transform and units.
- Provider scale/offset and harmonization status, applied exactly once.
- Pixel-level cloud, cirrus, cloud-shadow, snow, saturation and nodata masks.
- Declared band resampling/alignment and valid-pixel rules.
- Float32 scientific values retained separately from display stretching.

Do not assume DN/10000 is universally sufficient. Handle baseline-dependent offsets and provider transformations explicitly.

Use existing L2A surface reflectance without repeating atmospheric correction. If only L1C is available, use a documented appropriate correction workflow such as Sen2Cor and record its settings/version. Do not apply Sen2Cor to RGB screenshots or unrelated sensors.

Use distinct terminology for surface reflectance and super-resolution to avoid “SR” ambiguity. Do not describe upsampled 20 m bands as independently observed 10 m measurements.

4. REAL GEOGRAPHIC VALIDATION
Evaluate official WorldStrat and SEN2VENµS releases first; investigate other licensed aerial/satellite pairs and relevant Bhoonidhi products.

Select genuinely overlapping LR/HR acquisitions with:
- Compatible dates and stable land cover.
- Verified registration and measured residual alignment error.
- Compatible spectral responses, units and radiometric processing.
- Valid cloud/shadow/nodata intersection.
- Recorded native GSD and any pansharpening/resampling.
- Explicit licences permitting the intended research and website use.

Do not force naturally noninteger sensor-resolution ratios into an arbitrary ×2/×4 task without documenting resampling and consequences.

Separate:
A. Controlled synthetic-degradation evaluation.
B. Real cross-sensor paired evaluation.
C. Qualitative geographic inspection.

Google Earth Pro screenshots must not automatically become training data or quantitative HR ground truth. Verify terms and attribution before any use. Distinguish Google Earth Pro from Earth Engine scientific collections.

“Geotech satellite” is unresolved: request the exact satellite/provider/product only when necessary, without blocking independent work. Evaluate GeoEye only if that is the intended source; do not assume it is freely reusable or that its processing-level names mean the same as Sentinel-2.

Use fresh geographically independent regions under a committed protocol. Preserve the old frozen test. Split geography before patches; prevent temporal/geographic leakage and test-fitted radiometric matching.

5. FAIR EVALUATION
Compare the seven variants with bicubic, gain/offset attenuation controls, a matched CNN and an appropriate official pretrained baseline.

Use consistent data, degradation, masks, metric ranges, borders and training budgets. Report parameter/compute differences, multiple seeds, scene-level paired confidence intervals and effect sizes. Patches are not independent scenes.

Include valid PSNR, SSIM, RMSE, MAE, SAM, per-band errors, edge/gradient fidelity and radial-spectrum analysis. Use LPIPS only for perceptual evaluation. Add downstream tasks only when suitable labels exist.

Without aligned HR, reference metrics must be unavailable. Model disagreement, sharpness and downsampling consistency do not prove accuracy. Preserve null findings and failure cases.

6. PROFESSIONAL WEBSITE
Redesign the presentation substantially while retaining useful existing functionality.

Provide clear pages/sections for:
Home; Model Explorer; Enhance & Compare; Results; Forty Mini-projects; Satellite Data & Validation; Research & Report.

Use responsive navigation, accessible contrast, consistent typography, light/dark support, polished model cards and clear loading/error states.

Show content directly on-site:
- Input, bicubic and model-output galleries.
- Before/after slider and synchronized crop/zoom.
- Shared display contrast and reference-aware heatmaps.
- Training curves, ablations, metric charts and comparison tables.
- Architecture diagrams and mathematical explanations.
- Data provenance, limitations, failure cases and report preview.

No notebook viewer, code cells, terminal commands or raw JSON as the main visitor experience. Keep optional technical downloads in a secondary reproducibility area.

Each result must show model/checkpoint, scale method, dimensions, source, runtime and evidence status. Explain “What this shows / What it cannot show.” Provide metric formulas, symbol definitions, units and computed interpretations.

Use genuine recorded outputs for the research showcase; label them precomputed. Live inference uses a simple upload/select/enhance workflow. Never run training or expensive inference on navigation.

7. ALL 40 MINI-PROJECTS
Complete each project’s actual learning and experimental requirements. Present objective, concept, mathematics, inputs, outputs and interpretation as native website content.

Show all relevant existing figures and results, not just a notebook link or the first image. Add appropriate interactive controls for bounded demonstrations.

Execute each canonical project independently in a clean process/kernel. Check meaningful outputs, not only exit codes. Record source hashes, environment, commands, seeds, errors and output validation.

Create a 40-project acceptance report. Missing artifacts remain visibly unavailable; do not fabricate outputs or treat scaffolding as completion.

8. PAPER, DEPLOYMENT AND ACCEPTANCE
Use my uploaded IEEE conference format where available; complete the paper, references, generated tables/figures, claim→evidence matrix, model/data cards, slides and examiner demo.

Test model/scale combinations, scientific metadata handling, metric edge cases, all website sections, mobile/desktop layouts, downloads and cache isolation.

Investigate the current public Streamlit access problem. Deploy through authorized existing services and verify signed-out access, actual inference and cold starts. Verify persistent storage separately; do not claim ephemeral hosting is durable.

Install required ordinary dependencies and proceed autonomously with reversible work. Before paid resources or substantial downloads/training, provide a concrete size/time/cost proposal. Never request secrets in chat.

Publish reviewable changes and CI evidence. Do not claim “100% perfect,” seven trained models, or public readiness without verification.

Before stopping, update docs/next_session.md with exact commit/PR state, completed work, remaining blockers, evidence paths and next commands. Return the verified website URL and a concise acceptance table.