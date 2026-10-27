Complete the existing project, preserving previous work:
Repository: https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR
Streamlit: https://mini-project-ml-lab-rssr.streamlit.app/

Inspect AGENTS.md, current git/PR/CI state, docs/next_session.md, submission_readiness.md, completion_brief.md, research_upgrade_brief.md, src/rssr/models.py and app/. Verify actual implementation before trusting completion labels. Continue from the latest checkpoint; do not restart.

Implement the following, retaining all previous research-integrity requirements.

1. SEVEN DISTINCT MODEL CONFIGURATIONS

Provide separately selectable, trainable, evaluable and runnable models:
M1 CNN
M2 Attention
M3 Transformer
M4 CNN + Attention
M5 CNN + Transformer
M6 Attention + Transformer
M7 CNN + Attention + Transformer

Define C, A and T precisely before implementation:
- C = explicit convolutional feature-extraction backbone.
- A = additional channel/spatial feature-attention module.
- T = Transformer blocks, including their inherent self-attention.
“Transformer without A” still contains self-attention.

Document necessary shared embedding/reconstruction layers. Do not claim an architecture is convolution-free if it uses convolution. Current kind="attention" already has a CNN backbone: preserve checkpoint compatibility and historical names while mapping them accurately.

Use a versioned architecture registry with unique IDs, configs, module diagrams, parameter counts, checkpoint hashes, training provenance, supported bands and scales. Test module presence/absence so seven names cannot resolve silently to the same architecture. Shared infrastructure is encouraged; duplicate labels or unrelated checkpoint reuse are not.

Train missing variants within available resources. A random-weight forward pass is only a smoke test, never a trained working model. Distinguish implemented, trained, evaluated and deployed status.

2. MULTISCALE EXECUTION

Support inputs up to 1024 pixels per side, including rectangular and odd dimensions, at 2×, 4×, 8×, 16×, 32× and 64× wherever execution resources allow.

For input W×H and scale s, output must be exactly sW×sH. Preserve aspect ratio. Clearly distinguish input limit, output dimensions, linear magnification and s² pixel-count increase.

Reuse existing scaling code. Prefer trained native-scale models where justified; otherwise use explicitly labelled cascaded learned SR. Record every stage/checkpoint and distinguish interpolation from learned reconstruction. Never present cascading as native 64× training or increased dimensions as recovered detail.

Create a seven-model × six-scale capability matrix showing:
native/cascaded method, checkpoint, actual output shape, execution status, timing, memory measurement method and evidence path.

Test all combinations on bounded fixtures and representative images. Validate feasible large exports separately; large-output planning is not execution evidence. Use tiled/disk-backed inference, overlap/seam tests, bounded previews, atomic exports and memory/disk guards. Account for tile-dependent global attention.

Keep all scales visible in Streamlit. If a job exceeds host limits, explain the limit and provide an executable local/Colab bundle with exact configuration. Never silently lower scale, crop or downsize. Preserve scientific values, masks, bands, CRS and extent in GeoTIFF exports.

3. VERIFY ALL 40 MINI-PROJECTS INDIVIDUALLY

Use the authoritative syllabus and project-folder mapping. Complete MP36–40 and the MP20 evidence gap according to current verified status.

For EACH MP01–MP40:
- Check its objective, prerequisites, concept, mathematics, worked example, code and expected outputs.
- Execute its canonical notebook/script in a fresh process/kernel without relying on another project's hidden state.
- Validate meaningful outputs, not only exit code: expected dimensions, finite values, relevant numerical properties, readable figures and nonempty artifacts.
- Record source hash, command, environment, seed, duration, exit status, output hashes and limitations.
- Fix reproducible errors; do not suppress exceptions or mark skips as passes.
- Separate teaching demonstrations, real-data experiments and research evidence.

Create a machine-readable execution report plus a readable 40-row status table. Reuse existing runners where suitable. Run the final candidate batch after changes; avoid repeatedly rerunning unchanged expensive experiments or frozen held-out evaluations.

4. INTERACTIVE STREAMLIT MINI-PROJECT LAB

Extend the existing seven-section app, not a replacement website.

Give every mini-project a dedicated interactive view with:
objective → concept → mathematics → controls → Run → outputs → interpretation → downloads.

Provide relevant sliders/selectors and visualizations: image/band histograms, convolution kernels, degradation previews, training curves, attention maps, spectral plots, metric comparisons or other outputs appropriate to that project. Show all relevant generated outputs, not only the first PNG.

Use explicit Run buttons, input validation, progress, retryable errors, stable widget keys and source/configuration-aware session caching. Navigation must not trigger training or expensive inference.

Allow bounded demonstrations live. Show costly experiments as clearly labelled precomputed results with provenance and reproducible commands. Never pretend recorded results were computed live.

Add searchable project navigation, stage filters, completion status and downloadable code/notebooks/results.

5. MODEL COMPARISON AND RESULT EXPLANATIONS

Provide “Run selected model” and “Compare selected models” for all seven valid configurations. Run comparisons sequentially when needed to respect memory limits.

Show input, bicubic, selected model outputs and genuine aligned HR reference when available. Include before/after sliders, synchronized crops/zoom, shared contrast, downloadable full outputs and a filterable model/scale results table.

For every result explain:
- What operation happened and which modules contributed.
- Input/output dimensions, bands, dtype, range, scale method, checkpoint, runtime and available memory measurements.
- What the result shows and what it cannot establish.

For applicable metrics provide formula, symbol meanings, units, direction, actual computed value and limitations:
MSE, RMSE, MAE, PSNR, SSIM, SAM, per-band errors and edge/gradient fidelity.
Include ΔPSNR and percentage RMSE improvement with valid denominator guards.

Use identical alignment, masks, borders and data ranges for comparisons. Do not compute HR-reference metrics without a real aligned reference. Separate error maps from model-disagreement maps. Attention visualizations are not causal explanations or confidence estimates.

Keep Scientific and Perceptual modes visibly separate. LPIPS and optional enhancement belong in Perceptual mode. Generated images are not ground truth; avoid claims of guaranteed ChatGPT-like reconstruction.

6. DEFENSIBLE RESEARCH

Preserve existing frozen NAIP results. Do not tune on the old test or rerun it simply for UI work.

Maintain preregistered research questions and protocol v2 before new experiments:
- Do C/A/T combinations help under geographic holdout and comparable budgets?
- Does synthetic training transfer to real cross-sensor pairs?
- Do pixel metrics agree with edge/spectral and justified downstream results?

Use geographic separation, matched training/data budgets, disclosed parameter/compute differences, multiple seeds, scene-level paired bootstrap intervals and effect sizes. Patches and seeds are not independent geographic samples.

Include bicubic, train/validation-fitted gain/offset attenuation control, matched CNN and official SwinIR with pretraining caveats. Keep negative results. Do not assume M7 must win.

Finish official-source WorldStrat/SEN2VENµS feasibility, licensing, alignment/calibration and bounded acquisition. Use available authorized resources; ask only when a concrete cost, major download/compute commitment or unavailable account requires user action. Record infeasible work as blocked while continuing independently.

Add failure cases, edge/gradient and radial-spectrum analysis, valid spectral metrics, and downstream evaluation only when suitable data exists. No unsupported novelty/SOTA claims.

7. FOLLOW MY UPLOADED IEEE FORMAT

Uploaded files in ChatGPT Library /mini project:
conference_041818.tex
conference_041818.pdf
IEEEtran.cls
IEEEtran_HOWTO.pdf

Use these supplied files when accessible; do not assume ChatGPT Library paths exist in the Codex filesystem. If unavailable there, continue using the existing paper and explicitly record the missing template transfer.

Use \documentclass[conference]{IEEEtran}. Preserve prescribed columns, margins, fonts and spacing. Remove every sample paragraph, fake author, sample reference and placeholder figure. Do not invent author/contact/funding details or venue page limits.

Finish paper/main.tex and references.bib with:
abstract, introduction, related work, data, seven-model definitions, scale methodology, protocol, measured results, ablations, efficiency, failures, limitations, ethics and reproducibility.

Generate tables/figures from one versioned evidence source. Maintain a claim→evidence matrix. Compile the PDF and visually inspect every page for clipped equations, unreadable figures, overflow, missing citations and template leftovers.

Complete slides, examiner guide, three-minute demo script, model/data cards, CITATION.cff, license audit, pinned environment and one-command reproduction package.

8. PRESENTATION, PERSISTENCE AND DEPLOYMENT

Use a consistent accessible theme, colorblind-safe charts, clear typography, light/dark support and responsive phone layout. Add measured/precomputed/unavailable status chips and persistent Scientific/Perceptual mode indicators.

Verify existing database migrations and experiment provenance. Distinguish live acquisition, cached data and offline fixtures. Claim durable hosted storage only after configured storage survives restart.

Investigate current Streamlit signed-out no-access/not-found response. Verify deployment branch, app/streamlit_app.py, dependencies, assets/checkpoints, resource limits and visibility using authorized access. Do not claim the public app works from local tests alone.

Test every section with AppTest and actual browser smoke tests at phone/desktop sizes. Verify model selection, mini-project controls, output changes, downloads, invalid inputs, cache isolation, resource limits and cold starts. Save evidence under reports/.

9. FINISH AND RESUME

Work autonomously on authorized changes, install required ordinary dependencies, fix failures and publish a reviewable commit/PR. Respect existing merge rules and do not submit to an institution or journal.

Acceptance requires:
- Seven honestly defined model configurations with individually recorded status.
- A verified model/scale matrix.
- All 40 final mini-projects independently checked with meaningful outputs.
- Interactive project views connected to real computations or labelled recorded artifacts.
- An evidence-backed IEEE package.
- CI on the actual final candidate and public deployment evidence, or explicit owner blockers.

Do not promise universal error-free behavior or mark blocked/unverified work complete.

Keep updates concise. Before a context/token limit, save docs/next_session.md with exact git/PR state, completed tasks, evidence paths, commands, failures and next actions. Finish with a short verified/not-verified/blocked table and the working public URL only if tested.