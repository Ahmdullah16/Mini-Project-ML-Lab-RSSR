Continue and finish the existing project:
Repo: https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR
App: https://mini-project-ml-lab-rssr.streamlit.app/

Work from the current checkout and existing progress. Inspect AGENTS.md, git/PR/CI state, docs/next_session.md, docs/submission_readiness.md, docs/completion_brief.md and app/. Reconcile stale documentation against code and evidence. Preserve completed work; do not restart or rebuild the app.

PRIORITY: defensible research evidence → submission package → interface and deployment. Research-grade means defensible, including negative findings, not flawless or necessarily novel.

RULES
- Implement and verify, not just propose. Use targeted reads/tests and concise updates.
- Never fabricate results, citations, datasets, model performance or completion.
- Distinguish implemented, locally tested, publicly verified and blocked.
- Use existing authorized resources. Ask before large downloads, long training or paid services, specifying size, compute, cost and purpose. Never request tokens in chat.
- Preserve frozen data/protocols and existing user work.
- “Done” requires relevant, verified evidence, not merely a file.
- Keep owner-dependent access/storage blockers explicit while completing independent tasks.

RESEARCH-GRADE UPGRADE
These requirements override earlier instructions where they conflict.

A. Define the claim before new experiments
Commit docs/research_questions.md with:
1. Do attention/Transformer modules help under geographic holdout with matched parameter and compute budgets?
2. Does synthetic-degradation training transfer to real cross-sensor pairs?
3. Do PSNR/SSIM agree with edge, spectral and justified downstream measures?
Frame this as a rigorous evaluation study unless evidence supports stronger claims. Do not claim novelty or SOTA without evidence.

B. Strengthen evidence
- Verify the current limitations: synthetic NAIP degradation, one frozen held-out scene, no completed real paired cross-sensor evaluation, and attenuation potentially explaining pilot gains.
- Finish MP36 feasibility: verify WorldStrat and SEN2VENµS using official sources, including license, download size, bands, dates, alignment, registration and sensor differences. Seek approval for large resource use. If infeasible, document the exact blocker and continue.
- Commit protocol v2 before inspecting new test imagery/results. Add fresh geographically independent regions and appropriate geographic k-fold validation. Prevent scene/patch overlap and tuning leakage. Keep the old frozen test untouched and report it separately.
- Use multiple seeds, scene-level paired bootstrap confidence intervals and effect sizes. Scenes—not patches or seeds—are independent sampling units. Disclose small-sample limitations and retain null results.
- Include bicubic, gain/offset attenuation fitted on training/validation only, parameter-matched CNN and official SwinIR controls. Reuse verified existing runs when compatible. Record checkpoint origin, pretraining and domain mismatch; disclose imperfect compute matching.
- Add edge/gradient fidelity, radial power spectrum, SAM and per-band RMSE where physically valid. Add a failure-case atlas. Use downstream evaluation only with suitable labels/data; LPIPS belongs only in Perceptual mode.
- Complete remaining MP36–40 work and the MP20 independent SRCNN evidence gap after checking their actual status. Do not repeat completed experiments unnecessarily.

C. Super-resolution and scale correctness
- Accept supported images with each input dimension ≤1024, including odd and rectangular sizes.
- Offer 2×, 4×, 8×, 16×, 32× and 64× linear scaling. For input W×H and scale s, output must be exactly sW×sH. Explain pixel count grows by s².
- Treat 1024×1024 as the input limit, not a universal output limit: 16×16 at 64× becomes 1024×1024; 1024×1024 at 64× becomes 65536×65536.
- Verify existing scale implementation and its publication/CI status before modifying it. Label native learned SR, cascaded learned SR and interpolation accurately.
- Use tiled inference with overlap, seam checks, disk-backed outputs, bounded previews and memory/disk preflight. Preserve scientific float values, masks, bands, georeferencing and spatial extent.
- Keep every scale visible. When hosting limits prevent a job, explain why and provide an executable local/Colab bundle. Never silently resize, crop or substitute a smaller scale.
- Do not describe a planned huge output as executed. Validate feasible outputs and separately test large-job planning and resource guards.
- Investigate normalization, dtype, clipping, channel order, checkpoint compatibility, degradation mismatch and display settings before claiming quality improvements.
- Larger dimensions do not prove recovered detail. Generative enhancement may invent detail. ChatGPT-generated images are not ground truth.

D. Results and mathematical explanations
- Keep Scientific and Perceptual modes visibly separate; preserve raw scientific outputs and label optional sharpening/generative enhancement.
- Compare compatible existing models and baselines using aligned crops, identical display contrast and synchronized zoom.
- Show input, bicubic, model outputs and ground truth only when genuinely available.
- Each result must identify model/checkpoint, source, mode, native/requested/effective scale, dimensions, dtype, bands, range, timing and provenance; include physical pixel spacing only when supported by metadata.
- Explain every displayed metric conceptually and mathematically: formula, symbols, units, valid range, preferred direction, computed value and limitations.
- Include appropriate MSE, RMSE, MAE, PSNR, SSIM, SAM and per-band errors. Document PSNR data range, SSIM settings, masks, alignment and SAM zero-vector handling.
- Explain ΔPSNR as subtraction and percentage RMSE improvement with a zero-denominator guard. Do not average incompatible units or bands.
- Without aligned HR reference, show reference metrics as unavailable. Label model/bicubic differences as disagreement, not accuracy. Sharpness and consistency are not proof of fidelity.
- Use error maps only with valid reference data; give maps colorbars and units. Explain attention maps without presenting them as causal explanations or calibrated confidence.
- Under each result include “What this shows” and “What it can’t show.”

E. Paper and submission package
- Build an IEEE-style LaTeX evaluation paper with references.bib: abstract, related work, data, method, preregistered protocol, results, limitations, ethics and reproducibility checklist.
- Maintain a claim→evidence matrix linking every substantive claim to report/figure IDs.
- Generate paper/app tables and figures by script from one versioned evidence source. No hand-entered conflicting numbers.
- Finish model/data cards, CITATION.cff, license audit, pinned dependencies or Docker, and a one-command reproduction workflow. Include expected artifact hashes where deterministic and documented numerical tolerances otherwise.
- Complete notebooks, research explanations, slides, examiner guide and a three-minute demo script. Clearly separate measured work, proposals and blocked experiments.
- Validate installation, downloads, paths, documentation links and clean-environment reproduction.

F. Improve Streamlit by extending app/
- Preserve the seven sections: Overview, Live SR, Results, Method/Data, Mini Projects, Satellite Explorer and Downloads/Reproducibility.
- Use .streamlit/config.toml and small shared components for consistent typography, colorblind-safe colors, AA contrast, light/dark support and phone-friendly single-column layouts.
- Add persistent Scientific/Perceptual badges and measured/precomputed/unavailable result chips.
- Use explicit Run buttons, staged progress, retryable errors, useful empty states, lazy loading and memory guards.
- Key results/cache by input hash, checkpoint, preprocessing, scale and settings; prevent stale or mismatched comparisons.
- Add before/after sliders, synchronized zoom, shared contrast controls, colorbar heatmaps and mathematical metric expanders.
- Present the research questions, architecture, data provenance, experiments, ablations, uncertainty, failures and limitations clearly. Make existing research discoverable, not just visually decorative.
- Test all seven sections with AppTest; run browser smoke tests at phone and desktop sizes and save screenshots under reports/.

G. Database and public deployment
- Verify existing database/storage adapters, migrations, restart durability and provenance tracking. Clearly distinguish live provider data, cached data and synthetic/offline fixtures.
- Reuse authorized persistent services; do not claim local ephemeral storage is durable or offline adapter tests prove production operation.
- Investigate Streamlit’s signed-out access error: deployment branch, entrypoint, dependencies, checkpoint loading, resource limits and visibility.
- Deploy using existing authorized access if available. Otherwise finish all local work and document the exact owner action needed without claiming public verification.
- Verify the final URL signed out, including inference, result downloads, errors and restart behavior. Keep oversized jobs on the documented portable execution path.

FINISH
Run relevant tests and CI, publish authorized changes, and update readiness/next-session documents with evidence links, commit/PR status, reproducible commands, remaining blockers and the verified live URL if available. Do not mark the project 100% complete while research, execution or public-access requirements remain unverified.

Before the context/token limit, save a concise checkpoint in docs/next_session.md: completed work, exact next actions, changed files, experiment IDs, commands, failures, blockers and git/PR state. Make resuming possible without rereading the entire repository.