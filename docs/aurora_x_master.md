# MASTER CODEX PROMPT v3.0 — RSSR AURORA X
## Build a premium, evidence-first Earth Intelligence & AI Super-Resolution Studio

> **FOR CODEX — IMPLEMENT IN THE EXISTING REPOSITORY, NOT AS A DESIGN-ONLY MOCKUP.** This document supersedes the supplied RSSR AURORA v2 design brief while preserving its scientific integrity, repository constraints, managed authentication safeguards, and Streamlit-first deployment strategy.

---

## 0. MISSION AND OPERATING CONTRACT

Act as a coordinated team of **principal Python/Streamlit engineer, geospatial ML researcher, senior product designer, scientific visualisation engineer, accessibility specialist, security engineer, and QA lead**.

**Mission:** Evolve the working Remote Sensing Super-Resolution (RSSR) application into **RSSR AURORA X**: a modern, fast, premium **Earth Intelligence Laboratory** where anyone can explore satellite-image super-resolution, compare the seven real model families, understand their mathematics, inspect reproducible evidence, and run supported experiments—all inside the existing Streamlit application.

The experience should feel like a thoughtful combination of a modern AI research product, a geospatial mission-control console, and an interactive university laboratory—not like default Streamlit widgets hidden under decorative CSS. Beautiful presentation must increase clarity and trust, never replace them.

**Priority, in descending order:**
1. Scientific correctness and provenance
2. Security, privacy, and access control
3. Preserve working inference, learning modules, satellite integrations, and tests
4. Accessibility and graceful degradation
5. Correct, responsive interactions and performance
6. Design distinction, motion, and delight

**Execution rules:**
- **Upgrade; do not rewrite.** Reuse working architecture, models, checkpoints, processing and tests. Do not retrain, alter canonical experiments, or modify scientific results for visual appeal.
- Inspect `main` HEAD, all open PRs/branches, CI, deployment and the docs before coding. Protect any active MP27–MP31 Transformer lesson work; coordinate or isolate changes rather than overwrite it.
- Work in a fresh `feat/rssr-aurora-x` branch (or conflict-free equivalent). Make small reviewable commits and open a PR. **Do not force-push or merge into `main`.**
- **Do not stop at a static design proposal.** Implement as much verified functionality as possible in the current environment, with tested fallbacks. For anything requiring credentials, provider setup or browser access, implement a safe config-gated skeleton and document precisely what remains.
- Record evidence at every milestone and continue unless blocked. Do not invent successful tests, deployed features, benchmarks, account flows, paper publications, or live data.
- Use one of: `VERIFIED`, `NOT VERIFIED`, `NEEDS OWNER ACTION`, or `BLOCKED` for every acceptance claim.
- If this prompt conflicts with actual repository interfaces or hosting capabilities, **inspect and adapt**, preserving the intent. Document decisions and avoid fabricated API support.

### Project identity (do not alter)
- **GitHub:** https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR
- **Live app:** https://mini-project-ml-lab-rssr.streamlit.app/
- **Research:** Attention- and Transformer-Based Deep Learning for Super-Resolution of Remote Sensing Images
- **Researcher:** Ahmdullah — M.Tech, Signal Processing and Machine Learning, IIT Guwahati
- **Email:** exammotivation0816@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/exammotivation/
- **Seven family labels:** CNN; Attention; Transformer; CNN + Attention; CNN + Transformer; Attention + Transformer; CNN + Attention + Transformer.
- **Configured scale candidates:** ×2, ×4, ×8, ×16, ×32, ×64. Display actual support separately for each checkpoint and runtime. Never imply all combinations are scientifically validated.
- **Learning scope:** Approximately 40 mini-projects across nine stages; **derive the displayed counts from the live lesson registry**.
- **Historical checkpoint only:** `459f5bd` on 2026-11-01 reportedly passed 42/42 bounded small-scale cases. Re-check HEAD; do not use this as contemporary performance evidence.
- Show IIT Guwahati as academic background only, **never as institutional endorsement**.

---

## 1. THE PRODUCT IDEA: THREE INTELLIGENT MODES, ONE CONSISTENT PLATFORM

Create three selectable interface **lenses**, not three duplicate apps:

1. **Explore** — beginner-friendly project story, clear model explanations, guided examples, honest demo results.
2. **Build & Compare** — hands-on experiment workbench, runtime constraints, seven-family comparison, image tools, export.
3. **Research** — dataset protocols, provenance, reproducibility, metrics, ablations, architecture equations, experiment history.

The navigation and shared state remain consistent. A visitor may switch lenses without losing a safe in-session selection. Provide plain-language and scientific interpretations beside each other where helpful. Do not hide required safety, resource or scientific warnings in beginner mode.

**Distinctive product capabilities** (implemented only against genuine data):
- **Evidence Drawer:** a one-click, human-readable provenance panel for every displayed result, metric and experiment, containing its source, model/checkpoint, processing, protocol, time and known limitations.
- **Comparison Wall:** selectable real outputs from up to seven model families, aligned to the same input/scale/protocol; graceful omission and clear explanation for unavailable results.
- **Experiment Passport:** a downloadable machine-readable manifest accompanying every completed run, reproducible to the degree supported by available artifacts.
- **Trust Signals:** clearly distinguish `configured`, `implemented`, `executed`, and `scientifically validated`; never equate a UI option with a completed experiment.
- **Guided Science Layer:** expandable equation explanations, tooltips, “what changed?” explanations, and visual interpretations derived only from computed results.
- **Command Search / Quick Actions:** keyboard-accessible navigation to models, learning lessons, settings and samples. In native Streamlit if possible; do not inject unsafe global JavaScript.
- **Research Workspace:** optional saved collections of configurations, runs, notes and lesson progress, only when secure user persistence is configured. Otherwise, state that the workspace is session-only and provide export.

Do not add an LLM chatbot merely for novelty. If a **Research Copilot** is introduced, start with a deterministic, cited explainer over real metadata, equations and computed outputs. Any future LLM integration must be optional, explicitly labelled, provider-configured, privacy reviewed, source-grounded and prohibited from inventing scientific conclusions.

---

## 2. PHASE ZERO: REAL REPOSITORY + STREAMLIT AUDIT

Before implementation:
1. Inspect `AGENTS.md`, README, `completion_brief.md`, `next_session.md`, `submission_readiness.md`, docs, test folders, architecture/model registry, satellite interfaces, lesson dispatch, and open PRs when present.
2. Identify current Streamlit version, Python/runtime and pinned dependencies; map app entry point and navigation.
3. Run relevant baseline tests and record pass/fail/skipped with commands and environment. Do not treat skipped tests as passes.
4. Verify actual inference entry points, checkpoint availability, model sizes, scales, resource limits, current comparisons, dataset licensing, provenance and lesson completeness.
5. Evaluate existing components for reuse; prioritize additive changes and stable compatibility adapters.
6. Create `docs/aurora_x_audit.md` with a gap matrix: Existing | Reusable | Must build | Blocked | Risk | Evidence.

### Streamlit capability lab
Build a **dev-only** capability page and `docs/streamlit_capability_matrix.md`. On pinned Streamlit and, where possible, Community Cloud, verify:
- `st.navigation`, `st.Page`, `st.page_link`, supported top/grouped navigation, page deep links;
- `st.container(key=...)` styling hooks, `st.html`, CSS injection, style scoping, inline SVG;
- `st.fragment`, `st.dialog`, `st.status`, `st.toast`, `st.popover`, `st.pills` or other current primitives;
- `st.context` theme signals if supported, native light/dark theming, `.streamlit/config.toml`;
- `st.components` supported iframe and newer component interfaces; measured event exchange/height/scrolling and safe data handoff;
- `st.query_params`, file upload/download, map/AOI integration, cache behavior, static assets, memory constraints;
- mobile layout and keyboard navigation, content-security constraints and reduced-motion behavior.

Assume **JavaScript cannot control the parent Streamlit document** unless proven supported by an official interface. Do not rely on fragile DOM hacks, copied hashed Streamlit selectors, or a CDN that may be unavailable. Every advanced viewer must have a simpler functional fallback.

---

## 3. FLAGSHIP VISUAL DIRECTION — “THE LUMINOUS OBSERVATORY”

**Visual personality:** precise, cinematic, editorial, high-tech, spacious and trustworthy. Inspired by Earth-observation telemetry, scientific imaging and high-quality software—not generic rainbow gradients or maximalist glassmorphism.

### Design language
- **Deep-space canvas:** navy-black `#070B16` / `#0B1220` with layered surfaces `#172339` / `#1E2D4A` and readable white `#F8FAFC`; refined light mode with `#F5F8FF`, white cards and slate text.
- **Aurora accent:** cyan `#22D3EE` → indigo `#6366F1` → fuchsia `#E879F9`; use once as a visual anchor in a viewport, not on every element.
- **Semantic accents:** emerald for geospatial/success, amber for attention/warnings, rose for failures; matching icons and text always accompany colour.
- **Family identity:** CNN cyan/grid; Attention amber/target; Transformer fuchsia/tokens; hybrids use paired outlines and combined glyphs. Distinguish series via labels/marker shapes/line styles for colour-vision accessibility.
- **Typography:** elegant display sans (Manrope or Inter), readable body, JetBrains Mono/system mono for scientific data. Only self-host fonts with verified licences or fallback to system fonts. No forced network font requests.
- **Geometry:** 8px spacing scale, 14–20px card radius, 1px hairline borders, subtle shadows, ample whitespace, max content width about 1320px. Responsive 12-column design logic that collapses to one column on phones.
- **Navigation:** persistent, light-touch top utility bar, clear active state, concise primary navigation, accessibility skip link, never hide navigation behind animations.
- **Data first:** every numeric datum is legible, selectable and copyable; aligned units, reasonable precision, skeletons only for actual loads, explicit empty states.

### Interaction quality bar
Build a reusable, testable `rssr_ui/` package with a single token source generating CSS and chart themes. Components include `AppShell`, `Hero`, `MetricTile`, `ModelFamilyCard`, `StatusPill`, `CompareViewer`, `EvidenceDrawer`, `ExperimentPassport`, `WorkflowRail`, `EquationExplainer`, `DatasetCard`, `LearningRoadmap`, `EmptyState`, `ErrorBoundary`, `ResearcherCard`, and `Footer`. Adapt names to the repository's conventions; keep current app modules intact.

For each component specify visible hover/focus/active/disabled/loading/empty/error states, keyboard operation, dark/light variants, responsive behavior and test cases. Never fake active controls.

### Signature motion (tasteful and optional)
- Hero aurora drifts very slowly with static fallback; one orbit/swath **illustration**, clearly labelled as such.
- Headline and project workflow enter once with gentle stagger; cards use 2–4px hover elevation, never large disorienting movement.
- Data-flow diagrams use SVG arrows to communicate stages. Long animations have pause controls; nothing essential is encoded through motion.
- Comparator handle visibly invites user drag but is also usable by keyboard and touch.
- Running state visualises **real pipeline progress** only; no animated pretend model execution.
- Disable nonessential effects with both `prefers-reduced-motion` and an in-app toggle; no flashing, particle storm, autoplay video or ambient sound.
- CSS Tier A: lightweight CSS transforms/opacity. Tier B: annotated SVG diagrams. Tier C: isolated, sandboxed JS image interaction only if capability probe validates it; provide Streamlit fallback.
- Content remains visible with CSS disabled, JavaScript disabled or components blocked. Do not obscure scientific pixels with glow, blur, filters or opacity overlays.

**Accessibility:** WCAG 2.2 AA target; minimum contrast 4.5:1 normal text, 3:1 controls/large text; keyboard focus clearly visible; chart text alternatives; touch targets around 44px; user-controlled animations; descriptive headings; localized formatting of dates/units where appropriate. Validate in browser, not just from design intentions.

---

## 4. INFORMATION ARCHITECTURE — SIX PRIMARY DESTINATIONS

1. **Overview** — story, trustworthy highlights, featured real comparison, model access and science overview.
2. **AI Studio** — upload/choose scene, configure real inference, run, inspect, compare and export.
3. **Satellite Explorer** — geospatial search, AOI, scene provenance, imagery and supported SR action.
4. **Model Observatory** (can retain “Model Lab” route for compatibility) — models, equations, comparisons, efficiency and experiments.
5. **Learning Lab** — 40-mini-project journey, interactive math and native lessons.
6. **Research & Documentation** — methods, datasets, provenance, reproducibility, progress, publications only if verified, project identity.

**Optional Account / Workspace** is a profile menu, not a mandatory seventh public page. Public pages must work without signing up. Preserve existing bookmarks/routes or provide redirects where supported.

Global utility bar: brand mark + compact name, section navigation, `Explore / Build / Research` lens, search/quick actions if supported, light/dark selection, reduce-motion setting, **Open AI Studio** CTA, and optional account control. Keep mobile menu navigable by keyboard. Use actual capabilities rather than leaving nonfunctional buttons.

---

## 5. OVERVIEW — A 30-SECOND STORY WITH RESEARCH DEPTH

Make a beautifully composed, fast first render. Design ten sections with progressive disclosure:

**01 — Hero / Mission Control.**
- Eyebrow: `EARTH INTELLIGENCE · IMAGE SUPER-RESOLUTION`.
- Headline: **“See More. Compare Better. Understand Every Pixel.”**
- Concise subhead: `Explore CNN, Attention and Transformer approaches to satellite-image super-resolution through transparent, interactive experiments.`
- CTAs: **Launch AI Studio** and **Explore Seven Architectures**.
- Feature a large **verified real image pair** with a draggable split and visible model/scale/dataset badge, or an elegant honest empty state and clearly labelled diagram if no verified pair exists.
- A small researcher identification line; do not dominate hero with a giant biography.
- An optional side-rail reveals source, resolution, scale and scientific limitations without covering imagery.

**02 — Live Registry Highlights.** Four compact cards sourced from a typed facts ledger: implemented model families, supported scale options, native lessons, verified experiment coverage. Show the difference between configured, executed and validated. Never hard-code progress or count-up target numbers.

**03 — Why SR Matters.** Simple side-by-side conceptual pixel-grid illustration explaining LR versus reconstruction, and a prominent one-sentence warning that inferred details are estimates, not confirmed real-world facts.

**04 — Explainable Pipeline.** Input → selection/acquisition → preprocessing → model/checkpoint → inference → output → metric eligibility → visual analysis/export. Each stage links to its actual page; animated connector only for explanation.

**05 — Model Constellation.** Seven distinctive cards with family glyph, actual checkpoint status, short principle, trade-offs, supported real operations and details. A drill-down modal/section shows a real architecture summary, mathematical building blocks, resources and evidence.

**06 — The Comparison Theatre.** Verified same-scene gallery and accessible compare viewer with bicubic baseline and model outputs only when available; support side-by-side and synchronized zoom/pan. Ground truth is included only if correctly aligned and licensed.

**07 — Research Snapshot.** Real PSNR/SSIM/MSE and edge/spectral diagnostics only when applicable; show the dataset, protocol and reference availability. Provide plot + readable table. Missing experiment data gets a meaningful empty state, not sample numbers.

**08 — Learning Journey.** Nine-stage roadmap (registry-derived) into native lessons. Display actual progress, prerequisites and one-click continue where session/account progress is supported.

**09 — Methods and Evidence.** Small editorial blocks for data, evaluation method, reproducibility, experiments and known limitations. Strong links into full Research page and GitHub.

**10 — Researcher / Footer.** Compact professional profile, contact, LinkedIn, repository, licence/attribution and deployment/version details from real metadata.

**Visual acceptance:** looks composed and compelling at 1920, 1440, 1024, 768 and 390px; no clipped hero, unreadable figure, giant empty whitespace or horizontal page overflow. A first-time visitor should find an action and understand the core distinction between enhancing an image and verifying real details within 30 seconds.

---

## 6. AI STUDIO — REAL EXPERIMENT WORKBENCH

Create an interactive journey with a clear progress rail: **1 Choose Input → 2 Configure → 3 Run → 4 Compare → 5 Export**. Maintain workflow state across reruns without losing uploads or generating duplicate inferences.

### Inputs
- Existing demo/sample sources, PNG/JPEG and supported GeoTIFF uploads, and real satellite acquisitions if configured.
- Show image dimensions, bands, CRS if applicable, size, source and licence/attribution where available.
- Validate file type from content, not extension alone; bound file size/pixel count; safely handle corrupt or malicious files and unusual bands.
- Provide safe cropping/region-of-interest tools if feasible; expose how a crop affects evaluation.

### Model and runtime controls
- Select one of seven family categories and an actual implementation/checkpoint; show availability per checkpoint, not merely per abstract family.
- Real support matrix for scale ×2/4/8/16/32/64, output bounds, device, seed, tiles, memory/runtime estimate **only if calibrated**. Use candid warnings for unsupported combinations and large generated output.
- Guided and Advanced views; reproducibility defaults visible in Advanced view. Allow multi-model compare only if estimated resource cost is within tested bounds. Use queued sequential execution rather than loading seven heavy models together.
- Persist presets only with a secure account backend; otherwise allow downloadable/importable configuration JSON with validation.

### Execution
- Single controlled run request and stable run ID. Show real progress/step state, cancellation only if truly implementable, and error recovery with actionable text.
- Do not leak stack traces, host paths, private coordinates, tokens or user image content into public logs.
- On Streamlit Community Cloud, preserve memory limits and avoid unsupported promises of background jobs, GPU execution, permanent disk or 24/7 uptime.

### Output and scientific interpretation
- Show input, bicubic, each actually computed model output and aligned GT if supplied; maintain native pixel integrity.
- Accessible **Comparison Wall**: split, side-by-side, checkerboard, synchronized zoom/pan, single-step pan/keyboard support, pixel inspector and matched crops. Blink view must be user-triggered and stay below accessibility flash limits.
- Derived views only from true data: absolute error map, SSIM map if correct reference exists, Sobel edges, histograms, frequency spectrum, line profile; always label axes, units and color scales.
- Metrics: MSE, PSNR, SSIM only with verified alignment, preprocessing protocol and valid reference. Show reason when unavailable. No-reference measurements belong in a separate labelled section and must not be named PSNR/SSIM.
- A **Metric Explainer** beside each value: simple meaning, mathematical formula, direction of improvement, protocol and caveats. Never state “more realistic” based solely on a sharp-looking output.
- Export real output image(s), valid aligned crop pairs, configuration, metrics and manifest. Include optional `experiment_passport.json` with provenance, hashes, timestamps and code/dependency version where available.

**Novel capability:** an **Experiment Replay Card** displays an earlier run manifest and detects differences against the current model, scale, protocol or input hash. It should warn about mismatched comparisons and rerun only after explicit user action. A manifest without runnable artifacts is reviewable, not “replay guaranteed.”

---

## 7. SATELLITE EXPLORER — GEOSPATIAL CONTROL CENTER

Design a polished two-pane experience: map and AOI controls on one side, scene metadata/preview/processing on the other; stack correctly on mobile.

- Search a location or enter coordinates, inspect basemaps with proper attribution, and draw/select bounding boxes if a supported map component allows it.
- User-facing AOI coordinates are strictly **`[west, south, east, north]` in WGS84**. Validate ordering, latitude/longitude range, area/edge constraints from the repo, antimeridian and CRS conversions. Do not silently change inputs.
- Dates, cloud threshold, collection, bands and temporal selection follow the actual provider's supported API; show service errors and unavailable scenes honestly.
- Scene Evidence Card: provider, collection, item ID, acquisition time, processing level, source spatial resolution, CRS, band information, licence, cloud metadata and processing/calibration provenance **when returned**.
- Native-versus-super-resolved resolution must be explicitly different. A generated pixel count does not establish new sensor-observed detail.
- If aligned HR ground truth is absent, show **no-reference-only** assessment, not fabricated PSNR or SSIM.
- Include a **Spatial Integrity Check** before export: coordinate fidelity, dimensions, resampling settings, raster metadata preservation and output CRS; if not technically supported, call it out and do not mislabel an RGB export as georeferenced GeoTIFF.
- Optional **Temporal Scene Rail** displays only real retrieved dated scenes; no faux live satellite video or made-up timeline points.
- Run super-resolution using the existing real pipeline; provider secrets remain server-side.

---

## 8. MODEL OBSERVATORY — SEVEN ARCHITECTURES, ONE FAIR SCIENCE STANDARD

An explainable, research-grade architecture and benchmarking experience:

### Architecture Atlas
Each real family has one consistent detail view: What it is → Why it works → Mathematical core → Processing diagram → Model/checkpoint details → Computational cost → Training evidence → Validation/limitations. Use `st.latex` for correct equations, not image-only equations.

Cover at least: convolution and receptive field, channel/spatial attention, self-attention (`QK^T / sqrt(d_k)`), Transformer patch/token interactions, reconstruction and upsampling. Add short “in easy language” interpretations adjacent to derivations.

### Fair Comparison Engine
- Filter by **same dataset, split, source image, scale, preprocessing, alignment, checkpoint protocol, hardware/runtime and metrics**. Display a comparison-eligibility message and disable invalid aggregate rankings.
- Table and interactive charts for quality (PSNR/SSIM where valid), latency, memory, model parameters, inference cost and edge/spectral measures with clear units and actual sources.
- **Pareto Frontier:** quality vs runtime/size only over truly comparable observations. Different hardware or protocols show separate facets or are flagged as non-comparable.
- **Ablation Ledger:** display real experiments comparing components; do not generate invented ablation numbers or causal conclusions.
- **Confidence & Limitations Panel:** state sample coverage, uncertainty and dataset shift whenever evidence supports these; do not invent confidence intervals. High-scale results explicitly warn of hallucinated spatial detail.

**Novel capability:** a **Reproducibility Matrix** shows what is known/missing for each experiment: dataset licence, seed, split, preprocessing, code SHA, checkpoint hash, aligned GT, hardware and evaluation protocol. It is a transparent checklist, not a fabricated quality score.

---

## 9. LEARNING LAB — FROM PIXELS TO TRANSFORMERS

Preserve existing lessons and the active native MP27–MP31 implementation. Prefer on-site interactive learning to serving notebooks as the website experience.

- A visually coherent nine-stage route with clickable nodes and **real** mini-project registry counts, prerequisites and status.
- Each lesson follows: intuitive idea → why it matters to satellite SR → basic mathematics → derived equation → numerical example → interactive playground → model connection → practice questions → verified resources → conclusion.
- Enable simple/advanced explanation toggle without changing the underlying facts.
- Native interactive playground examples: convolution-window explorer; interpolation and aliasing comparison; attention weight matrix with normalization checks; patch/token embedding demonstration; PSNR/SSIM calculator on a tiny ground-truth pair.
- For the numerical exercises, calculate real values from displayed inputs rather than using canned figures presented as outputs.
- Saved progress reflects explicit completion events; anonymous progress is session-only unless the user exports it. Secure persistent progress requires verified auth/database configuration.
- Mobile layouts display formulas and figures without truncation; provide step-through controls and a reduced-motion version of animated math.
- Tests must explicitly protect lesson dispatch and MP27–MP31.

---

## 10. RESEARCH, DOCUMENTATION AND TRANSPARENT STATUS

Build the Research page as a credible project portfolio, not marketing copy.

- Problem, objectives, methodology, datasets, preprocessing, architecture trade-offs, experiment setup, training/evaluation, results with provenance, constraints, limitations and future directions.
- **Research Timeline** sourced from versioned milestones or Git metadata, with no fake publication dates or achievements.
- **Evidence Library:** a searchable index of verified experiment reports, model cards, dataset cards, documentation and citations, with links only to real assets.
- **Experiment Registry:** read-only by default, with exports of reproducibility manifests; avoid public write permissions.
- References and `CITATION.cff` for software if appropriate; do not list an unpublished paper as published.
- Developer documentation, usage/privacy and dataset licences; show the deployed version/commit only if actually discoverable.
- **Researcher:** Ahmdullah, M.Tech SPML at IIT Guwahati. Display `mailto:exammotivation0816@gmail.com` and https://www.linkedin.com/in/exammotivation/ with accessible names. Use an initials avatar, not a fabricated portrait. Explicitly disclaim institutional endorsement.

---

## 11. EVIDENCE-FIRST DATA CONTRACT — NON-NEGOTIABLE

Create a small typed, reusable data layer (dataclass/Pydantic only if already justified by dependencies) with separation among:

- `SiteFact`: `id`, `value`, `unit`, `status`, `source`, `commit`, `generated_at`, `limitations`.
- `ModelRecord`: `family`, `implementation_id`, `checkpoint_id/hash`, `supported_scales`, `input_constraints`, `validated_scenarios`, `license`, `availability`.
- `SceneRecord`: `provider`, `collection`, `item_id`, `acquired_at`, `bands`, `crs`, `pixel_size`, `license`, `attribution`, `processing`.
- `ExperimentRecord`: `run_id`, `input_hash`, `dataset/split`, `model/checkpoint`, `scale`, `seed`, `preprocess`, `evaluation_protocol`, `gt_reference`, `hardware`, `runtime`, `software_commit`, `metrics`, `artifact_ids`, `status`.
- `ShowcaseRecord`: references a real input/output pair, licence, applicable metrics and provenance; it cannot contain unchecked invented imagery.

Use these as **conceptual schemas** and adapt field names to the existing project. Every UI-facing count, badge, image, score and run detail must come from registered evidence, not arbitrary hard-coded cards. Exception: clearly labelled pedagogical hypothetical numbers in lessons. Separate facts from illustrations and examples.

### Scientific guards
- Preserve original scientific image pixel values in view/export; never apply CSS image enhancement to pretend the model improved them.
- Do not compare outputs at different native input crops, upscaling methods, colour spaces, alignments or protocol without prominent qualification.
- Show PSNR/SSIM only with aligned GT and disclosed range, colour space, crop/shave/border handling, scale and interpolation protocol.
- Figure colour maps must be perceptually appropriate with a colour bar/units; avoid jet/rainbow maps for quantitative image errors.
- Each demo/checkpoint has honest labels: `Validated`, `Implemented`, `In progress`, `Planned`, `Unavailable` (use separate computed/evidence fields where necessary).
- `×16/×32/×64` are **highly inferential outputs**: show a persistent context warning about unverified geographic detail.
- Do not claim satellite sources, data licences, training datasets, performance benchmarks, user counts, live processing status or reproducible research without evidence.

---

## 12. OPTIONAL AUTHENTICATION — STREAMLIT-NATIVE EXPERIENCE, MANAGED SECURITY

The entire public scientific site must be accessible without a Streamlit account or a project account, subject to documented guest compute constraints. User sign-up, if enabled, should appear **inside the web app** where supportable.

- Evaluate current managed authentication options (Supabase Auth/Firebase/Auth0/Clerk) for actual Streamlit support, email verification, password reset, SMTP limits, privacy, free-tier constraints, account deletion and cost. Document the selection and trade-offs in `docs/auth_decision.md`.
- Do **not** confuse Streamlit OIDC `st.login()` with a built-in email/password sign-up system. Do not build your own password storage, email reset token or unsafely issued cookies.
- Account flows when configured: Sign Up, Log In, Verify Email, Forgot Password, Logout, Profile, Delete/Export Account, Settings. Support clear loading/errors and avoid account enumeration.
- No special account privilege based merely on matching the public contact email. Use server-side authorization claims and database row-level security.
- User-specific history, image uploads, experiment artifacts, notes and learning progress must be isolated and private; never put them into globally shared caches or public Streamlit session names.
- Community Cloud local storage is ephemeral: use managed persistence or do **not** promise saved accounts/history. Document retention, deletion and user consent.
- Secrets belong in deployment secret management, never code, logs, screenshots, config examples or URLs. Include `secrets.example.toml` with placeholders only.
- If keys/provider/SMTP/database are absent, **hide/disable dependent functionality gracefully** and keep guest browsing and experiment export functional. Create `docs/OWNER_SETUP.md` with exact setup tasks and redirect URIs, not fabricated “working sign-up” evidence.
- Newsletter or email notifications are opt-in, privacy reviewed and provider-configured; never use the researcher's email mailbox as an unauthorised sending backend.

---

## 13. ARCHITECTURE, PERFORMANCE AND RELIABILITY

**Streamlit-first.** Keep Python + Streamlit as the primary site and avoid React/Next migration, Node server, client-side API secrets or unsanctioned browser automation. Add a custom component only if the capability probe proves clear value.

- Prefer zero or minimal pinned dependencies; record licences and Community Cloud compatibility. Use vendored local assets or supported components rather than runtime CDN reliance.
- Organize visual components and adapters under `rssr_ui/` or project-appropriate `app/` modules, without moving mature model or lesson modules unnecessarily.
- Cache immutable shared assets and public model resources cautiously. Never cache private user images/data in globally shared caches. Load at most one heavy model at a time under a bounded resource policy.
- Avoid rerunning expensive inference due to interface interactions; gate inference behind a real submit action and hash configuration inputs.
- Use thumbnail/pyramid/tile previews for large images; keep original bytes for correct export where feasible. Do not pass full-resolution raster data into browser widgets as huge base64 strings.
- All external integrations have timeout, retry/backoff only when safe, clear errors, and empty/offline states. Never silently substitute another dataset.
- Test file type, dimensions, decompression bombs, invalid AOIs, unsupported scales, model absence, memory pressure, service failure, auth provider absence, and cross-user data isolation.
- Set performance **targets**, then report real measured values on stated hardware/viewport; do not invent 60fps or first-load timing. Keep ambient animation count low and avoid mobile GPU-intensive glows.
- Keep CSS in one tokenized/scoped source, avoid unsupported Streamlit selectors and record upgrade hazards. Support `?lite=1` or equivalent feature switch for low-memory/low-motion sessions without weakening scientific validation.
- Observe privacy-by-default; avoid collecting analytics or emails without consent; do not put private geographic regions or uploaded images into telemetry.

---

## 14. IMPLEMENTATION ROADMAP WITH REAL EXIT CRITERIA

### M0 — Audit and Capability Gate
Baseline branch/PR/HEAD analysis; tests; Streamlit capability matrix; data and auth gap audit. Deliver `docs/aurora_x_audit.md` and `docs/streamlit_capability_matrix.md`.

### M1 — Design Foundation + Flagship Overview
Build design tokens, responsive shell, core cards/diagrams, theme, accessibility foundation, real facts and showcase registries, Overview and component gallery. **Exit:** both themes, mobile layout, reduced-motion fallback, no fabricated hero results, working nav.

### M2 — Research-Grade AI Studio
Integrate existing inference into new workbench; real progress, bounded config, compare wall with pure-Streamlit fallback, metrics gating, evidence drawer and experiment passport. **Exit:** working real inference case, valid manifest, no image manipulation, clean failure states.

### M3 — Satellite + Model Observatory
Preserve AOI/acquisition functions, geospatial metadata, real scene control center, model atlas, comparable experiments, Pareto and reproducibility matrix. **Exit:** AOI guard tests and same-protocol comparison logic verified.

### M4 — Learning + Research
Improve lesson navigation and interactive numerics; preserve MP27–MP31 dispatch; build evidence library, researcher page, citations, licences and docs. **Exit:** existing lessons run; results and statuses are source-backed.

### M5 — Optional Auth + Secure Workspaces
Configure provider adapter and safe opt-in account UI, RLS/data isolation tests, setup docs, gated saved runs. **Exit:** either fully tested and configured, or clearly recorded as `NEEDS OWNER ACTION` without disrupting public access.

### M6 — Hardening, Performance and Deployment
Responsive and keyboard browser audit, visual screenshots, contrast, security scan, performance tests, regression suite and live deployment verification when accessible. **Exit:** clear evidence table, known blockers, next action, PR ready for review.

For each milestone: record changed files, exact commands, test results, screenshots/paths where created, remaining issues and compatibility decisions. Continue to the next safe milestone without repeatedly asking for confirmation. Do not mark an unfinished milestone complete merely because the interface renders.

---

## 15. AUTOMATED AND MANUAL ACCEPTANCE MATRIX

### Functional and scientific
- Existing tests remain green; all actual model families and only truly supported scales are represented.
- No regression in canonical inference, satellite validation, lesson navigation and MP27–MP31.
- No metric is rendered without eligible data and a named protocol. Invalid comparisons are rejected or explicitly qualified.
- Every showcased image has a provenance record. No hard-coded achievements/progress, fake ground truth or invented benchmark values.
- Model execution follows memory limits. Results and manifests match actual runs.

### UX and accessibility
- Manual screenshot matrix: widths `1920 / 1440 / 1024 / 768 / 390`, themes dark/light, with normal/reduced motion.
- No horizontal page overflow, unreadable equations, clipped controls, unusable dialog, inaccessible comparator or hidden content with animations off.
- Full keyboard walkthrough for navigation, dialogs, model selection, actions, charts/data tables and comparison handles.
- Contrast tested programmatically. Visible focus, screen reader labels, non-colour status labels, reduced-motion toggle and browser preference both work.
- All buttons have a verified action; empty and error states are meaningful.

### Security and data isolation
- Public Overview and permitted guest demos load without authentication.
- File validation, secret scanning, logging hygiene, safe provider defaults and privacy-safe cache behavior verified.
- Managed sign-up/login/recovery and database isolation tested only if configured; otherwise report `NEEDS OWNER ACTION`.
- No private uploads shared between sessions. No privileged actions granted based on contact email.

### Deployment/performance
- Dependencies install against pinned versions; documented Community Cloud startup path works where tested.
- Lighthouse/Playwright/browser timings recorded **only if tools available**; state hardware and methodology. No invented FPS or uptime.
- Broken provider and offline conditions show helpful fallback, not a fake live result.
- Deployment verified from actual site if accessible; otherwise mark `NOT VERIFIED`.

---

## 16. REQUIRED DELIVERABLES

1. A clean reviewable feature branch and PR (only if repository write access is available).
2. Implemented Streamlit-native UI modules, compatible with existing project files and minimal dependency changes.
3. `docs/aurora_x_audit.md`, `docs/streamlit_capability_matrix.md`, `docs/design_system.md`, `docs/scientific_ui_contract.md`, `docs/auth_decision.md` if evaluated, `docs/OWNER_SETUP.md` if external setup is required.
4. Source-backed model, dataset, showcase and experiment registries, with tests.
5. Responsive screenshots, component gallery and real test output where tooling permits.
6. Updated README, architecture notes, acceptance matrix, issue/progress notes and `next_session.md` without overwriting ongoing work.
7. A concise final report with: implemented scope, file changes, tests, visual QA, actual performance, real deployment verification, guest access, auth status, limitations, security risks, owner setup requirements and **exact next action**.

## 17. FINAL CODEX RESPONSE FORMAT

Return:

**A. Current verified repository baseline** — branch, HEAD, open PRs and existing tests.

**B. Implemented product improvements** — organized by milestone and exact paths.

**C. Evidence / tests** — commands, pass/fail/skips, screenshot artifacts, scientific guard checks.

**D. Deployment and access** — verified site behavior vs local-only behavior; guest access, signup and persistence separately.

**E. Blockers / owner actions** — credential/provider limits, Community Cloud restrictions and anything untested.

**F. PR and next concrete action** — link if created, or reason not created, and smallest verified next step.

Every major claim must be tagged `VERIFIED`, `NOT VERIFIED`, `NEEDS OWNER ACTION` or `BLOCKED`.

---

## 18. THE NORTH STAR

**For a beginner:** “I understand why satellite images can be enhanced—and why enhanced details are not automatically true.”

**For a student:** “I can understand CNNs, Attention and Transformers through maths, experiments and interactive teaching.”

**For a researcher:** “I can inspect the exact dataset, checkpoint, evaluation protocol, provenance and validity of each comparison.”

**For a technical user:** “I can run a supported model, compare the real output, reproduce the configuration and export a trustworthy record.”

**The product should look extraordinary because its scientific evidence is easier to understand, not because the interface exaggerates what AI can do.**

### IMMEDIATE FIRST ACTION
Audit the current repository and open work, verify the existing tests, build the Streamlit capability matrix, and implement the highest-value unblocked milestone. Preserve ML correctness and all active lesson work. Report only what you truly verified.
