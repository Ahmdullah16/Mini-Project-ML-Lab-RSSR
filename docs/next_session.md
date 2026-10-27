# Resume checkpoint — 2026-10-27

## Current milestones and publication

- Local branch feat/mp27-attention. Last committed lesson milestone 9786787, tree a8e13e5796dfb6263fbca38365599f2494546cf6. Published as draft PR28 https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/pull/28, branch feat/mp38-mp40-research-lessons, head41957f3c3f639a5447895e59e4d2660e4ca83e5f, identical tree. CI40504668700 SUCCESS. PR attached; do not merge automatically. Last fetched origin/main5263e3c (external merge PR27). Inspect current remote before integrating.
- Latest user request preserved verbatim in docs/seven-model-completion-brief.md and linked from completion_brief.md. It adds seven C/A/T architectures, final 40-project execution verification, interactive labs and full evidence-backed submission acceptance. Do not mistake earlier completion labels for this expanded acceptance.
- NEW registry milestone: src/rssr/architecture_registry.py, docs/architecture-registry.md, tests/test_architecture_registry.py. Seven structurally distinct configurations, legacy code unchanged. Historical attention = C+A. M2 A-only feature path and M6 A+T are new. Shared embedding/reconstruction convolutions remain in all models. Registry configs are UNTRAINED; checkpoint/provenance null until explicitly bound.
- Actual validation: 20 registry tests passed13.11s (module presence, distinct signatures, finite gradients/shapes, strict historical compatibility, four-band native4). scripts/check_architecture_matrix.py executed42 small random-weight disk-export combinations in2.96s; reports/architecture-smoke-matrix.json records source/env/seed/timings/shapes/hashes and before/after RSS, NOT peak. Synthetic5x3 input; output up to320x192. No checkpoint, trained quality, new maximum-size or seam claims. Temporary TIFFs removed normally. Rasterio correctly warned fixture lacks georeferencing.
- Structure931links,40foldercontracts and7claimhashes passed. No frozen experiment/checkpoint/artifact regenerated. Publication of registry follow-up pending at this checkpoint; inspect git log for commit and later note.

## Exact next tasks

1. Publish/review registry follow-up and inspect CI on its actual head. Preserve externally merged changes and local checkpoint. No automatic PR merge.
2. Bind historical checkpoints only after matching full configuration, hash and training provenance. Implement explicit registry config in training/evaluation/CLI; M2/M6 have no trained artifact. Do matched parameter/MAC preflight before new study. Keep statuses implemented/trained/evaluated/deployed separate. No unrelated checkpoint reuse.
3. Complete new-variant tiled seam tests, representative-input/maximum-input planning and final trained seven-by-six capability matrix. Existing smoke matrix is not final acceptance. App currently uses its original trained catalog; do not expose random weights as working inference.
4. Continue MP36 v2 archive lineage and exact byte-count/finalURL gates; independent-region metadata and protocol budget matching; MP20 independent SRCNN evidence gap. MP38–40 expanded lessons passed but scientific criteria remain open. Reuse existing reports; do not rerun frozen holdout.
5. Paper visual inspection and evidence-derived failure/edge/spectrum analysis; finish submission package; then interactive mini-project views, UI/browser/host verification. Final40-project fresh-run batch should occur after changes, not repeated expensive audits.

## Commands and environment

Repository outputs/Mini-Project-ML-Lab-RSSR. PowerShell: `$env:PYTHONPATH=(Resolve-Path src).Path`; `$env:PYTHONUTF8=1`. Python `../../work/clean-venv/Scripts/python.exe`; deployment env `../../work/streamlit-venv/Scripts/python.exe`.

```text
python -m pytest tests/test_architecture_registry.py -q
python scripts/check_architecture_matrix.py
python scripts/check_structure.py
python scripts/check_claim_evidence.py
python scripts/check_project_folders.py
python scripts/build_paper_evidence.py --check
```

Use targeted checks after changes, not blind regeneration. Publication helper ../../work/publish_text_entries.py builds text tree entries from origin/main..HEAD. GitHub connector create_tree/create_commit/branch/PR used because shell push previously hung; verify remote/local trees match. Always attach PR.

## Preserved evidence and limitations

- MP37 complete implementation/teaching scope:7tests and3-cell real kernel; historical learner mastery unassessed. MP38:5tests/kernel; MP39:7tests/kernel; MP40:5tests/kernel and visually reviewed SVG. Reports mp37-fusion.json, mp38-architecture.json, mp39-ablation-analysis.json, mp40-benchmark-analysis.json. Historical quality data unchanged.
- PR26 CI40451432276success, externally merged0774232; PR27 CI40452296078success, externally merged5263e3c. Bounded localZIP inspection/extraction21tests and single-attempt transfer7offline tests passed. No live archive downloaded.
- Original SEN2VENUS methodology and correction read via publisher-provided ResearchGate texts; dated addendum reports/mp36-feasibility.md.15m is match/shift filtering, not measured residual tolerance. Correction changes public deposit URL, not conclusions. V2 archive properties remain unverified. C06hash intentionally refreshed, status unavailable.
- v2 protocol committed b667078 before new experiments: design only, no eligible frozen independent-region manifest, no new study executed.5folds≥15regions,5km buffer,seeds17/23/41,matched CNN/hybrid parameters5% MAC10%; budget proposal not authorization for long training. Patches/seeds not independent geography. Keep pilot negative attenuation findings.
- Paper compiled successfully after original-method/correction update; reports/intake-paper-validation-20261027.json. All-page visual review pending. Uploaded IEEE template files are not yet transferred/located; authors/contact details absent. Do not invent them.
- Existing scales preserve ≤1024 input per side, all6scales, exact dimensions, disk-backed stages and host cap1,048,576outputpixels.1024square→2048square executed historically;65536square planned only. No public execution claim from local checks.
- Public URL last signed-out result noaccess/notfound; no owner dashboard session. Do not ask for same URL again. Hosted durable storage blocked on external S3/Postgres configuration and live restart evidence. Four source example uploads requested earlier remain absent. No paid services or large acquisitions authorized implicitly.

## Latest follow-up publication and integration

- Registry milestone07deca7 published as draft PR29 https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/pull/29, branchfeat/seven-architecture-registry, head5ab193bfb129d8db8baa6d51487642a89d0d1f03, tree7edbe226f5e62f8336e95ecac8edccaea910ea02 equals local. Stacked on PR28; CI40505351814 SUCCESS. Both PRs attached; no agent merge.
- NEXT milestone now implements model_factory.py routing in existing train/load/evaluate/CLI paths, and stores versioned ID in database.9 integration/regression tests passed8.83s: one Adam step per architecture on synthetic ramps, strict reload, validation evaluation and checkpoint-hash database records; existing deterministic resume unchanged. Initial7 failures were missing fixture tile registration, fixed by registering fixtures; no foreign-key weakening. These are temporary optimizer smoke checkpoints, NOT trained scientific models. Source changes require follow-up publication/CI.
- Continue historical checkpoint provenance binding and matched parameter/MAC preflight, then new seam/representative fixtures. Prior random-weight42-row report remains source-current because registry/models/scaling were not modified by factory wiring. No frozen data/evaluations changed.
