# Resume checkpoint — 2026-10-27

## Exact git / PR / CI state

- Local branch feat/mp27-attention. Historical-binding implementationaf9e7b8, external-main integration9796842. Last fetched main8d3a0766ae8cd0532acd920858bbfece1484bc43 (external mergePR31). Working tree clean before this checkpoint update. Do not merge PRs automatically.
- PR30 headb45b953 passed CI40505931536 and externally merged981ec51; integrated12c3285 preserving local validation. PR31 budget/seam milestoneb76cfa2 published headd6b6056521e91197441f1a78a470a5cd5d3a1e7c, tree82a03b7546616dd4fef59a4bfbb3363df784449b. CI40506986000 SUCCESS, externally merged8d3a076, integrated9796842 retaining newer binding docs.
- Current draft PR32 https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/pull/32, branchfeat/historical-architecture-bindings, head37a0aaffe87ef7cf61561b727bc471b349a8e37a, tree415b15cc974d887e3ed6fe9f8bc6c966ccbf0dbb exactly equals local9796842. Attached. Exact-head CI40507399011 SUCCESS, verified after publication. This checkpoint is a local follow-up commit, not in that published tree.

## Newly executed evidence (do not repeat without changes)

- Versioned registry and train/load/evaluate/CLI wiring preserved. Previous20architecture tests and9optimizer/checkpoint integration tests passed;42tiny random-weight scale/export combinations passed. Those are implementation smoke checks, not trained scientific models.
- src/rssr/model_cost.py: partial Conv/Linear/MHA accounting, explicit exclusions, restores hooks/modes.4tests originally passed3.50s. scripts/preflight_architecture_budgets.py executed448 candidates:7models×32even widths2–64×RGB/four-band. FixedM7width16/window4/native2/input64square reference. Initial4.90s, final serializer rerun3.65s, identical choices. reports/architecture-budget-preflight.json includes source hashes/all candidates.
- Partial-budget eligible widths (both band settings):M1=20,M3=26,M4=20,M5=16,M6=26,M7=16. M2 has NO eligible match in searched range. Limits5%parameters/10%partialMAC. Never label full compute/runtime matched; no new training executed.
- scripts/check_registry_seams.py executed7untrained native2 comparisons on65x97→130x194, width4,tile32,halo32,seed76. Non-A models passed predefined max discrepancy1e-6. A models have measured discrepancies, no universal equality/tolerance. reports/architecture-seam-smoke.json retains output hashes/strip means. M6's tiny observed difference is not equivalence. No trained/representative/cascade/maximum-input seam claim.
- src/rssr/legacy_architecture.py + scripts/bind_legacy_architectures.py:60existing records inspected;39distinct runs verified checkpoint SHA256, full config/training identity, declared manifest hash consistency, strict registry load and bit-exact synthetic5x7 output.21gated/alternate-fusion records excluded. M1=9,M3=3,M4=6,M5=6,M7=15;M2/M6=0. reports/historical-architecture-bindings.json. No dataset reopened, artifact rewritten/uploaded, original report changed or deployment catalog mutation.
-11mapping tests originally passed0.11s. Final combined cost/mapping suite15passed3.00s.935relative links,40folder contracts,7claimhashes passed. Docs/architecture-preflight.md and architecture-registry.md explain math, actual results and limits. Historical binding applies only to each exact config, not all widths/bands/scales.

## Next unfinished work, in order

1. Check PR32/main/CI and preserve any external merge; publish remaining local checkpoint only with the next reviewable milestone. Do not repeat full audit.
2. Build trained capability entries from exact verified historical configurations/hashes. Keep original reports and weights untouched. Five IDs have historical pilot bindings; M2/M6 require bounded training design, not relabelled weights. Preserve M2's partial-budget mismatch. Develop representative-input and cascade seam/export checks with resource preflight; existing42-row random matrix is not trained capability evidence.
3. Finish protocol-v2 independent-region metadata and acquisition gates before research training. Full training resource estimates/approval remain required. No test tuning or old frozen rerun.
4. Finish MP20 independent SRCNN evidence gap and MP36–40 scientific criteria; all canonical folders already contain MP01-style deliverables. Final40-project independent fresh-kernel/output-hash batch belongs after substantive changes, not repeated frozen/expensive runs. Existing runner scripts/execute_project_notebooks.py can be extended.
5. Paper visual inspection/evidence-derived failure/edge/spectrum work and submission package, then interactive labs/UI/deployment. Use docs/seven-model-completion-brief.md (latest complete user request), completion_brief.md, research_upgrade_brief.md and submission_readiness.md; do not infer full completion from earlier teaching-only labels.

## Commands / environment

Repo outputs/Mini-Project-ML-Lab-RSSR. Dev Python ../../work/clean-venv/Scripts/python.exe; deployment ../../work/streamlit-venv/Scripts/python.exe. PowerShell set PYTHONPATH=(Resolve-Path src).Path and PYTHONUTF8=1.

```text
python -m pytest tests/test_model_cost.py tests/test_legacy_architecture.py -q
python scripts/preflight_architecture_budgets.py
python scripts/check_registry_seams.py
python scripts/bind_legacy_architectures.py
python scripts/check_structure.py
python scripts/check_claim_evidence.py
python scripts/check_project_folders.py
python scripts/build_paper_evidence.py --check
```

Run only relevant changed checks. GitHub connector publication uses create_tree/commit/branch/PR, comparing local and remote tree SHAs; helper ../../work/publish_text_entries.py emits origin/main..HEAD text entries. Shell push previously hung; read-only fetch works. Always attach PRs.

## Preserved scientific and external gates

- MP37 implementation/teaching complete; MP38architecture,MP39historical analysis,MP40historical robustness lessons executed; full scientific acceptance pending. Reports mp37-fusion.json,mp38-architecture.json,mp39-ablation-analysis.json,mp40-benchmark-analysis.json unchanged. Learner mastery unassessed.
- Protocol-v2 committedb667078 before new experiments: design only, no eligible frozen independent-region manifest.5folds≥15regions,5km buffer,seeds17/23/41. No new study executed. Patches/seeds are not independent geography; retain negative attenuation findings.
- MP36 original SEN2VENUS paper/correction read, dated addendum reports/mp36-feasibility.md.15m filtering is not residual-registration guarantee. V2archive/pair properties still unverified. Bounded localZIP21tests/single-attempt transfer7offline tests passed; no live archive acquired. Exact URL/bytes, licensing/pair validation and explicit large-transfer approval remain gates.
- Paper compilation passed reports/intake-paper-validation-20261027.json; all-page visual review pending. Uploaded IEEE template transfer and author/contact details unavailable. Do not invent them.
- Input≤1024eachside; six scales; exact dimensions; disk-backed outputs; host cap1,048,576outputpixels. Historical1024square→2048square ran;65536square planned only. Existing scientific value/mask/CRS rules remain.
- Public app last signed-out noaccess/notfound; no owner dashboard session. No public working/deployment claim; do not request sameURL again. External S3/Postgres configuration and live restart evidence absent. Four original example uploads unavailable. No paid resources activated.

## Latest upgrade milestone — 2026-10-28 (supersedes prior public-access blocker)

- User's latest brief preserved docs/scientific-website-upgrade-brief.md, linked completion_brief.md. Advanced family/ablations, full scientific ingestion and broad native-site redesign are additional unfinished requirements. Historical educational models/frozen evidence preserved.
- PR32 externally merged mainc167b962d4512945f782dee29d6f03d4a3e93f56; integrated6342e81 preserving checkpoint. Prior exact-head CI40507399011success. Current changes to publish after commit.
- Implemented src/rssr/surface_reflectance.py: bounded explicit L2A conversion, no inferred/default scale/offset, required harmonization/review metadata, no second conversion, conservative SCL4/5/6 and all-band validity/saturation/nodata intersection, grid mismatch rejection, unclipped float32 values and atomic no-overwrite masked GeoTIFF.24synthetic tests passed0.77s incl1024square and export round-trip. No spectral acquisition/real-provider calibration/alignment or inference executed. docs/scientific-sentinel.md explains contract/limitations and official sources.
- UI: model_explorer_ui.py native7cards/checkpoint selection/math/status in existing Methodology & Data section; M2/M6 unavailable, no random output. All matching project PNGs shown. Satellite Explorer explicitly labels old acquisition display RGB. Existing7-section AppTest and new7-model selection AppTest passed with network blocked; new check addedCI. reports/scientific-ui-validation.json pins source hashes.937links/40foldercontracts/7claimhashes passed. Full redesign, advanced models and final40acceptance unfinished.
- NEW PUBLIC EVIDENCE: new in-app browser tab reached existing app without login and executed licensed CNN100step example16x16→32x32 native2; displayedCPU.002s. reports/public-access-20261028.json and local work/public-inference-20261028.png. Old no-access blocker superseded for this request only. Deployed interface older/different than local; today's changes not deployed. Deployment commit/cold start/durable storage/download-content checks still pending. No public URL failure asserted from webtool fetch error; actualbrowser succeeded.
- NEXT: verify/publicize current PRCI; implement provider-specific reviewed spectral metadata adapter and explicit quality/grid alignment before claiming scientific acquisition. Add executable local scientific ingestion entry point; no learned-reflectance inference without compatible checkpoint/data evidence. Continue new trained M2/M6 resource design, seven-model trained scale evidence, native40lessons/final execution report, advanced-family preregistered ablations and paper. Keep original large-transfer/training approval gates.
- Commands: PYTHONPATH=src; dev python -m pytest tests/test_surface_reflectance.py -q; deployment python scripts/check_model_explorer.py and scripts/check_showcase.py. Do not repeat frozen experiments or prior40lesson batches without changes.

## Publication and browser follow-up — 2026-10-28

- Local milestone68afcda published draft PR33 https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/pull/33, branchfeat/scientific-reflectance-model-explorer, headb9ea56f9542d0bc9f4b38fe074af2943fcabc72f, treeb1acfc503beeaae3164829f98e2fb0578ebcfd6f identical local. Attached; Exact-head CI40572712377 SUCCESS, verified after browser checks. Local preview process stopped after verification.
- Actual local browser visual smoke completed for Model Explorer at1280x800 and390x844: desktop3-column and phone stacked cards, explicit M2/M6 missing-training labels. reports/model-explorer-browser-20261028.json pins screenshot hashes; screenshots in work/. This is focused card-layout inspection, not exhaustive section/accessibility/mobile certification.
- Public example inference succeeded separately on the older deployed UI; reports/public-access-20261028.json. New code remains PR-only until externally merged/deployed. No hosting/persistence/cold-start claim. Continue local scientific adapter and remaining website/research acceptance tasks from latest brief.

## Latest executable ingestion milestone — 2026-10-28

- PR33 exact-head CI40572712377 SUCCESS rechecked; externally merged origin/main880d96c, integrateda5c5ab5 preserving browser checkpoint. No agent PR merge. Current milestone publication follows this commit.
- NEW src/rssr/reflectance_job.py + `rssr prepare-reflectance REQUEST OUTPUT [--execute]`: local hash-pinned self-contained GeoTIFF assets only; explicit bounded window, header preflight,512MiB full-source verification cap, array estimate, disk reserve, same-grid rejection, per-band quality/saturation and quality-coverage intersection, float32 atomic export. uint8 spectral display inputs, nonbinary flags, unpinned sidecars and out-of-bounds windows rejected. No network/resampling/model inference.
- Actual final validation:36combined ingestion/conversion tests passed2.34s, including fresh CLI subprocess plan-only/no output and --execute/two-band3x2output. Correct shifted origin500010,2999990; synthetic DN2000 with scale.0001offset-.1 gives.1reflectance.940links/40foldercontracts/7claimhashes passed. reports/reflectance-job-validation.json pins environment/source hashes; docs/local-reflectance-job.md has schema/commands/bounds. No real provider product or frozen result touched.
- NEXT: inspect current milestone PRCI, then provider-specific metadata reconciliation and explicit reviewed mixed-resolution quality alignment; do not default offsets/harmonization. Local command now exists but real scientific ingestion remains unverified. In parallel continue trained model/scale capability evidence from39historical bindings; M2/M6training, advanced-family ablations, final40notebook acceptance, paper and fullsite redesign remain open. New training/large acquisitions still require concrete resource review.
- Reproduce changed tests only: PYTHONPATH=src; ../../work/clean-venv/Scripts/python.exe -m pytest tests/test_reflectance_job.py tests/test_surface_reflectance.py -q. No reason to rerun frozen research or previous lesson batches for this CLI change.
