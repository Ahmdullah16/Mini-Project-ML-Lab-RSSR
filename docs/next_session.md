# Latest continuation — 2026-10-21

PR #11 is a draft on `feat/submission-integration`. Published commit `1baf02121ca9814e3533ec9ed41957846cb5f7b8` passed both Linux CI jobs, run 39578629874, including all three hosted checkpoints and max-size inference. Its tree matches local `ce10a0f`. The pre-test local freeze `9109766` is preserved as tree-equivalent remote parent `f9e4880`; remote publication was later, not evidence of test timing. No PR was merged by the agent.

MP26 now expanded and executed successfully; its six recorded development curves and compute table are in `reports/mp26-attention-sr.json`. No new training or test evaluation. MP01–MP26 are expanded; MP27–MP40 still need full expansion, and earlier project-specific exceptions such as MP20's independent SRCNN test table remain explicit.

`scripts/rebuild_frozen_data.py` downloaded 16,883,208 raw bytes plus 142,290 manifest bytes into a new empty directory. All 144 RGB and 144 RGBNIR tiles matched frozen checksums and complete manifest structures. Original-byte manifests restored only after equality checks. SQLite integrity and foreign keys passed; repeat used zero raw bytes; controlled corruption was rejected, then restoration recovered successfully. Evidence: `reports/frozen-data-rebuild.json`; instructions: `docs/frozen-reproduction.md`. Linux rebuild added to CI, pending the next published head. Existing held-out result remains unchanged.

Next: publish this MP26/rebuild follow-up and verify its Linux CI; continue MP27–MP40, portable four-band research checkpoints, normalized external database, paired cross-sensor feasibility, final report/slides/package and final all-notebook gates. Existing public pilot remains available; new model selector has not been verified deployed. No new host/paid resource is needed. Durable external account credentials remain absent.

---
Earlier checkpoint (superseded where noted):

# Current checkpoint — 2026-10-21

Preserve branch `feat/submission-integration`. Main advanced to `5d6a2ea` through external merge PR #10 (frozen RGB artifacts); no open PRs at this check. Do not merge PRs or overwrite existing work. The source integration remains local until published and CI verified.

MP01–MP22 expanded and executed; MP23–MP25 have executed attention comparisons and expanded notebooks; all three passed real kernels. MP20's separate held-out SRCNN table remains unfulfilled; none of these statuses asserts learner mastery. MP26–MP40 still need full lesson expansion, even where underlying research experiments exist.

Research: 36 development records completed, with 30 new and six reused runs. Freeze protocol was committed at `9109766` BEFORE the single held-out evaluation. `reports/final-heldout.json` contains the executed results; do not retune or overwrite. RGB held-out mean PSNR: bicubic 48.5183, CNN 50.2614, hybrid 50.2459, official SwinIR 48.9897 dB. One scene and synthetic degradation cannot establish cross-sensor/general geographic validity. No hybrid superiority claim. Official SwinIR has different pretraining/degradation. Robustness/spectral/CPU-efficiency evidence is in `reports/development-benchmark.json`.

Public app https://mini-project-ml-lab-rssr.streamlit.app/ recovered at the same URL. Recorded tests passed sample, uploaded RGB, repeat-byte equality, maximum 512→1024, corrupt/transparent/oversized input rejection, and actual Sentinel acquisition with georeferenced downloadable output. This evidence is from the existing pilot deployment; no new deployment is claimed. Exact deployed source identity remains unverified.

New local Streamlit model selector preserves the pilot and adds genuine frozen RGB CNN/hybrid. Immutable artifact commit `b21b797a2c43f3a45879cd6778be4cd81600f2e7` contains downloadable SHA-verified weights and original-byte dataset manifests. All three models passed AppTest and actual maximum-size inference in the deployment-only environment; RSS after inference, not peak, was 259–283 MB. The new selector has not been verified on the public host. New code reports source identity and gives friendly bounded-upload errors. Preserve Gradio and stateless hosting; local caches/SQLite are not durable.

Ingestion scheduler/schema v3 is committed: leases, durable pending windows, page/cursor recovery, crash reconciliation and overlap watermark. Four scheduler tests and live metadata-only pagination passed; no external always-running scheduler was installed. No external-storage account credentials are configured. Offline S3/PostgreSQL tests do not establish live durability or complete the normalized external database requirement.

Validation before attention additions: 83 tests plus four subtests passed in 12.01 seconds. MP19–MP22 passed real kernels. Attention comparison ran CPU seed23, 300 updates, four train/four validation scenes: reconstruction PSNR off35.0253, fixed0.5 35.1894, channel35.1721, spatial35.6038, combined35.6899 dB. This is same-grid teaching reconstruction, not SR or a new research test. See `reports/mp23-mp25-attention.json`; focused attention tests passed.

Latest checks: 85 tests and four subtests passed in 34.13 seconds; 412 relative links/40 IDs passed; existing held-out result was preserved without repeat evaluation.

Next: commit validated checkpoint, integrate main's additive artifact merge, publish a reviewable draft PR and verify exact-head CI. Continue MP26–MP40, frozen dataset/checkpoint rebuild in a fresh checkout (Windows separators fixed in script consumers; original manifest byte/hash portability still needs validation), four-band portable checkpoints, normalized durable DB, real paired satellite feasibility, final report/slides/package and final all-notebook gates. Do not mark whole project submission-ready.

Environment: isolated `work/clean-venv` points editably at an older clone; set PYTHONPATH to current src. Use PYTHONUTF8=1 and writable work/tmp for Jupyter. CPU only; zero GPU. Keep within original 2 GB downloads/2 GPU-hours authorization; prior retry download overhead is incompletely measured. No new imagery/downloads for attention lessons.

---
Historical notes follow; the checkpoint above supersedes conflicting old status claims.

# Active continuation — 2026-10-19

Branch feat/submission-integration, based on merged PR #9/main 382be52. User request attachment 48604ea0-96d1-42f7-8537-002ae67434d3 says finish all feasible requirements, preserve the existing public deployment https://mini-project-ml-lab-rssr.streamlit.app/, prioritize final model/public verification/artifacts. No merging or paid services.

Uncommitted work: scheduler.py plus schema v3 ingestion_streams migration and CLI refresh-incremental. Eleven affected scheduler/resume/schema/catalog tests passed. Includes partial-page recovery, leases, hard-crash replay, completed-job reconciliation, expired-cursor replay and overlap watermark. Full suite passed 79 tests and four subtests in 10.69 s with zero warnings; 405 links valid. Live incremental run reached completed and advanced watermark after three pages. Public sample, uploaded 32x48 RGB -> 64x96, repeat-byte equality and actual Sentinel 512x512 georeferenced output/download passed; report public-browser-validation.json. App briefly redirected to not_found, user reconfirmed same URL and it loads again. Pending public invalid/maximum-size uploads and exact deployed source identity. Public checkpoint hash matches genuine old CNN pilot; no hybrid relabeling.

Submission study finished all 36 records (30 new + six reused): ten 4-band variants × three seeds, RGB CNN/hybrid × three seeds, 1000 steps, validation only. No held-out evaluation. scripts/submission_study.py resumes by checkpoint/config hashes. New training recorded ~999 CPU seconds, zero GPU. Mean validation delta dB: 4-band attention 1.79466, CNN 1.57092, hybrid 1.33370, learned gate 1.66575, fixed half 1.62262, addition 1.55239, consistency 1.37736, no-attention .99042, Transformer .61800, educational SRCNN -4.93602. RGB CNN 1.66969 vs hybrid 1.66992 (essentially equal; no superiority claim). Robustness/spectral/efficiency analysis completed in reports/development-benchmark.json. Official SwinIR validation executed, mean delta +0.84155 dB. configs/final-evaluation.json now freezes checkpoint/config/metric decisions; commit it before scripts/final_evaluation.py accesses test. Portable RGB checkpoints and demo selection remain pending.

Official SwinIR source downloaded only to work/swinir-source at commit 6545850fbf8df298df73d81f3e8cba638787c8bd; Apache-2.0. Official lightweight x2 checkpoint downloaded 17,147,989 bytes under data/official-swinir; SHA 193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7 recorded configs/official-swinir.json. timm/torchvision installation was launched in clean-venv; verify availability. Attributed unmodified source and Apache license copied to third_party/swinir. Optional loader and validation experiment ran successfully. timm 1.0.30 and torchvision 0.29.1+cpu installed; upstream deprecation warnings documented.

No RSSR external-storage environment keys present. Existing adapter tests use doubles; normalized external schema integration still pending. MP19–MP40 full expansions, final report/slides/package and final all-notebook/test/CI gates remain. Preserve existing negative research results and all completed teaching work. Current clean-venv has editable old clone, so set PYTHONPATH to current src. Windows UTF-8 and work/tmp caches required. Resource budget remains 2 GB new downloads / 2 GPU-hours; recent SwinIR weight download measured, historical retry overhead unknown.

---
Historical checkpoint follows; active notes above supersede it.

# Next session — 2026-10-19

Current branch feat/convolution-playground. PR #8 was merged externally at bc66a60; published MP16 head 85c770f95e299a75e17e6ea76b5daf84fabbc5ae passed both CI jobs (run 39247942583). Preserve existing work. PR #7 Streamlit/MP12–MP15 was merged externally at main 1283cf9; its exact published head 6c118ea6bc110385a826bbd2bf7e9f76118f5d7b passed both Linux jobs, including clean deployment install and real checkpoint inference (Actions run 39247571971). No automatic merges.

MP16 now has an expanded Dataset/DataLoader, loss accounting and checkpoint recovery lesson. `python scripts/pipeline_experiment.py` executed: seeded regression final MSE 1.91e-14; validation MSE 4.19e-14; two runs had identical parameters. Eight-step production training exactly matched four steps resumed to eight, and rejected an inference-only checkpoint. Notebook passed a real isolated kernel with exit 0. Synthetic recovery splits intentionally repeat a ramp and cannot establish generalization. Existing real geographic runs are referenced without retraining. MP17 and MP18 are now expanded and ran in isolated kernels with exit 0. Their scripts executed synthetic and existing licensed real-crop experiments; figures were inspected. Convolution matches PyTorch exactly in tested cases; learned Sobel imitation is not semantic-edge accuracy. Newest full suite: 75 tests plus four subtests, 14.44 s, no warnings. Next full teaching task is MP19; MP19–MP40 remain incomplete.

Streamlit settings: repository Ahmdullah16/Mini-Project-ML-Lab-RSSR, branch main (PR #7 is merged), entry point app/streamlit_app.py, Python 3.12. No secrets needed for stateless inference. User must sign in/connect GitHub through the Streamlit dashboard. No public URL or public request verified. Genuine pinned trained weights and example download independently; two empty-cache processes produced identical results with sampled RSS at most 300.4 MB. Local browser sample and actual Sentinel/GeoTIFF download passed. See docs/streamlit-deployment.md and reports/streamlit-*.json. Preserve Gradio.

External S3/PostgreSQL adapter has offline tests only, is disconnected from demo, and does not replace full durable normalized database migration. Exact required secrets are documented. Do not claim local files survive hosted restarts. Remaining work includes scheduler/watermark recovery, cross-sensor calibration/alignment, strong baselines, robustness/spectral studies, final frozen selection and untouched held-out evaluation, report/slides/package.

Research: six geographic development runs over four train/four validation scenes and three seeds gave mean CNN +1.5709 dB and hybrid +1.3337 dB over bicubic. No hybrid superiority or novelty claim. Original single-scene negative results preserved. Original test remains unused.

Local full suite at Streamlit checkpoint: 73 tests plus four subtests passed in 7.46 s without warnings; 40 IDs/394 links. MP16 subsequently executed its assertions and notebook. Notebook hashes reflect Windows working-file line endings. No learner mastery or overall completion claim.

Environment: work/clean-venv is isolated but editable to work/submission-clean; set PYTHONPATH to this repo/src. work/streamlit-venv is the isolated deployment-only environment. Use PYTHONUTF8=1 and writable work/tmp caches. CPU only, zero GPU-hours; historical retry overhead incompletely measured; remain within 2 GB authorized download budget. Publish MP17–MP18 checkpoint as a new draft PR and verify exact-head CI. PRs #7 and #8 were merged externally; do not update those closed PRs.

The user added .devcontainer/devcontainer.json on main at a7ae3cd. It was preserved and corrected to Python 3.12, the tested app/requirements.txt and enabled request protections. JSON/startup assertions passed; Docker is unavailable, so container startup remains unverified.
