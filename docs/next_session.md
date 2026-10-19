# Next session — current checkpoint, 2026-10-19

Continue feat/streamlit-submission. Historical PR #5 (now merged externally): https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/pull/5. Published checkpoint 79b8e5aa55cc4535dbf4a1be953d4635a99397b2 passed CI. Additional Streamlit and MP12–MP15 changes are validated locally and pending publication; preserve them. PRs 1–6 were merged externally; artifact files were incorporated on main by the user. Do not merge automatically.

## Current priority: Streamlit deployment

User explicitly chose Streamlit Community Cloud and authorized a stateless first deployment. They will connect their GitHub account through that dashboard; do not request tokens or infer hosting access from the connector. Entry point app/streamlit_app.py, Python 3.12, branch feat/streamlit-submission. Keep Gradio and the durable database requirement.

Genuine trained RGB pilot and licensed example published on artifacts/rgb-pilot-v1 at immutable commit b94966013b046b7f2b8e542d6161786d67e750c0. configs/hosted-model.json pins URLs/checksums. Model download (101637 bytes) verified and loaded. Public artifact exception is explicitly for user-requested portable model hosting; normal source checkout ignores weights.

Streamlit deployment-only clean Python 3.12 installation completed in work/streamlit-venv; resolved packages are in requirements/streamlit-windows-py312-lock.txt. Two independent processes with empty caches downloaded verified assets and produced identical outputs. Maximum 512x512 input produced 1024x1024 output; sampled peak RSS was 292.4 and 300.4 MB. Local browser inference, contrast previews, PNG download, and real Sentinel acquisition/GeoTIFF download passed. The Sentinel request transferred 1,900,544 bytes and returned EPSG:32611 output. Evidence: reports/streamlit-restart.json and reports/streamlit-satellite-browser.json. Linux CI job added; final published candidate must be checked. Public URL remains unverified; dashboard requires sign-in.

External storage adapter src/rssr/external_storage.py is optional and disconnected. Offline doubles test idempotency and transaction failure. S3 credentials/bucket and PostgreSQL URI are documented in docs/streamlit-deployment.md. It stores artifacts/provenance, not the full migrated normalized research schema. Full durable database remains incomplete; do not claim persistence across Streamlit restarts.

## Verified curriculum and research

MP05–MP15 expanded, with actual synthetic and licensed real training-crop experiments. MP12 SVD rank/error/storage; MP13 learning-rate divergence; MP14 NumPy chain rule; MP15 PyTorch gradient equivalence. Finite-difference maximum error 4.289e-11; NumPy/PyTorch final parameter difference 4.441e-16. Real red-to-NIR validation MSE .0098226 is worse than training-mean baseline .0095690; preserve this negative result. MP12–MP15 notebooks passed isolated kernels with exit 0. Next teaching task is MP16, then MP17 onward.

12 initialization diagnostics preserved. Six geographic validation-only runs completed over four train/four validation scenes and three seeds: mean PSNR gain over bicubic +1.5709 dB CNN, +1.3337 dB hybrid. No hybrid superiority, novelty or held-out claim. Original test remains unused. Study resume reverified hashes without retraining.

Last complete isolated suite: 73 tests and four subtests passed in 7.46 seconds without warnings. Structure check: 40 IDs and 394 relative links valid. Earlier full 40-notebook batch exited 0; MP12–MP15 reran after expansion. Notebook raw file hashes reflect the Windows working files; Git can normalize line endings. Jupyter emits a local TCP warning.

## Environments and next actions

Workspace work/venv points at this repo; work/clean-venv is isolated and editable to work/submission-clean, so set PYTHONPATH to current src when validating current code. work/streamlit-venv is the deployment-only clean install. Use PYTHONUTF8=1 and writable TEMP/TMP/Jupyter caches under work/tmp on Windows. All jobs CPU; zero GPU-hours; new Streamlit dependencies plus clean reinstall remain within authorized 2 GB download cap; exact historic retry overhead unknown.

Commit the validated Streamlit/MP12–MP15 changes and open a new draft PR from feat/streamlit-submission; PR #5 is already merged. Do not force-push or merge. Verify Linux deployment CI on actual head. If Streamlit dashboard is authenticated, deploy and verify public inference; otherwise give exact remaining dashboard steps. Continue MP16–MP40 and remaining research/report/presentation work independently of hosting authorization.
