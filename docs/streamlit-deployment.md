# Streamlit Community Cloud deployment

The first deployment target is Streamlit Community Cloud. The app is deliberately stateless: uploads, output bytes, local SQLite, and satellite crops are temporary. Restart recovery rebuilds model/example caches from pinned public artifacts; it does not restore user sessions or research records. Gradio remains available separately.

## Dashboard settings

The existing app is https://mini-project-ml-lab-rssr.streamlit.app/. Preserve it; do not create another deployment. Its configured settings are:

- Repository: `Ahmdullah16/Mini-Project-ML-Lab-RSSR`
- Branch: `main`
- Entry point: `app/streamlit_app.py`
- Advanced settings: Python **3.12**
- Secrets: **none required** for the stateless demo

Use only the free Community Cloud deployment. The adjacent `app/requirements.txt` pins CPU PyTorch and runtime dependencies; no GPU, training job or paid service starts on import. Recorded public sample, upload, error/size, repeat and Sentinel export checks passed. The new model selector is not yet deployed/verified publicly. Keep main until the integration PR is reviewed; after an authorized update, verify Application version, chosen model/checkpoint hash and an actual output. Empty-cache recovery is locally verified; hosted cold restart is not yet verified.

Official documentation checked 2026-10-19: [deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) and [dependency discovery](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies). Community Cloud uses Linux and searches the entry-point directory for its dependency file before the repository root. The local environment is Windows; Linux installation is checked by the review branch CI job, not assumed from Windows success.

## Local clean installation

```bash
python -m venv .venv-streamlit
# Activate this environment using your shell.
python -m pip install -r app/requirements.txt
python -m streamlit run app/streamlit_app.py
python scripts/check_streamlit.py
```

The entry point adds the repository `src` directory explicitly; it does not require a prior editable install. The check launches two separate processes with empty asset caches, downloads and verifies the actual trained checkpoint, runs Streamlit AppTest inference and maximum-size tiled inference, compares restart outputs, and records sampled process memory. It does not claim to be a public HTTP test.

## Published model and example

`configs/hosted-model.json` pins immutable artifact commit `b94966013b046b7f2b8e542d6161786d67e750c0`, not a mutable branch URL. Artifact branch `artifacts/rgb-pilot-v1` contains the genuine 100-step RGB CNN checkpoint, a licensed NAIP training example and model card. The checkpoint is 101,637 bytes with SHA-256 `56656672c50c222105cc159738b63957dee84b1ef394106f6dc2c782a53d415e`. Downloads have byte limits and are verified before loading with the repository's restricted checkpoint loader. This explicit deployment artifact is isolated from normal source commits and preserves the original local training record.

The pilot is not a final research model. Its public example was acquired on 2021-09-17 and synthetically degraded; it is clearly labeled as cached. No HR-reference score is fabricated for uploads or newly acquired satellite imagery. Model output can contain incorrect detail. Display scaling does not establish physical reflectance calibration.

## Limits and recovery

Uploads accept opaque RGB/grayscale PNG or JPEG, at most 512 pixels per side and 8 MB. Inference uses 128-pixel tiles with a 32-pixel halo and serialized CPU access. The optional Sentinel workflow explicitly queries a bounded interval and AOI, downloads at most 32 MB, validates masks/bands/CRS, then offers a GeoTIFF and provenance. Its SQLite scratch directory is removed after the request; acquisition, ingestion, inference and refresh timestamps remain in the downloadable metadata.

Clear the app's temporary result or restart the app to discard a session. A restart can rebuild public model/example caches. If GitHub artifact access is unavailable or checksum validation fails, the UI reports failure instead of substituting random weights. A provider failure does not become a successful fresh-data result.

## Durable storage remains a separate requirement

`src/rssr/external_storage.py` is an opt-in S3-compatible object/PostgreSQL provenance adapter. It is not invoked by the public demo and never persists user uploads automatically. Its tests use injected local doubles; no live external persistence is claimed. Full normalized research-schema migration and live restart/durability validation remain incomplete.

An operator needs an existing authorized object bucket and PostgreSQL database, with no paid services activated by this project. Store these values in the host secret manager, never Git or chat:

- `RSSR_S3_ENDPOINT`: HTTPS endpoint for the operator's S3-compatible service.
- `RSSR_S3_BUCKET`: pre-created private bucket.
- `RSSR_S3_REGION`: service region (optional, defaults to us-east-1).
- `RSSR_S3_ACCESS_KEY_ID` and `RSSR_S3_SECRET_ACCESS_KEY`: restricted credentials permitting PutObject and HeadObject under the `rssr/` prefix.
- `RSSR_DATABASE_URL`: PostgreSQL connection URI with TLS required (`sslmode=require`) and permission to create/insert/select `rssr_external_artifacts`.

Install the optional SDKs only when enabling this adapter: `python -m pip install boto3 "psycopg[binary]"`. Inject the secret values into environment variables, call `from_environment()`, then explicitly call `persist(content, provenance)` for approved research outputs. Object upload precedes the database transaction. A database failure can leave a content-addressed orphan object but cannot report successful persistence; retries reuse its content key. Administrative cleanup of orphan objects is an operator task. Do not infer that this artifact adapter replaces the complete durable research database.

## Frozen RGB model catalog

`configs/hosted-models.json` pins genuine CNN/hybrid inference-only checkpoints at immutable artifact commit b21b797a2c43f3a45879cd6778be4cd81600f2e7, preserving original training-checkpoint provenance. The original pilot remains the default. `python scripts/check_model_catalog.py` downloads/verifies every model in an empty temporary cache and executes sample plus maximum 512-pixel CPU inference. See `reports/model-catalog-check.json`; RSS is sampled after inference, not a guaranteed peak. None of this claims local SQLite survives host restarts.
