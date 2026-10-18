# Catalog and maintenance

SQLite schema v1 lives in src/rssr/db.py; foreign keys are enabled per connection.
Repeated initialization is idempotent, unknown versions are rejected. Pair/scene
relations use pair_sources rather than an ambiguous list. Local runs use one writer.
Use `rssr export data/catalog.sqlite reports/export` for JSON and CSV snapshots.
Keep bulk databases and downloaded assets outside Git. Back up SQLite after closing
writers or via SQLite's backup API; do not copy an actively changing database blindly.

`rssr refresh configs/stac.json --root data` ingests bounded STAC metadata. Provider
collection+item IDs deduplicate repeated overlapping queries. It records job start,
end, failures, acquisition time and provider-created field. Retries use exponential
backoff and honor numeric Retry-After, capped at 30 seconds. Pagination stops at the
configured budget, records a next cursor, and reports partial rather than complete.
Resume a partial, failed, or interrupted job explicitly with `rssr refresh configs/stac.json --root data --resume-job JOB_ID`. The configuration must match exactly. Each request is checkpointed before fetching; replay deduplicates committed scene IDs. POST next-link merge bodies are supported. Expired provider cursors require restarting the overlapping query; automatic timestamp watermarks remain pending. Downloaded assets
use checksummed resumable downloads with three attempts and atomic final replacement.

Default proposed scheduler interval: 24 hours; configure an OS scheduler to run the
explicit refresh command only on an always-on research host. No scheduler is installed
on the user's machine. Hosting sleep can delay refreshes. Last successful refresh is
the latest completed ingestion_jobs.end_time; show it and acquisition time to users.
Cached observations are not live sensing. Inference and retraining are separate,
explicit commands. Fresh observations without an HR reference have no PSNR/SSIM.

The local Gradio demo uses no database writer. Public deployment needs an account
and verified free compute entitlement. Checked 2026-10-18: Hugging Face documentation
states ordinary compute Spaces require a paid plan even though CPU Basic has no hourly cost; eligible free personal accounts can host up to two ZeroGPU Gradio Spaces. That exception still needs an authenticated account and verified eligibility:
https://huggingface.co/docs/hub/spaces-overview . Do not activate a plan automatically.
Disk persistence must be checked: https://huggingface.co/docs/hub/spaces-storage .
Never use ephemeral storage as the durable research database. A hosted SQL/object
storage adapter and public deployment remain pending; no live URL is claimed.

Local RGB demo: set RSSR_CHECKPOINT to a validated three-band checkpoint, then run
`python app/demo.py`. Four-band scientific rasters use `rssr infer` and GeoTIFF output.
Upload cap is 8 MB, input edge cap 1024, queue 2, concurrency 1. Gradio temporary
cache expiry is 60 seconds; uploaded data are not added to research records.
