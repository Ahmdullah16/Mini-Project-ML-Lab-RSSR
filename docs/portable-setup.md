# Portable setup and examiner smoke demo

Use Python 3.12 in a new virtual environment. Do not enable system site packages. Run commands from the repository root; use that environment's Python throughout.

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/windows-py312-lock.txt
python -m pip install --no-deps -e .
python scripts/portable_quickstart.py
python scripts/launch_demo.py --data-root data/submission-pilot --host 127.0.0.1 --port 7860
```

The resolved lock was executed on Windows x64/Python 3.12. It is not a verified Linux lock. On another platform, install the appropriate CPU PyTorch wheel and `python -m pip install -e ".[dev,demo]"`, then record `python -m pip freeze`; platform-specific reproducibility remains to be verified. No Hugging Face credentials are needed for the pinned public dataset used by the quickstart. A network connection is required for installation and the first data download.

Quickstart downloads three checksum-pinned, licensed NAIP scenes (5,627,736 bytes), prepares RGB synthetic pairs, trains a 100-step CPU teaching checkpoint, evaluates validation only, writes a georeferenced estimate and checks SQLite integrity. The resulting `data/submission-pilot/demo.json` gives relative artifact paths and checkpoint/output SHA-256 values. This is a training fallback, not a downloaded final research model. Checkpoint bytes can differ across package versions; the newly generated manifest records the actual hash. Copy the entire bundle when moving it. Legacy experiment database rows may retain old absolute paths; prepared manifests and new fresh-inference records support relocation.

The launcher verifies the checkpoint hash and only starts inference. It does not redownload imagery or retrain. Keep `--data-root` on persistent storage for server-side satellite outputs and SQLite. Uploaded user images use temporary Gradio cache, with 60-second expiry. Configure host and port with launcher arguments or RSSR_HOST/RSSR_PORT. For multiple operator-controlled RGB checkpoints, RSSR_MODELS names a JSON object mapping model labels to paths relative to that JSON file. Uploaded model files are never accepted. RSSR_SAMPLES can name a list of licensed sample records with `path`, acquisition date and attribution; sample paths are relative to the sample JSON file.

Examiner sequence: open the printed local address; upload an opaque uint8 RGB image; compare bicubic and pilot outputs; download PNG and provenance JSON; verify the shown checkpoint hash against demo.json. In the Sentinel accordion choose a small AOI (at most one degree on each side), a bounded acquisition interval and cloud threshold, then explicitly acquire. That action can download up to 32 MB. Download the georeferenced SR and provenance; compare CRS and extent with the source. Dates represent acquisition, ingestion and inference separately. There is no HR reference for this live image, so no PSNR/SSIM is reported. The NAIP-trained display model is not scientifically validated on Sentinel.

Validation commands:

```bash
python -m pytest -q
python scripts/check_structure.py
python -m ipykernel install --sys-prefix --name python3
python scripts/execute_notebooks.py
```

Restricted Windows environments may require TEMP, TMP, JUPYTER_RUNTIME_DIR, IPYTHONDIR and MPLCONFIGDIR to point inside a writable workspace. Local socket permission is required for Jupyter and Gradio. Do not interpret a socket startup failure as a notebook pass.

Public deployment is not complete. No authenticated free inference host or persistent volume has been provisioned and no public URL is verified. The remaining external prerequisite is access to an authorized free inference host with persistent storage; do not put credentials into notebooks or chat. See [readiness](submission_readiness.md) for remaining engineering, teaching and research work. MP12–MP40 and the final report/presentation package still need substantive completion.
