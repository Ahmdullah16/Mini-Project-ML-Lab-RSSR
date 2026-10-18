# Local and Colab setup

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,demo]"
python -m pytest
```

Use the official PyTorch selector for a GPU environment. The initial local pilot
uses CPU, float32 and two threads. No batch/patch changes happen silently. Reduce
batch size explicitly in a new configuration if memory is insufficient and record
the changed configuration. Never call that run directly comparable without disclosure.

For Jupyter, install `jupyterlab`, run `python -m jupyterlab`, open notebooks/MP02.ipynb,
and run cells in order. MP01 remains in its original stage folder to preserve its
helper imports. Notebook execution evidence is recorded separately from mastery.

In Colab, clone the implementation target and install it:

```python
!git clone https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR.git
%cd /content/Mini-Project-ML-Lab-RSSR
!pip install -e '.[dev,demo]'
```

The initial pilot PR #1 was merged. Continuation work is on `feat/notebook-demo-validation`; check out that branch until its PR is merged.
Save `--root` to a persistent mounted location for training. Downloads, prepared
manifests, SQLite and checkpoints must all be copied together; prepared manifests
currently contain absolute local tile paths, so rerun preparation after relocation.
Resume with `rssr train CONFIG MANIFEST --root DATA --resume DATA/runs/ID/last.pt`.
Use `best.pt` for inference, never resume. Best validation L1 selects the checkpoint.

Restricted Windows environments may need TEMP and TMP set to a writable workspace
folder before invoking Python. The measured environment used a venv with read-only
access to bundled NumPy/Pillow and isolated Matplotlib plus isolated installed dependencies;
the lock records resolved versions, rather than claiming a pristine base install.
