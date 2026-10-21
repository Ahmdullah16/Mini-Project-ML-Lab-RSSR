"""Report source identity without requiring credentials or a network call."""
import hashlib
import subprocess
from pathlib import Path


def build_identity(root):
    root=Path(root)
    try:
        result=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,timeout=3,check=True)
        commit=result.stdout.strip()
        if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):commit='unavailable'
    except (OSError,subprocess.SubprocessError):commit='unavailable'
    source=root/'app/streamlit_app.py'
    return {'git_commit':commit,'entrypoint_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
