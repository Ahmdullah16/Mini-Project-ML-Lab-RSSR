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
    modules = ['app/satellite_scale_ui.py', 'src/rssr/inference.py', 'src/rssr/fresh.py', 'app/scale_ui.py', 'src/rssr/scaling.py', 'src/rssr/scale_job.py', 'app/showcase.py', 'app/inference_ui.py', 'app/compare_ui.py', 'src/rssr/evidence.py', 'src/rssr/interpretation.py', 'src/rssr/comparison.py']
    return {'git_commit':commit,'entrypoint_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'module_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                             for name in modules if (root/name).is_file()}}
