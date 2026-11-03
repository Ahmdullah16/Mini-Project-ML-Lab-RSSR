"""Community Cloud entry point; showcase keeps inference lazy."""
from pathlib import Path
import sys
import runpy
import importlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
import deployment_bootstrap
# The refresh policy itself can change during a warm source deployment.
importlib.reload(deployment_bootstrap).prepare_imports(Path(__file__).resolve().parents[1])
# Execute the renderer on each Streamlit rerun, without caching its UI as an import.
runpy.run_path(str(Path(__file__).resolve().parent/'showcase.py'),run_name='__main__')
