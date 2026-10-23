"""Community Cloud entry point; showcase keeps inference lazy."""
from pathlib import Path
import sys
import runpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
# Execute the renderer on each Streamlit rerun, without caching its UI as an import.
runpy.run_path(str(Path(__file__).resolve().parent/'showcase.py'),run_name='__main__')
