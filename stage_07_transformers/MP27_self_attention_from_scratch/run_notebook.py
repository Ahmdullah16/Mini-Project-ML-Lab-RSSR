"""Run the canonical lesson in a real Jupyter kernel; preserve other records."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(root/'scripts/execute_notebooks.py'), '--only', 'MP27'], cwd=root, check=True)
