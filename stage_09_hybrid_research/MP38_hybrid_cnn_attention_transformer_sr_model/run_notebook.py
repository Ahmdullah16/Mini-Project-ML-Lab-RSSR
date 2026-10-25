from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))

from rssr.project_runner import run_project

if __name__=='__main__':
    run_project(ROOT,'MP38')
