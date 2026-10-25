"""Run project-folder notebooks without regenerating research artifacts."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rssr.project_runner import run_project

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', help='Project IDs; defaults to all 40')
    args = parser.parse_args()
    projects = json.loads((ROOT / 'docs/project_folders.json').read_text())
    ids = [p['id'] for p in projects]
    selected = args.only or ids
    if set(selected) - set(ids):
        parser.error('Unknown project ID')
    for ident in selected:
        run_project(ROOT, ident)
