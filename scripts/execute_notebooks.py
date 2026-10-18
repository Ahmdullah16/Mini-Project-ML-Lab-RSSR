"""Execute real Jupyter kernels; failures stop the run. No exec fallback here."""
import json
import hashlib
import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
import nbformat
from nbclient import NotebookClient

if sys.platform=='win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

root=Path(__file__).resolve().parents[1]
out=root/'data'/'executed_notebooks'; out.mkdir(parents=True,exist_ok=True)
paths=[root/'stage_01_python_numpy/MP01_digital_image_explorer/digital_image_explorer.ipynb',*sorted((root/'notebooks').glob('MP*.ipynb'))]
parser=argparse.ArgumentParser()
parser.add_argument('--only',help='Notebook stem, e.g. MP02; preserve other execution records')
args=parser.parse_args()
if args.only:
    paths=[p for p in paths if p.stem==args.only]
    if not paths: parser.error('Unknown notebook stem')
evidence=[]
report=root/'reports/notebook-execution.json'
report.parent.mkdir(parents=True,exist_ok=True)
if args.only and report.exists():
    evidence=[r for r in json.loads(report.read_text()) if r['source'] not in {str(p.relative_to(root)) for p in paths}]
for path in paths:
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    record={'source':str(path.relative_to(root)), 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'started_at':datetime.now(timezone.utc).isoformat()}
    try:
        NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(path.parent)}}).execute()
    except Exception as error:
        record.update(status='failed',error=str(error))
        evidence.append(record)
        report.write_text(json.dumps(evidence,indent=2))
        nbformat.write(notebook,out/(path.stem+'.failed.ipynb'))
        raise
    target=out/(path.stem+'.executed.ipynb'); nbformat.write(notebook,target)
    record.update(executed=str(target.relative_to(root)),code_cells=sum(c.cell_type=='code' for c in notebook.cells),status='passed-real-jupyter-kernel',finished_at=datetime.now(timezone.utc).isoformat())
    evidence.append(record)
    report.write_text(json.dumps(evidence,indent=2))
    print(path.name,'passed',flush=True)
(root/'reports/notebook-execution.json').write_text(json.dumps(evidence,indent=2))
