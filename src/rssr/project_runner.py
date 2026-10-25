"""Execute a canonical project-folder notebook in its own directory."""
import hashlib
import json
import sys
import asyncio
from datetime import datetime,timezone
from pathlib import Path


def run_project(root,ident):
    import nbformat
    from nbclient import NotebookClient
    root=Path(root)
    if sys.platform=='win32':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    entry=next(p for p in json.loads((root/'docs/project_folders.json').read_text()) if p['id']==ident)
    source=root/entry['notebook'];notebook=nbformat.read(source,as_version=4);nbformat.validate(notebook)
    report=root/'reports/project-notebook-execution.json'
    records=json.loads(report.read_text()) if report.exists() else []
    records=[r for r in records if r['id']!=ident]
    record={'id':ident,'source':entry['notebook'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'started_at':datetime.now(timezone.utc).isoformat()}
    output=root/'data/executed_notebooks/project_folders';output.mkdir(parents=True,exist_ok=True)
    try:
        NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(source.parent)}}).execute()
    except Exception as error:
        record.update(status='failed',error=str(error));records.append(record)
        report.write_text(json.dumps(records,indent=2)+'\n');nbformat.write(notebook,output/f'{ident}.failed.ipynb');raise
    nbformat.write(notebook,output/f'{ident}.executed.ipynb')
    record.update(status='passed-real-jupyter-kernel',code_cells=sum(c.cell_type=='code' for c in notebook.cells),finished_at=datetime.now(timezone.utc).isoformat())
    records.append(record);report.write_text(json.dumps(records,indent=2)+'\n')
    print(ident,'project-folder kernel passed',flush=True)
