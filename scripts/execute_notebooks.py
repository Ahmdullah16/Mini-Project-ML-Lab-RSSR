"""Execute real Jupyter kernels; failures stop the run. No exec fallback here."""
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
out=root/'data'/'executed_notebooks'; out.mkdir(parents=True,exist_ok=True)
paths=[root/'stage_01_python_numpy/MP01_digital_image_explorer/digital_image_explorer.ipynb',*sorted((root/'notebooks').glob('MP*.ipynb'))]
evidence=[]
for path in paths:
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(path.parent)}}).execute()
    target=out/(path.stem+'.executed.ipynb'); nbformat.write(notebook,target)
    evidence.append({'source':str(path.relative_to(root)),'executed':str(target.relative_to(root)),'code_cells':sum(c.cell_type=='code' for c in notebook.cells),'status':'passed-real-jupyter-kernel'})
    print(path.name,'passed',flush=True)
(root/'reports/notebook-execution.json').write_text(json.dumps(evidence,indent=2))
