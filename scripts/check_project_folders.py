"""Validate MP01-style deliverables without asserting scientific completion."""
import ast
import json
import re
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
projects = json.loads((ROOT / 'docs/project_folders.json').read_text())
assert [p['id'] for p in projects] == [f'MP{i:02d}' for i in range(1, 41)]
for project in projects:
    folder = ROOT / project['directory']
    for name in ['README.md', 'VALIDATION.md', 'run_notebook.py', project['tools'], *project['tests']]:
        path = folder / name
        assert path.is_file(), path
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    source = ROOT / project['notebook']
    assert source.parent == folder
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            for link in re.findall(r'\]\(([^)]+)\)', cell.source):
                link = link.split('#')[0]
                if link and '://' not in link and not link.startswith('mailto:'):
                    assert (folder / link).exists(), (source, link)
print('All 40 project folders contain README, validation, notebook, tools, runner and tests.')
