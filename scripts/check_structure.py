import json,re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
syllabus=json.loads((root/'docs/syllabus.json').read_text(encoding='utf-8'))
assert [x['id'] for x in syllabus]==[f'MP{i:02d}' for i in range(1,41)]
broken=[]; checked=0
for file in root.rglob('*.md'):
    if any(part in ('data','.git','.venv') for part in file.relative_to(root).parts): continue
    for link in re.findall(r'\]\(([^)]+)\)',file.read_text(encoding='utf-8')):
        link=link.split('#')[0]
        if not link or '://' in link or link.startswith('mailto:'): continue
        checked+=1
        if not (file.parent/link).exists(): broken.append((str(file.relative_to(root)),link))
assert not broken,broken
print('40 unique sequential IDs;',checked,'relative Markdown links valid')
