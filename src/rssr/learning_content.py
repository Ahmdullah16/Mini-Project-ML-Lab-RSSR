"""Read existing teaching prose without executing or displaying notebook code."""
import hashlib
import json
import posixpath
import re
from pathlib import Path
from urllib.parse import quote

REPO = 'https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/'


def prose(text, source):
    """Keep mathematical Markdown; omit fenced code and notebook UI directions."""
    text = re.sub(r'```[^\n]*\n.*?```', '', text, flags=re.S)
    text = re.sub(r'~~~[^\n]*\n.*?~~~', '', text, flags=re.S)
    paragraphs = text.split('\n\n')
    text = '\n\n'.join(p for p in paragraphs if not any(
        marker in p for marker in ('**How to use this lesson:**', 'Shift+Enter', 'Restart Kernel')))

    def link(match):
        label, target = match.groups()
        if re.match(r'^(https?://|mailto:|#)', target):
            return match.group(0)
        resolved = posixpath.normpath(posixpath.join(str(Path(source).parent).replace('\\', '/'), target))
        return f'[{label}]({REPO}{quote(resolved, safe="/#")})'

    return re.sub(r'\[([^\]]+)\]\(([^\s)]+)\)', link, text).strip()


def load_lesson(root, ident):
    root = Path(root)
    rows = json.loads((root/'docs/project_folders.json').read_text(encoding='utf-8'))
    row = next((r for r in rows if r['id'] == ident), None)
    if row is None:
        raise ValueError('Unknown project ID')
    notebook = row['notebook']
    readme = row['directory']+'/README.md'
    validation = row['directory']+'/VALIDATION.md'
    doc = json.loads((root/notebook).read_text(encoding='utf-8'))
    cells = [prose(''.join(c['source']), notebook) for c in doc['cells'] if c['cell_type'] == 'markdown']
    sources = [readme, notebook, validation]
    return {'id': ident, 'scope': row['scope'],
            'overview': prose((root/readme).read_text(encoding='utf-8'), readme),
            'explanations': '\n\n'.join(c for c in cells if c),
            'validation': prose((root/validation).read_text(encoding='utf-8'), validation),
            'source_hashes': {p: hashlib.sha256((root/p).read_text(encoding='utf-8').replace('\r\n','\n').encode()).hexdigest() for p in sources},
            'coverage': 'Existing prose migrated; complete parameter/derivation/visual coverage not yet individually accepted.'}
