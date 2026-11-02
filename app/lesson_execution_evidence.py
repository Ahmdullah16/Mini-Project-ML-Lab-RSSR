"""Read pinned teaching-check evidence without upgrading scientific acceptance."""
import hashlib
import json
from pathlib import Path


def execution_status(root,ident):
    root=Path(root)
    try:
        record=json.loads((root/'reports/native-lesson-execution-index.json').read_text(encoding='utf-8'))
        row=next(x for x in record['lessons'] if x['id']==ident)
        if record['verification']['conclusion']!='success' or not all(
                p in row['source_sha256'] for p in (row['module'],row['check'])):
            return 'Native-example verification record incomplete; no pass is inferred.'
        paths=dict(record['shared_source_sha256'],**row['source_sha256'])
        if any(hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=sha for p,sha in paths.items()):
            return 'Native example changed since the recorded check; fresh verification is required.'
        check=record['verification']
        return (f"Native example: recorded CI check passed on {check['date']} "
                f"(run {check['run_id']}, revision {check['head'][:8]}). "
                'This verifies the teaching interaction, not learner mastery or scientific project completion.')
    except (OSError,ValueError,KeyError,StopIteration,TypeError):
        return 'Native-example verification record unavailable; no pass is inferred.'
