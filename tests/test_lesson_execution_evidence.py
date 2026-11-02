"""Pinned execution evidence must fail closed when lesson sources change."""
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from lesson_execution_evidence import execution_status


def test_all40_have_distinct_valid_execution_bindings():
    record=json.loads((ROOT/'reports/native-lesson-execution-index.json').read_text(encoding='utf-8'))
    assert [r['id'] for r in record['lessons']]==[f'MP{i:02d}' for i in range(1,41)]
    workflow=(ROOT/'.github/workflows/test.yml').read_text(encoding='utf-8')
    for row in record['lessons']:
        assert row['check'] in workflow
        assert 'recorded CI check passed' in execution_status(ROOT,row['id'])
        assert 'not learner mastery' in execution_status(ROOT,row['id'])


def test_missing_and_changed_evidence_not_reported_as_pass(tmp_path):
    assert 'unavailable' in execution_status(tmp_path,'MP01')
    record=json.loads((ROOT/'reports/native-lesson-execution-index.json').read_text(encoding='utf-8'))
    row=record['lessons'][0]
    for path in set(record['shared_source_sha256'])|set(row['source_sha256']):
        dest=tmp_path/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,dest)
    (tmp_path/'reports').mkdir(exist_ok=True)
    (tmp_path/'reports/native-lesson-execution-index.json').write_text(json.dumps(record),encoding='utf-8')
    assert 'passed' in execution_status(tmp_path,'MP01')
    (tmp_path/row['module']).write_text('# changed\n',encoding='utf-8')
    assert 'changed since' in execution_status(tmp_path,'MP01')
