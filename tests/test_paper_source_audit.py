import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paper_audit',ROOT/'scripts/audit_paper_sources.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_measured_rows_match_original_with_explicit_units_and_limits():
    result=module.audit(ROOT)
    source=json.loads((ROOT/'reports/final-heldout.json').read_text())
    assert len(result['measured'])==len(source['summary'])
    for item,row in zip(result['measured'],source['summary']):
        assert item['bands']==row['bands'] and item['model']==row['model']
        assert all(row[k]==v for k,v in item['metrics'].items())
        assert item['scenes']==1 and 'no geographic CI' in item['limitation']
    assert {x['id']:x['historical_bindings'] for x in result['models']}=={
        'cat-v1/M1':9,'cat-v1/M2':0,'cat-v1/M3':3,'cat-v1/M4':6,
        'cat-v1/M5':6,'cat-v1/M6':0,'cat-v1/M7':15}


@pytest.mark.parametrize('failure',['citation','protocol'])
def test_audit_rejects_unresolved_citation_or_protocol_drift(monkeypatch,failure):
    original=module.read
    def altered(root,path):
        text=original(root,path)
        if failure=='citation' and path=='paper/main.tex':text+=r'\cite{nonexistent_key}'
        if failure=='protocol' and path=='reports/paper-evidence.json':
            doc=json.loads(text);doc['frozen_protocol_hash']='changed';text=json.dumps(doc)
        return text
    monkeypatch.setattr(module,'read',altered)
    with pytest.raises(ValueError):module.audit(ROOT)
