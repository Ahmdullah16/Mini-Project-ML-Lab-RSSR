from pathlib import Path
import pytest
from rssr.evidence import build_index
from rssr.interpretation import explain_selection,metric_definition,CONTEXT

def test_every_record_has_definition_and_comparable_summary():
    index=build_index(Path(__file__).resolve().parents[1]);groups={}
    for row in index['rows']:
        assert metric_definition(row['metric'])['direction']!='unspecified'
        groups.setdefault(tuple(row[k] for k in CONTEXT),[]).append(row)
    for rows in groups.values():
        result=explain_selection(rows)
        assert result['available'] and result['sources']
        for model in result['summary']:
            original=[r['value'] for r in rows if r['model']==model['model']]
            assert model['value']==pytest.approx(sum(original)/len(original))

def test_filter_context_and_no_silent_pooling():
    rows=build_index(Path(__file__).resolve().parents[1])['rows']
    row=rows[0]
    for key in CONTEXT:
        changed=dict(row);changed[key]='different'
        with pytest.raises(ValueError):explain_selection([row,changed])
    assert not explain_selection([])['available']
    changed=dict(row,value=None)
    assert not explain_selection([changed])['available']
    result=explain_selection([dict(row,model='only-this-model',value=-2)])
    assert 'only-this-model: -2' in result['observation']
    assert metric_definition('rmse_band_3')['title'].endswith('near infrared')
