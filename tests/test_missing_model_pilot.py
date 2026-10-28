import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('missingpilot',ROOT/'scripts/train_missing_models.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_subset_excludes_all_test_rows_and_preserves_development_identity(tmp_path):
    source=tmp_path/'source.json';target=tmp_path/'derived'/'manifest.json'
    doc={'tiles':[{'id':s,'split':s,'path':s+'.npz'} for s in ('train','validation','test')],
         'scenes':[{'split':s} for s in ('train','validation','test')]}
    source.write_text(json.dumps(doc))
    result=module.subset_manifest(source,target)
    assert [r['id'] for r in result['tiles']]==['train','validation']
    assert all(r['split']!='test' for r in result['scenes'])
    assert (target.parent/result['tiles'][0]['path']).resolve()==tmp_path/'train.npz'
    assert json.loads(source.read_text())==doc
    assert module.subset_manifest(source,target)==result
    target.write_text('{}')
    with pytest.raises(ValueError):module.subset_manifest(source,target)


def test_committed_pilot_is_bounded_and_not_budget_matched():
    p=json.loads((ROOT/'configs/missing-model-pilot.json').read_text())
    assert p['architectures']==['cat-v1/M2','cat-v1/M6']
    assert p['steps']==100 and len(p['seeds'])==3
    assert p['allowed_splits']==['train','validation']
    assert 'not parameter/MAC matched' in p['limits']

def test_recorded_pilots_are_distinct_and_complete():
    report=json.loads((ROOT/'reports/missing-model-pilot.json').read_text())
    assert report['test_pixels_accessed'] is False
    rows=report['runs']
    assert {(r['architecture_id'],r['seed']) for r in rows}=={(a,s) for a in ('cat-v1/M2','cat-v1/M6') for s in (17,23,41)}
    assert len({r['checkpoint_sha256'] for r in rows})==6
    for row in rows:
        assert row['reload_output_shape']==[3,32,32]
        assert row['selected_step'] in (50,100)
        assert row['seconds_training']>0
        assert row['within_run_selection']=='minimum validation L1'
        assert row['validation_scene_means']


def test_trained_scale_evidence_preserves_gaps():
    r=json.loads((ROOT/'reports/trained-rgb-capability.json').read_text())
    assert len(r['rows'])==42
    executed=[x for x in r['rows'] if x['status'].startswith('executed')]
    assert len(executed)==24
    assert {x['architecture_id'] for x in executed}=={'cat-v1/M1','cat-v1/M2','cat-v1/M6','cat-v1/M7'}
    for row in executed:
        p=row['provenance']
        assert (p['output_width'],p['output_height'])==(5*row['scale'],3*row['scale'])
        assert p['runtime_seconds']>0 and row['sampled_process_rss_max_bytes']>0
    assert r['maximum_request_preflight_only']['output_width']==65536
