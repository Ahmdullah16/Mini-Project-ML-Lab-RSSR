import json
from pathlib import Path
import pytest
from rssr.evidence import build_index

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def index():return build_index(ROOT)

def test_frozen_metrics_are_exact_and_partitioned(index):
    report=json.loads((ROOT/'reports/final-heldout.json').read_text())
    rows=[r for r in index['rows'] if r['study']=='Frozen held-out evaluation']
    for original in report['summary']:
        for metric in ['psnr','ssim','sam_rad']:
            matched=[r for r in rows if r['bands']==original['bands'] and r['model']==original['model'] and r['metric']==metric]
            assert len(matched)==1
            assert matched[0]['value']==original[metric]
            assert matched[0]['split']=='test' and matched[0]['scenes']==1
            assert matched[0]['tiles']==16 and matched[0]['protocol']==report['protocol_hash']
    assert all(r['split']!='test' for r in index['rows'] if r['study']!='Frozen held-out evaluation')

def test_sources_projects_and_negative_results_preserved(index):
    assert len(index['projects'])==40
    assert len({p['stage'] for p in index['projects']})==9
    for ident in ['MP27','MP28','MP29','MP30','MP31']:
        assert next(p for p in index['projects'] if p['id']==ident)['status'].startswith('Complete implementation')
    for project in index['projects']:
        assert (ROOT/project['brief']).is_file() and (ROOT/project['notebook']).is_file()
        assert Path(project['notebook']).parent == Path(project['brief']).parent
    negative=[r for r in index['rows'] if r['metric']=='delta_psnr' and r['value']<0]
    assert negative and any(r['model']=='srcnn_educational' for r in negative)
    for row in index['rows']:
        assert (ROOT/row['source']).is_file()
        assert row['unit']!='declared in source'
    assert all(a['bytes']>0 and len(a['sha256'])==64 for a in index['artifacts'])
    json.dumps(index,allow_nan=False)


def test_latest_lesson_scopes_and_nested_figures(index):
    projects={p['id']:p for p in index['projects']}
    assert projects['MP35']['status'].startswith('Complete implementation')
    assert projects['MP37']['status'].startswith('Complete implementation')
    assert 'mastery unassessed' in projects['MP37']['status']
    for ident in ['MP36','MP38','MP39','MP40']:
        assert projects[ident]['status'].startswith('Expanded')
        assert 'pending' in projects[ident]['status']
        assert projects[ident]['scope_source']=='docs/project_folders.json'
    assert projects['MP20']['exception']=='Independent held-out SRCNN table pending'
    assert any(a['path']=='reports/figures/mp40-robustness.svg' for a in index['artifacts'])
