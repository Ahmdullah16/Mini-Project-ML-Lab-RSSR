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
    for project in index['projects']:
        assert (ROOT/project['brief']).is_file() and (ROOT/project['notebook']).is_file()
    negative=[r for r in index['rows'] if r['metric']=='delta_psnr' and r['value']<0]
    assert negative and any(r['model']=='srcnn_educational' for r in negative)
    for row in index['rows']:
        assert (ROOT/row['source']).is_file()
        assert row['unit']!='declared in source'
    assert all(a['bytes']>0 and len(a['sha256'])==64 for a in index['artifacts'])
    json.dumps(index,allow_nan=False)
