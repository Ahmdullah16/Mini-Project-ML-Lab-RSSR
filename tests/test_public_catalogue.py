import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from public_catalogue import records,select

def test_manifest_preserves_splits_hashes_and_unknown_geometry():
    import json
    canonical=json.loads((ROOT/'configs/naip-development-manifest.json').read_text())
    rows=records(ROOT)
    assert len(rows)==len(canonical['scenes'])
    for row,scene in zip(rows,canonical['scenes']):
        assert row['sha256']==scene['sha256'] and row['split']==scene['split']
        assert row['bbox'] is None and row['crs'] is None
    assert all(r['split']=='train' for r in select(rows,split='train'))
    assert not select(rows,'Taj Mahal')
    assert not select(rows,'no-such-scene')
