import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from public_catalogue import records,select

def test_metadata_helpers_import_without_streamlit():
    import subprocess
    code = '''
import importlib.abc, sys
from pathlib import Path
class NoStreamlit(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'streamlit' or fullname.startswith('streamlit.'):
            raise ModuleNotFoundError('Streamlit intentionally unavailable')
sys.meta_path.insert(0, NoStreamlit())
sys.path.insert(0, str(Path.cwd()/'app'))
from public_catalogue import records, select
assert records(Path.cwd())
assert select(records(Path.cwd()), split='train')
assert 'streamlit' not in sys.modules
'''
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True,timeout=30)

def test_manifest_preserves_splits_hashes_and_unknown_geometry():
    import json
    canonical=json.loads((ROOT/'configs/naip-development-manifest.json').read_text())
    rows=records(ROOT)
    assert len(rows)==len(canonical['scenes'])
    for row,scene in zip(rows,canonical['scenes']):
        assert row['sha256']==scene['sha256'] and row['split']==scene['split']
        if row['region']=='ROI_00006':
            assert row['crs']=='EPSG:32611' and row['pixel_spacing']==[2.5,2.5]
            assert 'Full canonical HR' in row['geometry_status']
        else:
            assert row['bbox'] is None and row['crs'] is None
    assert all(r['split']=='train' for r in select(rows,split='train'))
    assert not select(rows,'Taj Mahal')
    assert not select(rows,'no-such-scene')
