"""Actual bounded checkpoint inference; does not rewrite historical reports."""
import hashlib
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import rasterio
from rasterio.transform import from_origin
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'app')]
from rssr_ui.passport import parse_passport
import json

app = AppTest.from_file(str(ROOT/'app/streamlit_app.py'), default_timeout=60).run()
app.radio(key='section').set_value('Live SR').run()
image = np.full((3, 5, 3), 80, np.uint8)
app.session_state['scaled_liveprepared'] = {'image': image, 'source': {'kind': 'synthetic passport fixture'}}
app.run()
app.button(key='scaled_liverun').click().run(timeout=120)
assert not app.exception and not app.error, (app.exception, app.error)
result = app.session_state['scaled_liveresults']['outputs'][0]
passport = result['passport']
assert result['pixels'].shape == (6, 10, 3)
assert passport['input_sha256'] == hashlib.sha256(image.tobytes()).hexdigest()
assert passport['output_sha256'] == hashlib.sha256(result['tiff']).hexdigest()
assert parse_passport(json.dumps(passport).encode())['scale'] == 2
with patch('scale_ui.cascade', side_effect=AssertionError('Unexpected rerun')):
    app.selectbox(key='aurora_lens').set_value('Research').run()
    app.button(key='scaled_liverun').click().run()
    assert app.session_state['scaled_liveresults']['outputs'][0]['passport'] == passport

with rasterio.io.MemoryFile() as mem:
    with mem.open(driver='GTiff',width=5,height=3,count=3,dtype='uint8',
                  crs='EPSG:32632',transform=from_origin(100,200,10,10)) as ds:
        ds.write(image.transpose(2,0,1))
    source = mem.read()
app.radio(key='section').set_value('Satellite Explorer').run()
app.session_state['fresh_prepared'] = {'geotiff':source, 'metadata':{'fixture':True}}
app.radio(key='satellite_workflow').set_value('Compare models').run()
selected = app.multiselect(key='sat_compare_models').value[:1]
app.multiselect(key='sat_compare_models').set_value(selected).run()
app.button(key='sat_compare_run').click().run(timeout=120)
assert not app.exception and not app.error, (app.exception, app.error)
result = next(iter(app.session_state['sat_compare_results'].values()))
assert result['passport']['input_sha256'] == hashlib.sha256(source).hexdigest()
assert result['passport']['output_sha256'] == hashlib.sha256(result['geotiff']).hexdigest()
print('VERIFIED: actual CNN 5x3→10x6 RGB and GeoTIFF inference; passport hashes, import schema and rerun stability; synthetic inputs, no acquisition/HR metrics')
