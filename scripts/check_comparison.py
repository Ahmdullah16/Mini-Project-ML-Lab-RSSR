"""Actual published model comparisons through the exact-scale Streamlit workflow."""
import json,sys,time
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=60).run()
app.radio(key='section').set_value('Live SR').run()
app.radio(key='inference_workflow').set_value('Compare models').run()
key='scaled_compare'
assert len(app.selectbox(key=key+'scale').options)==6
app.button(key=key+'prepare').click().run()
app.selectbox(key=key+'scale').set_value(4).run()
app.button(key=key+'run').click().run(timeout=180)
assert not app.exception and not app.error,([x.value for x in app.exception],[x.value for x in app.error])
batch=app.session_state[key+'results'];assert len(batch['outputs'])==3
for result in batch['outputs']:
    assert result['pixels'].shape==(64,64,3)
    assert result['metadata']['scale']==4 and len(result['metadata']['stage_outputs'])==2
keys=[r['key'] for r in batch['outputs']]
with patch('scale_ui.cascade',side_effect=AssertionError('Cache miss')):
    app.button(key=key+'run').click().run()
    assert not app.session_state[key+'results']['errors']
app.selectbox(key=key+'scale').set_value(64).run()
assert [r['key'] for r in app.session_state[key+'results']['outputs']]==keys
# The prepared 16x16 sample at x64 is exactly the configured limit.
assert not app.button(key=key+'run').disabled
import numpy as np
app.session_state[key+'prepared']={'image':np.zeros((17,1024,3),np.uint8),'source':{'kind':'synthetic resource fixture'}}
app.run();assert app.button(key=key+'run').disabled and app.warning
app.selectbox(key=key+'scale').set_value(2).run()
app.session_state[key+'prepared']={'image':np.zeros((3,5,3),np.uint8),'source':{'kind':'synthetic failure fixture'}}
app.run()
from rssr.scaling import cascade as original
labels=app.multiselect(key=key+'models').value

def fail_one(image,model,directory,scale,identity):
    if identity['model']['label']==labels[0]:raise ValueError('Controlled model failure')
    return original(image,model,directory,scale,identity)
with patch('scale_ui.cascade',side_effect=fail_one):
    app.button(key=key+'run').click().run(timeout=120)
    partial=app.session_state[key+'results'];assert len(partial['outputs'])==2 and len(partial['errors'])==1
app.button(key=key+'run').click().run(timeout=120)
assert len(app.session_state[key+'results']['outputs'])==3 and not app.session_state[key+'results']['errors']
(ROOT/'reports/scale-comparison-validation.json').write_text(json.dumps({'scope':'Local actual published RGB comparison; not public deployment or HR-quality evaluation','models':[r['metadata'] for r in batch['outputs']],'cache_reused':True,'selection_keeps_labels':True,'host_rejection':True,'partial_failure_retry':True},indent=2),encoding='utf-8')
print('Three actual models x4, cache, exact dimensions, host rejection and partial failure/retry passed')
