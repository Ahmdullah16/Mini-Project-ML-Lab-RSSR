"""Execute the real published models; verify cache identity and selection changes."""
import json,sys,time
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=60).run()
app.radio(key='section').set_value('Live SR').run()
app.radio(key='inference_workflow').set_value('Compare models').run()
app.multiselect(key='compare_models').set_value([]).run()
app.button[0].click().run()
assert any('Select at least one' in e.value for e in app.error)
catalog=[json.loads((ROOT/'configs/hosted-model.json').read_text(encoding='utf-8'))]+json.loads((ROOT/'configs/hosted-models.json').read_text(encoding='utf-8'))['models']
app.multiselect(key='compare_models').set_value([s['label'] for s in catalog]).run()
assert app.multiselect(key='compare_models').value==[s['label'] for s in catalog]
start=time.perf_counter();app.button[0].click().run(timeout=180)
assert not app.exception and not app.error,([e.value for e in app.exception],[e.value for e in app.error])
batch=app.session_state['comparison'];assert len(batch['results'])==3 and not batch['errors']
for result,spec in zip(batch['results'],catalog):
    assert result['sr'].shape==(32,32,3)
    assert result['metadata']['checkpoint_sha256']==spec['checkpoint']['sha256']
    assert result['metadata']['model_name']==spec['label']
keys=[r['key'] for r in batch['results']]
times=[r['metadata']['computed_at'] for r in batch['results']]
app.multiselect(key='compare_models').set_value([catalog[-1]['label']]).run()
assert [r['key'] for r in app.session_state['comparison']['results']]==keys
with patch('rssr.display.predict_rgb',side_effect=AssertionError('Cached identity should not recompute')):
    app.button[0].click().run(timeout=60)
    assert not app.exception and not app.session_state['comparison']['errors']
assert app.session_state['comparison']['results'][0]['metadata']['computed_at']==times[-1]
from rssr.display import predict_rgb as actual_predict
def fail_one(image,model,state,sha,**kwargs):
    if sha==catalog[0]['checkpoint']['sha256']:raise ValueError('Controlled model failure')
    return actual_predict(image,model,state,sha,**kwargs)
app.multiselect(key='compare_models').set_value([catalog[0]['label'],catalog[-1]['label']]).run()
with patch('rssr.display.predict_rgb',side_effect=fail_one):
    app.button[0].click().run(timeout=60)
    partial=app.session_state['comparison']
    assert len(partial['results'])==1 and partial['results'][0]['metadata']['model_name']==catalog[-1]['label']
    assert len(partial['errors'])==1 and 'Controlled model failure' in partial['errors'][0]['error']
app.button[0].click().run(timeout=60)
assert len(app.session_state['comparison']['results'])==2 and not app.session_state['comparison']['errors']
report={'scope':'Local actual published RGB sample comparison; not public deployment or quality-gain evaluation',
        'models':[r['metadata'] for r in batch['results']],'total_check_seconds':time.perf_counter()-start,
        'selection_change_preserves_labels':True,'matching_identity_cache_reused':True,'partial_failure_and_retry_passed':True}
(ROOT/'reports/comparison-validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('All 3 real models, aligned outputs, selection labels and identity cache passed')
