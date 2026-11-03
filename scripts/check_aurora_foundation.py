"""Offline native-shell checks; no hosted/browser acceptance inferred."""
import os
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected network')):
    app = AppTest.from_file(str(ROOT/'app/streamlit_app.py'), default_timeout=60).run()
    assert not app.exception
    assert len(app.radio(key='section').options) == 7
    app.selectbox(key='aurora_lens').set_value('Research').run()
    assert not app.exception
    assert any('reference eligibility' in c.value for c in app.caption)
    app.selectbox(key='lab_appearance').set_value('Dark').run()
    app.checkbox(key='aurora_reduce_motion').uncheck().run()
    assert not app.exception
    assert any('translateY' in m.value for m in app.markdown)
    app.query_params['lite'] = '1'
    app.run()
    assert not any('translateY' in m.value for m in app.markdown)
    app.selectbox(key='aurora_quick_action').set_value('Mini-projects').run()
    assert app.radio(key='section').value == 'Mini-projects' and not app.exception
    assert app.selectbox(key='aurora_lens').value == 'Research'
    assert len(app.selectbox(key='onsite_lesson').options) == 40
    disabled = AppTest.from_file(str(ROOT/'app/aurora_capability_lab.py')).run()
    assert not disabled.exception and any('disabled' in i.value for i in disabled.info)
    with patch.dict(os.environ, RSSR_DEV_CAPABILITIES='1'):
        probe = AppTest.from_file(str(ROOT/'app/aurora_capability_lab.py')).run()
        assert not probe.exception
        assert probe.pills[0].value == 'A'
print('VERIFIED: offline lenses, seven routes, all40 selection, scoped themes, lite/reduced motion and gated capability probes')
