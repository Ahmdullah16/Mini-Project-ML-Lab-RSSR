"""Recorded model explorer navigation must not execute a model or fetch data."""
from pathlib import Path
import sys
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
with patch('urllib.request.urlopen',side_effect=AssertionError('Network on model navigation')):
    app=AppTest.from_file(str(root/'app/streamlit_app.py'),default_timeout=60).run()
    app.radio(key='section').set_value('Methodology & Data').run()
    for key in ['M1','M2','M3','M4','M5','M6','M7']:
        app.selectbox(key='registry_architecture').set_value(key).run()
        assert not app.exception and not app.error,key
        labels=[s.key for s in app.selectbox]
        assert 'registry_checkpoint_'+key in labels
        if key in ('M2','M6'):
            assert any('Development-only pilot' in item.value for item in app.warning)
        assert any('4096' in item.value for item in app.info)
print('Seven model selections passed; bounded pilot records distinguished from historical bindings; no network')
