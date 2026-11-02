"""Presentation contract: guest access, theme/navigation and no implicit network."""
from pathlib import Path
import sys
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected network')):
    app = AppTest.from_file(str(ROOT/'app/streamlit_app.py'), default_timeout=60).run()
    assert not app.exception, app.exception
    assert any('Understand every pixel' in m.value for m in app.markdown)
    assert any('About the researcher' in m.value and 'Ahmdullah' in m.value for m in app.markdown)
    assert len(app.radio(key='section').options) == 7
    assert any(m.label == 'Configured scales' for m in app.metric)
    app.selectbox(key='lab_appearance').set_value('Dark').run()
    assert not app.exception
    assert any('--lab-bg:#0B1220' in m.value for m in app.markdown)
    app.button(key='overview-route-Mini-projects').click().run()
    assert app.radio(key='section').value == 'Mini-projects'
    assert not app.exception
    app.radio(key='section').set_value('Overview').run()
    assert app.selectbox(key='lab_appearance').value == 'Dark'
    app.selectbox(key='lab_appearance').set_value('Light').run()
    assert not app.exception
    assert any('--lab-bg:#F5F8FC' in m.value for m in app.markdown)
    app.button[0].click().run()
    assert app.radio(key='section').value == 'Live SR'
    assert not app.exception
print('Guest overview, evidence cards, both appearances and existing navigation verified offline')
