"""Bounded offline MP23–26 Streamlit smoke and source contract."""
import sys
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
with patch('urllib.request.urlopen',side_effect=AssertionError('Unexpected network')):
    app=AppTest.from_file(str(root/'app/streamlit_app.py'),default_timeout=90).run()
    app.radio(key='section').set_value('Mini-projects').run()
    for ident in ('MP23','MP24','MP25','MP26'):
        app.selectbox(key='onsite_lesson').set_value(ident).run()
        assert not app.exception and not app.success,ident
        app.button(key=ident+'_run').click().run()
        assert not app.exception,ident
        assert app.success,ident
        app.slider(key=ident+'_amplitude').set_value(0.).run()
        assert not app.success
print('MP23-26 seeded synthetic attention controls and stale clearing passed; no network')
