"""Catalogue navigation/filtering must not download imagery or access frozen pixels."""
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
with patch('urllib.request.urlopen',side_effect=AssertionError('Unexpected network')):
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=60).run()
    for route in ['Overview','Live SR','Satellite Explorer']:
        app.radio(key='section').set_value(route).run()
        key='catalogue-'+route
        assert not app.exception
        app.selectbox(key=key+'split').set_value('train').run()
        app.text_input(key=key+'query').set_value('no-such-scene').run()
        assert any('No public samples match' in x.value for x in app.info)
        assert not app.exception
print('Public catalogue renders on three routes and filters offline; no imagery/frozen pixels loaded')
