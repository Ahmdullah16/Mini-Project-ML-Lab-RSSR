"""All forty native lessons render without notebook execution or network."""
from pathlib import Path
import sys
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
with patch('urllib.request.urlopen',side_effect=AssertionError('Network during teaching navigation')):
    app=AppTest.from_file(str(root/'app/streamlit_app.py'),default_timeout=60).run()
    app.radio(key='section').set_value('Mini-projects').run()
    for i in range(1,41):
        ident=f'MP{i:02d}'
        app.selectbox(key='onsite_lesson').set_value(ident).run()
        assert not app.exception and not app.error,ident
        assert not app.code,ident
        assert not any('github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR' in w.value or '.ipynb)' in w.value for w in app.markdown),ident
        assert any('not yet individually accepted' in w.value for w in app.info),ident
    app.selectbox(key='onsite_lesson').set_value('MP01').run()
    app.slider(key='mp01_brightness').set_value(20).run()
    assert any('clipped [255, 20, 20]' in w.value for w in app.markdown)
    assert any('Histogram count sum=6144' in w.value for w in app.caption)
    app.slider(key='mp01_brightness').set_value(-255).run()
    assert any('clipped [0, 0, 0]' in w.value for w in app.markdown)
    app.slider(key='teaching_width').set_value(256)
    app.slider(key='teaching_height').set_value(128)
    app.selectbox(key='teaching_scale').set_value(8).run()
    assert any('2048 × 1024; 2,097,152' in w.value for w in app.markdown)
    app.slider(key='teaching_width').set_value(1024)
    app.slider(key='teaching_height').set_value(1024)
    app.selectbox(key='teaching_scale').set_value(64).run()
    assert any('65536 × 65536; 4,294,967,296' in w.value for w in app.markdown)
print('40 lesson selections and exact rectangular/maximum teaching arithmetic passed; no code cells/network')
