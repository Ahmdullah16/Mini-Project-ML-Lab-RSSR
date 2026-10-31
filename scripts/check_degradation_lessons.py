"""One batched AppTest for MP06–08; no training or frozen-artifact regeneration."""
from pathlib import Path
import re
import sys
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
with patch('urllib.request.urlopen',side_effect=AssertionError('Teaching navigation used network')):
    app=AppTest.from_file(str(root/'app/streamlit_app.py'),default_timeout=60).run()
    app.radio(key='section').set_value('Mini-projects').run()
    app.selectbox(key='onsite_lesson').set_value('MP06').run()
    assert not app.success
    for scale,shape in [(2,'24 × 16'),(4,'12 × 8')]:
        app.selectbox(key='mp06_scale').set_value(scale).run()
        app.button(key='mp06_run').click().run()
        assert not app.exception
        assert any(f'Point mean=0; area mean=0.5; output={shape}' in w.value for w in app.markdown)
    app.selectbox(key='onsite_lesson').set_value('MP07').run()
    app.slider(key='mp07_noise').set_value(0.).run()
    app.button(key='mp07_run').click().run()
    assert not app.exception
    assert any('Noise-after MSE=0.00000000; noise-before MSE=0.00000000' in w.value for w in app.markdown)
    app.slider(key='mp07_noise').set_value(.1).run()
    assert not app.success
    app.button(key='mp07_run').click().run()
    original=next(w.value for w in app.markdown if 'Noise-after MSE=' in w.value)
    after,before=map(float,re.findall(r'MSE=([\d.]+)',original))
    assert .006<after<.014 and .001<before<.004
    app.button(key='mp07_run').click().run()
    assert original in [w.value for w in app.markdown]
    app.number_input(key='mp07_seed').set_value(18).run()
    app.button(key='mp07_run').click().run()
    assert original not in [w.value for w in app.markdown]
    app.selectbox(key='onsite_lesson').set_value('MP08').run()
    app.button(key='mp08_run').click().run()
    assert not app.exception
    line=next(w.value for w in app.markdown if 'Inverse max error=' in w.value)
    values=list(map(float,re.findall(r'=([0-9]+(?:\.[0-9]+)?(?:e[+\-]?[0-9]+)?)',line)))
    assert values[0]<1e-12 and values[1]<1e-9 and abs(values[2]-.5)<1e-12 and values[3]<1e-12
    app.slider(key='mp08_cutoff').set_value(.2).run()
    assert not app.success
    app.button(key='mp08_run').click().run()
    line=next(w.value for w in app.markdown if 'Inverse max error=' in w.value)
    assert 'filtered standard deviation=0.282843' in line
    assert not app.exception
print('MP06–08 batched AppTest passed: exact sampling shapes, seeded noise/JPEG, FFT roundtrip/Parseval/cutoff, stale clearing; no network')
