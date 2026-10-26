"""Navigation and empty states must not trigger provider/model downloads."""
from pathlib import Path
import sys
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
sections=['Overview','Live SR','Results','Methodology & Data','Mini-projects','Satellite Explorer','Downloads & Reproducibility']
with patch('urllib.request.urlopen',side_effect=AssertionError('Network access on navigation')):
    app=AppTest.from_file(str(root/'app/streamlit_app.py'),default_timeout=60).run()
    app.button[0].click().run()
    assert app.radio(key='section').value=='Live SR'
    for section in sections:
        print('Checking',section,flush=True)
        app.radio(key='section').set_value(section).run(timeout=60)
        assert not app.exception and not app.error,(section,app.exception,app.error)
    app.radio(key='section').set_value('Results').run(timeout=60)
    def choose(label,value):
        next(w for w in app.selectbox if w.label==label).set_value(value).run()
    choose('Study','CPU efficiency')
    choose('Metric','sampled_process_peak_rss_bytes')
    assert any('Not exact allocator peak' in w.value for w in app.markdown)
    choose('Study','Teaching experiments')
    assert any(w.label=='Protocol' for w in app.selectbox)
    choose('Study','Frozen held-out evaluation')
    choose('Metric','sam_rad')
    assert any('norm products' in w.value for w in app.markdown)
    app.text_input[0].set_value('no-such-model-xyz').run()
    assert not app.exception and any('No recorded values' in i.value for i in app.info)
    app.radio(key='result_view').set_value('Learning curves').run()
    assert not app.exception
    app.radio(key='result_view').set_value('Artifact library').run()
    assert not app.exception
    app.radio(key='section').set_value('Live SR').run()
    app.radio(key='scaled_livechoice').set_value('Upload').run()
    app.button[0].click().run()
    assert not app.exception and any('Choose an image' in e.value for e in app.error)
    app.radio(key='section').set_value('Satellite Explorer').run()
    app.date_input[0].set_value(app.date_input[1].value.replace(year=app.date_input[1].value.year+1)).run()
    app.button[0].click().run()
    assert not app.exception and any('Start date' in e.value for e in app.error)
    app.radio(key='section').set_value('Mini-projects').run()
    app.text_input[0].set_value('no-such-project-xyz').run()
    assert not app.exception and any('No projects match' in i.value for i in app.info)
print('Seven sections, learning curves, source library and empty searches passed; zero network on navigation')
