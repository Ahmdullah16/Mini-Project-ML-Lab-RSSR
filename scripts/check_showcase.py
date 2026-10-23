"""Navigation and empty states must not trigger provider/model downloads."""
from pathlib import Path
import sys
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
sections=['Overview','Live SR','Results','Methodology & Data','Mini-projects','Satellite Explorer','Downloads & Reproducibility']
with patch('urllib.request.urlopen',side_effect=AssertionError('Network access on navigation')):
    app=AppTest.from_file(str(root/'app/streamlit_app.py')).run(timeout=60)
    for section in sections:
        app.radio(key='section').set_value(section).run(timeout=60)
        assert not app.exception and not app.error,(section,app.exception,app.error)
    app.radio(key='section').set_value('Results').run(timeout=60)
    app.text_input[0].set_value('no-such-model-xyz').run()
    assert not app.exception and any('No recorded values' in i.value for i in app.info)
    app.radio(key='result_view').set_value('Learning curves').run()
    assert not app.exception
    app.radio(key='result_view').set_value('Artifact library').run()
    assert not app.exception
    app.radio(key='section').set_value('Live SR').run()
    app.radio(key='input_choice').set_value('Upload').run()
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
