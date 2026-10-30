from pathlib import Path
import sys
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
with patch('urllib.request.urlopen',side_effect=AssertionError('Unexpected network')):
 app=AppTest.from_file(str(root/'app/streamlit_app.py'),default_timeout=60).run()
 for section,prefix in [('Live SR','live'),('Satellite Explorer','satellite')]:
  app.radio(key='section').set_value(section).run()
  registry=next(d.value for d in app.dataframe if 'rgb_execution' in d.value.columns)
  assert len(registry)==7 and int(registry.rgb_execution.sum())==7
  for kind in ('Blur','Noise','Brightness','Contrast','Shift','Sharpening'):
   app.selectbox(key=prefix+'_distortion').set_value(kind).run()
   app.button(key=prefix+'_run_metrics').click().run()
   assert not app.exception and not app.error,(section,kind)
   assert any('For this displayed pair:' in w.value for w in app.markdown)
  app.slider(key=prefix+'_strength').set_value(0.).run()
  assert not any('For this displayed pair:' in w.value for w in app.markdown)
  app.button(key=prefix+'_run_metrics').click().run()
  assert any('MSE=0, RMSE=0, MAE=0' in w.value for w in app.markdown)
print('Both sections: six distortions executed, zero identity and stale-output clearing verified; no network')
