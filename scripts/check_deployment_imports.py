"""Simulate a warm worker retaining helpers from before a source deployment."""
from pathlib import Path
import sys,types
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'app')]
import rssr.learning_content as current
old=types.ModuleType('rssr.learning_content');old.__file__=current.__file__;old.load_lesson=current.load_lesson
sys.modules['rssr.learning_content']=old
try:
 from rssr.learning_content import visitor_prose
 raise AssertionError('Stale helper did not reproduce missing export')
except ImportError:pass
old_sat=types.ModuleType('rssr.inference');old_sat.__file__=str(ROOT/'src/rssr/inference.py')
sys.modules['rssr.inference']=old_sat
if hasattr(sys,'_rssr_deployment_import_state'):delattr(sys,'_rssr_deployment_import_state')
with patch('urllib.request.urlopen',side_effect=AssertionError('Navigation requested network')):
 app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=60).run()
 for section in ['Mini-projects','Satellite Explorer','Live SR']:
  app.radio(key='section').set_value(section).run()
  assert not app.exception and not app.error,section
 app.radio(key='section').set_value('Mini-projects').run()
 for ident in ['MP01','MP02','MP40']:
  app.selectbox(key='onsite_lesson').set_value(ident).run()
  assert not app.exception,ident
from rssr.learning_content import visitor_prose
from rssr.inference import geotiff_plan
from deployment_bootstrap import prepare_imports
before=sys.modules['rssr.learning_content']
assert not prepare_imports(ROOT)['imports_refreshed']
assert sys.modules['rssr.learning_content'] is before
print('Reproduced old missing exports; worker refresh recovered lessons/satellite/live without network; unchanged rerun retains imports')
