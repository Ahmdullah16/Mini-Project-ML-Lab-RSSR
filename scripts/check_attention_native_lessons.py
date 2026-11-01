"""Bounded offline MP23–26 Streamlit smoke and source contract."""
import sys
import numpy as np
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
        table=app.dataframe[0].value
        assert table.iloc[0]['MSE_vs_disabled']==0
        assert np.isclose(table.iloc[1]['mean'],table.iloc[0]['mean']/2)
        x=np.arange(192,dtype=float)/191*.5
        assert np.isclose(table.iloc[1]['MSE_vs_disabled'],np.mean((x*.5)**2),rtol=1e-5)
        assert 0<=table.iloc[2]['mean']<=table.iloc[0]['mean']
        if ident=='MP26': assert np.isclose(table.iloc[3]['mean'],.5)
        app.slider(key=ident+'_amplitude').set_value(0.).run()
        assert not app.success
        app.button(key=ident+'_run').click().run()
        assert not app.exception
        assert np.allclose(app.dataframe[0].value['mean'],0)
print('MP23-26 seeded synthetic attention controls and stale clearing passed; no network')
