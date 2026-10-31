"""Batched native MP09–11 checks with a known reference, never frozen data."""
from pathlib import Path
import sys
import numpy as np
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
with patch('urllib.request.urlopen',side_effect=AssertionError('Unexpected network')):
    app=AppTest.from_file(str(root/'app/streamlit_app.py'),default_timeout=60).run()
    app.radio(key='section').set_value('Mini-projects').run()
    for ident in ['MP09','MP10','MP11']:
        app.selectbox(key='onsite_lesson').set_value(ident).run()
        assert not app.success
        app.button(key=ident+'_metric_run').click().run()
        assert not app.exception
        table=next(w.value for w in app.dataframe if 'MSE' in w.value.columns)
        bright=table[table['case']=='brightness'].iloc[0]
        assert abs(bright['MSE']-.01)<1e-12 and abs(bright['PSNR_dB']-20)<1e-10
        assert any('SSIM map=42 × 26 × 3' in w.value for w in app.markdown)
        if ident=='MP11':
            assert len(table)==4 and table.iloc[0]['MSE']==0 and np.isinf(table.iloc[0]['PSNR_dB'])
        app.slider(key=ident+'_metric_strength').set_value(0.).run()
        assert not app.success
        app.selectbox(key=ident+'_metric_window').set_value(3).run()
        app.button(key=ident+'_metric_run').click().run()
        assert not app.exception
        table=next(w.value for w in app.dataframe if 'MSE' in w.value.columns)
        assert np.allclose(table.MSE,0) and np.allclose(table.SSIM,1)
        assert any('SSIM map=46 × 30 × 3' in w.value for w in app.markdown)
print('MP09–11 batched AppTest passed: brightness hand example, identity, four-row comparison, window dimensions and stale clearing; no network')
