"""Offline MP32–36 execution, expected failure/retry, exports and stale clearing."""
import socket
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
original_connect=socket.socket.connect


def local_loopback_only(sock,address):
    if isinstance(address,tuple) and address[0] in ('127.0.0.1','::1'):
        return original_connect(sock,address)
    raise AssertionError('Unexpected external network')


with ExitStack() as stack:
    for target in ('urllib.request.urlopen','socket.create_connection','socket.socket.connect_ex'):
        stack.enter_context(patch(target,side_effect=AssertionError('Unexpected network')))
    stack.enter_context(patch('socket.socket.connect',local_loopback_only))
    app=AppTest.from_file(str(root/'app/streamlit_app.py'),default_timeout=90).run()
    app.radio(key='section').set_value('Mini-projects').run()
    assert len(app.selectbox(key='onsite_lesson').options)==40
    for ident in ('MP32','MP33','MP34','MP35','MP36'):
        app.selectbox(key='onsite_lesson').set_value(ident).run()
        assert not app.exception and not app.success,ident
        app.button(key=ident+'_run').click().run()
        assert not app.exception and app.success,ident
        assert len(app.get('download_button'))==2,ident
        if ident=='MP32':
            assert list(app.dataframe[4].value['mean'])==[3,4]
            app.selectbox(key='MP32_mask').set_value('all invalid').run()
        elif ident=='MP33':
            assert app.dataframe[4].value.iloc[0]['mean']==3
            app.selectbox(key='MP33_split').set_value('validation').run()
        elif ident=='MP34':
            assert app.dataframe[1].value.iloc[0,0]==2.5
            app.checkbox(key='MP34_invalid').set_value(True).run()
        elif ident=='MP35':
            assert app.dataframe[1].value.iloc[0]['max_axis_clearance_degrees']==1
            app.selectbox(key='MP35_buffer').set_value(1.).run()
        else:
            text=' '.join(x.value for x in list(app.markdown)+list(app.warning))
            assert 'NOT a measured residual alignment error' in text
            assert 'official paper/correction detail is unavailable' not in text
            assert 'No VENµS HR reference' in list(app.dataframe[0].value['status'])
            app.selectbox(key='MP36_offset').set_value(2.5).run()
        assert not app.success and not app.get('download_button'),ident
        app.button(key=ident+'_run').click().run()
        assert not app.exception,ident
        if ident in ('MP33','MP34','MP35'):
            assert app.error and not app.success and not app.get('download_button'),ident
            if ident=='MP33':app.selectbox(key='MP33_split').set_value('train')
            elif ident=='MP34':app.checkbox(key='MP34_invalid').set_value(False)
            else:app.selectbox(key='MP35_buffer').set_value(.02)
            app.run();app.button(key=ident+'_run').click().run()
            assert app.success and not app.exception,ident
        else:
            assert app.success,ident
print('MP32–36 offline AppTests passed: computations, masks, stale clearing, rejection/retry and exports;40 choices retained')
