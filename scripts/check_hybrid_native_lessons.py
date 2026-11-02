"""Offline native MP37–40 execution, numerical contracts and failure recovery."""
import socket
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
original_connect=socket.socket.connect


def local_loopback_only(sock,address):
    if isinstance(address,tuple) and address[0] in ('127.0.0.1','::1'):return original_connect(sock,address)
    raise AssertionError('Unexpected external network')


with ExitStack() as stack:
    for target in ('urllib.request.urlopen','socket.create_connection','socket.socket.connect_ex'):
        stack.enter_context(patch(target,side_effect=AssertionError('Unexpected network')))
    stack.enter_context(patch('socket.socket.connect',local_loopback_only))
    app=AppTest.from_file(str(root/'app/streamlit_app.py'),default_timeout=90).run()
    app.radio(key='section').set_value('Mini-projects').run()
    assert len(app.selectbox(key='onsite_lesson').options)==40
    for ident in ('MP37','MP38','MP39','MP40'):
        app.selectbox(key='onsite_lesson').set_value(ident).run()
        assert not app.success and not app.exception
        app.button(key=ident+'_run').click().run()
        assert app.success and not app.exception,ident
        assert len(app.get('download_button'))==2
        if ident=='MP37':
            assert list(app.dataframe[0].value['mean_projection'])==[2,3]
            app.selectbox(key='MP37_mode').set_value('concat').run()
        elif ident=='MP38':
            assert app.dataframe[1].value.shape==(10,14)
            app.selectbox(key='MP38_bands').set_value(4).run()
        elif ident=='MP39':
            assert len(app.dataframe[0].value)==8
            app.selectbox(key='MP39_units').set_value(1).run()
        else:
            assert set(app.dataframe[0].value['bands'])=={3}
            app.selectbox(key='MP40_bands').set_value(4).run()
        assert not app.success and not app.get('download_button')
        app.button(key=ident+'_run').click().run()
        assert not app.exception
        if ident=='MP39':
            assert app.error and not app.success
            app.selectbox(key='MP39_units').set_value(3).run()
            app.button(key='MP39_run').click().run()
        assert app.success and not app.exception
    app.selectbox(key='MP40_axis').set_value('noise').run()
    app.button(key='MP40_run').click().run()
    assert app.success and not app.exception
    assert all(app.dataframe[0].value['sigma_hr_pixels']==.8)
print('MP37–40 offline AppTests passed: actual outputs, protocol selection, stale clearing, exports and rejected one-unit CI')
