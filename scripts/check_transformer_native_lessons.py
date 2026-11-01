"""Offline on-site MP27–31 execution, failure recovery and stale-output checks."""
import socket
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))
original_connect = socket.socket.connect


def local_event_loop_only(sock, address):
    # Windows asyncio implements socketpair via a loopback connection.
    if isinstance(address, tuple) and address[0] in ('127.0.0.1', '::1'):
        return original_connect(sock, address)
    raise AssertionError('Unexpected external network: '+str(address))


with ExitStack() as stack:
    for target in ('urllib.request.urlopen', 'socket.create_connection', 'socket.socket.connect_ex'):
        stack.enter_context(patch(target, side_effect=AssertionError('Unexpected network')))
    stack.enter_context(patch('socket.socket.connect', local_event_loop_only))
    app = AppTest.from_file(str(root/'app/streamlit_app.py'), default_timeout=90).run()
    app.radio(key='section').set_value('Mini-projects').run()
    assert len(app.selectbox(key='onsite_lesson').options) == 40
    for ident in ('MP27','MP28','MP29','MP30','MP31'):
        app.selectbox(key='onsite_lesson').set_value(ident).run()
        assert not app.exception and not app.success, ident
        app.button(key=ident+'_run').click().run()
        assert not app.exception and app.success, ident
        assert len(app.get('download_button')) >= 2, ident
        if ident == 'MP27':
            assert np.isclose(app.dataframe[3].value.iloc[0,0],13.3023845)
            app.checkbox(key='MP27_mask').set_value(True).run()
        elif ident == 'MP28':
            assert app.dataframe[1].value.shape == (16,4)
            app.selectbox(key='MP28_heads').set_value(4).run()
        elif ident == 'MP29':
            assert app.dataframe[3].value.shape == (16,4)
            app.checkbox(key='MP29_positions').set_value(True).run()
        elif ident == 'MP30':
            assert list(app.dataframe[1].value.iloc[0]) == [0,1,4,5]
            app.selectbox(key='MP30_patch').set_value(4).run()
        else:
            assert len(app.dataframe[0].value) == 2
            app.slider(key='MP31_height').set_value(1).run()
        assert not app.success and not app.get('download_button'), ident
        app.button(key=ident+'_run').click().run()
        assert not app.exception and app.success, ident
    # Missing data is explicit, recovers after retry, and does not show stale success.
    original_read = Path.read_text
    def missing_benchmark(path, *args, **kwargs):
        if path.name == 'development-benchmark.json':
            raise FileNotFoundError('Teaching evidence unavailable')
        return original_read(path, *args, **kwargs)
    with patch.object(Path, 'read_text', missing_benchmark):
        app.button(key='MP31_run').click().run()
        assert app.error and not app.success and not app.exception
    app.button(key='MP31_run').click().run()
    assert app.success and not app.exception
print('MP27–31 offline AppTests passed: actual outputs, exports, stale clearing and retry; 40 choices retained')
