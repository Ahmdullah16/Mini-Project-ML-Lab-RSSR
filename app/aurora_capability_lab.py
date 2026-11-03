"""Developer-only probe. Run separately with RSSR_DEV_CAPABILITIES=1.

Not dispatched from the public app; never accepts executable user content.
"""
import inspect
import os
import sys
from pathlib import Path
import streamlit as st

if os.environ.get('RSSR_DEV_CAPABILITIES') != '1':
    st.info('Developer capability lab is disabled. No public configuration changes are available.')
    st.stop()

st.title('AURORA X · capability laboratory')
st.caption('Local API probes are not browser, accessibility or Community Cloud verification.')
names = ('navigation', 'Page', 'page_link', 'html', 'fragment', 'dialog', 'status',
         'toast', 'popover', 'pills', 'context', 'query_params', 'login')
st.json({'streamlit': st.__version__, 'available': {n: hasattr(st, n) for n in names},
         'navigation_signature': str(inspect.signature(st.navigation))})
with st.container(key='aurora_probe', border=True):
    st.markdown('Keyed container: native layout, no private selectors.')
    st.html('<svg role="img" aria-label="Illustrative processing arrow" width="120" height="40" viewBox="0 0 120 40"><path d="M4 20H100L88 8M100 20L88 32" fill="none" stroke="currentColor" stroke-width="3"/></svg>')
with st.popover('Open probe details'):
    st.write('Popover content is static and contains no secrets.')
st.pills('Probe selection', ['A', 'B'], default='A')
with st.status('Local probe rendered', state='complete'):
    st.write('This status describes rendering only, not model execution.')
st.download_button('Download probe text', 'local capability probe', 'probe.txt')
st.info('Iframe event exchange, parent control, mobile keyboard and hosted limits: NOT VERIFIED. No custom JavaScript component enabled.')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_design import apply_theme, cards, hero
st.subheader('Component gallery · illustrative interface fixtures')
mode = st.selectbox('Gallery panel appearance', ['Light', 'Dark'])
apply_theme(mode)
hero()
cards([('NOT VERIFIED', 'Empty result', 'No inference has run in this capability page.'),
       ('NEEDS OWNER ACTION', 'Saved workspace', 'Provider configuration is absent; guest export remains independent.')])
st.warning('Illustrative warning state; not a measured model result.')
