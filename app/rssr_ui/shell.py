"""Native keyboard-operable utilities; session state only, no shared user cache."""
import streamlit as st

LENSES = {
    'Explore': 'Start with a licensed sample. Extra pixels are estimates, not confirmed geographic detail.',
    'Build & Compare': 'Prepare → inspect resource limits → explicitly run → compare the same crop → export provenance.',
    'Research': 'Inspect source, checkpoint, processing and reference eligibility before interpreting a metric. Missing evidence stays unavailable.',
}


def utilities(sections, labels):
    st.selectbox('Workspace lens', list(LENSES), key='aurora_lens')
    st.checkbox('Reduce motion', value=True, key='aurora_reduce_motion',
                help='Browser reduced-motion preference also takes precedence. No essential content uses animation.')
    def choose_route():
        target = st.session_state['aurora_quick_action']
        if target in sections:
            st.session_state['section'] = target
    st.selectbox('Quick navigation', ['Choose destination', *sections],
                 format_func=lambda x: labels.get(x, x), key='aurora_quick_action',
                 on_change=choose_route)
    st.caption('Session-only workspace. Appearance and lens never start inference. Native widget theme is available in Streamlit Settings.')


def guidance():
    st.caption(LENSES[st.session_state.get('aurora_lens', 'Explore')])
    st.markdown('<div id="aurora-content" tabindex="-1"></div>', unsafe_allow_html=True)


def evidence_drawer(records, key='aurora-facts'):
    import json
    with st.expander('Evidence drawer · registry facts'):
        st.dataframe(records, hide_index=True, width='stretch')
        st.download_button('Export registry evidence', json.dumps(records, indent=2),
                           'registry-evidence.json', 'application/json', key=key)
        st.caption('Counts describe current registry content. Unknown commit/time fields remain null rather than invented.')
