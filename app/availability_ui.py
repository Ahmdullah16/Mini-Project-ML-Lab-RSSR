"""Same registry status in both inference sections; execution uses pinned catalog."""
from pathlib import Path
import streamlit as st
from rssr.model_availability import availability

def render():
    with st.expander('All seven models · checkpoint compatibility and blockers',expanded=True):
        rows=availability(Path(__file__).resolve().parents[1])
        st.dataframe(rows,hide_index=True)
        st.caption('The execution selectors offer only pinned hosted RGB checkpoints, including the preserved early CNN pilot. Unavailable families cannot execute. Local pilot counts do not imply hosting or scientific acceptance. No multispectral input is silently converted to RGB. Full seven-model execution acceptance is still pending.')
