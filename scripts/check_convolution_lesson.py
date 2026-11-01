"""MP17 offline Streamlit AppTest and stale-result check."""
import sys
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/"src"))
with patch("urllib.request.urlopen",side_effect=AssertionError("Unexpected network")):
    app=AppTest.from_file(str(root/"app/streamlit_app.py"),default_timeout=60).run()
    app.radio(key="section").set_value("Mini-projects").run()
    app.selectbox(key="onsite_lesson").set_value("MP17").run()
    assert not app.success
    app.button(key="mp17_run").click().run()
    assert not app.exception
    assert any("Input=9 × 9; output=9 × 9" in w.value for w in app.markdown)
    assert any("Actual canonical cross-correlation" in w.value for w in app.success)
    app.selectbox(key="mp17_stride").set_value(2).run()
    assert not app.success
    app.button(key="mp17_run").click().run()
    assert not app.exception
    assert any("output=5 × 5" in w.value for w in app.markdown)
print("MP17 native lesson: actual canonical synthetic cross-correlation and stale clearing passed")
