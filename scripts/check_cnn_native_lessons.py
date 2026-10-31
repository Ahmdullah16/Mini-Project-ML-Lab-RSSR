"""MP18-22 offline Streamlit AppTest: actual synthetic outputs; never network."""
from pathlib import Path
import sys
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"src"));sys.path.insert(0,str(root))
with patch("urllib.request.urlopen",side_effect=AssertionError("Unexpected network")):
    app=AppTest.from_file(str(root/"app/streamlit_app.py"),default_timeout=90).run()
    app.radio(key="section").set_value("Mini-projects").run()
    for ident in ("MP18","MP19","MP20","MP21","MP22"):
        app.selectbox(key="onsite_lesson").set_value(ident).run()
        assert not app.exception and not app.success,ident
        app.button(key=ident.lower()+"_run").click().run()
        assert not app.exception, (ident,[e.message for e in app.exception])
        assert app.success, ident
        assert any("Download measurements" in w.label for w in app.button),ident
        if ident=="MP18":
            assert any("Input=(8, 10)" in w.value for w in app.markdown)
        if ident=="MP19":
            assert any("MSE=" in w.value for w in app.markdown)
        if ident=="MP20":
            assert any("output max=0.000000" in w.value for w in app.markdown)
        if ident=="MP21":
            assert any("MSE=0.000000" in w.value for w in app.markdown)
        if ident=="MP22":
            assert any("max |output" in w.value for w in app.markdown)
            app.selectbox(key="mp22_mode").set_value("transpose").run()
            assert not app.success
            app.button(key="mp22_run").click().run()
            assert not app.exception
print("MP18-22 UI batch: five canonical synthetic lessons and stale result clearing passed.")
