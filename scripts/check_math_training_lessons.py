"""MP12–16 native Streamlit batch: offline, controlled actual execution."""
from pathlib import Path
import sys
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
with patch("urllib.request.urlopen",side_effect=AssertionError("Unexpected network")):
    app=AppTest.from_file(str(ROOT/"app/streamlit_app.py"),default_timeout=90).run()
    app.radio(key="section").set_value("Mini-projects").run()
    for project,button in [
        ("MP12","mp12_run"),("MP13","mp13_run"),("MP14","mp14_run"),
        ("MP15","mp15_run"),("MP16","mp16_run")
    ]:
        app.selectbox(key="onsite_lesson").set_value(project).run()
        assert not app.exception and not app.error,project
        assert not app.success,project
        app.button(key=button).click().run()
        assert not app.exception and not app.error,project
        assert app.success,project
        assert any("Download numerical table" in x.label for x in app.button),project
        if project=="MP12":
            assert any("tail energy=5.000000" in x.value for x in app.markdown)
            app.slider(key="mp12_rank").set_value(3).run()
            assert not app.success
            app.button(key="mp12_run").click().run()
            assert any("error norm=0.000000" in x.value for x in app.markdown)
        if project=="MP13":
            assert any("Initial MSE=17.000000" in x.value for x in app.markdown)
            assert any("final MSE=1.685000" in x.value for x in app.markdown)
            app.slider(key="mp13_steps").set_value(0).run()
            assert not app.success
        if project=="MP14":
            assert any("gradient finite-difference error=" in x.value for x in app.markdown)
            app.selectbox(key="mp14_hidden").set_value(2).run()
            assert not app.success
        if project=="MP15":
            assert any("maximum derivative mismatch=" in x.value for x in app.markdown)
            app.slider(key="mp15_steps").set_value(0).run()
            assert not app.success
        if project=="MP16":
            assert any("exact seed-matched parameters=True" in x.value for x in app.markdown)
            app.selectbox(key="mp16_batch").set_value(3).run()
            assert not app.success
    assert not app.exception
print("MP12-16 native UI: actual bounded calculations and stale-output clearing passed; network forbidden")
