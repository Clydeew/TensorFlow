from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_interface_is_english_only_with_indonesian_data(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    app = AppTest.from_file(str(root / "src/tensorforge/ui.py"), default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "TensorFlow"
    assert not app.selectbox
    assert app.tabs[0].label == "Build a notebook"
    assert app.text_input[0].value == "examples/indonesia.yaml"
    app.button[0].click().run()
    assert not app.exception
    assert any("validated" in item.value for item in app.success)
