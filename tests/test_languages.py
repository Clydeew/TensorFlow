import json
from pathlib import Path

import pandas as pd
import pytest

from tensorforge.cli import main
from tensorforge.config import load_config, validate
from tensorforge.errors import WorkbenchError
from tensorforge.generator import generate, verify
from tensorforge.languages import PRESETS, validate_language
from tensorforge.templates import runtime as rt

ROOT = Path(__file__).resolve().parents[1]


def test_indonesian_english_and_regional_notebook(tmp_path):
    cfg, root = load_config(ROOT / "examples/indonesia.yaml")
    assert {w["id"] for w in validate(cfg, root, require_approval=False)} == {"NO_SUBMISSION_SPEC"}
    bundle = tmp_path / "indonesia"
    generate(cfg, root, bundle)
    verify(bundle)
    report = json.loads((bundle / "artifacts/execution_report.json").read_text(encoding="utf-8"))
    assert report["config"]["language"]["primary"] == "id"
    assert report["config"]["language"]["additional"] == ["en", "jv", "su"]
    predictions = pd.read_csv(bundle / "artifacts/predictions.csv")
    assert len(predictions) == 5
    assert set(predictions["prediction"]) <= {"olahraga", "teknologi"}
    diagnostics = pd.read_csv(bundle / "artifacts/validation_predictions.csv")
    assert set(diagnostics["language"]) <= {"id", "en", "jv", "su", "id-en-jv"}


def test_native_script_and_diacritics_are_not_ascii_stripped(monkeypatch):
    monkeypatch.setattr(rt, "CONFIG", {"language": {"normalization": "NFC", "preserve_case": True}})
    texts = ["ꦧꦱ ꦗꦮ", "ꦧꦸꦏꦸ ꦲꦚꦂ", "Café Java", "Cafe java"]
    vec = rt.vectorizer().fit(texts)
    assert any("ꦧ" in term for term in vec.vocabulary_)
    assert "fé" in vec.vocabulary_
    assert "Jav" in vec.vocabulary_ and "jav" in vec.vocabulary_
    assert rt.text_error(["café"], ["cafe\u0301"]) == 0
    assert rt.normalize_text("ꦧꦱ ꦗꦮ") == "ꦧꦱ ꦗꦮ"


def test_custom_regional_tag_requires_no_english_fallback():
    config = {"primary": "gor", "additional": ["id", "en"], "script": "Latn", "code_switching": True}
    assert validate_language(config) == config
    assert {"id", "en", "jv", "su", "ban", "bug", "min", "ace", "bjn"} <= set(PRESETS)


@pytest.mark.parametrize(
    "language",
    [
        {"primary": "Indonesia"},
        {"additional": "en"},
        {"additional": None},
        {"script": "Javanese"},
        {"normalization": "ASCII"},
        {"code_switching": "yes"},
        {"preserve_case": 1},
    ],
)
def test_bad_language_config_rejected(language, classification):
    cfg, root = classification
    cfg["language"] = language
    with pytest.raises(WorkbenchError) as exc:
        validate(cfg, root)
    assert exc.value.code == "INVALID_LANGUAGE_CONFIG"


def test_cli_language_catalog(capsys):
    assert main(["languages", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["jv"]["ocr"] == "jav"
