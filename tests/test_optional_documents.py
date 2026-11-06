"""Real local document/OCR tests; no model downloads during tests."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tensorforge.generator import generate, verify

ROOT = Path(__file__).resolve().parents[1]


def config(task, manifest):
    return {
        "schema_version": 1,
        "project_root": ".",
        "task": task,
        "data": {"test": str(manifest)},
        "columns": {"id": "id", "image": "image"},
        "metric": "none",
        "seed": 42,
        "params": {},
        "review": {"approved": True, "acknowledged_warnings": ["EXPERIMENTAL_BACKEND", "NO_SUBMISSION_SPEC"]},
    }


def test_real_printed_image_and_scanned_pdf_ocr(tmp_path):
    pytest.importorskip("pytesseract")
    pytest.importorskip("pypdfium2")
    from PIL import Image, ImageDraw, ImageFont

    directory = ROOT / "models/tesseract"
    if not shutil.which("tesseract") or not (directory / "eng.traineddata").is_file():
        pytest.skip("Stage Tesseract English data using scripts/stage_ocr.py")
    image = Image.new("RGB", (1000, 180), "white")
    ImageDraw.Draw(image).text(
        (30, 40), "Hello TensorFlow", font=ImageFont.load_default(size=60), fill="black"
    )
    image.save(tmp_path / "sample.png")
    image.save(tmp_path / "scan.pdf", "PDF")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {"id": ["0001", "0002"], "image": [str(tmp_path / "sample.png"), str(tmp_path / "scan.pdf")]}
    ).to_csv(manifest, index=False)
    cfg = config("ocr", manifest)
    cfg["params"] = {"language": "eng", "tessdata_path": str(directory)}
    generate(cfg, tmp_path, tmp_path / "bundle", experimental=True)
    verify(tmp_path / "bundle", timeout=120)
    output = pd.read_csv(tmp_path / "bundle/artifacts/predictions.csv", dtype=str)
    assert output["id"].tolist() == ["0001", "0002"]
    assert all("Hello" in text for text in output["prediction"])


def test_docx_html_and_plain_text_extraction(tmp_path):
    pytest.importorskip("docx")
    pytest.importorskip("bs4")
    pytest.importorskip("pypdf")
    from docx import Document

    doc = Document()
    doc.add_paragraph("A document paragraph")
    doc.add_table(rows=1, cols=1).cell(0, 0).text = "A table cell"
    doc.save(tmp_path / "source.docx")
    (tmp_path / "source.html").write_text("<h1>Visible heading</h1><script>do_not_execute()</script>")
    (tmp_path / "source.txt").write_text("Exact plain text")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {"id": ["d", "h", "t"], "image": [str(tmp_path / f"source.{ext}") for ext in ["docx", "html", "txt"]]}
    ).to_csv(manifest, index=False)
    cfg = config("document_extraction", manifest)
    generate(cfg, tmp_path, tmp_path / "bundle", experimental=True)
    verify(tmp_path / "bundle", timeout=120)
    rows = pd.read_csv(tmp_path / "bundle/artifacts/predictions.csv").set_index("id")["prediction"]
    assert "A table cell" in rows["d"]
    assert rows["h"] == "Visible heading"
    assert rows["t"] == "Exact plain text"
    assert (
        json.loads((tmp_path / "bundle/artifacts/execution_report.json").read_text())["measured_metrics"]
        == {}
    )
