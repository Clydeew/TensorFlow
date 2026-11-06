import json
import socket

import nbformat
import pandas as pd
import pytest
from conftest import ROOT
from PIL import Image

from tensorforge.config import load_config
from tensorforge.generator import generate, verify
from tensorforge.registry import TASKS

EXAMPLES = [
    "classification",
    "regression",
    "pairs",
    "multilabel",
    "clustering",
    "retrieval",
    "seq2seq",
    "ner",
    "rendering",
]

# Only loopback traffic used by the local Jupyter kernel is allowed.
NETWORK_GUARD = """import socket
_original_connect = socket.socket.connect
def _local_connect(self, address):
    if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError("External networking disabled in offline acceptance test")
    return _original_connect(self, address)
socket.socket.connect = _local_connect
"""


@pytest.mark.parametrize("name", EXAMPLES)
def test_generated_notebook_offline_end_to_end(name, tmp_path, monkeypatch):
    cfg, root = load_config(ROOT / "examples" / f"{name}.yaml")

    def deny_external(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "localhost", "::1"):
            raise AssertionError("Generator attempted external networking")
        return original(sock, address)

    original = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", deny_external)
    bundle = tmp_path / name
    result = generate(cfg, root, bundle)
    assert not result["executed"]
    assert not (bundle / "artifacts").exists()
    report = json.loads((bundle / "generation_report.json").read_text())
    assert report["executed"] is False
    notebook = nbformat.read(bundle / "baseline.ipynb", as_version=4)
    nbformat.validate(notebook)
    sources = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "import tensorforge" not in sources and "from tensorforge" not in sources
    notebook.cells.insert(0, nbformat.v4.new_code_cell(NETWORK_GUARD))
    nbformat.write(notebook, bundle / "baseline.ipynb")
    verify(bundle, timeout=120)
    execution = json.loads((bundle / "artifacts/execution_report.json").read_text())
    assert execution["task"] == cfg["task"]
    assert execution["measured_metrics"] or name in ("clustering", "rendering")
    if name == "classification":
        result = pd.read_csv(bundle / "artifacts/submission.csv")
        assert result["id"].tolist() == ["x1", "x2"]
        assert set(result["label"]) <= {"sport", "tech"}
    elif name == "retrieval":
        ranked = pd.read_csv(bundle / "artifacts/rankings.csv")
        assert ranked.iloc[0]["document_id"] == "d1"
        assert execution["measured_metrics"]["mrr"] == 1
    elif name == "rendering":
        manifest = json.loads((bundle / "artifacts/manifest.json").read_text())
        with Image.open(bundle / "artifacts" / manifest[0]["artifact"]) as im:
            assert im.size == (1024, 256)
    else:
        assert (bundle / "artifacts/predictions.csv").is_file()


def test_all_verified_adapters_have_execution_fixture():
    examples = {load_config(ROOT / "examples" / f"{name}.yaml")[0]["task"] for name in EXAMPLES}
    assert examples == {t.id for t in TASKS.values() if t.status == "verified"}


def test_pair_regression(tmp_path):
    cfg, root = load_config(ROOT / "examples/pairs.yaml")
    cfg["params"]["variant"] = "regression"
    cfg["metric"] = "mae"
    cfg["split"]["strategy"] = "random"
    cfg["review"]["acknowledged_warnings"].append("RANDOM_SPLIT")
    generate(cfg, root, tmp_path / "pair-regression")
    verify(tmp_path / "pair-regression")
    assert (
        json.loads((tmp_path / "pair-regression/artifacts/execution_report.json").read_text())[
            "measured_metrics"
        ]["mae"]
        >= 0
    )
