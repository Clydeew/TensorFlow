from pathlib import Path

import pytest

from tensorforge.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def classification():
    return load_config(ROOT / "examples/classification.yaml")


@pytest.fixture(autouse=True)
def bounded_threads(monkeypatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
