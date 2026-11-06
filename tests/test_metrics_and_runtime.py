import math

import numpy as np
import pandas as pd
import pytest

from tensorforge.templates import runtime as rt


@pytest.fixture(autouse=True)
def runtime_config(monkeypatch):
    monkeypatch.setattr(rt, "CONFIG", {"review": {"acknowledged_warnings": []}})
    monkeypatch.setattr(rt, "COLS", {"text": "text", "target": "label", "tokens": "tokens", "tags": "tags"})
    monkeypatch.setattr(rt, "PARAMS", {})


def test_edit_distance_cer_and_wer():
    assert rt.edit_distance("kitten", "sitting") == 3
    assert rt.text_error(["abc", "de"], ["axc", "d"]) == 2 / 5
    assert rt.text_error(["hello world"], ["hello there"], words=True) == 0.5
    with pytest.raises(ValueError, match="empty reference"):
        rt.text_error([""], [""])


def test_entity_metric_is_strict_span_metric():
    assert rt.entities(["B-PER", "I-PER", "O", "B-LOC"]) == {(0, 2, "PER"), (3, 4, "LOC")}
    assert rt.entity_f1([["B-PER", "I-PER"]], [["B-PER", "O"]]) == 0
    assert rt.entity_f1([["B-PER", "O"]], [["B-PER", "O"]]) == 1
    assert rt.safe_bio(["I-PER", "I-PER", "I-LOC"]) == ["B-PER", "I-PER", "B-LOC"]


def test_ranking_metrics_against_hand_calculation():
    result = rt.retrieval_metrics({"q": ["b", "a", "c"]}, {"q": {"a": 1, "c": 1}}, 2)
    assert result["mrr"] == 0.5
    assert result["recall_at_k"] == 0.5
    assert result["ndcg"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_no_tfidf_fit_on_validation_text(monkeypatch):
    monkeypatch.setattr(rt, "CONFIG", {"metric": "accuracy"})
    train = pd.DataFrame(
        {
            "text": ["football sport", "tennis sport", "computer tech", "software tech"],
            "label": ["s", "s", "t", "t"],
        }
    )
    test = pd.DataFrame({"text": ["heldout_unseen_xyz"]})
    original = rt.vectorizer
    fitted = []

    def capture():
        obj = original()
        fitted.append(obj)
        return obj

    monkeypatch.setattr(rt, "vectorizer", capture)
    rt.predict_core(train, test, "text_classification")
    assert "xyz" not in fitted[0].vocabulary_


def test_group_and_time_boundaries(monkeypatch):
    frame = pd.DataFrame(
        {
            "text": [f"row {i}" for i in range(8)],
            "group": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "time": [f"2026-01-0{i + 1}" for i in range(8)],
        }
    )
    monkeypatch.setattr(rt, "CONFIG", {"split": {"strategy": "group", "column": "group", "test_size": 0.25}})
    a, b = rt.split_indices(frame)
    assert not set(frame.iloc[a]["group"]) & set(frame.iloc[b]["group"])
    rt.CONFIG["split"] = {"strategy": "time", "column": "time", "test_size": 0.25}
    a, b = rt.split_indices(frame)
    assert max(frame.iloc[a]["time"]) < min(frame.iloc[b]["time"])


def test_duplicate_source_requires_acknowledgement(monkeypatch):
    monkeypatch.setattr(
        rt,
        "CONFIG",
        {"split": {"strategy": "random", "test_size": 0.5}, "review": {"acknowledged_warnings": []}},
    )
    with pytest.raises(ValueError, match="DUPLICATE_TRAIN"):
        rt.split_indices(pd.DataFrame({"text": ["same"] * 8}))


def test_ner_misalignment_fails():
    frame = pd.DataFrame({"tokens": [["Alice", "here"]], "tags": [["B-PER"]]})
    with pytest.raises(ValueError, match="length mismatch"):
        rt.predict_core(frame, frame, "token_classification")


def test_sample_id_mismatch_cannot_export(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rt, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(rt, "COLS", {"id": "id"})
    sample = tmp_path / "sample.csv"
    sample.write_text("id,label\nwrong,\n")
    monkeypatch.setattr(
        rt,
        "CONFIG",
        {"submission": {"sample": "sample.csv", "id_column": "id", "prediction_column": "label"}},
    )
    with pytest.raises(ValueError, match="ID mismatch"):
        rt.write_predictions(pd.DataFrame({"id": ["correct"]}), ["A"])
    assert not (tmp_path / "submission.csv").exists()


def test_stratification_does_not_fallback(monkeypatch):
    monkeypatch.setattr(rt, "CONFIG", {"split": {"strategy": "stratified", "test_size": 0.25}})
    frame = pd.DataFrame({"text": ["one", "two", "three", "four"], "label": ["a", "a", "a", "b"]})
    with pytest.raises(ValueError):
        rt.split_indices(frame)


def test_numeric_nonfinite_fails():
    with pytest.raises(ValueError, match="Nonfinite"):
        rt.numeric([1, np.nan])
