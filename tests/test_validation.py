import copy
import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest
import yaml

from tensorforge.cli import main
from tensorforge.config import load_config, validate
from tensorforge.data import profile_file, read_frame
from tensorforge.errors import WorkbenchError
from tensorforge.generator import generate


def test_approval_and_warnings(classification, tmp_path):
    cfg, root = classification
    cfg["review"]["approved"] = False
    with pytest.raises(WorkbenchError, match="approved"):
        generate(cfg, root, tmp_path / "bundle")
    cfg["review"]["approved"] = True
    cfg.pop("submission")
    with pytest.raises(WorkbenchError) as exc:
        generate(cfg, root, tmp_path / "bundle")
    assert exc.value.code == "WARNING_ACK_REQUIRED"
    assert not (tmp_path / "bundle").exists()


def test_collision_and_safe_overwrite(classification, tmp_path):
    cfg, root = classification
    bundle = tmp_path / "bundle"
    generate(cfg, root, bundle)
    before = (bundle / "baseline.ipynb").read_bytes()
    with pytest.raises(WorkbenchError) as exc:
        generate(cfg, root, bundle)
    assert exc.value.code == "OUTPUT_EXISTS"
    assert (bundle / "baseline.ipynb").read_bytes() == before
    generate(cfg, root, bundle, overwrite=True)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "important.txt").write_text("keep me")
    with pytest.raises(WorkbenchError) as exc:
        generate(cfg, root, unrelated, overwrite=True)
    assert exc.value.code == "UNSAFE_OVERWRITE"
    assert (unrelated / "important.txt").read_text() == "keep me"


def test_planned_task_fails(classification):
    cfg, root = classification
    cfg["task"] = "inpainting"
    with pytest.raises(WorkbenchError) as exc:
        validate(cfg, root)
    assert exc.value.code == "TASK_NOT_IMPLEMENTED"


def test_metric_split_and_id_contracts(classification, tmp_path):
    base, root = classification
    cfg = copy.deepcopy(base)
    cfg["metric"] = "cer"
    with pytest.raises(WorkbenchError) as exc:
        validate(cfg, root)
    assert exc.value.code == "METRIC_INCOMPATIBLE"
    cfg = copy.deepcopy(base)
    cfg["split"] = {"strategy": "group", "column": "absent"}
    with pytest.raises(WorkbenchError) as exc:
        validate(cfg, root)
    assert exc.value.code == "MISSING_SPLIT_COLUMN"
    cfg = copy.deepcopy(base)
    bad = tmp_path / "bad.csv"
    bad.write_text("id,text\nx,some text\nx,other text\n")
    cfg["data"]["test"] = str(bad)
    with pytest.raises(WorkbenchError) as exc:
        validate(cfg, root)
    assert exc.value.code == "INVALID_IDS"


def test_config_strings_are_data_not_source(classification, tmp_path):
    cfg, root = classification
    payload = "\"\"\"); __import__('pathlib').Path('PWNED').touch(); #"
    cfg["params"]["unused"] = payload
    bundle = tmp_path / "safe"
    generate(cfg, root, bundle)
    source = "\n".join(c.source for c in nbformat.read(bundle / "baseline.ipynb", 4).cells)
    assert payload not in source
    assert json.loads((bundle / "config.json").read_text())["params"]["unused"] == payload
    assert not (Path.cwd() / "PWNED").exists()


@pytest.mark.parametrize(
    "content", ["x: 1\nx: 2", "!!python/object/apply:os.system ['touch PWNED']", "- not-a-mapping", "x: .nan"]
)
def test_yaml_rejected(content, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    with pytest.raises(WorkbenchError):
        load_config(path)


def test_bounded_profile_and_formats(tmp_path):
    frame = pd.DataFrame({"id": range(7), "text": [f"text {i}" for i in range(7)]})
    for extension in ("csv", "jsonl", "parquet"):
        path = tmp_path / f"data.{extension}"
        if extension == "csv":
            frame.to_csv(path, index=False)
        elif extension == "jsonl":
            frame.to_json(path, orient="records", lines=True)
        else:
            frame.to_parquet(path)
        profile = profile_file(path, limit=3)
        assert profile["sample_rows"] == 3
        assert profile["scope"] == "sample"
        assert profile["total_rows"] is None
        assert len(read_frame(path)) == 7


def test_malformed_jsonl(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"text": "ok"}\n{bad json}\n')
    with pytest.raises(WorkbenchError) as exc:
        read_frame(path)
    assert exc.value.code == "DATA_READ_FAILED"


def test_inspect_cli_emits_unapproved_draft(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("id,text,label\n1,hello,A\n2,world,B\n")
    out = tmp_path / "inspect"
    assert main(["inspect", str(data), "--output", str(out)]) == 0
    draft = yaml.safe_load((out / "config.draft.yaml").read_text())
    assert draft["review"]["approved"] is False
    assert not (out / "artifacts").exists()


def test_experimental_opt_in_before_dependencies(classification):
    cfg, root = classification
    cfg["task"] = "text_to_image"
    cfg["metric"] = "none"
    with pytest.raises(WorkbenchError) as exc:
        validate(cfg, root)
    assert exc.value.code == "EXPERIMENTAL_OPT_IN"


def test_leading_zero_ids_are_preserved(tmp_path):
    path = tmp_path / "ids.csv"
    path.write_text("id,text\n0001,hello\n0002,world\n")
    assert read_frame(path)["id"].tolist() == ["0001", "0002"]


def test_misaligned_ner_fails_generation(tmp_path):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text(json.dumps({"id": "a", "tokens": ["A", "B"], "tags": ["O"]}) + "\n")
    test.write_text(json.dumps({"id": "b", "tokens": ["C"]}) + "\n")
    cfg = {
        "schema_version": 1,
        "task": "token_classification",
        "metric": "entity_f1",
        "data": {"train": str(train), "test": str(test)},
        "columns": {"id": "id", "tokens": "tokens", "tags": "tags"},
    }
    with pytest.raises(WorkbenchError) as exc:
        validate(cfg, tmp_path, require_approval=False)
    assert exc.value.code == "TOKEN_TAG_ALIGNMENT"


def test_sample_ids_fail_before_notebook_generation(classification, tmp_path):
    cfg, root = classification
    sample = tmp_path / "sample.csv"
    sample.write_text("id,label\nunknown,\n")
    cfg["submission"]["sample"] = str(sample)
    with pytest.raises(WorkbenchError) as exc:
        generate(cfg, root, tmp_path / "bundle")
    assert exc.value.code == "SUBMISSION_IDS"
    assert not (tmp_path / "bundle").exists()


def test_infeasible_stratification_rejected_before_generation(classification, tmp_path):
    cfg, root = classification
    train = tmp_path / "rare.csv"
    train.write_text("id,text,label\n1,one,A\n2,two,A\n3,three,A\n4,four,B\n")
    cfg["data"]["train"] = str(train)
    with pytest.raises(WorkbenchError) as exc:
        generate(cfg, root, tmp_path / "bundle")
    assert exc.value.code == "INFEASIBLE_STRATIFICATION"
    assert not (tmp_path / "bundle").exists()
