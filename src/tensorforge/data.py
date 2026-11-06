"""Bounded inspection. Full training loads happen only during notebook execution."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .errors import WorkbenchError


def resolve(root, value):
    path = Path(value)
    return (Path(root) / path).resolve() if not path.is_absolute() else path.resolve()


def read_frame(path, limit=None):
    path = Path(path)
    try:
        if path.suffix.lower() == ".csv":
            # Preserve lexical IDs such as 00012; numeric targets are parsed explicitly.
            frame = pd.read_csv(path, nrows=limit, keep_default_na=False, dtype=object)
        elif path.suffix.lower() == ".jsonl":
            rows = []
            with path.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if limit is not None and i >= limit:
                        break
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"Line {i + 1}: expected JSON object")
                    rows.append(row)
            frame = pd.DataFrame(rows)
        elif path.suffix.lower() == ".parquet":
            source = pq.ParquetFile(path)
            batches = source.iter_batches(batch_size=limit or 65536)
            if limit:
                batch = next(batches, None)
                frame = batch.to_pandas() if batch is not None else pd.DataFrame()
            else:
                frame = source.read().to_pandas()
        else:
            raise WorkbenchError("UNSUPPORTED_FORMAT", f"Use CSV, JSONL or Parquet manifests: {path}")
        if frame.empty:
            raise WorkbenchError("EMPTY_DATA", f"No records: {path}")
        return frame
    except WorkbenchError:
        raise
    except (OSError, ValueError, TypeError) as e:
        raise WorkbenchError("DATA_READ_FAILED", f"{path}: {e}") from e


def fingerprint(path, max_bytes=100_000_000):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        return {"sha256": None, "reason": "File exceeds bounded hashing limit"}
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest()}


def profile_file(path, limit=1000):
    frame = read_frame(path, limit + 1)
    truncated = len(frame) > limit
    frame = frame.head(limit)
    columns = {}
    for col in frame:
        series = frame[col]
        texts = series.map(
            lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        )
        columns[str(col)] = {
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "empty": int(texts.eq("").sum()),
            "unique": int(texts.nunique()),
            "mean_length": round(float(texts.str.len().mean()), 2),
            "max_length": int(texts.str.len().max()),
        }
    canonical = frame.map(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else str(x))
    return {
        "path": str(path),
        "sample_rows": len(frame),
        "sample_cap": limit,
        "scope": "sample" if truncated else "full",
        "total_rows": None if truncated else len(frame),
        "columns": columns,
        "duplicate_rows": int(canonical.duplicated().sum()),
        "fingerprint": fingerprint(path),
    }


def propose(profile):
    cols = list(profile["columns"])
    text = next((c for c in cols if c.lower() in ("text", "content", "sentence", "review")), cols[-1])
    target = next((c for c in cols if c.lower() in ("label", "target", "class")), None)
    task = "text_classification" if target else "clustering"
    return {
        "schema_version": 1,
        "project_root": ".",
        "task": task,
        "data": {"train": Path(profile["path"]).name, "test": None},
        "columns": {"text": text, "target": target, "id": "id" if "id" in cols else None},
        "metric": "macro_f1" if target else "none",
        "seed": 42,
        "split": {"strategy": "stratified" if target else "none", "test_size": 0.25},
        "params": {},
        "review": {"approved": False, "acknowledged_warnings": []},
    }
