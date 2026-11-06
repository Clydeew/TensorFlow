import importlib.util
import json
import math
import shutil
import subprocess
from pathlib import Path

import yaml

from .data import read_frame, resolve
from .errors import WorkbenchError
from .languages import validate_language
from .registry import TASKS


class UniqueSafeLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    pairs = loader.construct_pairs(node, deep=deep)
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate YAML keys are not allowed")
    return dict(pairs)


UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def load_config(path):
    path = Path(path).resolve()
    try:
        if path.stat().st_size > 2_000_000:
            raise ValueError("Config exceeds 2 MB limit")
        cfg = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueSafeLoader)
        if not isinstance(cfg, dict):
            raise ValueError("Configuration must be a mapping")
        json.dumps(cfg, allow_nan=False)  # Reject YAML objects/dates, non-finite values, recursive aliases.
        root = resolve(path.parent, cfg.get("project_root", "."))
        return cfg, root
    except (OSError, ValueError, TypeError, RecursionError, yaml.YAMLError) as e:
        raise WorkbenchError("INVALID_CONFIG", str(e)) from e


def warn(ident, message):
    return {"id": ident, "message": message}


def validate(cfg, root, *, require_approval=True, experimental=False):
    try:
        return _validate(cfg, root, require_approval=require_approval, experimental=experimental)
    except WorkbenchError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise WorkbenchError("INVALID_CONFIG", str(e)) from e


def _validate(cfg, root, *, require_approval, experimental):
    if cfg.get("schema_version") != 1:
        raise WorkbenchError("SCHEMA_VERSION", "schema_version must be 1")
    try:
        validate_language(cfg.get("language", {}))
    except ValueError as e:
        raise WorkbenchError("INVALID_LANGUAGE_CONFIG", str(e)) from e
    task = TASKS.get(cfg.get("task"))
    if not task:
        raise WorkbenchError("UNKNOWN_TASK", "Run tensorforge list-tasks for valid IDs")
    if task.status == "planned":
        raise WorkbenchError("TASK_NOT_IMPLEMENTED", f"{task.id}: {task.note}")
    if task.status == "experimental" and not experimental:
        raise WorkbenchError(
            "EXPERIMENTAL_OPT_IN", "Pass --experimental to use an unverified local-model template"
        )
    if cfg.get("metric") not in task.metrics:
        raise WorkbenchError("METRIC_INCOMPATIBLE", f"{task.id} allows {task.metrics}")
    for section in ("params", "limits", "review"):
        if section in cfg and not isinstance(cfg[section], dict):
            raise ValueError(f"{section} must be a mapping")
    for name, default in (("max_rows", 100000), ("max_image_pixels", 25000000), ("max_pages", 100)):
        value = cfg.get("limits", {}).get(name, default)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"limits.{name} must be a positive integer")
    if cfg.get("params", {}).get("device", "cpu") not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda or mps")
    if not isinstance(cfg.get("seed", 42), int):
        raise ValueError("seed must be an integer")
    data, cols = cfg.get("data", {}), cfg.get("columns", {})
    if not isinstance(data, dict) or not isinstance(cols, dict):
        raise ValueError("data and columns must be mappings")
    warnings, profiles = [], {}
    sampled_roles = set()
    train_task = task.id in {
        "text_classification",
        "text_regression",
        "multilabel_classification",
        "text_pair",
        "clustering",
        "seq2seq",
        "token_classification",
    }
    needed = (
        ["train", "test"] if train_task else (["corpus", "queries"] if task.id == "retrieval" else ["test"])
    )
    if task.backend == "clip":
        needed = ["corpus", "queries"]
    for role in needed + (["qrels"] if data.get("qrels") else []):
        if not data.get(role):
            raise WorkbenchError("MISSING_DATA_ROLE", f"data.{role} is required")
        path = resolve(root, data[role])
        frame = read_frame(path, 1001)
        profiles[role] = frame.head(1000)
        if len(frame) > 1000:
            sampled_roles.add(role)
            warnings.append(
                warn(
                    "SAMPLED_VALIDATION",
                    "Only the first 1000 records were inspected; notebook revalidates full data",
                )
            )
    required = {
        "text_classification": ["text", "target"],
        "text_regression": ["text", "target"],
        "multilabel_classification": ["text", "target"],
        "text_pair": ["text", "text_b", "target"],
        "clustering": ["text"],
        "seq2seq": ["text", "target"],
        "token_classification": ["tokens", "tags"],
        "text_rendering": ["text"],
        "ocr": ["image"],
        "document_extraction": ["image"],
        "image_captioning": ["image"],
        "visual_qa": ["image", "question"],
        "text_to_image": ["text"],
        "image_editing": ["image", "text"],
    }.get(task.id, [])
    for role, frame in profiles.items():
        if task.id == "retrieval":
            keys = {
                "corpus": ["id", "text"],
                "queries": ["id", "text"],
                "qrels": ["query_id", "document_id", "relevance"],
            }[role]
        elif task.backend == "clip":
            is_image = (role == "corpus") == (task.id == "text_image_retrieval")
            keys = ["id", "image" if is_image else "text"]
        else:
            keys = [k for k in required if role == "train" or k not in ("target", "tags")]
        for key in keys:
            column = cols.get(key)
            if column not in frame.columns:
                raise WorkbenchError("MISSING_COLUMN", f"columns.{key} must name a field in {role}")
            if frame[column].isna().any():
                raise WorkbenchError("NULL_FIELD", f"{role}.{column} contains null values")
            if (
                key in ("text", "text_b", "question")
                and not frame[column].map(lambda v: isinstance(v, str)).all()
            ):
                raise WorkbenchError("INVALID_TEXT", f"{role}.{column} must contain text strings")
            if key == "image":
                for value in frame[column]:
                    if not resolve(root, value).is_file():
                        raise WorkbenchError("MISSING_ASSET", f"{role}: {value}")
        ident = cols.get("id")
        if (
            ident
            and role != "qrels"
            and (
                ident not in frame
                or frame[ident].isna().any()
                or frame[ident].astype(str).eq("").any()
                or frame[ident].astype(str).duplicated().any()
            )
        ):
            raise WorkbenchError("INVALID_IDS", f"{role}: IDs must be present, non-null and unique")
    if task.id in ("token_classification", "multilabel_classification"):

        def parse_list(value):
            parsed = json.loads(value) if isinstance(value, str) else value
            if not isinstance(parsed, list) or not all(isinstance(v, str) for v in parsed):
                raise WorkbenchError("INVALID_SEQUENCE", "Expected a list of strings or JSON array")
            return parsed

        if task.id == "token_classification":
            for role, frame in profiles.items():
                for _, row in frame.iterrows():
                    tokens = parse_list(row[cols["tokens"]])
                    if not tokens:
                        raise WorkbenchError("INVALID_SEQUENCE", "Empty token sequence")
                    if role == "train":
                        tags = parse_list(row[cols["tags"]])
                        if len(tokens) != len(tags):
                            raise WorkbenchError(
                                "TOKEN_TAG_ALIGNMENT", "Tokens and tags have different lengths"
                            )
                        previous = None
                        for tag in tags:
                            if tag != "O" and not (len(tag) > 2 and tag[:2] in ("B-", "I-")):
                                raise WorkbenchError("INVALID_BIO", f"Invalid tag: {tag}")
                            if tag.startswith("I-") and previous != tag[2:]:
                                raise WorkbenchError(
                                    "INVALID_BIO", "I-tag starts a new entity; fix training labels"
                                )
                            previous = tag[2:] if tag != "O" else None
        else:
            for value in profiles["train"][cols["target"]]:
                parse_list(value)
    if cfg.get("submission"):
        spec = cfg["submission"]
        if (
            task.id in ("retrieval", "text_rendering", "text_to_image", "image_editing")
            or task.backend == "clip"
        ):
            raise WorkbenchError(
                "UNSUPPORTED_SUBMISSION",
                "This adapter exports a manifest/rankings, not scalar submission rows",
            )
        sample = read_frame(resolve(root, spec["sample"]), 1001)
        ident, target = spec["id_column"], spec["prediction_column"]
        if set(sample.columns) != {ident, target} or not cols.get("id"):
            raise WorkbenchError(
                "SUBMISSION_SCHEMA", "Use exactly the ID and prediction columns and set columns.id"
            )
        if sample[ident].astype(str).duplicated().any() or sample[ident].astype(str).eq("").any():
            raise WorkbenchError("SUBMISSION_IDS", "Sample IDs must be nonempty and unique")
        if (
            len(sample) <= 1000
            and "test" not in sampled_roles
            and set(sample[ident].astype(str)) != set(profiles["test"][cols["id"]].astype(str))
        ):
            raise WorkbenchError("SUBMISSION_IDS", "Sample/test ID sets differ")
    if task.id == "text_regression" or (
        task.id == "text_pair" and cfg.get("params", {}).get("variant") == "regression"
    ):
        for v in profiles["train"][cols["target"]]:
            if not math.isfinite(float(v)):
                raise ValueError("Regression targets must be finite numeric values")
    if task.id == "text_pair":
        regression = cfg.get("params", {}).get("variant") == "regression"
        if cfg.get("params", {}).get("variant", "classification") not in ("classification", "regression"):
            raise ValueError("Pair variant must be classification or regression")
        if regression and cfg.get("split", {}).get("strategy") == "stratified":
            raise WorkbenchError("INVALID_SPLIT", "Pair regression cannot use class stratification")
        if (cfg["metric"] in ("mae", "spearman")) != regression:
            raise WorkbenchError(
                "METRIC_INCOMPATIBLE", "Pair regression requires variant: regression and mae/spearman"
            )
        warnings.append(
            warn(
                "PAIR_GROUP_LEAKAGE",
                "Reused pair texts may cross validation; configure group split or acknowledge",
            )
        )
    if train_task:
        split = cfg.get("split", {})
        strategy = split.get("strategy")
        allowed = {"none"} if task.id == "clustering" else {"random", "stratified", "group", "time"}
        if strategy not in allowed:
            raise WorkbenchError("INVALID_SPLIT", f"{task.id}: choose split.strategy from {sorted(allowed)}")
        if strategy != "none" and not 0 < float(split.get("test_size", 0.25)) < 1:
            raise ValueError("split.test_size must be between 0 and 1")
        for key in ("group", "time"):
            if strategy == key and split.get("column") not in profiles["train"]:
                raise WorkbenchError("MISSING_SPLIT_COLUMN", f"split.column required for {key}")
        if strategy == "stratified" and task.id not in ("text_classification", "text_pair"):
            raise WorkbenchError(
                "INVALID_SPLIT", "Stratification is only supported for single-label classification"
            )
        if strategy == "stratified" and "train" not in sampled_roles:
            from sklearn.model_selection import train_test_split

            try:
                train_test_split(
                    range(len(profiles["train"])),
                    test_size=float(split.get("test_size", 0.25)),
                    random_state=cfg.get("seed", 42),
                    stratify=profiles["train"][cols["target"]],
                )
            except ValueError as e:
                raise WorkbenchError("INFEASIBLE_STRATIFICATION", str(e)) from e
        if strategy == "random":
            warnings.append(
                warn(
                    "RANDOM_SPLIT",
                    "A row split does not isolate shared sources; confirm this task permits it",
                )
            )
        key = cols.get("text") or cols.get("tokens")
        if key:

            def stringify(v):
                return json.dumps(v, ensure_ascii=False) if isinstance(v, list) else str(v)

            train = profiles["train"][key].map(stringify)
            test = profiles["test"][key].map(stringify)
            if train.duplicated().any():
                warnings.append(
                    warn(
                        "DUPLICATE_TRAIN",
                        "Repeated training inputs may leak across validation; use grouped split",
                    )
                )
            if set(train) & set(test):
                warnings.append(
                    warn("TRAIN_TEST_OVERLAP", "Exact input overlap found in inspected train/test samples")
                )
    if task.backend != "core":
        for dep in task.dependencies:
            if importlib.util.find_spec(dep) is None:
                raise WorkbenchError(
                    "MISSING_DEPENDENCY", f"Install optional dependencies; missing import: {dep}"
                )
        if task.id == "ocr" and shutil.which("tesseract") is None:
            raise WorkbenchError(
                "MISSING_TESSERACT", "Install Tesseract and language data; see docs/PORTABILITY.md"
            )
        if task.id == "ocr":
            command = [shutil.which("tesseract"), "--list-langs"]
            directory = cfg.get("params", {}).get("tessdata_path")
            if directory:
                command += ["--tessdata-dir", str(resolve(root, directory))]
            probe = subprocess.run(command, capture_output=True, text=True, timeout=10)
            available = set(probe.stdout.splitlines()[1:])
            requested = set(cfg.get("params", {}).get("language", "eng").split("+"))
            if probe.returncode or not requested <= available:
                raise WorkbenchError(
                    "MISSING_OCR_LANGUAGE", "Stage requested language data; see scripts/stage_ocr.py"
                )
            if cfg["metric"] != "none" and cols.get("target") not in profiles["test"]:
                raise WorkbenchError(
                    "MISSING_COLUMN", "OCR evaluation requires columns.target in the test manifest"
                )
        if task.backend in ("vision", "diffusion", "clip"):
            language = cfg.get("language", {})
            if language.get("primary", "en") != "en" or language.get("additional"):
                warnings.append(
                    warn(
                        "MODEL_LANGUAGE_UNVERIFIED",
                        "Verify local model language/script coverage with language-specific fixtures; Unicode input support is not proof of language competence",
                    )
                )
            model = cfg.get("params", {}).get("model_path")
            if not model or not resolve(root, model).is_dir():
                raise WorkbenchError(
                    "MISSING_MODEL", "params.model_path must point to a pre-staged local model directory"
                )
            metadata = "model_index.json" if task.backend == "diffusion" else "config.json"
            if not (resolve(root, model) / metadata).is_file():
                raise WorkbenchError("INCOMPLETE_MODEL", f"Missing local model metadata: {metadata}")
        warnings.append(warn("EXPERIMENTAL_BACKEND", task.note))
    if task.id == "retrieval" and not data.get("qrels"):
        warnings.append(
            warn("NO_QRELS", "Rankings will be produced; evaluation unavailable without relevance judgments")
        )
    if not cfg.get("submission"):
        warnings.append(
            warn(
                "NO_SUBMISSION_SPEC",
                "Outputs are generic predictions, not an official competition submission",
            )
        )
    warnings = list({w["id"]: w for w in warnings}.values())
    review = cfg.get("review", {})
    if require_approval:
        if review.get("approved") is not True:
            raise WorkbenchError(
                "REVIEW_REQUIRED", "Set review.approved: true after inspecting the proposed task and contract"
            )
        missing = {w["id"] for w in warnings} - set(review.get("acknowledged_warnings", []))
        if missing:
            raise WorkbenchError(
                "WARNING_ACK_REQUIRED", f"Acknowledge stable warning IDs: {', '.join(sorted(missing))}"
            )
    return warnings
