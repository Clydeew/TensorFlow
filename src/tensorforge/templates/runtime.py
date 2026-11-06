"""Trusted runtime embedded verbatim in generated notebooks. No generator import."""

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
for _threads in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# The notebook configuration cell supplies these; defaults allow isolated metric tests.
CONFIG = globals().get("CONFIG", {})
PROJECT_ROOT = globals().get("PROJECT_ROOT", Path.cwd())
COLS = globals().get("COLS", {})
PARAMS = globals().get("PARAMS", {})
TASK = globals().get("TASK", "")
SEED = globals().get("SEED", 42)
ARTIFACT_DIR = globals().get("ARTIFACT_DIR", Path.cwd())
METRICS = globals().get("METRICS", {})


def check(condition, message):
    if not condition:
        raise ValueError(message)


def path_for(value):
    p = Path(value)
    return (PROJECT_ROOT / p).resolve() if not p.is_absolute() else p.resolve()


def load_data(role):
    path = path_for(CONFIG["data"][role])
    limit = int(CONFIG.get("limits", {}).get("max_rows", 100_000))
    check(limit > 0, "max_rows must be positive")
    if path.suffix == ".csv":
        frame = pd.read_csv(path, nrows=limit + 1, keep_default_na=False, dtype=object)
    elif path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                check(i < limit, f"{role} exceeds max_rows; increase explicitly")
                row = json.loads(line)
                check(isinstance(row, dict), "Each JSONL line must be an object")
                rows.append(row)
        frame = pd.DataFrame(rows)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        check(pq.ParquetFile(path).metadata.num_rows <= limit, f"{role} exceeds max_rows")
        frame = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported manifest format")
    check(0 < len(frame) <= limit, f"{role} is empty or exceeds max_rows; no silent truncation")
    frame = frame.reset_index(drop=True)
    id_col = COLS.get("id")
    if id_col and role != "qrels":
        check(id_col in frame, f"Missing ID column: {id_col}")
        check(not frame[id_col].isna().any(), "Null IDs")
        check(not frame[id_col].astype(str).eq("").any(), "Empty IDs")
        check(not frame[id_col].astype(str).duplicated().any(), "Duplicate IDs")
    return frame


def strings(frame, key="text"):
    col = COLS[key]
    check(col in frame, f"Missing {key} column")
    check(not frame[col].isna().any(), f"Null {key} values")
    check(frame[col].map(lambda x: isinstance(x, str)).all(), f"{key} must contain strings")
    return frame[col].tolist()


def ids(frame):
    return frame[COLS["id"]].astype(str).tolist() if COLS.get("id") else [str(i) for i in range(len(frame))]


def ack(ident, message):
    check(ident in CONFIG.get("review", {}).get("acknowledged_warnings", []), f"{ident}: {message}")
    print(f"Warning {ident}: {message}")


def split_indices(frame):
    split = CONFIG["split"]
    strategy, fraction = split["strategy"], float(split.get("test_size", 0.25))
    ix = np.arange(len(frame))
    check(len(frame) >= 4, "Need at least four training records for holdout")
    if strategy == "group":
        groups = frame[split["column"]]
        check(not groups.isna().any(), "Null split groups")
        a, b = next(
            GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=SEED).split(ix, groups=groups)
        )
        check(not set(groups.iloc[a]) & set(groups.iloc[b]), "Group boundary crossed")
    elif strategy == "time":
        times = pd.to_datetime(frame[split["column"]], errors="raise", utc=True)
        check(not times.isna().any(), "Null timestamps")
        order = np.argsort(times.to_numpy(), kind="stable")
        cut = max(1, min(len(ix) - 1, int(len(ix) * (1 - fraction))))
        a, b = order[:cut], order[cut:]
        check(
            times.iloc[a].max() < times.iloc[b].min(),
            "Tied timestamps cross split; choose explicit time/group boundary",
        )
    else:
        y = frame[COLS["target"]] if strategy == "stratified" else None
        a, b = train_test_split(ix, test_size=fraction, random_state=SEED, stratify=y)
    # Duplicate source sequences/text must not silently cross holdout boundaries.
    key = COLS.get("text") or COLS.get("tokens")
    if key:

        def canon(v):
            return json.dumps(v, ensure_ascii=False) if isinstance(v, list) else str(v)

        left, right = set(frame.iloc[a][key].map(canon)), set(frame.iloc[b][key].map(canon))
        if left & right:
            ack(
                "DUPLICATE_TRAIN",
                "Exact training inputs cross validation; group them or acknowledge this limitation",
            )
    return a, b


def vectorizer():
    return TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=30_000, sublinear_tf=True)


def numeric(y):
    arr = np.asarray(y, dtype=float)
    check(np.isfinite(arr).all(), "Nonfinite numeric target/prediction")
    return arr


def scalar_metric(y, pred, metric=None):
    metric = metric or CONFIG["metric"]
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    if metric in ("macro_f1", "micro_f1"):
        return float(f1_score(y, pred, average=metric.split("_")[0], zero_division=0))
    if metric == "mae":
        return float(mean_absolute_error(numeric(y), numeric(pred)))
    if metric == "rmse":
        return float(math.sqrt(mean_squared_error(numeric(y), numeric(pred))))
    if metric == "spearman":
        value = float(spearmanr(numeric(y), numeric(pred)).statistic)
        check(math.isfinite(value), "Spearman undefined for constant targets/predictions")
        return value
    raise ValueError(f"Unsupported metric {metric}")


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def text_error(refs, preds, words=False):
    pairs = [(a.split(), b.split()) if words else (a, b) for a, b in zip(refs, preds, strict=True)]
    denominator = sum(len(a) for a, _ in pairs)
    check(denominator > 0, "CER/WER undefined for empty reference corpus")
    return sum(edit_distance(a, b) for a, b in pairs) / denominator


def as_list(value):
    if isinstance(value, str):
        value = json.loads(value)
    check(
        isinstance(value, list) and all(isinstance(x, str) for x in value),
        "Expected list of strings or JSON array",
    )
    return value


def safe_bio(tags):
    result, previous = [], None
    for tag in tags:
        check(tag == "O" or (len(tag) > 2 and tag[:2] in ("B-", "I-")), f"Invalid BIO tag {tag}")
        if tag.startswith("I-") and previous != tag[2:]:
            tag = "B-" + tag[2:]
        result.append(tag)
        previous = tag[2:] if tag != "O" else None
    return result


def entities(tags):
    tags = safe_bio(tags)
    out, start, typ = set(), None, None
    for i, tag in enumerate(tags + ["O"]):
        if start is not None and (tag == "O" or tag.startswith("B-") or tag[2:] != typ):
            out.add((start, i, typ))
            start = None
        if tag.startswith("B-"):
            start, typ = i, tag[2:]
    return out


def entity_f1(refs, preds):
    tp = fp = fn = 0
    for ref, pred in zip(refs, preds, strict=True):
        a, b = entities(ref), entities(pred)
        tp += len(a & b)
        fp += len(b - a)
        fn += len(a - b)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def predict_core(train, test, task):
    if task == "token_classification":
        lexicon = defaultdict(Counter)
        for _, row in train.iterrows():
            tokens, tags = as_list(row[COLS["tokens"]]), as_list(row[COLS["tags"]])
            check(len(tokens) == len(tags), "Token/tag length mismatch")
            check(tags == safe_bio(tags), "Training BIO sequence invalid; fix labels explicitly")
            for token, tag in zip(tokens, tags, strict=True):
                lexicon[token.lower()][tag] += 1
        return [
            safe_bio(
                [lexicon[t.lower()].most_common(1)[0][0] if t.lower() in lexicon else "O" for t in as_list(v)]
            )
            for v in test[COLS["tokens"]]
        ]
    if task == "text_pair":
        a, b = strings(train), strings(train, "text_b")
        vec = vectorizer().fit(a + b)

        def features(frame):
            x, y = vec.transform(strings(frame)), vec.transform(strings(frame, "text_b"))
            difference = abs(x - y)
            components = (
                [x + y, difference, x.multiply(y)]
                if PARAMS.get("symmetric", True)
                else [x, y, difference, x.multiply(y)]
            )
            return sparse.hstack(components, format="csr")

        regression = PARAMS.get("variant") == "regression"
        model = (
            Ridge(alpha=1, solver="lsqr")
            if regression
            else LogisticRegression(max_iter=1000, random_state=SEED)
        )
        y = numeric(train[COLS["target"]]) if regression else train[COLS["target"]].astype(str)
        check(regression or len(set(y)) >= 2, "Classification needs at least two classes in training split")
        return model.fit(features(train), y).predict(features(test))
    vec = vectorizer()
    x, xt = vec.fit_transform(strings(train)), None
    xt = vec.transform(strings(test))
    if task == "clustering":
        count = int(PARAMS.get("n_clusters", 3))
        check(1 < count <= len(train), "n_clusters must be between 2 and training row count")
        model = MiniBatchKMeans(n_clusters=count, random_state=SEED, n_init=10, batch_size=256).fit(x)
        terms = vec.get_feature_names_out()
        topics = {
            str(i): terms[c.argsort()[-10:][::-1]].tolist() for i, c in enumerate(model.cluster_centers_)
        }
        (ARTIFACT_DIR / "cluster_terms.json").write_text(json.dumps(topics, indent=2), encoding="utf-8")
        return model.predict(xt)
    if task == "seq2seq":
        refs = strings(train, "target")
        # Bounded batch multiplication avoids an unbounded test-by-train dense matrix.
        out = []
        for row in range(0, xt.shape[0], 64):
            scores = (xt[row : row + 64] @ x.T).toarray()
            out.extend(refs[i] for i in scores.argmax(axis=1))
        return out
    if task == "multilabel_classification":
        labels = [as_list(v) for v in train[COLS["target"]]]
        encoder = MultiLabelBinarizer()
        y = encoder.fit_transform(labels)
        check(y.shape[1] >= 2, "Multilabel baseline needs at least two distinct labels")
        estimator = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=SEED)).fit(x, y)
        probabilities = np.asarray(estimator.predict_proba(xt))
        threshold = float(PARAMS.get("threshold", 0.5))
        check(0 < threshold < 1, "threshold must be strictly between zero and one")
        return [list(v) for v in encoder.inverse_transform((probabilities >= threshold).astype(int))]
    regression = task == "text_regression"
    y = numeric(train[COLS["target"]]) if regression else train[COLS["target"]].astype(str)
    check(regression or len(set(y)) >= 2, "Classification needs at least two classes in training split")
    estimator = (
        Ridge(alpha=1, solver="lsqr") if regression else LogisticRegression(max_iter=1000, random_state=SEED)
    )
    return estimator.fit(x, y).predict(xt)


def evaluate_core(frame, pred, task):
    if task == "token_classification":
        refs = [as_list(v) for v in frame[COLS["tags"]]]
        for ref, guess, toks in zip(refs, pred, frame[COLS["tokens"]], strict=True):
            check(len(ref) == len(guess) == len(as_list(toks)), "Token/tag output misalignment")
        return entity_f1(refs, pred)
    if task == "seq2seq":
        refs = strings(frame, "target")
        return text_error(refs, pred) if CONFIG["metric"] == "cer" else float(np.mean(np.array(refs) == pred))
    if task == "multilabel_classification":
        refs = [as_list(v) for v in frame[COLS["target"]]]
        vocab = sorted(set(x for row in refs + pred for x in row))
        encoder = MultiLabelBinarizer(classes=vocab)
        return scalar_metric(encoder.fit_transform(refs), encoder.transform(pred))
    regression = task == "text_regression" or (task == "text_pair" and PARAMS.get("variant") == "regression")
    refs = numeric(frame[COLS["target"]]) if regression else frame[COLS["target"]].astype(str)
    return scalar_metric(refs, pred)


def write_predictions(frame, predictions):
    check(len(frame) == len(predictions), "Prediction row count mismatch")
    values = [json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple)) else v for v in predictions]
    output = pd.DataFrame({"id": ids(frame), "prediction": values})
    submission = CONFIG.get("submission")
    if submission:
        check(COLS.get("id"), "Official submission mapping requires columns.id")
        sample = pd.read_csv(path_for(submission["sample"]), dtype=str, keep_default_na=False)
        ident, target = submission["id_column"], submission["prediction_column"]
        check(set(sample.columns) == {ident, target}, "Only two-column ID/prediction submissions supported")
        check(not sample[ident].duplicated().any(), "Duplicate sample submission IDs")
        check(set(sample[ident]) == set(output["id"]), "Sample/test ID mismatch")
        sample[target] = sample[ident].map(output.set_index("id")["prediction"])
        check(not sample[target].isna().any(), "Missing submission prediction")
        sample.to_csv(ARTIFACT_DIR / "submission.csv", index=False)
    output.to_csv(ARTIFACT_DIR / "predictions.csv", index=False)
    return output


def retrieval_metrics(rankings, qrels, k):
    values = []
    for query, ranked in rankings.items():
        relevant = qrels.get(query, {})
        if not any(v > 0 for v in relevant.values()):
            continue
        hits = [i for i, doc in enumerate(ranked[:k], 1) if relevant.get(doc, 0) > 0]
        recall = len(hits) / sum(v > 0 for v in relevant.values())
        dcg = sum((2 ** relevant.get(doc, 0) - 1) / math.log2(i + 1) for i, doc in enumerate(ranked[:k], 1))
        ideal = sum(
            (2**value - 1) / math.log2(i + 1)
            for i, value in enumerate(sorted(relevant.values(), reverse=True)[:k], 1)
        )
        values.append(
            {"mrr": 1 / hits[0] if hits else 0, "recall_at_k": recall, "ndcg": dcg / ideal if ideal else 0}
        )
    check(values, "No judged queries with positive relevance; metrics unavailable")
    return {key: float(np.mean([v[key] for v in values])) for key in values[0]}


def run_retrieval():
    check(
        not CONFIG.get("submission"),
        "Retrieval uses long-format rankings; tabular submission mapping unsupported",
    )
    corpus, queries = load_data("corpus"), load_data("queries")
    vec = vectorizer()
    matrix = vec.fit_transform(strings(corpus))
    query_matrix = vec.transform(strings(queries))
    doc_ids, query_ids = ids(corpus), ids(queries)
    k = int(PARAMS.get("top_k", 10))
    check(k > 0, "top_k must be positive")
    rankings, rows = {}, []
    for qi, query in enumerate(query_ids):
        scores = (query_matrix[qi] @ matrix.T).toarray().ravel()
        order = sorted(range(len(doc_ids)), key=lambda i: (-float(scores[i]), doc_ids[i]))[:k]
        rankings[query] = [doc_ids[i] for i in order]
        rows.extend(
            {"query_id": query, "document_id": doc_ids[i], "rank": rank, "score": float(scores[i])}
            for rank, i in enumerate(order, 1)
        )
    pd.DataFrame(rows).to_csv(ARTIFACT_DIR / "rankings.csv", index=False)
    if CONFIG["data"].get("qrels"):
        judgments = load_data("qrels")
        mapping = defaultdict(dict)
        for _, r in judgments.iterrows():
            query, doc = str(r[COLS["query_id"]]), str(r[COLS["document_id"]])
            check(query in query_ids and doc in doc_ids, "Qrels refer to unknown query/document")
            check(doc not in mapping[query], "Duplicate qrel pair")
            rel = float(r[COLS["relevance"]])
            check(math.isfinite(rel) and 0 <= rel <= 20, "Relevance must be finite in [0,20]")
            mapping[query][doc] = rel
        METRICS.update(retrieval_metrics(rankings, mapping, k))
    else:
        print("Evaluation unavailable: no qrels")


def open_image(value):
    p = path_for(value)
    max_pixels = int(CONFIG.get("limits", {}).get("max_image_pixels", 25_000_000))
    with Image.open(p) as source:
        check(source.width * source.height <= max_pixels, "Image exceeds max_image_pixels")
        source.load()
        return source.convert("RGB")


def run_rendering():
    frame = load_data("test")
    width, height = int(PARAMS.get("width", 1024)), int(PARAMS.get("height", 512))
    check(
        0 < width * height <= int(CONFIG.get("limits", {}).get("max_image_pixels", 25_000_000)),
        "Invalid image size",
    )
    font = (
        ImageFont.truetype(str(path_for(PARAMS["font_path"])), int(PARAMS.get("font_size", 28)))
        if PARAMS.get("font_path")
        else ImageFont.load_default(size=28)
    )
    rows = []
    for i, (ident, text) in enumerate(zip(ids(frame), strings(frame), strict=True)):
        check(
            PARAMS.get("font_path") or text.isascii(),
            "Non-ASCII rendering requires an explicit local font covering those glyphs",
        )
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        box = draw.multiline_textbbox((20, 20), text, font=font)
        check(
            box[2] <= width - 20 and box[3] <= height - 20,
            "Text would be clipped; increase dimensions or wrap explicitly",
        )
        draw.multiline_text((20, 20), text, font=font, fill="black")
        name = f"image-{i:06d}.png"
        image.save(ARTIFACT_DIR / name)
        rows.append({"id": ident, "artifact": name, "text": text})
    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


def run_documents():
    frame = load_data("test")
    texts = []
    pages_limit = int(CONFIG.get("limits", {}).get("max_pages", 100))
    tess_config = ""
    if PARAMS.get("tessdata_path"):
        directory = str(path_for(PARAMS["tessdata_path"]))
        check('"' not in directory, "Tessdata paths cannot contain double quotes")
        tess_config = f'--tessdata-dir "{directory}"'
    for value in frame[COLS["image"]]:
        path = path_for(value)
        if TASK == "ocr":
            import pytesseract

            if path.suffix.lower() == ".pdf":
                import pypdfium2 as pdfium

                with pdfium.PdfDocument(path) as pdf:
                    check(len(pdf) <= pages_limit, "PDF exceeds page limit")
                    pages = []
                    for index in range(len(pdf)):
                        page = pdf[index]
                        check(
                            page.get_width() * page.get_height() * 4
                            <= int(CONFIG.get("limits", {}).get("max_image_pixels", 25_000_000)),
                            "PDF rendering exceeds image budget",
                        )
                        bitmap = page.render(scale=2)
                        try:
                            pages.append(
                                pytesseract.image_to_string(
                                    bitmap.to_pil(), lang=PARAMS.get("language", "eng"), config=tess_config
                                )
                            )
                        finally:
                            bitmap.close()
                            page.close()
                    text = "\n\f\n".join(pages)
            else:
                text = pytesseract.image_to_string(
                    open_image(value), lang=PARAMS.get("language", "eng"), config=tess_config
                )
        elif path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(path)
            check(not reader.is_encrypted, "Encrypted PDF is unsupported")
            check(len(reader.pages) <= pages_limit, "PDF exceeds page limit")
            text = "\n\f\n".join(p.extract_text() or "" for p in reader.pages)
            check(text.strip(), "No usable PDF text layer; select OCR explicitly")
        elif path.suffix.lower() == ".docx":
            from docx import Document

            document = Document(path)
            text = "\n".join(
                [p.text for p in document.paragraphs]
                + ["\t".join(c.text for c in r.cells) for t in document.tables for r in t.rows]
            )
        elif path.suffix.lower() in (".html", ".htm"):
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
            for elem in soup(["script", "style"]):
                elem.decompose()
            text = soup.get_text("\n", strip=True)
        else:
            check(path.suffix.lower() in (".txt", ".md", ".xml"), "Unsupported document format")
            text = path.read_text(encoding="utf-8")
        texts.append(text)
    if CONFIG["metric"] != "none":
        refs = strings(frame, "target")
        METRICS[CONFIG["metric"]] = text_error(refs, texts, words=CONFIG["metric"] == "wer")
    return write_predictions(frame, texts)


def run_local_vision():
    import torch
    from transformers import AutoProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

    frame = load_data("test")
    model_path = path_for(PARAMS["model_path"])
    device = PARAMS.get("device", "cpu")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    cls = BlipForConditionalGeneration if TASK == "image_captioning" else BlipForQuestionAnswering
    model = cls.from_pretrained(model_path, local_files_only=True).to(device).eval()
    predictions = []
    for _, row in frame.iterrows():
        args = {"images": open_image(row[COLS["image"]]), "return_tensors": "pt"}
        if TASK == "visual_qa":
            args["text"] = str(row[COLS["question"]])
        inputs = processor(**args).to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=int(PARAMS.get("max_new_tokens", 64)))
        predictions.append(processor.decode(output[0], skip_special_tokens=True))
    return write_predictions(frame, predictions)


def run_diffusion():
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline

    frame = load_data("test")
    device = PARAMS.get("device", "cpu")
    cls = StableDiffusionPipeline if TASK == "text_to_image" else StableDiffusionInstructPix2PixPipeline
    pipeline = cls.from_pretrained(
        path_for(PARAMS["model_path"]), local_files_only=True, torch_dtype=torch.float32, use_safetensors=True
    ).to(device)
    count = int(PARAMS.get("images_per_prompt", 1))
    width, height = int(PARAMS.get("width", 512)), int(PARAMS.get("height", 512))
    check(1 <= count <= 8, "images_per_prompt must be in [1,8]")
    check(
        width > 0 and height > 0 and width % 8 == height % 8 == 0,
        "Dimensions must be positive multiples of 8",
    )
    check(
        width * height <= int(CONFIG.get("limits", {}).get("max_image_pixels", 25_000_000)),
        "Output exceeds pixel limit",
    )
    steps = int(PARAMS.get("steps", 20))
    check(1 <= steps <= 100, "steps must be in [1,100]")
    rows = []
    for i, (_, row) in enumerate(frame.iterrows()):
        for variant in range(count):
            seed = SEED + i * count + variant
            kwargs = {
                "prompt": str(row[COLS["text"]]),
                "generator": torch.Generator(device="cpu").manual_seed(seed),
                "num_inference_steps": steps,
                "guidance_scale": float(PARAMS.get("guidance", 7.5)),
            }
            if TASK == "image_editing":
                kwargs["image"] = open_image(row[COLS["image"]]).resize((width, height))
                kwargs["image_guidance_scale"] = float(PARAMS.get("image_guidance", 1.5))
            else:
                kwargs.update(width=width, height=height)
            result = pipeline(**kwargs)
            check(
                not getattr(result, "nsfw_content_detected", None) or not any(result.nsfw_content_detected),
                "Backend rejected this output",
            )
            name = f"image-{i:06d}-{variant}.png"
            result.images[0].save(ARTIFACT_DIR / name)
            rows.append({"id": ids(frame)[i], "variant": variant, "seed": seed, "artifact": name})
    (ARTIFACT_DIR / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def run_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    corpus, queries = load_data("corpus"), load_data("queries")
    device = PARAMS.get("device", "cpu")
    source = path_for(PARAMS["model_path"])
    processor = CLIPProcessor.from_pretrained(source, local_files_only=True)
    model = CLIPModel.from_pretrained(source, local_files_only=True).to(device).eval()

    def encode(frame, image):
        out = []
        for _, row in frame.iterrows():
            inputs = (
                processor(images=open_image(row[COLS["image"]]), return_tensors="pt")
                if image
                else processor(text=[str(row[COLS["text"]])], return_tensors="pt", truncation=False)
            )
            with torch.inference_mode():
                feats = (
                    model.get_image_features(**inputs.to(device))
                    if image
                    else model.get_text_features(**inputs.to(device))
                )
            out.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()[0])
        return np.array(out)

    image_corpus = TASK == "text_image_retrieval"
    c, q = encode(corpus, image_corpus), encode(queries, not image_corpus)
    rows, doc_ids = [], ids(corpus)
    k = int(PARAMS.get("top_k", 10))
    check(k > 0, "top_k must be positive")
    for query, vector in zip(ids(queries), q, strict=True):
        scores = c @ vector
        order = sorted(range(len(c)), key=lambda i: (-float(scores[i]), doc_ids[i]))[:k]
        rows.extend(
            {"query_id": query, "document_id": doc_ids[i], "rank": rank, "score": float(scores[i])}
            for rank, i in enumerate(order, 1)
        )
    pd.DataFrame(rows).to_csv(ARTIFACT_DIR / "rankings.csv", index=False)
    print("Cross-modal ranking diagnostics only; relevance evaluation adapter not implemented")
    return rows


def run_task():
    if TASK == "retrieval":
        return run_retrieval()
    if TASK == "text_rendering":
        return run_rendering()
    if TASK in ("ocr", "document_extraction"):
        return run_documents()
    if TASK in ("image_captioning", "visual_qa"):
        return run_local_vision()
    if TASK in ("text_to_image", "image_editing"):
        return run_diffusion()
    if TASK in ("text_image_retrieval", "image_text_retrieval"):
        return run_clip()
    train, test = load_data("train"), load_data("test")
    key = COLS.get("text") or COLS.get("tokens")

    def canon(v):
        return json.dumps(v, ensure_ascii=False) if isinstance(v, list) else str(v)

    if key and set(train[key].map(canon)) & set(test[key].map(canon)):
        ack("TRAIN_TEST_OVERLAP", "Exact train/test inputs overlap")
    if TASK != "clustering":
        a, b = split_indices(train)
        fitted, heldout = train.iloc[a], train.iloc[b]
        guesses = predict_core(fitted, heldout, TASK)
        METRICS[CONFIG["metric"]] = evaluate_core(heldout, guesses, TASK)
        if TASK == "seq2seq":
            refs = strings(heldout, "target")
            sources = strings(heldout)
            METRICS["identity_cer"] = text_error(refs, sources)
        diagnostic = heldout.copy()
        diagnostic["baseline_prediction"] = [
            json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for v in guesses
        ]
        diagnostic.to_csv(ARTIFACT_DIR / "validation_predictions.csv", index=False)
    else:
        print("Unsupervised clustering: no supervised score is fabricated")
    predictions = predict_core(train, test, TASK)
    return write_predictions(test, predictions)
