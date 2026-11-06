"""Coverage is explicit; catalogued scenarios are not implementation claims."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    backend: str
    status: str
    metrics: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    note: str = ""


TASKS = {}


def register(task):
    if task.id in TASKS:
        raise ValueError(f"Duplicate adapter: {task.id}")
    TASKS[task.id] = task


for ident, title, metrics in [
    ("text_classification", "Single-label text classification", ("accuracy", "macro_f1")),
    ("multilabel_classification", "Multiple labels per text", ("micro_f1", "macro_f1")),
    ("text_regression", "Numeric prediction from text", ("mae", "rmse", "spearman")),
    ("text_pair", "Pair classification or numeric similarity", ("macro_f1", "accuracy", "mae", "spearman")),
    ("clustering", "Document clustering", ("none",)),
    ("retrieval", "Query-to-document ranking", ("mrr", "recall_at_k", "ndcg")),
    ("seq2seq", "Normalization/transliteration nearest-neighbor baseline", ("cer", "exact_match")),
    ("token_classification", "Pretokenized BIO entity tagging", ("entity_f1",)),
    ("text_rendering", "Render exact text as an image", ("none",)),
]:
    register(Task(ident, title, "core", "verified", metrics))

for ident, title, backend, deps, metrics in [
    (
        "ocr",
        "Printed text OCR and scanned/digital PDFs",
        "ocr",
        ("pytesseract", "pypdf", "pypdfium2"),
        ("cer", "wer", "none"),
    ),
    (
        "document_extraction",
        "Digital PDF, DOCX, HTML and plain text extraction",
        "ocr",
        ("pypdf", "docx", "bs4"),
        ("none",),
    ),
    ("image_captioning", "Local image-to-caption model", "vision", ("torch", "transformers"), ("none",)),
    ("visual_qa", "Local image-plus-question answer model", "vision", ("torch", "transformers"), ("none",)),
    (
        "text_to_image",
        "Local text-to-image diffusion pipeline",
        "diffusion",
        ("torch", "diffusers", "transformers", "accelerate"),
        ("none",),
    ),
    (
        "image_editing",
        "Local instruction-based image editing",
        "diffusion",
        ("torch", "diffusers", "transformers", "accelerate"),
        ("none",),
    ),
    ("text_image_retrieval", "CLIP text-to-image ranking", "clip", ("torch", "transformers"), ("none",)),
    ("image_text_retrieval", "CLIP image-to-text ranking", "clip", ("torch", "transformers"), ("none",)),
]:
    register(
        Task(
            ident,
            title,
            backend,
            "experimental",
            metrics,
            deps,
            "Template implemented; full model/asset fixture execution not verified on this machine. Requires --experimental.",
        )
    )

PLANNED = {
    "hierarchical_classification": "Hierarchical/ordinal classification",
    "structured_extraction": "Relations, events, key-value extraction, entity linking/coreference",
    "question_answering": "Extractive/generative multi-document QA",
    "summarization": "Abstractive/extractive summarization",
    "translation": "Neural translation and text correction",
    "anomaly_detection": "Text/log anomalies and authorship analysis",
    "segmentation": "Text and topic boundary detection",
    "text_detection": "Text boxes, polygons and end-to-end spotting",
    "handwriting_recognition": "Dedicated handwriting recognition",
    "layout_analysis": "Layout and reading order",
    "table_extraction": "Table structure recognition",
    "form_understanding": "Receipts/invoices/forms",
    "formula_recognition": "Equation image to symbolic expression",
    "visual_grounding": "Phrase localization and segmentation",
    "multimodal_classification": "Text-plus-image classification/regression",
    "image_text_matching": "Pair similarity and contrastive matching",
    "reference_generation": "Reference/control-conditioned image generation",
    "inpainting": "Masked image editing and outpainting",
    "interleaved_generation": "Ordered mixed text/image output",
    "visual_structure": "Text to diagrams, charts, SVG or layout",
    "preference_scoring": "Image/caption/edit ranking under explicit rubric",
    "speech_to_text": "Audio transcription/translation",
    "text_to_speech": "Speech synthesis",
    "audio_qa": "Text-linked audio QA/retrieval",
    "video_captioning": "Video subtitles/captioning/summarization",
    "text_to_video": "Text-driven video generation/editing",
    "temporal_grounding": "Text to video time intervals",
    "structured_generation": "Code, SQL, graphs, scene/3D specifications",
    "metadata_prediction": "Text with tabular, temporal or graph features",
    "composed_pipeline": "Arbitrary multi-stage contracts (requires a new tested adapter)",
}
for ident, title in PLANNED.items():
    register(
        Task(
            ident,
            title,
            "unavailable",
            "planned",
            (),
            note="No runnable adapter yet; generation fails explicitly.",
        )
    )


def catalog():
    return [asdict(t) | {"competition_eligibility": "pending"} for t in TASKS.values()]
