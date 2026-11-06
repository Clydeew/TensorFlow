# Task coverage

Implementation status is independent from competition eligibility, which is pending for every task. Verified means maintained core notebook fixtures pass, not competitive quality. Experimental templates are executable code with unverified pretrained-model assets; planned means generation is unavailable.

| ID | Task | Status | Backend |
|---|---|---|---|
| `text_classification` | Single-label text classification | verified | core |
| `multilabel_classification` | Multiple labels per text | verified | core |
| `text_regression` | Numeric prediction from text | verified | core |
| `text_pair` | Pair classification or numeric similarity | verified | core |
| `clustering` | Document clustering | verified | core |
| `retrieval` | Query-to-document ranking | verified | core |
| `seq2seq` | Normalization/transliteration nearest-neighbor baseline | verified | core |
| `token_classification` | Pretokenized BIO entity tagging | verified | core |
| `text_rendering` | Render exact text as an image | verified | core |
| `ocr` | Printed text OCR and scanned/digital PDFs | experimental | ocr |
| `document_extraction` | Digital PDF, DOCX, HTML and plain text extraction | experimental | ocr |
| `image_captioning` | Local image-to-caption model | experimental | vision |
| `visual_qa` | Local image-plus-question answer model | experimental | vision |
| `text_to_image` | Local text-to-image diffusion pipeline | experimental | diffusion |
| `image_editing` | Local instruction-based image editing | experimental | diffusion |
| `text_image_retrieval` | CLIP text-to-image ranking | experimental | clip |
| `image_text_retrieval` | CLIP image-to-text ranking | experimental | clip |
| `hierarchical_classification` | Hierarchical/ordinal classification | planned | unavailable |
| `structured_extraction` | Relations, events, key-value extraction, entity linking/coreference | planned | unavailable |
| `question_answering` | Extractive/generative multi-document QA | planned | unavailable |
| `summarization` | Abstractive/extractive summarization | planned | unavailable |
| `translation` | Neural translation and text correction | planned | unavailable |
| `anomaly_detection` | Text/log anomalies and authorship analysis | planned | unavailable |
| `segmentation` | Text and topic boundary detection | planned | unavailable |
| `text_detection` | Text boxes, polygons and end-to-end spotting | planned | unavailable |
| `handwriting_recognition` | Dedicated handwriting recognition | planned | unavailable |
| `layout_analysis` | Layout and reading order | planned | unavailable |
| `table_extraction` | Table structure recognition | planned | unavailable |
| `form_understanding` | Receipts/invoices/forms | planned | unavailable |
| `formula_recognition` | Equation image to symbolic expression | planned | unavailable |
| `visual_grounding` | Phrase localization and segmentation | planned | unavailable |
| `multimodal_classification` | Text-plus-image classification/regression | planned | unavailable |
| `image_text_matching` | Pair similarity and contrastive matching | planned | unavailable |
| `reference_generation` | Reference/control-conditioned image generation | planned | unavailable |
| `inpainting` | Masked image editing and outpainting | planned | unavailable |
| `interleaved_generation` | Ordered mixed text/image output | planned | unavailable |
| `visual_structure` | Text to diagrams, charts, SVG or layout | planned | unavailable |
| `preference_scoring` | Image/caption/edit ranking under explicit rubric | planned | unavailable |
| `speech_to_text` | Audio transcription/translation | planned | unavailable |
| `text_to_speech` | Speech synthesis | planned | unavailable |
| `audio_qa` | Text-linked audio QA/retrieval | planned | unavailable |
| `video_captioning` | Video subtitles/captioning/summarization | planned | unavailable |
| `text_to_video` | Text-driven video generation/editing | planned | unavailable |
| `temporal_grounding` | Text to video time intervals | planned | unavailable |
| `structured_generation` | Code, SQL, graphs, scene/3D specifications | planned | unavailable |
| `metadata_prediction` | Text with tabular, temporal or graph features | planned | unavailable |
| `composed_pipeline` | Arbitrary multi-stage contracts (requires a new tested adapter) | planned | unavailable |
