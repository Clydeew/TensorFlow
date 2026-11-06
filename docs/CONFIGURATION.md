# Configuration contracts

Start with a YAML under examples/. All mappings use `schema_version: 1`. `project_root` resolves relative to the YAML file; data/asset paths resolve relative to that root. Example YAML files deliberately specify `project_root: ..`.

- `task`: an exact registry ID, not a free-form instruction.
- `data`: `train` and `test` manifests for learned core tasks; `corpus`, `queries` and optional `qrels` for retrieval; `test` for OCR/rendering/generation.
- `columns`: map semantic roles to actual manifest fields. `id`, `text`, `target`, `text_b`, `tokens`, `tags`, `image`, `question`, `query_id`, `document_id`, `relevance` are recognized as appropriate. Paths live in manifest image fields; file names do not become executable code.
- `metric`: choose from the selected adapter's list (`tensorforge list-tasks --json`). Direction is explicit in execution reports. `none` means not applicable/unavailable, never zero accuracy.
- `split`: `stratified` for single-label classes; `random` with explicit warning acknowledgement; `group` with a source column; `time` with a strict timestamp boundary; `none` for clustering/inference. No fallback when stratification fails.
- `seed`: integer random seed. Reproducibility is bounded by the selected numerical/model backend; GPU image bytes are not promised identical across devices.
- `limits`: positive integers `max_rows` (default 100000), `max_image_pixels` (25000000) and `max_pages` (100). Over-limit execution fails, never silently drops output rows. These are safeguards, not a universal training time/memory guarantee.
- `review`: set `approved: true` only after reviewing the task, metric, split, assets and output contract. `acknowledged_warnings` contains exact stable IDs reported by validate. A sampled schema check is not full-data acceptance; execution revalidates records.
- `submission`: optional `sample`, `id_column`, `prediction_column`. Only two-column scalar/text submissions are supported. IDs must match, remain unique and preserve sample order. Generic predictions are always retained. Ranking/image manifests require their own serializers and cannot use this mapping.

## Task parameters

Classification/regression use character TF-IDF (2–5 grams, up to 30000 features) with a fixed LogisticRegression/Ridge baseline. Only trusted templates generate code; arbitrary Python callbacks/import strings are not supported.

- `text_pair`: `variant: classification` (default) or `regression`; `symmetric: true` uses sum/absolute difference/product features; false preserves direction. Pair regression supports mae or spearman and must not use stratified splits.
- `multilabel_classification`: JSON arrays of strings in target fields (native JSONL arrays or CSV JSON strings). `threshold` defaults to .5 and must stay in (0,1). Tune only on a declared calibration partition, not final test labels.
- `clustering`: `n_clusters`, default 3. Current support is KMeans cluster IDs and character-ngram top terms, not NMF topic mixtures.
- `seq2seq`: train-only nearest-neighbor target mapping and identity comparator. It is a weak normalization/transliteration baseline, not a neural summarizer or translator.
- `token_classification`: pretokenized `tokens` and aligned BIO `tags` arrays. Invalid training BIO fails; predicted I-tags with no compatible predecessor are reconstructed safely. Raw character-offset annotations are planned.
- `retrieval`: `top_k`, default 10; a fixed allowed corpus and optional qrels with nonnegative relevance in [0,20]. Ties break by document ID. Evaluation uses judged queries with positive relevance; missing judgments never produce fake metrics.
- `text_rendering`: `width`, `height`, optional `font_path`, `font_size`. Explicit local font required for non-ASCII. Text must fit; clipped content is rejected. This is exact text rendering, distinct from learned text-to-image synthesis.
- `ocr`: install OCR extra and system Tesseract, stage language data; `language` defaults eng and `tessdata_path` can point to models/tesseract. OCR of images/scanned PDFs is explicit execution; character/word error evaluation additionally needs reference text in `columns.target`.
- `document_extraction`: image-column manifest paths point to PDF/DOCX/HTML/TXT/MD/XML. Digital PDF text must be usable; scanned PDFs require the OCR adapter. No implicit fallback.
- `image_captioning` / `visual_qa`: compatible BLIP `model_path`, `device`, optional `max_new_tokens` (64). Generic scene QA, not arbitrary model architectures.
- `text_to_image`: compatible local Stable Diffusion `model_path`, `device`, `width/height` (multiples of 8), `steps` (1–100), `guidance`, `images_per_prompt` (1–8). Artifacts have generated safe filenames and an ID/variant/seed manifest.
- `image_editing`: local InstructPix2Pix model, source image and instruction text; also `image_guidance`. The source is resized to configured dimensions. Masks, inpainting and exact preservation constraints are not implemented by this adapter.
- `text_image_retrieval` / `image_text_retrieval`: CLIP model and processor, `top_k`, `device`; corpus/queries contain IDs plus image paths or text according to direction. Relevance-evaluation adapter remains planned.

Optional model templates require `--experimental`; the UI also has an explicit opt-in. No remote model IDs, API keys or implicit downloads are used. Planned catalog entries always refuse generation rather than emitting a placeholder notebook.
