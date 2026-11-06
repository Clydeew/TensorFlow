# Indonesian English and regional languages

The browser interface and CLI use English only. Dataset language is configured independently and can be Indonesian, English or another language. Configuration field names, stable errors and CLI diagnostics retain their technical English identifiers so scripts work consistently on all platforms.

The core character TF-IDF baselines accept UTF-8 Unicode text without an English vocabulary, English stopword list, ASCII transliteration or accent stripping. This allows Indonesian, English, Javanese, Sundanese and other local-language data, including mixed-language/code-switched sentences. It does **not** establish competitive quality, automatic language detection, automatic translation or dialect understanding. Evaluate on representative local-language data using source/group-aware splits.

`examples/indonesia.yaml` is a runnable synthetic classification example containing Indonesian, English, Latin-script Javanese and Latin-script Sundanese, plus mixed Indonesian/English/Javanese rows. Text samples demonstrate contracts rather than a production language benchmark; local-language wording/dialect coverage should be reviewed by fluent speakers before reuse as evaluation material. Labels and IDs are preserved.

```yaml
language:
  primary: id
  additional: [en, jv, su]
  script: Latn
  normalization: NFC
  code_switching: true
  preserve_case: false
```

- `primary` and `additional` use two/three-letter language tags, optionally with subtags, such as id, en, jv, su, ban, bug, min, ace, bjn, bbc, mak, mad, sas or ms. Custom valid tags are accepted; no preset is required to process text.
- `script` is explicit metadata such as Latn, Java (Aksara Jawa), Sund, Bali, Bugi, Arab or mixed. It does not choose or validate a font/model automatically.
- `normalization` defaults NFC to make canonically equivalent Unicode sequences comparable without dropping diacritics. NFD or none are available explicitly. CER uses Unicode code points, not grapheme clusters; WER uses whitespace splitting. These conventions may differ from the official scorer, especially for combining marks and non-spaced scripts.
- `preserve_case: true` disables TF-IDF lowercasing when letter case matters. No Indonesian stemmer, slang dictionary, translation, or regional-language morphology is applied implicitly. Learned language preprocessing must be fitted only on allowed training data.
- `code_switching` documents the intended data; it does not run a language detector or pretend to annotate span-level languages.

Run `tensorflow-workbench languages` for presets and candidate OCR codes. Presets describe language/script metadata, not model-language verification. BLIP/CLIP/diffusion templates add MODEL_LANGUAGE_UNVERIFIED for explicitly configured non-English or mixed data; acknowledge it only after checking the selected model and task fixtures.

## OCR and fonts

`python scripts/stage_ocr.py` explicitly downloads pinned fast Tesseract language files for English (eng), Indonesian (ind), Javanese (jav) and Sundanese (sun), verifies hashes, and writes source/revision/license metadata. It never runs during notebook generation. Select `params.language: ind+eng+jav+sun` and `params.tessdata_path: models/tesseract` for a deliberate mixed-language OCR configuration. Validation checks that the selected data files are actually available. Adding more languages can change accuracy and performance; compare with task-specific references rather than always enabling every language.

The existence of jav or sun language packs is **not a claim of tested Aksara Jawa or Aksara Sunda recognition**. Native scripts require a suitable recognizer, appropriate text rendering/fonts and annotated script-specific fixtures. Current OCR remains experimental. Other regional languages without a staged OCR pack require a separate model/data source, not a silent fallback to Indonesian.

Deterministic text-image rendering requires an explicit local font for non-ASCII text. Choose a font covering the actual script and verify the rendered glyphs; Unicode input support alone cannot prevent missing-glyph boxes. Regional-language model weights/fonts are not bundled or guessed. See the official pinned [tessdata_fast source](https://github.com/tesseract-ocr/tessdata_fast/tree/87416418657359cb625c412a48b6e1d6d41c29bd) and [Tesseract data documentation](https://tesseract-ocr.github.io/tessdoc/Data-Files).
