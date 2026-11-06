# TensorFlow

An independent offline workbench for creating baseline notebooks from text and multimodal competition data. **This is not Google's TensorFlow library.** The Python distribution is `tensorforge-workbench` and the command/import name is `tensorforge`, so both projects can coexist.

TensorFlow inspects a local dataset, validates a human-reviewed task configuration, and generates a standalone editable Jupyter notebook. Generating a notebook never trains, evaluates, runs OCR, downloads models or writes predictions. Notebook execution is an explicit separate action.

## Language and operating system support

The interface and CLI use English. Dataset language is independent: character baselines preserve Unicode and accept Indonesian/English/Javanese/Sundanese and code-switched text, with explicit language/script configuration and a runnable `examples/indonesia.yaml`. Regional-language input support is separate from pretrained model/OCR/font competence. See [language contracts](docs/LANGUAGES.md).

Separate setup scripts are provided for Linux (`setup-linux.sh`), macOS (`setup-macos.sh`) and Windows (`setup-windows.ps1` / `setup-windows.cmd`). Each installs the same **tensorflow-workbench** CLI; root `.sh`, `.cmd` and `.ps1` launchers work without environment activation. See [OS setup and CLI](docs/SETUP.md).

## Quick start

If uv is not already installed, the portable bootstrap keeps its tools inside the project:

```bash
python3 scripts/setup.py          # Windows: python scripts/setup.py
python3 scripts/run.py ui
# Include all optional Python libraries:
python3 scripts/setup.py --extras all
```

Requires **Python 3.12 or 3.13** for the app. Python 3.12 is recommended and pinned in `.python-version`. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then from this repository:

```bash
uv python install 3.12
uv sync --locked
uv run tensorforge doctor
uv run tensorforge ui
```

The interface runs locally at http://127.0.0.1:8501. It reads local paths; keep it bound to localhost. No account or external API is needed.

Run the included example from the repository root:

```bash
uv run tensorforge validate --config examples/classification.yaml
uv run tensorforge generate --config examples/classification.yaml --output runs/demo
uv run tensorforge verify --bundle runs/demo
```

The fixture config is pre-approved only for the included synthetic demonstration data. Review and approve each real competition configuration yourself. Predictions and measured scores appear in `runs/demo/artifacts/` only after `verify` executes the notebook.

## What works

Nine core adapters have end-to-end notebook fixtures: single-label classification, multilabel classification, numeric text regression, text-pair classification/similarity, clustering, retrieval/ranking, nearest-neighbor text transformation, pretokenized BIO entity tagging, and deterministic text-to-image rendering. The baselines establish correctness rather than competitive performance.

Eight optional templates cover OCR/document extraction, BLIP captioning/visual QA, Stable Diffusion text-to-image, InstructPix2Pix editing, and CLIP text↔image retrieval. They are **experimental**: they require matching local model assets/system dependencies and are not model-execution-verified by the core test suite. Explicit `--experimental` opt-in is required. No vision weights are bundled or automatically downloaded. OCR language data can be staged explicitly with `python scripts/stage_ocr.py`.

The expanded scenario registry retains planned work such as table/layout/form extraction, neural translation/summarization, grounding, inpainting, mixed generation and text-linked speech/video. **The full SRS scenario catalog is not implemented.** Planned adapters refuse generation; installed packages alone do not enable a task. See [coverage](docs/COVERAGE.md) and `tensorforge list-tasks`.

## Workflow

```bash
uv run tensorforge inspect examples/data/train.csv --output runs/inspection
uv run tensorforge list-tasks
uv run tensorforge validate --config YOUR_CONFIG.yaml
uv run tensorforge generate --config YOUR_CONFIG.yaml --output runs/experiment
# Separate explicit execution; can train models or produce images:
uv run tensorforge verify --bundle runs/experiment --timeout 600
```

`inspect` reads a bounded sample and emits `profile.json` plus an **unapproved** `config.draft.yaml`. Set test data, task, fields, metric, split and output contract, then acknowledge stable warning IDs and set `review.approved: true`. `validate` reports warnings without requiring approval. `generate` enforces approval and acknowledgements.

Configuration paths resolve relative to the YAML's `project_root`, which itself resolves from the YAML's directory. Bundles store a relative path back to that root. Move the whole project with its data, or adjust `_project_root` in bundle/config.json. Open Jupyter from the bundle directory. Generated notebooks embed their helpers and do not import `tensorforge`.

Use `uv run --with jupyterlab jupyter lab` from a bundle directory, or use an already installed notebook editor with this project's virtual environment as kernel. CLI verification uses the same environment automatically.

Outputs are protected by default. `--overwrite` can replace only a marked generated bundle, never arbitrary directories. Existing datasets must not be inside the output bundle. A `submission` mapping supports a two-column ID/prediction sample CSV; other serializers remain generic outputs with explicit limitations. Retrieval outputs long-form rankings; rendered/generated images use a manifest.

## Another device and offline use

`pyproject.toml` declares dependencies; **`uv.lock` pins the complete dependency graph** and is committed. On another device run `uv sync --locked`; it resolves the correct platform wheels. Do not copy `.venv` between machines. Setup needs internet unless you prepare an offline package first. See [portable and offline setup](docs/PORTABILITY.md), including Windows/macOS/Linux, optional models, Docker and wheelhouse transfer.

```bash
# Optional document/OCR packages (plus system Tesseract for actual OCR)
uv sync --locked --extra ocr
# Optional local vision/generation packages (large download; weights separate)
uv sync --locked --extra vision
# All Python extras, still without model weights or Tesseract
uv sync --locked --all-extras
```

After installing optional extras, use `python scripts/run.py ...` to preserve that environment. If you use `uv run`, pass the same `--extra ocr` / `--extra vision` / `--all-extras` flags; a plain uv run synchronizes only the default dependencies.

## Development and GitHub

```bash
uv sync --locked --group dev
uv run ruff check .
uv run pytest
uv build
```

CI runs lint and tests on Linux, macOS and Windows with Python 3.12. Example data is synthetic. Real datasets, model weights, secrets, virtual environments and generated runs are ignored. The repository is initialized locally on `codex/initial-app`; **no remote is configured and nothing has been pushed**. See [GitHub setup](docs/GITHUB.md) when you are ready.

Based on the user's SRS version 1.2. See [requirements and limitations](docs/REQUIREMENTS.md). Competition eligibility of code, dependencies and weights remains separate from software correctness.
