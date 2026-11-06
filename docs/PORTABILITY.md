# Portable setup and offline transfer

## Supported baseline

Use Python 3.12 (recommended) or 3.13 on Windows, Linux or macOS. The core app needs no Google TensorFlow installation, CUDA, API credentials or model downloads. `uv.lock` locks direct and transitive Python dependencies for the supported Python range. Lock resolution is cross-platform; execution is tested locally on Linux. CI is configured for all three operating systems but is not claimed to have run remotely before a push.

Online setup on a new device:

```bash
uv python install 3.12
uv sync --locked
uv run tensorforge doctor
uv run tensorforge generate --config examples/classification.yaml --output runs/demo
uv run tensorforge verify --bundle runs/demo
```

PowerShell uses the same uv commands. For manual activation: `.venv\Scripts\Activate.ps1` on Windows; `source .venv/bin/activate` on Unix. Copy/clone this repository including uv.lock; do not transfer a virtual environment. Preserve relative data paths and use the project root as the working directory for CLI examples.

## Optional OCR and documents

```bash
uv sync --locked --extra ocr
```

OCR additionally requires the **Tesseract executable and selected trained language data**, which Python package installers cannot supply:

- Debian/Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-eng` (add `tesseract-ocr-ind` for Indonesian).
- macOS with Homebrew: `brew install tesseract tesseract-lang`.
- Windows: install a Tesseract distribution and add its executable folder to PATH. Confirm `tesseract --version` and `tesseract --list-langs` before use.

For a project-local alternative to system language packages, run `python scripts/stage_ocr.py` explicitly while online. It downloads pinned English/Indonesian data from the official tessdata_fast repository, verifies SHA-256 hashes, and records provenance under ignored `models/tesseract/`. Set `params.tessdata_path: models/tesseract` (as in examples/ocr.yaml), then copy that folder separately for offline use. The Tesseract executable is still required.

The Docker image includes English/Indonesian Tesseract and document extras. `pypdfium2` supplies its rendering library with the platform wheel. Unsupported platform wheels may require additional native setup. DOCX extraction currently concatenates paragraphs and table text; original interleaving, coordinates and layout are not preserved. The OCR template does whole-page OCR; it is not a layout/table/handwriting recognition adapter. These templates remain experimental until tested with actual local assets.

## Optional vision and generation

```bash
uv sync --locked --extra vision
```

This extra is large and includes PyTorch, Transformers, Diffusers, Accelerate and Safetensors. These are libraries, **not model weights**. CPU runs are possible but diffusion can be very slow; device-specific GPU/driver requirements remain external. Linux/Windows PyTorch and Torchvision use the explicit official CPU wheel index; macOS uses platform wheels from PyPI. Specialized GPU installations need a separate validated lock/environment. `device` is explicit (`cpu`, `cuda`, `mps`), with no fallback. Baseline diffusion uses float32.

Supported experimental model architectures are BLIP conditional captioning and BLIP QA, CLIP, Stable Diffusion, and Stable Diffusion InstructPix2Pix. Other architectures are not interchangeable. Stage a complete compatible `save_pretrained` directory under ignored `models/` while online, subject to its license/access rules. Include model configs, tokenizer/processor assets, Safetensors weights and diffusion component/scheduler configs. A model ID such as `organization/name` is not accepted as a substitute for a local directory.

Record model name, exact revision, source, license, file hashes, expected task, memory/VRAM and versions in a separate manifest. Copy that directory separately to the target device. Set `params.model_path` in an example YAML, validate with `--experimental`, review warnings and approve. The runtime sets Hugging Face offline flags, loads only local assets and never enables remote model code. Missing/incompatible assets fail rather than download. Runtime validation and semantic quality still require asset-specific tests.

## Offline Python wheelhouse

Prepare on an **online machine matching the destination OS, CPU architecture and Python minor version**. This is necessary because native wheels differ. First install uv, sync the chosen extras, and use:

```bash
# Core; export the exact locked dependency closure
uv export --locked --no-dev --no-emit-project --no-hashes --format requirements-txt --output-file requirements-core.txt
# Export selected extras instead if needed:
uv export --locked --no-dev --extra ocr --no-emit-project --no-hashes --format requirements-txt --output-file requirements-ocr.txt
# Include all Python libraries (large CPU vision wheels):
uv export --locked --no-dev --all-extras --no-emit-project --emit-index-url --no-hashes --format requirements-txt --output-file requirements-all.txt
uv build --wheel
uv run --with pip python -m pip download --only-binary=:all: -r requirements-core.txt --dest wheelhouse
uv run --with pip python -m pip download --only-binary=:all: pip --dest wheelhouse
```

Use the matching requirements file for the selected profile in the download/install commands. Pre-exported hashed requirements-core.txt, requirements-ocr.txt and requirements-all.txt are also committed.

Copy the repository, `dist/*.whl`, wheelhouse, requirements export and any external assets to the offline machine. A compatible Python interpreter must already be available; `uv python install` cannot download it offline. Create an environment and install without accessing an index:

```bash
python -m venv .venv
# Linux/macOS; Windows uses .venv\Scripts\python.exe
.venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements-core.txt
.venv/bin/python -m pip install --no-index --no-deps dist/tensorforge_workbench-0.1.0-py3-none-any.whl
.venv/bin/tensorforge doctor
```

Wheelhouse preparation is explicit and platform-specific, not a claim that arbitrary offline hardware has all dependencies. Build a checksum manifest (`sha256sum wheelhouse/* dist/* > checksums.sha256` on Linux) and verify after transfer. System Tesseract/language data, GPU drivers, fonts, model weights and licenses are separate. The repository intentionally does not contain gigabytes of wheel/model binaries.

## Docker

```bash
docker build -t tensorforge:0.1.0 .
docker run --rm -p 8501:8501 -v "$PWD:/workspace" tensorforge:0.1.0
```

The container runs as a non-root user and serves the UI on the container interface. The bind-mounted project lives at `/workspace`; generated files belong to container UID 1000. Prefer Docker Desktop on Windows/macOS, or match UID in Linux if needed. The image includes core/document extras, not vision models. Build while online, then use `docker save`/`docker load` to transfer an offline image. The Dockerfile is supplied, but image build/runtime verification requires an available Docker daemon. The OS base and system packages are not pinned to a digest/snapshot; uv.lock pins the Python dependency graph.
