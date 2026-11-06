# Local verification

Verified on 6 November 2026 with Python 3.12.14 on Linux x86_64. Remote CI and other hardware are not claimed as tested.

- Full local suite: **40 passed**, no skips/failures (125.26 seconds). This included nine core generated notebooks, pair regression, independent metric references, malicious config strings, malformed inputs, warning/approval gates, output protection, group/time boundaries, full-data runtime checks, Streamlit rendering/validation, and actual image/scanned-PDF OCR plus DOCX/HTML/TXT extraction.
- A final validation improvement rejects infeasible stratification before generation. Its new test and the affected configuration/runtime/UI suites passed: **28 passed** (8.98 seconds). The current suite therefore contains 41 tests; the additional test was included in the targeted run rather than a second full run.
- The core notebook matrix executed with external socket connections blocked; loopback connections were permitted for local Jupyter kernels. Generating a bundle produced no artifacts or predictions. Every generated notebook passed nbformat validation and imported no generator package.
- Built both wheel and source distribution with uv. Installed the wheel into a separate virtual environment after syncing the hashed core requirements from the local cache with `--offline`. The installed command passed doctor, generated the example and executed its notebook with the clean environment's interpreter.
- All core, OCR and vision Python packages are installed in the local project environment. BLIP, CLIP, Stable Diffusion and InstructPix2Pix model classes imported successfully. This verifies imports, not pretrained-model inference or output quality. No large vision weights have been downloaded.
- English and Indonesian Tesseract fast language files are staged locally with pinned source revision and checked SHA-256 hashes. The image/scanned-PDF OCR acceptance test executed English OCR successfully. Other language/domain quality has not been benchmarked.
- The local UI was opened in the browser and visually inspected at http://127.0.0.1:8501. Its health endpoint returned ok. Headless startup avoids Streamlit's interactive onboarding prompt; telemetry is disabled.
- Ruff check and formatter check passed. The dependency lock passed freshness checking. The portable setup script completed with `--extras all --offline` using the already populated local cache.
- Dockerfile and cross-platform CI are supplied. Docker execution was not tested because the local Docker daemon is unavailable. Windows/macOS CI runs only after the user publishes the repository.

Local raw JUnit outputs and generated notebook artifacts are under ignored runs/. Reproduce with `.venv/bin/pytest -q` (Windows: use the environment's python -m pytest). See docs/REQUIREMENTS.md and release-manifest.json for explicit implementation gaps and status boundaries.
