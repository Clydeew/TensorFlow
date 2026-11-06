# Setup and CLI for Linux Windows and macOS

All three platforms install the same `tensorflow-workbench` CLI from the same uv.lock. The historical `tensorforge` command remains an alias. Python's package namespace stays tensorforge, independent of Google's TensorFlow library. The browser UI is optional; inspect, validate, generate, verify, doctor, languages and list-tasks work entirely from the terminal.

The bootstrap requires an existing Python 3 interpreter with venv support, or an installed uv executable. It obtains Python 3.12 using uv when needed, creates a local .venv, and installs locked dependencies without modifying system Python. Initial setup needs internet unless the interpreter/tool/cache or a platform-matched wheelhouse is already staged. See PORTABILITY.md for fully offline transfer.

| Platform | Setup core | Setup all Python extras | CLI launcher without activation |
|---|---|---|---|
| Linux | `bash setup-linux.sh` | `bash setup-linux.sh --extras all` | `./tensorflow-workbench.sh doctor` |
| macOS Intel or Apple Silicon | `bash setup-macos.sh` | `bash setup-macos.sh --extras all` | `./tensorflow-workbench.sh doctor` |
| Windows Command Prompt | `setup-windows.cmd` | `setup-windows.cmd --extras all` | `tensorflow-workbench.cmd doctor` |
| Windows PowerShell | `.\setup-windows.ps1` | `.\setup-windows.ps1 -Extras all` | `.\tensorflow-workbench.ps1 doctor` |

If PowerShell's existing execution policy prevents .ps1 execution, use the .cmd files (also callable from PowerShell). No script changes execution policy. On Unix, `bash tensorflow-workbench.sh ...` works if the checkout has lost executable bits. Paths containing spaces are quoted by the launchers. Start from the repository for setup; CLI launchers themselves locate the repository and work from other directories.

For document Python packages only, use `--extras ocr` or `-Extras ocr`. Fully cached setup adds `--offline` or `-Offline`. The all profile installs the optional libraries, not large pretrained vision weights or the system Tesseract executable. Run `python scripts/stage_ocr.py` while online to provision checked English/Indonesian/Javanese/Sundanese language packs under ignored models/tesseract.

## Terminal workflow

Linux/macOS:

```bash
./tensorflow-workbench.sh doctor
./tensorflow-workbench.sh languages
./tensorflow-workbench.sh list-tasks
./tensorflow-workbench.sh inspect examples/data/indonesia_train.csv --output runs/inspection-id
./tensorflow-workbench.sh validate --config examples/indonesia.yaml
./tensorflow-workbench.sh generate --config examples/indonesia.yaml --output runs/indonesia
# Explicit execution performs training/inference:
./tensorflow-workbench.sh verify --bundle runs/indonesia
# Optional browser interface:
./tensorflow-workbench.sh ui
```

Windows Command Prompt or PowerShell (prefix .\ in PowerShell):

```bat
tensorflow-workbench.cmd doctor
tensorflow-workbench.cmd languages
tensorflow-workbench.cmd validate --config examples/indonesia.yaml
tensorflow-workbench.cmd generate --config examples/indonesia.yaml --output runs/indonesia
tensorflow-workbench.cmd verify --bundle runs/indonesia
tensorflow-workbench.cmd ui
```

After activating .venv, the installed `tensorflow-workbench` console command works on all three platforms without a wrapper. You can also run `python scripts/run.py ...`. Stable JSON fields, warning IDs, task IDs and configuration keys are the same across platforms; input text, labels and artifacts retain their original language. Errors return nonzero status. `generate` never trains, runs OCR or produces predictions; `verify` explicitly executes the notebook.

Linux launcher and bootstrap are executed locally in acceptance checks. macOS/Windows launchers are supplied and included in the CI matrix; they are not claimed to have been executed on those OSes before remote CI runs. The setup does not install operating-system packages, GPU drivers, fonts or licensed model weights automatically.
