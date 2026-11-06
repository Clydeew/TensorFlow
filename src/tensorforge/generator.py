import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path

import nbformat

from . import __version__
from .config import validate
from .data import fingerprint, resolve
from .errors import WorkbenchError
from .registry import TASKS

BOOTSTRAP = """from pathlib import Path
import json, os
BUNDLE_ROOT = Path.cwd().resolve()
CONFIG = json.loads((BUNDLE_ROOT / "config.json").read_text(encoding="utf-8"))
PROJECT_ROOT = (BUNDLE_ROOT / CONFIG.pop("_project_root")).resolve()
COLS = CONFIG.get("columns", {})
PARAMS = CONFIG.get("params", {})
TASK = CONFIG["task"]
SEED = int(CONFIG.get("seed", 42))
ARTIFACT_DIR = BUNDLE_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
METRICS = {}
"""
EXECUTE = """RESULT = run_task()
print("Artifacts:", ARTIFACT_DIR)
print("Metrics:", METRICS if METRICS else "Unavailable or not applicable")
"""
EXPORT = """import importlib.metadata, platform
report = {
    "task": TASK, "seed": SEED, "metric": CONFIG["metric"],
    "direction": "minimize" if CONFIG["metric"] in ("mae", "rmse", "cer", "wer") else "maximize",
    "measured_metrics": METRICS, "config": CONFIG,
    "python": platform.python_version(),
    "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "Pillow")},
}
(ARTIFACT_DIR / "execution_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
print("Execution report saved. Review validation_predictions.csv for errors where available.")
"""


def generate(cfg, root, output, *, overwrite=False, experimental=False):
    warnings = validate(cfg, root, experimental=experimental)
    output = Path(output).resolve()
    if output == Path(root).resolve() or output == Path.cwd().resolve():
        raise WorkbenchError("UNSAFE_OUTPUT", "Choose a new bundle directory, not the project root")
    if output.is_symlink() or output.exists():
        if not overwrite:
            raise WorkbenchError(
                "OUTPUT_EXISTS", f"Preserving {output}; use --overwrite for a generated bundle"
            )
        if not output.is_dir() or not (output / ".tensorforge-bundle").is_file():
            raise WorkbenchError(
                "UNSAFE_OVERWRITE", "Only marked TensorFlow bundle directories can be overwritten"
            )
    inputs = dict(cfg["data"])
    if cfg.get("submission"):
        inputs["submission_sample"] = cfg["submission"]["sample"]
    for key in ("model_path", "font_path", "tessdata_path"):
        if cfg.get("params", {}).get(key):
            inputs[key] = cfg["params"][key]
    for role, value in inputs.items():
        if value and resolve(root, value).is_relative_to(output):
            raise WorkbenchError("UNSAFE_OUTPUT", f"Output would contain input data ({role})")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".tensorforge-", dir=output.parent))
    try:
        config = json.loads(json.dumps(cfg))
        config["_project_root"] = os.path.relpath(root, output)
        (temp / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        runtime = resources.files("tensorforge").joinpath("templates/runtime.py").read_text(encoding="utf-8")
        task = TASKS[cfg["task"]]
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    "# TensorFlow baseline notebook\n\nTask: **" + task.id + "**. "
                    "Run from this bundle directory. This notebook contains its runtime and does not import the generator. "
                    "Model training/inference starts only when execution reaches the baseline cell."
                ),
                nbformat.v4.new_markdown_cell(
                    "## Configuration and provenance\n"
                    "Configuration is read as JSON data, never interpolated into Python source. "
                    "The relative project root is stored in config.json; adjust it if you move the bundle independently."
                ),
                nbformat.v4.new_code_cell(BOOTSTRAP),
                nbformat.v4.new_markdown_cell(
                    "## Dataset audit and methodology\n"
                    "Full data and IDs are validated at execution, with row/image/page limits. "
                    "The configured holdout is applied before vectorizer fitting. Learned preprocessing uses only its training partition. "
                    "Final predictions refit on all allowed training rows. Fixed-corpus retrieval fits on the declared search corpus. "
                    "Review generation_report.json warnings and inspect source/group/time boundaries."
                ),
                nbformat.v4.new_markdown_cell(
                    "## Baseline and metric implementation\n"
                    "All helper code follows. Classical baselines are validity floors. Local-model templates require matching pre-staged assets; "
                    "their runtime is experimental until asset-specific execution tests pass. No downloads are requested."
                ),
                nbformat.v4.new_code_cell(runtime),
                nbformat.v4.new_markdown_cell(
                    "## Execute validation and predictions\n"
                    "This cell trains or performs inference. Outputs are written under artifacts/. "
                    "Do not tune against final test labels. Task metrics are measured only when appropriate reference data exists."
                ),
                nbformat.v4.new_code_cell(EXECUTE),
                nbformat.v4.new_markdown_cell(
                    "## Diagnostics and artifact export\n"
                    "Inspect validation predictions, cluster terms, rankings or image manifests as appropriate. "
                    "No official submission is claimed unless an explicit supported sample contract was supplied."
                ),
                nbformat.v4.new_code_cell(EXPORT),
                nbformat.v4.new_markdown_cell(
                    "## Limitations\n"
                    "No leaderboard guarantee. Exact overlap audits do not guarantee semantic independence. "
                    "Nearest-neighbor text transformation and lexicon NER are deliberately weak. "
                    "Clustering has no fabricated accuracy. Optional visual outputs have file-contract checks, not verified semantic quality. "
                    "Competition eligibility of prior code, packages and weights must be confirmed independently."
                ),
            ],
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "language_info": {"name": "python"},
            },
        )
        nbformat.validate(notebook)
        nbformat.write(notebook, temp / "baseline.ipynb")
        hashes = {role: fingerprint(resolve(root, value)) for role, value in cfg["data"].items() if value}
        (temp / "generation_report.json").write_text(
            json.dumps(
                {
                    "generator_version": __version__,
                    "task": task.id,
                    "adapter_status": task.status,
                    "competition_eligibility": "pending",
                    "executed": False,
                    "warnings": warnings,
                    "input_hashes": hashes,
                    "seed": cfg.get("seed", 42),
                    "config": config,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # An exact snapshot of the actual installed runtime; uv.lock in the source repo is canonical.
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        if freeze.returncode == 0:
            pinned = "\n".join(
                line
                for line in freeze.stdout.splitlines()
                if "tensorforge" not in line.lower() and not line.startswith("-e ")
            )
        else:
            from importlib.metadata import distributions

            pinned = "\n".join(
                sorted(
                    f"{d.metadata['Name']}=={d.version}"
                    for d in distributions()
                    if "tensorforge" not in d.metadata["Name"].lower()
                )
            )
        if "+cpu" in pinned:
            pinned = "--extra-index-url https://download.pytorch.org/whl/cpu\n" + pinned
        (temp / "runtime_requirements.txt").write_text(pinned + "\n", encoding="utf-8")
        (temp / "README.md").write_text(
            "# Generated TensorFlow bundle\n\n"
            "Run Jupyter from this folder and open baseline.ipynb, or use `tensorforge verify --bundle PATH`.\n"
            "Data paths are relative to the original project root recorded in config.json (_project_root). "
            "Copy the whole project with its data, or update that relative path after moving. "
            "Notebook dependencies are listed in runtime_requirements.txt for this platform; the source repository's uv.lock "
            "is the portable dependency authority. Install matching optional extras and stage model assets before going offline.\n\n"
            "generate did not train, execute cells, perform inference, or write predictions. "
            "The artifacts directory is created only by explicit execution.\n",
            encoding="utf-8",
        )
        (temp / ".tensorforge-bundle").write_text("1\n", encoding="utf-8")
        if output.exists():
            # Preserve an existing bundle if publishing the replacement fails.
            backup = output.with_name(output.name + ".previous")
            if backup.exists():
                raise WorkbenchError("BACKUP_EXISTS", f"Remove or move {backup} before overwriting")
            output.rename(backup)
            try:
                temp.rename(output)
            except OSError:
                backup.rename(output)
                raise
            shutil.rmtree(backup)
        else:
            temp.rename(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return {
        "bundle": str(output),
        "notebook": str(output / "baseline.ipynb"),
        "warnings": warnings,
        "executed": False,
    }


def verify(bundle, timeout=600):
    from nbclient import NotebookClient

    bundle = Path(bundle).resolve()
    if not (bundle / ".tensorforge-bundle").is_file():
        raise WorkbenchError("NOT_A_BUNDLE", str(bundle))
    notebook = nbformat.read(bundle / "baseline.ipynb", as_version=4)
    try:
        client = NotebookClient(
            notebook, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(bundle)}}
        )
        manager = client.create_kernel_manager()
        # A user's global python3 kernelspec may point at an unrelated environment.
        # This changes only this manager's in-memory spec, never a system file.
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client.execute()
    except Exception as e:
        raise WorkbenchError("NOTEBOOK_EXECUTION_FAILED", str(e)) from e
    nbformat.write(notebook, bundle / "baseline.executed.ipynb")
    return {"bundle": str(bundle), "executed": True}
