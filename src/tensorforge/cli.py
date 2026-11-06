import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from . import __version__
from .config import load_config, validate
from .data import profile_file, propose
from .errors import WorkbenchError
from .generator import generate, verify
from .languages import PRESETS
from .registry import TASKS, catalog


def doctor():
    packages = [
        "numpy",
        "pandas",
        "sklearn",
        "pyarrow",
        "yaml",
        "nbformat",
        "nbclient",
        "ipykernel",
        "PIL",
        "streamlit",
    ]
    optional = ["pytesseract", "pypdf", "pypdfium2", "torch", "transformers", "diffusers"]
    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "core": {p: importlib.util.find_spec(p) is not None for p in packages},
        "optional": {p: importlib.util.find_spec(p) is not None for p in optional},
        "tesseract": shutil.which("tesseract"),
        "note": "Model weights and competition permission are not bundled. See docs/PORTABILITY.md.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="tensorflow-workbench",
        description="TensorFlow offline baseline workbench (independent project)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list-tasks", help="Show verified, experimental and planned coverage")
    ls.add_argument("--json", action="store_true")
    sub.add_parser("doctor", help="Check local dependency availability")
    languages = sub.add_parser("languages", help="Dataset language metadata and OCR codes")
    languages.add_argument("--json", action="store_true")
    inspect = sub.add_parser("inspect", help="Bounded manifest profile and unapproved configuration draft")
    inspect.add_argument("data")
    inspect.add_argument("--output", default="runs/inspection")
    inspect.add_argument("--limit", type=int, default=1000)
    for name in ("validate", "generate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--experimental", action="store_true")
        if name == "generate":
            p.add_argument("--output", required=True)
            p.add_argument("--overwrite", action="store_true")
    run = sub.add_parser(
        "verify", help="Explicitly execute generated notebook; may train or perform inference"
    )
    run.add_argument("--bundle", required=True)
    run.add_argument("--timeout", type=int, default=600)
    ui = sub.add_parser("ui", help="Open local browser interface")
    ui.add_argument("--port", type=int, default=8501)
    args = parser.parse_args(argv)
    try:
        if args.command == "languages":
            if args.json:
                print(json.dumps(PRESETS, indent=2, ensure_ascii=False))
            else:
                for tag, info in PRESETS.items():
                    print(
                        f"{tag:5} {info['name']:32} OCR: {info['ocr'] or 'unavailable'} | {', '.join(info['scripts'])}"
                    )
                print(
                    "Custom language tags are accepted. OCR codes do not imply support for every native script."
                )
            return 0
        if args.command == "list-tasks":
            if args.json:
                print(json.dumps(catalog(), indent=2))
            else:
                for task in TASKS.values():
                    print(f"{task.id:30} {task.status:13} {task.title}")
            return 0
        if args.command == "doctor":
            result = doctor()
            print(json.dumps(result, indent=2))
            return 0 if all(result["core"].values()) else 1
        if args.command == "inspect":
            if not 1 <= args.limit <= 100_000:
                raise WorkbenchError("INVALID_LIMIT", "Use --limit between 1 and 100000")
            output = Path(args.output).resolve()
            if output.exists():
                raise WorkbenchError("OUTPUT_EXISTS", str(output))
            profile = profile_file(args.data, args.limit)
            draft = propose(profile)
            # Portable path from draft's directory to the inspected data, without leaking into code.
            import os

            draft["project_root"] = os.path.relpath(Path(args.data).resolve().parent, output)
            output.mkdir(parents=True)
            (output / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
            (output / "config.draft.yaml").write_text(
                yaml.safe_dump(draft, sort_keys=False), encoding="utf-8"
            )
            result = {
                "output": str(output),
                "approved": False,
                "note": "Set data.test, verify task/metric/split and acknowledge warnings before approving",
            }
        elif args.command in ("generate", "validate"):
            cfg, root = load_config(args.config)
            if args.command == "validate":
                warnings = validate(cfg, root, require_approval=False, experimental=args.experimental)
                result = {
                    "valid": True,
                    "warnings": warnings,
                    "approved": cfg.get("review", {}).get("approved") is True,
                }
            else:
                result = generate(
                    cfg, root, args.output, overwrite=args.overwrite, experimental=args.experimental
                )
        elif args.command == "verify":
            result = verify(args.bundle, args.timeout)
        else:
            app = Path(__file__).with_name("ui.py")
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(app),
                    "--server.address=127.0.0.1",
                    "--server.headless=true",
                    "--theme.base=dark",
                    f"--server.port={args.port}",
                    "--browser.gatherUsageStats=false",
                ]
            )
        print(json.dumps(result, indent=2))
        return 0
    except WorkbenchError as e:
        print(json.dumps(e.as_dict()), file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as e:
        print(json.dumps(WorkbenchError("OPERATION_FAILED", str(e)).as_dict()), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
