import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_installed_console_alias():
    from importlib.metadata import distribution

    entries = {entry.name for entry in distribution("tensorforge-workbench").entry_points}
    assert {"tensorflow-workbench", "tensorforge"} <= entries


def test_platform_scripts_exist_and_are_not_binary():
    for name in [
        "setup-linux.sh",
        "setup-macos.sh",
        "setup-windows.ps1",
        "setup-windows.cmd",
        "tensorflow-workbench.sh",
        "tensorflow-workbench.ps1",
        "tensorflow-workbench.cmd",
        ".gitignore",
    ]:
        content = (ROOT / name).read_bytes()
        assert b"\x00" not in content
        assert content


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell syntax test")
def test_posix_setup_and_launcher_syntax():
    subprocess.run(
        ["bash", "-n", "setup-linux.sh", "setup-macos.sh", "tensorflow-workbench.sh"], cwd=ROOT, check=True
    )


def test_launcher_runs_from_another_directory(tmp_path):
    if os.name == "nt":
        command = ["cmd", "/c", str(ROOT / "tensorflow-workbench.cmd"), "list-tasks", "--json"]
    else:
        command = ["bash", str(ROOT / "tensorflow-workbench.sh"), "list-tasks", "--json"]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=True)
    assert "text_classification" in result.stdout
