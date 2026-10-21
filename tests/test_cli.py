import subprocess
import sys

from deepinfer_mac.cli import main


def test_help_and_portable_model_listing(capsys):
    assert main(["models", "--json"]) == 0
    assert "qwen3-4b" in capsys.readouterr().out
    result = subprocess.run([sys.executable, "-m", "deepinfer_mac.cli", "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert all(command in result.stdout for command in ("doctor", "models", "run", "serve", "bench"))


def test_reject_unsafe_bind_and_empty_prompt(capsys):
    assert main(["serve", "--host", "0.0.0.0"]) == 1
    assert "bearer token" in capsys.readouterr().err
    assert main(["run", ""]) == 1
    assert "empty" in capsys.readouterr().err
