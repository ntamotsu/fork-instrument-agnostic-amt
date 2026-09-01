from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    temporary_root = tmp_path_factory.mktemp("package-distribution")
    output_directory = temporary_root / "dist"
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to verify the project distribution"
    result = subprocess.run(
        [
            uv,
            "build",
            "--offline",
            "--no-python-downloads",
            "--python",
            sys.executable,
            "--wheel",
            "--out-dir",
            str(output_directory),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = tuple(output_directory.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_wheel_contains_runtime_package_data_without_repository_files(
    built_wheel: Path,
) -> None:
    with zipfile.ZipFile(built_wheel) as wheel:
        members = set(wheel.namelist())

    assert "instrument_agnostic_amt/__init__.py" in members
    assert "instrument_agnostic_amt/py.typed" in members
    assert "instrument_agnostic_amt/taxonomy/instrument_merge.json" in members
    assert "instrument_agnostic_amt/taxonomy/gm_instrument_classes.json" in members
    assert not any(member.startswith("tests/") for member in members)
    assert not any("__pycache__" in member for member in members)
    assert not any(member.endswith((".pyc", ".DS_Store")) for member in members)
    assert "Colab_Inference.ipynb" not in members
    assert "infer.py" not in members


def test_wheel_preserves_the_supported_python_range(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as wheel:
        metadata_path = next(
            member for member in wheel.namelist() if member.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser(policy=default).parsebytes(wheel.read(metadata_path))

    assert set(metadata["Requires-Python"].split(",")) == {">=3.10", "<3.15"}


@pytest.fixture(scope="module")
def installed_package(
    built_wheel: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    target = tmp_path_factory.mktemp("installed-distribution")
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to verify the project distribution"
    result = subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--offline",
            "--no-index",
            "--no-deps",
            "--no-python-downloads",
            "--python",
            sys.executable,
            "--target",
            str(target),
            str(built_wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return target


def _run_from_consumer_directory(
    script: str,
    *,
    installed_package: Path,
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    consumer_directory = tmp_path / "consumer"
    consumer_directory.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(
        [sys.executable, "-I", "-c", script, str(installed_package)],
        cwd=consumer_directory,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_wheel_imports_with_package_data_outside_the_checkout(
    installed_package: Path,
    tmp_path: Path,
) -> None:
    result = _run_from_consumer_directory(
        """
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

installation = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installation))
import instrument_agnostic_amt
from instrument_agnostic_amt.beat_chord import key_only_candidates
from instrument_agnostic_amt.inference import compat
from instrument_agnostic_amt.taxonomy import instrument_classes

package_path = Path(instrument_agnostic_amt.__file__).resolve()
assert package_path.is_relative_to(installation)
assert version("instrument-agnostic-amt") == "0.1.0"
assert instrument_agnostic_amt.VelocityEstimator
assert instrument_classes.get_instrument_class_id_by_name("drums") >= 0
taxonomy = files("instrument_agnostic_amt.taxonomy")
assert taxonomy.joinpath("instrument_merge.json").is_file()
assert taxonomy.joinpath("gm_instrument_classes.json").is_file()
assert key_only_candidates.amt_infer is compat
""",
        installed_package=installed_package,
        tmp_path=tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_installed_inference_module_exposes_help_outside_the_checkout(
    installed_package: Path,
    tmp_path: Path,
) -> None:
    result = _run_from_consumer_directory(
        """
import runpy
import sys

sys.path.insert(0, sys.argv[1])
sys.argv = ["instrument_agnostic_amt.cli.infer", "--help"]
runpy.run_module("instrument_agnostic_amt.cli.infer", run_name="__main__")
""",
        installed_package=installed_package,
        tmp_path=tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "Downloading" not in result.stdout
