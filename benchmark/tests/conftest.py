"""Shared helpers of the tests of the runner and the report."""

import subprocess
import sys
from pathlib import Path

import pytest


BENCHMARK = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def script():
    """Return a function that runs a script of the benchmark.

    **Arguments:**

    - `name`: The script, such as `run.py`.
    - `arguments`: Its command-line arguments.

    **Returns:**

    The completed process, with its output captured as text.
    """

    def run(name, *arguments):
        return subprocess.run(
            [sys.executable, name, *map(str, arguments)],
            cwd=BENCHMARK,
            capture_output=True,
            text=True,
        )

    return run
