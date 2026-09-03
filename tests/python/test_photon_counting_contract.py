"""Charge-cloud and pileup configuration can be imported without numerical extras."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_photon_counting_cold_import_does_not_load_numerical_runtimes() -> None:
    source = Path(__file__).resolve().parents[2] / "python"
    code = "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {str(source)!r})",
            "import dpt.photon_counting",
            "import dpt.examples.photon_counting",
            "extras = {'warp', 'numpy', 'scipy', 'torch'}",
            "assert not any(name.split('.')[0] in extras for name in sys.modules)",
            "print('cold import passed')",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code], check=True, capture_output=True, text=True
    )
    assert completed.stdout.strip() == "cold import passed"
