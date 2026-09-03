"""Typed transport diagnostics and dependency-free public imports."""

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from dpt.transport import HistoryStatus, HistoryStatusHistogram
from dpt.transport.model import TransportError


def test_transport_simulation_import_keeps_optional_runtimes_lazy() -> None:
    source = Path(__file__).parents[2] / "python"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(source)!r}); "
        "from dpt.transport import simulate, prepare_simulation, SimulationResult; "
        "assert not {'warp', 'numpy', 'torch'}.intersection(sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True)


def test_terminal_histogram_is_a_typed_immutable_snapshot() -> None:
    counts: Any = [1, 2, 3, 4, 5, 6, 7]
    histogram = HistoryStatusHistogram(counts)
    counts[0] = 100
    assert histogram.total == 28
    assert histogram.count(HistoryStatus.ESCAPED) == 1
    assert histogram.count(HistoryStatus.ANGLE_BUDGET) == 7
    with pytest.raises(FrozenInstanceError):
        cast(Any, histogram).counts = (0,) * 7
    invalid: Any = 0
    with pytest.raises(TransportError, match="HistoryStatus"):
        histogram.count(invalid)


@pytest.mark.parametrize("counts", [(0,) * 6, (-1, 0, 0, 0, 0, 0, 0), (True, 0, 0, 0, 0, 0, 0)])
def test_terminal_histogram_rejects_invalid_counts(counts: Any) -> None:
    with pytest.raises(TransportError, match="seven nonnegative integer counts"):
        HistoryStatusHistogram(counts)
