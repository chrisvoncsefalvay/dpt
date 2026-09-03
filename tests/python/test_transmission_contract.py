"""The package boundary and policy validation do not require a CUDA runtime."""

import subprocess
import sys
from typing import Any

import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.transmission import (
    GradientRangeError,
    TransmissionError,
    TransmissionSpec,
    prepare_transmission,
)


def test_import_is_runtime_lazy() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dpt.transmission; assert 'warp' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("capacity", [-1, 2**31, True, 1.5])
def test_invalid_capacity_fails_before_runtime(capacity: Any) -> None:
    with pytest.raises(TransmissionError, match="max_pixels"):
        prepare_transmission(max_pixels=capacity)


@pytest.mark.parametrize("beam", ["none", "scalar"])
def test_active_beam_requires_device_storage(beam: Any) -> None:
    with pytest.raises(TransmissionError, match="active beam"):
        TransmissionSpec(beam=beam, active_beam=True)


def test_legacy_exceptions_participate_in_shared_boundary_hierarchy() -> None:
    assert issubclass(TransmissionError, ContractError)
    assert issubclass(GradientRangeError, NumericalError)
