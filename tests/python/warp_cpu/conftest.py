"""Use the common explicit CPU/CUDA device contract for native geometry tests."""

import pytest


@pytest.fixture
def device(warp_device: str) -> str:
    return warp_device
