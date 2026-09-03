"""CUDA ordered diagnostic controls using actual resident workspaces."""

import pytest

from helpers.material_iteration_status_cases import (
    test_snapshot_matches_nested_status as test_snapshot_matches_nested_status,
)
from helpers.material_iteration_status_cases import (
    test_trial_preserves_first_status as test_trial_preserves_first_status,
)

pytestmark = pytest.mark.gpu


@pytest.fixture
def device() -> str:
    return "cuda:0"
