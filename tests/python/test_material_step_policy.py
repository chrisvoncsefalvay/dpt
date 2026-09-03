"""Acceptance uses realised scalar loss/slope; no device preparation required."""

import math

from dpt.material_reconstruction import MaterialReconstructionSettings, StepPolicy


def test_armijo_and_nonfinite_trials():
    policy = StepPolicy(MaterialReconstructionSettings())
    assert policy.accepts(1.0, 0.5, -1.0)
    for loss, slope in (
        (1.0, -1.0),
        (0.99999, -1.0),
        (0.5, 0.0),
        (0.5, math.nan),
        (math.inf, -1.0),
    ):
        assert not policy.accepts(1.0, loss, slope)
    assert policy.backtrack(0.3) == 0.15
    assert policy.grow(0.3) == 0.6
    assert policy.grow(0.8) == 1.0
